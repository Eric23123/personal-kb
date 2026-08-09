"""Dense + lexical hybrid retrieval for Personal KB.

Combines OpenViking dense results with a local BM25 lexical index using
Reciprocal Rank Fusion (RRF), then optionally reranks with a cross-encoder.

Usage:
    python scripts/retrieval/hybrid_retrieval.py search "kanban" --index-path data/lexical_index.json
    python scripts/retrieval/hybrid_retrieval.py ab "DG/(1+DGH)" --index-path data/lexical_index.json
"""


from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from ..retrieval.lexical_index import DEFAULT_ROOT, LexicalIndex, build_lexical_index, load_index, save_index
    from ..core.openviking_backend import (
        PERSONAL_NAMESPACE,
        PersonalOpenVikingBackend,
        validate_metadata_filters,
    )
    from . import reranker as reranker_mod
    from .source_reader import DEFAULT_MAX_L2_TOKENS
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from scripts.retrieval.lexical_index import DEFAULT_ROOT, LexicalIndex, build_lexical_index, load_index, save_index
    from scripts.core.openviking_backend import (
        PERSONAL_NAMESPACE,
        PersonalOpenVikingBackend,
        validate_metadata_filters,
    )
    import reranker as reranker_mod
    from scripts.retrieval.source_reader import DEFAULT_MAX_L2_TOKENS
DEFAULT_OPENVIKING_URL = "http://127.0.0.1:1934"
DEFAULT_INDEX_PATH = Path("data/lexical_index.json")
DEFAULT_RRF_K = 60
DEFAULT_DENSE_WEIGHT = 1.0
DEFAULT_LEXICAL_WEIGHT = 1.0
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_LEXICAL_LIMIT = 20
DEFAULT_TOP_K = 5
DEFAULT_MAX_SOURCES = 8
DEFAULT_RERANK_TOP = 0
DEFAULT_EXPERIMENTAL_RERANK_TOP = 20
DEFAULT_GENERATION_CANDIDATE_LIMIT = 16
DEFAULT_GENERATION_USE_RRF = False
DEFAULT_RERANK_SCORE_DROP = 0.15
DEFAULT_RERANK_RELATIVE_FLOOR = 0.08
DEFAULT_FUSED_SCORE_DROP = 0.005


def _canonical_uri(uri: str) -> str:
    """Return the directory-level (source-document) URI for deduplication.

    OpenViking may split a single source file into multiple chunk resources
    (e.g. ``foo.json`` → ``foo_1.md``, ``foo_2.md``, …).  Both the lexical
    index and the dense side reference the same *parent directory*, so we
    strip the trailing filename to merge them in RRF.
    """
    if not uri:
        return uri
    relative = uri.removeprefix(f"{PERSONAL_NAMESPACE}/")
    if "/" not in relative:
        # A file directly under the namespace is already its own resource.
        return uri
    # Keep everything up to the last "/" — the resource container directory.
    return uri.rsplit("/", 1)[0]


def _normalize_dense_result(hit: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenViking search hit into the canonical candidate shape."""
    return {
        "uri": hit.get("uri", ""),
        "canonical_uri": _canonical_uri(hit.get("uri", "")),
        "score": round(hit.get("score", 0.0), 4),
        "level": hit.get("level", 0),
        "abstract": (hit.get("abstract") or "")[:500],
        "overview": (hit.get("overview") or "")[:1200],
        "metadata": hit.get("metadata") or {},
        "source": "dense",
    }


def _normalize_lexical_result(hit: dict[str, Any]) -> dict[str, Any]:
    """Convert a BM25 hit into the canonical candidate shape."""
    return {
        "uri": hit.get("uri", ""),
        "canonical_uri": _canonical_uri(hit.get("uri", "")),
        "score": round(hit.get("score", 0.0), 4),
        "level": 0,
        "abstract": hit.get("abstract", "")[:500],
        "metadata": hit.get("metadata") or {},
        "source": "lexical",
    }


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    *,
    k: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
) -> list[dict[str, Any]]:
    """Merge dense and lexical rankings using RRF.

    score(uri) = weight_dense * sum(1/(k + rank_dense)) +
                 weight_lexical * sum(1/(k + rank_lexical))

    Results are returned sorted by fused score descending. Each result includes
    the list of source ranks that contributed to the score.

    Deduplication uses ``canonical_uri`` (directory-level) so that OpenViking
    chunk resources and lexical index entries for the same source document are
    merged correctly.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores: dict[str, float] = {}
    provenance: dict[str, list[dict[str, Any]]] = {}
    best_uri: dict[str, str] = {}

    def _process(results: list[dict[str, Any]], weight: float, source: str) -> None:
        for rank, item in enumerate(results, start=1):
            can = item.get("canonical_uri") or item.get("uri", "")
            if not can:
                continue
            scores[can] = scores.get(can, 0.0) + weight * (1.0 / (k + rank))
            provenance.setdefault(can, []).append({
                "source": source,
                "rank": rank,
                "score": item.get("score"),
            })
            if can not in best_uri:
                best_uri[can] = item.get("uri", can)

    _process(dense_results, dense_weight, "dense")
    _process(lexical_results, lexical_weight, "lexical")

    # Build a lookup for the best abstract per canonical URI.
    abstracts: dict[str, str] = {}
    levels: dict[str, int] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for item in dense_results:
        can = item.get("canonical_uri") or item.get("uri", "")
        abstract = item.get("abstract", "") or ""
        if can and len(abstract) > len(abstracts.get(can, "")):
            abstracts[can] = abstract
            levels[can] = item.get("level", 0)
            metadata[can] = item.get("metadata", {}) or {}
    for item in lexical_results:
        can = item.get("canonical_uri") or item.get("uri", "")
        abstract = item.get("abstract", "") or ""
        if can and len(abstract) >= len(abstracts.get(can, "")):
            # Lexical hits often contain the complete source excerpt while
            # dense hits may expose only a short or OCR-damaged abstract.
            # Give the reranker the most informative representation; lexical
            # wins equal-length ties because it is source-text based.
            abstracts[can] = abstract
            levels[can] = item.get("level", 0)
            metadata[can] = item.get("metadata", {}) or {}

    merged = []
    for can, fused_score in scores.items():
        merged.append({
            "uri": best_uri.get(can, can),
            "canonical_uri": can,
            "fused_score": round(fused_score, 4),
            "abstract": abstracts.get(can, ""),
            "metadata": metadata.get(can, {}),
            "level": levels.get(can, 0),
            "provenance": provenance.get(can, []),
        })

    merged.sort(key=lambda x: x["fused_score"], reverse=True)
    return merged


def _read_source_text(backend: PersonalOpenVikingBackend, uri: str) -> str:
    raw = backend.read(uri, limit=50000)
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False)


def _prepare_readable_candidates(
    backend: PersonalOpenVikingBackend,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    context_chars: int = reranker_mod.DEFAULT_CONTEXT_CHARS,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Read candidate sources, attach focused context, and drop unreadable URIs."""
    readable: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for original in candidates[:limit]:
        candidate = dict(original)
        uri = str(candidate.get("uri", ""))
        try:
            source_text = _read_source_text(backend, uri)
            candidate["source_readable"] = True
            candidate["source_text_chars"] = len(source_text)
            candidate["rerank_context"] = reranker_mod.build_query_focused_context(
                query, candidate, source_text, max_chars=context_chars
            )
            readable.append(candidate)
        except Exception as error:
            failures.append({"uri": uri, "error": repr(error)})
    return readable, failures


def _combined_rerank_query(dense_query: str, lexical_query: str) -> str:
    if dense_query.strip().casefold() == lexical_query.strip().casefold():
        return dense_query
    return f"{dense_query}\nExact retrieval terms: {lexical_query}"


def lexical_search(
    query: str,
    index_path: Path,
    *,
    limit: int = DEFAULT_LEXICAL_LIMIT,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search the local BM25 index and return normalized candidates."""
    index = load_index(index_path)
    return [
        _normalize_lexical_result(r)
        for r in index.search(
            query,
            top_k=limit,
            filters=validate_metadata_filters(filters),
        )
    ]


def ensure_lexical_index(
    index_path: Path,
    root: Path,
    *,
    force_rebuild: bool = False,
) -> LexicalIndex:
    """Return a loaded BM25 index, building it first if it is missing or stale."""
    index_path = Path(index_path).expanduser()
    if not force_rebuild and index_path.is_file():
        return load_index(index_path)
    index = build_lexical_index(root)
    save_index(index, index_path)
    return index


def _dedupe_by_canonical_uri(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the highest-scoring candidate per canonical URI.

    OpenViking splits large source files into multiple chunk resources, each
    with its own score and rank.  Without deduplication a document with 5
    chunks would accumulate 5 RRF contributions and unfairly outrank a
    single-chunk document at the same semantic relevance.

    The input is assumed to be sorted by descending score (OpenViking returns
    results pre-ranked).  We keep the first occurrence of each canonical URI.
    """
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in results:
        can = item.get("canonical_uri") or item.get("uri", "")
        if not can or can in seen:
            continue
        seen.add(can)
        deduped.append(item)
    return deduped


def build_generation_candidate_pool(
    dense_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    *,
    candidate_limit: int = DEFAULT_GENERATION_CANDIDATE_LIMIT,
    k: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    use_rrf: bool = DEFAULT_GENERATION_USE_RRF,
    force_rrf: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the candidate set handed to the active generation model.

    Small unions preserve both retrieval rankings directly. RRF is only a
    candidate-budget guard when the deduplicated union exceeds the limit; it
    is never used as the final evidence decision.
    """
    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    union: dict[str, dict[str, Any]] = {}
    for source_name, results in (("dense", dense_results), ("lexical", lexical_results)):
        for rank, item in enumerate(results, start=1):
            can = item.get("canonical_uri") or item.get("uri", "")
            if not can:
                continue
            candidate = union.get(can)
            if candidate is None:
                candidate = dict(item)
                candidate["retrieval_sources"] = []
                union[can] = candidate
            if source_name not in candidate["retrieval_sources"]:
                candidate["retrieval_sources"].append(source_name)
            candidate[f"{source_name}_rank"] = rank
            candidate[f"{source_name}_score"] = item.get("score")
            if len(item.get("abstract", "") or "") > len(candidate.get("abstract", "") or ""):
                candidate["abstract"] = item.get("abstract", "")
            if len(item.get("overview", "") or "") > len(candidate.get("overview", "") or ""):
                candidate["overview"] = item.get("overview", "")
            candidate["metadata"] = candidate.get("metadata") or item.get("metadata") or {}

    union_count = len(union)
    if force_rrf or (use_rrf and union_count > candidate_limit):
        fused = reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            k=k,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )[:candidate_limit]
        for item in fused:
            item["retrieval_sources"] = sorted({
                contribution["source"]
                for contribution in item.get("provenance", [])
            })
        policy = (
            f"rrf_top{candidate_limit}_forced"
            if force_rrf
            else f"rrf_top{candidate_limit}_overflow"
        )
        return fused, {
            "policy": policy,
            "candidate_union_count": union_count,
            "candidate_limit": candidate_limit,
            "rrf_k": k,
        }

    if union_count > candidate_limit:
        source_orders: dict[str, list[str]] = {"dense": [], "lexical": []}
        for source_name, results in (("dense", dense_results), ("lexical", lexical_results)):
            seen_source: set[str] = set()
            for item in results:
                can = item.get("canonical_uri") or item.get("uri", "")
                if can and can in union and can not in seen_source:
                    source_orders[source_name].append(can)
                    seen_source.add(can)

        selected: list[dict[str, Any]] = []
        selected_keys: set[str] = set()
        offsets = {"dense": 0, "lexical": 0}
        while len(selected) < candidate_limit:
            progressed = False
            for source_name in ("dense", "lexical"):
                order = source_orders[source_name]
                while offsets[source_name] < len(order) and order[offsets[source_name]] in selected_keys:
                    offsets[source_name] += 1
                if offsets[source_name] >= len(order):
                    continue
                can = order[offsets[source_name]]
                offsets[source_name] += 1
                selected.append(union[can])
                selected_keys.add(can)
                progressed = True
                if len(selected) >= candidate_limit:
                    break
            if not progressed:
                break

        return selected, {
            "policy": f"union_top{candidate_limit}_no_rrf",
            "candidate_union_count": union_count,
            "candidate_limit": candidate_limit,
            "rrf_k": None,
        }

    return list(union.values()), {
        "policy": "union",
        "candidate_union_count": union_count,
        "candidate_limit": candidate_limit,
        "rrf_k": None,
    }


def _safe_generation_metadata(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    return {
        str(key): value
        for key, value in metadata.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def _read_l1_overview(
    backend: PersonalOpenVikingBackend,
    uri: str,
) -> tuple[str, str | None]:
    """Read only an OpenViking overview resource, never arbitrary L2 content."""
    if not uri.endswith("/.overview.md"):
        return "", None
    try:
        raw = backend.read(uri, limit=8000)
        overview = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        return overview[:8000], None
    except Exception as error:
        return "", repr(error)


def build_generation_candidate_manifest(
    backend: PersonalOpenVikingBackend,
    candidates: list[dict[str, Any]],
    *,
    candidate_limit: int = DEFAULT_GENERATION_CANDIDATE_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Expose navigation metadata to the answer model without L2 snippets.

    The returned manifest deliberately contains candidate IDs, L0/L1 fields,
    canonical source URIs, metadata, and retrieval provenance. It does not
    include ``rerank_context`` or arbitrary source text.
    """
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates[:candidate_limit], start=1):
        uri = str(candidate.get("uri", ""))
        l1_overview = str(candidate.get("overview", "") or "")[:8000]
        if not l1_overview:
            l1_overview, error = _read_l1_overview(backend, uri)
            if error:
                failures.append({"uri": uri, "error": error})
        provenance = list(candidate.get("provenance", []) or [])
        dense_contribution = next(
            (item for item in provenance if item.get("source") == "dense"), {}
        )
        lexical_contribution = next(
            (item for item in provenance if item.get("source") == "lexical"), {}
        )
        manifest.append({
            "candidate_id": f"C{index:02d}",
            "canonical_uri": candidate.get("canonical_uri") or _canonical_uri(uri),
            "resource_uri": uri,
            "l0_abstract": str(candidate.get("abstract", "") or "")[:4000],
            "l1_overview": l1_overview,
            "metadata": _safe_generation_metadata(candidate.get("metadata")),
            "retrieval_provenance": {
                "sources": list(candidate.get("retrieval_sources", []) or []),
                "dense_rank": candidate.get("dense_rank", dense_contribution.get("rank")),
                "lexical_rank": candidate.get("lexical_rank", lexical_contribution.get("rank")),
                "dense_score": candidate.get("dense_score", dense_contribution.get("score")),
                "lexical_score": candidate.get("lexical_score", lexical_contribution.get("score")),
                "rrf_score": candidate.get("fused_score"),
                "contributions": provenance,
            },
        })
    return manifest, failures


def build_source_selection_guidance(candidate_count: int) -> str:
    """Return instructions embedded in the active model's answer context."""
    return (
        f"You have {candidate_count} candidate source manifests. Each manifest has a "
        "candidate_id, L0 abstract, L1 overview, metadata, canonical source URI, "
        "and retrieval provenance. Select the smallest complete evidence set before "
        "answering. Use multiple sources/candidates when complementary evidence is needed. "
        "Reject duplicate, merely related, or misleading distractors. Preserve the "
        "canonical URI and cite only candidates actually used. L0/L1 are navigation "
        "evidence. After selecting IDs, invoke read_selected_generation_sources with "
        f"those IDs; it reads selected L2 evidence under a {DEFAULT_MAX_L2_TOKENS:,}-token total budget "
        "before making detailed claims. State when the selected evidence is incomplete."
    )


DEFAULT_BLEND_RRF_WEIGHT = 0.70
DEFAULT_BLEND_RERANKER_WEIGHT = 0.30
DEFAULT_BLEND_SCORE_DROP = 0.15


def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def blend_candidate_scores(
    candidates: list[dict[str, Any]],
    *,
    rrf_weight: float = DEFAULT_BLEND_RRF_WEIGHT,
    reranker_weight: float = DEFAULT_BLEND_RERANKER_WEIGHT,
) -> list[dict[str, Any]]:
    """Blend normalized RRF and reranker scores without changing provenance.

    RRF scores and cross-encoder logits have unrelated scales. Min-max
    normalization is applied within the same candidate pool before the
    weighted blend, and fused score remains the deterministic tie-breaker.
    """
    if not 0 <= rrf_weight <= 1 or not 0 <= reranker_weight <= 1:
        raise ValueError("weights must be in the range [0, 1]")
    if abs((rrf_weight + reranker_weight) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1")
    if not candidates:
        return []
    if any("fused_score" not in item or "rerank_score" not in item for item in candidates):
        raise ValueError("every candidate must contain fused_score and rerank_score")

    fused = [float(item["fused_score"]) for item in candidates]
    reranked = [float(item["rerank_score"]) for item in candidates]
    fused_normalized = _minmax_normalize(fused)
    reranker_normalized = _minmax_normalize(reranked)
    blended: list[dict[str, Any]] = []
    for item, fused_value, reranker_value in zip(
        candidates, fused_normalized, reranker_normalized,
    ):
        candidate = dict(item)
        candidate["rrf_score_normalized"] = round(fused_value, 6)
        candidate["reranker_score_normalized"] = round(reranker_value, 6)
        candidate["blend_score"] = round(
            rrf_weight * fused_value + reranker_weight * reranker_value, 6
        )
        candidate["blend_weights"] = {
            "rrf": rrf_weight,
            "reranker": reranker_weight,
        }
        blended.append(candidate)
    blended.sort(
        key=lambda item: (
            item["blend_score"],
            float(item.get("fused_score", 0.0)),
            float(item.get("rerank_score", 0.0)),
        ),
        reverse=True,
    )
    return blended


def select_final_sources(
    candidates: list[dict[str, Any]],
    *,
    default_limit: int = DEFAULT_TOP_K,
    max_sources: int = DEFAULT_MAX_SOURCES,
    reranked: bool = True,
    score_drop: float | None = None,
    relative_floor: float | None = None,
    score_key: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a bounded source set using a score-drop cutoff.

    ``default_limit`` is a target budget, not a minimum: candidates below the
    calibrated threshold are excluded even when that returns fewer sources.
    Cross-encoder and RRF scores are on different scales, so the default
    drop is intentionally separate for each. These provisional thresholds are
    returned in the cutoff metadata for later benchmark calibration.
    """
    if default_limit < 1 or max_sources < default_limit:
        raise ValueError("source limits must satisfy 1 <= default_limit <= max_sources")
    if not candidates:
        return [], {"selected_count": 0, "cutoff_reason": "no_candidates"}
    score_key = score_key or ("rerank_score" if reranked and "rerank_score" in candidates[0] else (
        "fused_score" if "fused_score" in candidates[0] else "score"
    ))
    if score_key not in candidates[0]:
        raise ValueError(f"score key {score_key!r} is missing from candidates")
    if score_drop is None:
        if score_key == "rerank_score":
            score_drop = DEFAULT_RERANK_SCORE_DROP
        elif score_key == "blend_score":
            score_drop = DEFAULT_BLEND_SCORE_DROP
        else:
            score_drop = DEFAULT_FUSED_SCORE_DROP
    if relative_floor is None and score_key == "rerank_score":
        relative_floor = DEFAULT_RERANK_RELATIVE_FLOOR
    if relative_floor is not None and not 0 < relative_floor <= 1:
        raise ValueError("relative_floor must be in the range (0, 1]")
    if score_drop < 0:
        raise ValueError("score_drop must be non-negative")

    best_score = float(candidates[0].get(score_key, 0.0))
    threshold = best_score - score_drop
    relative_threshold = None
    if relative_floor is not None and best_score > 0:
        relative_threshold = best_score * relative_floor
        threshold = max(threshold, relative_threshold)
    selected: list[dict[str, Any]] = []
    cutoff_reason = "candidate_limit"
    for candidate in candidates:
        if len(selected) >= max_sources:
            cutoff_reason = "max_sources"
            break
        score = float(candidate.get(score_key, 0.0))
        if selected and score < threshold:
            cutoff_reason = "score_drop"
            break
        selected.append(candidate)

    return selected, {
        "selected_count": len(selected),
        "score_key": score_key,
        "best_score": best_score,
        "score_threshold": threshold,
        "score_drop": score_drop,
        "relative_floor": relative_floor,
        "relative_threshold": relative_threshold,
        "cutoff_reason": cutoff_reason,
        "default_limit": default_limit,
        "max_sources": max_sources,
    }


def hybrid_search(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    lexical_limit: int = DEFAULT_LEXICAL_LIMIT,
    top_k: int = DEFAULT_TOP_K,
    k: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    rerank_top: int = DEFAULT_RERANK_TOP,
    rerank_server_url: str = "",
    max_sources: int = DEFAULT_MAX_SOURCES,
    rerank_score_drop: float | None = None,
    dense_query: str | None = None,
    lexical_query: str | None = None,
    context_chars: int = reranker_mod.DEFAULT_CONTEXT_CHARS,
    blend_rrf_weight: float | None = None,
    blend_reranker_weight: float = DEFAULT_BLEND_RERANKER_WEIGHT,
    filters: dict[str, Any] | None = None,
    generation_mode: bool = False,
    candidate_limit: int = DEFAULT_GENERATION_CANDIDATE_LIMIT,
    use_rrf: bool = DEFAULT_GENERATION_USE_RRF,
    force_rrf: bool = False,
) -> dict[str, Any]:
    """Search, fuse, read, and optionally rerank with production-safe source context.

    RRF-only is the default. Pass ``rerank_top > 0`` explicitly for the
    experimental reranker path; the reranker is retained for future course-data
    evaluation but is not part of normal retrieval.
    """
    dense_query = dense_query or query
    lexical_query = lexical_query or query
    filters = validate_metadata_filters(filters)
    rerank_query = _combined_rerank_query(dense_query, lexical_query)

    if filters:
        dense_raw = backend.search(dense_query, limit=search_limit, filters=filters)
        lexical_results = lexical_search(
            lexical_query,
            index_path,
            limit=lexical_limit,
            filters=filters,
        )
    else:
        dense_raw = backend.search(dense_query, limit=search_limit)
        lexical_results = lexical_search(lexical_query, index_path, limit=lexical_limit)
    dense_all = [_normalize_dense_result(h) for h in dense_raw.get("resources", [])]
    dense_results = _dedupe_by_canonical_uri(dense_all)

    result = {
        "query": query,
        "dense_query": dense_query,
        "lexical_query": lexical_query,
        "rerank_query": rerank_query,
        "filters": filters,
        "namespace": PERSONAL_NAMESPACE,
        "mode": "dense_plus_lexical",
        "dense_candidates": len(dense_results),
        "lexical_candidates": len(lexical_results),
        "results": [],
        "source_read_failures": [],
    }

    if generation_mode:
        start = time.monotonic()
        candidate_pool, candidate_selection = build_generation_candidate_pool(
            dense_results,
            lexical_results,
            candidate_limit=candidate_limit,
            k=k,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
            use_rrf=use_rrf,
            force_rrf=force_rrf,
        )
        result["fusion_latency_ms"] = round((time.monotonic() - start) * 1000, 2)
        manifest, l1_failures = build_generation_candidate_manifest(
            backend,
            candidate_pool,
            candidate_limit=candidate_limit,
        )
        result["source_read_failures"].extend(l1_failures)
        result["mode"] = "generation_candidates"
        result["generation_candidate_pool"] = True
        result["candidate_selection"] = candidate_selection
        result["l2_read_contract"] = {
            "operation": "read_selected_generation_sources",
            "required_before_detailed_answer": True,
            "input": {"selected_candidate_ids": "list[str]"},
            "max_total_l2_tokens": DEFAULT_MAX_L2_TOKENS,
            "uri_policy": "resolve IDs from this manifest; reject arbitrary model URIs",
        }
        result["generation_guidance"] = build_source_selection_guidance(len(manifest))
        result["results"] = manifest
        return result

    start = time.monotonic()
    fused = reciprocal_rank_fusion(
        dense_results,
        lexical_results,
        k=k,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
    )
    result["fusion_latency_ms"] = round((time.monotonic() - start) * 1000, 2)

    if rerank_top > 0 and fused:
        context_start = time.monotonic()
        rerank_candidates, failures = _prepare_readable_candidates(
            backend,
            rerank_query,
            fused,
            limit=min(rerank_top, len(fused)),
            context_chars=context_chars,
        )
        result["source_read_failures"].extend(failures)
        result["context_latency_ms"] = round((time.monotonic() - context_start) * 1000, 2)
        rerank_start = time.monotonic()
        reranked = reranker_mod.rerank_candidates(
            rerank_query,
            rerank_candidates,
            top_k=len(rerank_candidates),
            server_url=rerank_server_url,
        ) if rerank_candidates else []
        rerank_ms = round((time.monotonic() - rerank_start) * 1000, 2)
        has_rerank_scores = any("rerank_score" in c for c in reranked)
        blend_active = blend_rrf_weight is not None and has_rerank_scores
        if blend_active:
            reranked = blend_candidate_scores(
                reranked,
                rrf_weight=blend_rrf_weight,
                reranker_weight=blend_reranker_weight,
            )
            result["mode"] = "dense_lexical_blend"
            result["blend"] = {
                "rrf_weight": blend_rrf_weight,
                "reranker_weight": blend_reranker_weight,
                "normalization": "candidate_pool_minmax",
            }
        else:
            result["mode"] = "dense_lexical_reranker" if has_rerank_scores else "dense_lexical_fallback"
            if blend_rrf_weight is not None:
                result["blend"] = {
                    "enabled": False,
                    "reason": "reranker_scores_unavailable",
                    "rrf_weight": blend_rrf_weight,
                    "reranker_weight": blend_reranker_weight,
                }
        result["rerank_latency_ms"] = rerank_ms if has_rerank_scores else None
        result["rerank_candidates"] = len(reranked)
        selected, cutoff = select_final_sources(
            reranked,
            default_limit=top_k,
            max_sources=max_sources,
            reranked=has_rerank_scores and not blend_active,
            score_drop=DEFAULT_BLEND_SCORE_DROP if blend_active else rerank_score_drop,
            score_key="blend_score" if blend_active else None,
        )
        result["results"] = selected
        result["source_cutoff"] = cutoff
    else:
        readable, failures = _prepare_readable_candidates(
            backend,
            rerank_query,
            fused,
            limit=min(len(fused), max_sources + top_k + 5),
            context_chars=context_chars,
        )
        result["source_read_failures"].extend(failures)
        result["mode"] = "dense_plus_lexical"
        selected, cutoff = select_final_sources(
            readable,
            default_limit=top_k,
            max_sources=max_sources,
            reranked=False,
        )
        result["results"] = selected
        result["source_cutoff"] = cutoff
    return result


def hybrid_ab_comparison(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    lexical_limit: int = DEFAULT_LEXICAL_LIMIT,
    top_k: int = DEFAULT_TOP_K,
    k: int = DEFAULT_RRF_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
) -> dict[str, Any]:
    """Compare dense-only, lexical-only, and fused retrieval on the same query."""
    dense_raw = backend.search(query, limit=search_limit)
    _dense_all = [_normalize_dense_result(h) for h in dense_raw.get("resources", [])]
    dense_results = _dedupe_by_canonical_uri(_dense_all)
    lexical_results = lexical_search(query, index_path, limit=lexical_limit)
    fused = reciprocal_rank_fusion(
        dense_results, lexical_results, k=k,
        dense_weight=dense_weight, lexical_weight=lexical_weight,
    )

    return {
        "query": query,
        "dense_top_uris": [r["uri"] for r in dense_results[:top_k]],
        "lexical_top_uris": [r["uri"] for r in lexical_results[:top_k]],
        "fused_top_uris": [r["uri"] for r in fused[:top_k]],
        "dense_results": dense_results[:top_k],
        "lexical_results": lexical_results[:top_k],
        "fused_results": fused[:top_k],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="hybrid_retrieval",
        description="Dense + BM25 lexical hybrid retrieval for Personal KB",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser("search", help="Run hybrid retrieval")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--index-path", default=str(DEFAULT_INDEX_PATH),
        help="Path to the BM25 index JSON",
    )
    search_parser.add_argument(
        "--url", default=DEFAULT_OPENVIKING_URL,
        help="OpenViking server URL",
    )
    search_parser.add_argument(
        "--search-limit", type=int, default=DEFAULT_SEARCH_LIMIT,
        help="Dense candidate count",
    )
    search_parser.add_argument(
        "--lexical-limit", type=int, default=DEFAULT_LEXICAL_LIMIT,
        help="Lexical candidate count",
    )
    search_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="Default final source count (default 5)",
    )
    search_parser.add_argument(
        "--rrf-k", type=int, default=DEFAULT_RRF_K,
        help="RRF smoothing constant",
    )
    search_parser.add_argument(
        "--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT,
        help="Weight for dense ranks in RRF",
    )
    search_parser.add_argument(
        "--lexical-weight", type=float, default=DEFAULT_LEXICAL_WEIGHT,
        help="Weight for lexical ranks in RRF",
    )
    search_parser.add_argument(
        "--rerank-top", type=int, default=DEFAULT_RERANK_TOP,
        help="Fused candidates to rerank; default 0 (RRF-only); set >0 to opt in",
    )
    search_parser.add_argument(
        "--rerank-server", default="",
        help="Persistent reranker server host:port (e.g. localhost:1940)",
    )
    search_parser.add_argument(
        "--max-sources", type=int, default=DEFAULT_MAX_SOURCES,
        help="Hard maximum final source documents (default 8)",
    )
    search_parser.add_argument(
        "--build", action="store_true",
        help="Rebuild the lexical index before searching",
    )
    search_parser.add_argument(
        "--root", default=str(DEFAULT_ROOT),
        help="Personal KB root directory (for --build)",
    )

    ab_parser = subparsers.add_parser("ab", help="Compare dense vs lexical vs fused")
    ab_parser.add_argument("query", help="Search query")
    ab_parser.add_argument(
        "--index-path", default=str(DEFAULT_INDEX_PATH),
        help="Path to the BM25 index JSON",
    )
    ab_parser.add_argument(
        "--url", default=DEFAULT_OPENVIKING_URL,
        help="OpenViking server URL",
    )
    ab_parser.add_argument(
        "--search-limit", type=int, default=DEFAULT_SEARCH_LIMIT,
        help="Dense candidate count",
    )
    ab_parser.add_argument(
        "--lexical-limit", type=int, default=DEFAULT_LEXICAL_LIMIT,
        help="Lexical candidate count",
    )
    ab_parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="Results to compare",
    )

    args = parser.parse_args()

    backend = PersonalOpenVikingBackend(base_url=args.url, timeout=30, root=Path.cwd())
    index_path = Path(args.index_path)

    if args.command == "search":
        if args.build:
            ensure_lexical_index(index_path, Path(args.root), force_rebuild=True)
        result = hybrid_search(
            backend, args.query,
            index_path=index_path,
            search_limit=args.search_limit,
            lexical_limit=args.lexical_limit,
            top_k=args.top_k,
            k=args.rrf_k,
            dense_weight=args.dense_weight,
            lexical_weight=args.lexical_weight,
            rerank_top=args.rerank_top,
            rerank_server_url=args.rerank_server,
            max_sources=args.max_sources,
        )
    elif args.command == "ab":
        result = hybrid_ab_comparison(
            backend, args.query,
            index_path=index_path,
            search_limit=args.search_limit,
            lexical_limit=args.lexical_limit,
            top_k=args.top_k,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
