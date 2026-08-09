"""Cross-encoder reranking layer for Personal KB retrieval.

Retrieves a broad candidate set from OpenViking dense search, then applies a
stronger cross-encoder to reorder the top candidates. Falls back to dense
ranking if the reranker is unavailable or times out.

Usage:
    # In-process (loads model each call, ~19s overhead)
    python scripts/retrieval/reranker.py search "feedback stability" --limit 20 --rerank-top 5
    python scripts/retrieval/reranker.py ab "kanban" --limit 20  # A/B comparison

    # Server mode (model loaded once, <1s per query):
    #   1. Start the server:  python scripts/retrieval/reranker_server.py --port 1940
    #   2. Use with --server: python scripts/retrieval/reranker.py search "kanban" --server localhost:1940

Default model: BAAI/bge-reranker-base (local, ~278MB, CPU-compatible).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

try:
    from ..core.openviking_backend import PersonalOpenVikingBackend, PERSONAL_NAMESPACE
except ImportError:  # pragma: no cover
    from scripts.core.openviking_backend import PersonalOpenVikingBackend, PERSONAL_NAMESPACE

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_OPENVIKING_URL = "http://127.0.0.1:1934"
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_RERANK_TOP = 5
DEFAULT_RERANKER_SERVER = ""  # empty = in-process; set to "host:port" for server mode
DEFAULT_CONTEXT_CHARS = 2400

_DOMAIN_TERMS = {
    "control": {"control", "closed", "loop", "feedback", "nyquist", "bode", "stability", "stable", "transfer", "gain", "phase", "frequency", "routh", "hurwitz", "kp", "pole", "zero"},
    "lean": {"lean", "kanban", "jidoka", "takt", "inventory", "throughput", "pull", "wip", "manufacturing", "little"},
    "personal": {"personal", "hindsight", "obsidian", "lecture", "transcript", "knowledge", "pipeline", "study", "note"},
}

_FAMILY_BY_DOMAIN = {
    "control": {"control", "technical_reference", "lecture", "homework", "exam"},
    "lean": {"reference", "lecture", "homework", "exam"},
    "personal": {"project_or_metadata", "lecture", "reference"},
}


def infer_query_domain(query: str) -> str:
    """Infer a coarse query domain for context labeling, never as a hard filter."""
    tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
    scores = {domain: len(tokens & terms) for domain, terms in _DOMAIN_TERMS.items()}
    domain, score = max(scores.items(), key=lambda item: item[1])
    return domain if score >= 2 else "general"


def infer_document_metadata(candidate: dict[str, Any], source_text: str = "") -> dict[str, str]:
    """Derive stable, explainable source signals from URI and source text."""
    uri = str(candidate.get("uri", ""))
    identity = f"{uri} {candidate.get('abstract', '')}".lower()
    context_head = str(source_text)[:1200].lower()
    combined = f"{identity} {context_head}"
    # Prefer source identity and abstract names over incidental words in a long body.
    if any(term in identity for term in ("freq_response", "nyquist", "bode", "control")):
        family = "technical_reference"
    elif any(term in identity for term in ("homework", "hw1", "hw2", "hw3", "assignment")):
        family = "homework"
    elif any(term in identity for term in ("midterm", "final", "exam", "sample-final")):
        family = "exam"
    elif any(term in identity for term in ("lecture", "transcript", "source_verification")):
        family = "lecture"
    elif any(term in identity for term in ("skill", "lean-", "book_overview", "methodology")):
        family = "reference"
    elif any(term in identity for term in ("readme", "plan", "config", "catalog", "modelfile", ".sh")):
        family = "project_or_metadata"
    elif any(term in combined for term in ("diagram", "frequency response")):
        family = "technical_reference"
    else:
        family = "unknown"
    course_match = re.search(r"\b([A-Z]{2,8}\s?\d{4}[A-Z]?)\b", combined, re.IGNORECASE)
    suffix = uri.rsplit("/", 1)[-1].lower()
    source_type = suffix.rsplit(".", 1)[-1] if "." in suffix else "resource"
    return {
        "document_family": family,
        "course": course_match.group(1).upper().replace(" ", " ") if course_match else "unknown",
        "source_type": source_type,
    }


def document_family_priority(query: str, metadata: dict[str, str]) -> int:
    """Return a bounded domain-family signal for reranking tie resolution."""
    domain = infer_query_domain(query)
    family = metadata.get("document_family", "unknown")
    if domain == "general":
        return 0
    return 1 if family in _FAMILY_BY_DOMAIN.get(domain, set()) else (-1 if family != "unknown" else 0)


def build_query_focused_context(
    query: str,
    candidate: dict[str, Any],
    source_text: str = "",
    *,
    max_chars: int = DEFAULT_CONTEXT_CHARS,
) -> str:
    """Build reranker context from query-relevant source passages, not a head slice."""
    source_text = str(source_text or "")
    candidate_excerpt = str(candidate.get("abstract", "") or "")
    if source_text and candidate_excerpt and candidate_excerpt not in source_text:
        text = f"{candidate_excerpt}\n{source_text}"
    else:
        text = source_text or candidate_excerpt or candidate.get("uri", "")
    text = str(text)
    metadata = infer_document_metadata(candidate, text)
    candidate["source_metadata"] = metadata
    query_domain = infer_query_domain(query)
    query_tokens = set(re.findall(r"[a-z0-9_]+", query.lower()))
    segments = [segment.strip() for segment in re.split(r"\n+|(?<=[.!?])\s+", text) if segment.strip()]
    scored = []
    for index, segment in enumerate(segments):
        segment_tokens = set(re.findall(r"[a-z0-9_]+", segment.lower()))
        overlap = len(query_tokens & segment_tokens)
        phrase_bonus = sum(2 for token in query_tokens if len(token) > 4 and token in segment.lower())
        scored.append((overlap + phrase_bonus, -index, segment))
    selected: list[str] = []
    used = 0
    for _score, _position, segment in sorted(scored, reverse=True):
        if used + len(segment) + 1 > max_chars - 180:
            continue
        selected.append(segment)
        used += len(segment) + 1
        if used >= max_chars - 180:
            break
    if not selected:
        selected = [text[: max_chars - 180]]
    prefix = (
        f"[query_domain={query_domain} document_family={metadata['document_family']} "
        f"course={metadata['course']} source_type={metadata['source_type']}]\n"
    )
    context = prefix + "\n".join(selected)
    return context[:max_chars]

# Module-level cache so the model loads once per process.
_reranker_model: Any = None
_reranker_model_name: str | None = None


def _get_reranker(model_name: str = DEFAULT_RERANKER_MODEL) -> Any:
    """Lazy-load the cross-encoder model, cached per process."""
    global _reranker_model, _reranker_model_name
    if _reranker_model is not None and _reranker_model_name == model_name:
        return _reranker_model
    from sentence_transformers import CrossEncoder
    _reranker_model = CrossEncoder(model_name)
    _reranker_model_name = model_name
    return _reranker_model


def _rerank_via_server(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    server_url: str,
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Rerank by POSTing candidates to the persistent reranker server."""
    payload = json.dumps({
        "query": query,
        "candidates": candidates,
        "top_k": top_k,
    }, ensure_ascii=False).encode("utf-8")

    url = f"http://{server_url.rstrip('/')}/rerank"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        latency = data.get("latency_ms")
        if latency is not None:
            # Attach server-side latency for the caller to use.
            for r in results:
                r.setdefault("_server_latency_ms", latency)
        return results
    except Exception:
        # Fallback to dense ranking if server is unreachable.
        return sorted(
            candidates,
            key=lambda c: c.get("fused_score", c.get("score", 0.0)),
            reverse=True,
        )[:top_k]


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    model_name: str = DEFAULT_RERANKER_MODEL,
    top_k: int = DEFAULT_RERANK_TOP,
    timeout_seconds: float = 30.0,
    server_url: str = "",
) -> list[dict[str, Any]]:
    """Rerank OpenViking search results with a cross-encoder.

    Args:
        query: The original search query.
        candidates: List of search result dicts with 'uri', 'score', 'abstract'.
        model_name: HuggingFace cross-encoder model name (in-process mode only).
        top_k: Number of results to return after reranking.
        timeout_seconds: If reranking exceeds this, fall back to dense ranking.
        server_url: If non-empty, use the persistent reranker server instead of
            loading the model in-process (e.g. "localhost:1940").

    Returns:
        Reranked list of top_k candidates with added 'rerank_score' field.
    """
    if not candidates:
        return []

    if top_k < 1:
        raise ValueError("top_k must be positive")

    # Rerank even when the candidate list is shorter than top_k. The caller
    # may intentionally provide a small fused set, and quality—not avoiding a
    # few milliseconds of work—is the default policy.

    # Server mode: delegate to the persistent reranker server.
    if server_url:
        return _rerank_via_server(query, candidates, top_k, server_url, timeout_seconds)

    # In-process mode: load the model and score locally.
    try:
        model = _get_reranker(model_name)
        # Build (query, document) pairs for the cross-encoder.
        pairs = [
            (query, c.get("rerank_context") or c.get("abstract", "") or c.get("uri", ""))
            for c in candidates
        ]
        scores = model.predict(pairs)

        # Attach raw scores and a bounded source-family adjustment. The family
        # signal only participates when candidates are close; it never overrides
        # a clearly stronger cross-encoder score.
        raw_scores = [float(score) for score in scores]
        max_raw = max(raw_scores, default=0.0)
        for candidate, score in zip(candidates, raw_scores):
            metadata = candidate.get("source_metadata") or infer_document_metadata(
                candidate, candidate.get("rerank_context", "")
            )
            candidate["source_metadata"] = metadata
            candidate["rerank_score_raw"] = score
            candidate["family_priority"] = document_family_priority(query, metadata)
            adjusted = score
            if max_raw - score <= 0.05:
                if candidate["family_priority"] > 0:
                    adjusted += 0.05
                elif candidate["family_priority"] < 0:
                    adjusted -= 0.02
            candidate["rerank_score"] = adjusted
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]

    except Exception:
        # Fallback: return the top-k by fused score for hybrid candidates,
        # otherwise the original dense score.
        return sorted(
            candidates,
            key=lambda c: c.get("fused_score", c.get("score", 0.0)),
            reverse=True,
        )[:top_k]


def search_with_rerank(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_top: int = DEFAULT_RERANK_TOP,
    model_name: str = DEFAULT_RERANKER_MODEL,
    use_reranker: bool = True,
    server_url: str = "",
) -> dict[str, Any]:
    """Search OpenViking, optionally rerank, return both paths for comparison."""
    raw = backend.search(query, limit=search_limit)
    candidates = []
    for hit in raw.get("resources", []):
        candidates.append({
            "uri": hit.get("uri", ""),
            "score": round(hit.get("score", 0.0), 4),
            "level": hit.get("level", 0),
            "abstract": (hit.get("abstract") or "")[:500],
            "metadata": hit.get("metadata") or {},
        })

    source_read_failures = []
    readable_candidates = []
    for candidate in candidates:
        try:
            raw_source = backend.read(candidate["uri"], limit=50000)
            source_text = raw_source if isinstance(raw_source, str) else json.dumps(raw_source, ensure_ascii=False)
            candidate["source_readable"] = True
            candidate["source_text_chars"] = len(source_text)
            candidate["rerank_context"] = build_query_focused_context(query, candidate, source_text)
            readable_candidates.append(candidate)
        except Exception as error:
            source_read_failures.append({"uri": candidate["uri"], "error": repr(error)})

    result = {
        "query": query,
        "namespace": PERSONAL_NAMESPACE,
        "total_candidates": len(candidates),
        "readable_candidates": len(readable_candidates),
        "source_read_failures": source_read_failures,
        "results": readable_candidates[:rerank_top] if not use_reranker else [],
    }

    if not use_reranker:
        result["mode"] = "dense_only"
        return result

    # Rerank
    start = time.monotonic()
    reranked = rerank_candidates(
        query, readable_candidates, model_name=model_name, top_k=rerank_top,
        server_url=server_url,
    )
    elapsed_ms = round((time.monotonic() - start) * 1000, 1)

    # Check if reranker actually fired (has rerank_score) or fell back.
    has_rerank_scores = any("rerank_score" in c for c in reranked)

    result["mode"] = "dense_plus_reranker" if has_rerank_scores else "dense_fallback"
    result["rerank_latency_ms"] = elapsed_ms if has_rerank_scores else None
    result["results"] = reranked
    return result


def cmd_ab_comparison(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    search_limit: int = DEFAULT_SEARCH_LIMIT,
    rerank_top: int = DEFAULT_RERANK_TOP,
    model_name: str = DEFAULT_RERANKER_MODEL,
    server_url: str = "",
) -> dict[str, Any]:
    """Run dense-only and dense+reranker side by side for comparison."""
    dense_result = search_with_rerank(
        backend, query, search_limit=search_limit, rerank_top=rerank_top,
        use_reranker=False,
    )
    rerank_result = search_with_rerank(
        backend, query, search_limit=search_limit, rerank_top=rerank_top,
        model_name=model_name, use_reranker=True, server_url=server_url,
    )

    # Compare top-k ordering
    dense_uris = [c["uri"] for c in dense_result["results"]]
    rerank_uris = [c["uri"] for c in rerank_result["results"]]
    order_changed = dense_uris != rerank_uris

    return {
        "query": query,
        "search_limit": search_limit,
        "rerank_top": rerank_top,
        "dense_only": {
            "top_uris": dense_uris,
            "results": dense_result["results"],
        },
        "dense_plus_reranker": {
            "mode": rerank_result["mode"],
            "rerank_latency_ms": rerank_result.get("rerank_latency_ms"),
            "top_uris": rerank_uris,
            "results": rerank_result["results"],
        },
        "order_changed": order_changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="reranker",
        description="Cross-encoder reranking for Personal KB OpenViking retrieval",
    )
    parser.add_argument(
        "command",
        choices=["search", "ab"],
        help="search: retrieve+rerank | ab: compare dense-only vs dense+reranker",
    )
    parser.add_argument("query", help="Search query")
    parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT,
                        help="Broad candidate set size (default 20)")
    parser.add_argument("--rerank-top", type=int, default=DEFAULT_RERANK_TOP,
                        help="Results to return after reranking (default 5)")
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL,
                        help=f"Cross-encoder model (default {DEFAULT_RERANKER_MODEL})")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip reranking, return dense-only results")
    parser.add_argument("--server", default=DEFAULT_RERANKER_SERVER,
                        help="Reranker server host:port (e.g. localhost:1940). "
                             "Empty = load model in-process")
    parser.add_argument("--url", default=DEFAULT_OPENVIKING_URL,
                        help="OpenViking server URL")
    parser.add_argument("--timeout", type=float, default=30,
                        help="Request timeout in seconds")
    args = parser.parse_args()

    backend = PersonalOpenVikingBackend(base_url=args.url, timeout=args.timeout)

    if args.command == "search":
        result = search_with_rerank(
            backend, args.query,
            search_limit=args.limit,
            rerank_top=args.rerank_top,
            model_name=args.model,
            use_reranker=not args.no_rerank,
            server_url=args.server,
        )
    elif args.command == "ab":
        result = cmd_ab_comparison(
            backend, args.query,
            search_limit=args.limit,
            rerank_top=args.rerank_top,
            model_name=args.model,
            server_url=args.server,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())