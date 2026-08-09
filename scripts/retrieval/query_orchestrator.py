"""Execute validated Personal query plans without trusting planner output."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

try:
    from ..core.openviking_backend import PERSONAL_NAMESPACE, PersonalOpenVikingBackend
    from ..retrieval.hybrid_retrieval import (
        DEFAULT_EXPERIMENTAL_RERANK_TOP,
        build_source_selection_guidance,
        select_final_sources,
    )
    from ..retrieval.query_models import PlanningResult, QueryPlan
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from scripts.core.openviking_backend import PERSONAL_NAMESPACE, PersonalOpenVikingBackend
    from scripts.retrieval.hybrid_retrieval import (
        DEFAULT_EXPERIMENTAL_RERANK_TOP,
        build_source_selection_guidance,
        select_final_sources,
    )
    from scripts.retrieval.query_models import PlanningResult, QueryPlan

HindsightRecall = Callable[..., Any]


def execute_planned_query(
    planning: PlanningResult,
    backend: PersonalOpenVikingBackend,
    *,
    index_path: str | Path = "data/lexical_index.json",
    hindsight_recall: HindsightRecall | None = None,
    max_sources: int = 8,
    blend_rrf_weight: float | None = None,
    blend_reranker_weight: float = 0.30,
    enable_reranker: bool = False,
) -> dict[str, Any]:
    """Execute validated plan with one role-aware hybrid call per OpenViking plan."""
    plan, trace = planning.plan, planning.trace
    if max_sources < 1:
        raise ValueError("max_sources must be positive")
    merged: dict[str, dict[str, Any]] = {}
    hindsight_results: list[dict[str, Any]] = []
    generation_metadata: dict[str, Any] | None = None

    def merge_result(result: dict[str, Any], provenance: list[str]) -> None:
        for candidate in result.get("results", []):
            uri = candidate.get("canonical_uri") or candidate.get("uri", "")
            if not uri or not uri.startswith(PERSONAL_NAMESPACE):
                continue
            candidate = dict(candidate)
            candidate.setdefault("query_provenance", []).extend(provenance)
            existing = merged.get(uri)
            if existing is None or _candidate_score(candidate) > _candidate_score(existing):
                merged[uri] = candidate

    openviking_queries = [
        query for query in plan.queries
        if "openviking" in ((query.backend,) if query.backend else plan.backends)
    ]
    if openviking_queries:
        started = time.monotonic()
        record: dict[str, Any] = {
            "query": plan.original_query,
            "query_role": "hybrid" if len(openviking_queries) > 1 else openviking_queries[0].role,
            "queries": [
                {"text": query.text, "role": query.role}
                for query in openviking_queries
            ],
            "original_query": plan.original_query,
            "trace_id": plan.trace_id,
            "backend": "openviking",
            "filters": plan.filters.to_dict(),
            "status": "started",
        }
        try:
            lexical = next((q for q in openviking_queries if q.role == "lexical"), None)
            semantic = next((q for q in openviking_queries if q.role == "semantic"), None)
            retrieval_query = (
                lexical.text if lexical else semantic.text if semantic else plan.original_query
            )
            result = _execute_openviking(
                backend,
                retrieval_query,
                plan,
                index_path=index_path,
                dense_query=semantic.text if semantic else retrieval_query,
                lexical_query=lexical.text if lexical else retrieval_query,
                blend_rrf_weight=blend_rrf_weight,
                blend_reranker_weight=blend_reranker_weight,
                enable_reranker=enable_reranker,
                generation_mode=not enable_reranker,
                filters=plan.filters.to_dict(),
            )
            if result.get("generation_candidate_pool"):
                generation_metadata = {
                    "candidate_selection": result.get("candidate_selection", {}),
                    "l2_read_contract": result.get("l2_read_contract", {}),
                    "generation_guidance": result.get("generation_guidance", ""),
                }
            provenance = [query.text for query in openviking_queries]
            merge_result(result, provenance)
            record["result_count"] = len(result.get("results", []))
            record["mode"] = result.get("mode")
            record["source_read_failures"] = result.get("source_read_failures", [])
            record["status"] = "completed"
        except Exception as error:
            record["status"] = "failed"
            trace.add_error(f"openviking query failed: {error}")
        record["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
        trace.execution.append(record)

    for planned_query in plan.queries:
        selected_backends = (planned_query.backend,) if planned_query.backend else plan.backends
        for backend_name in selected_backends:
            if backend_name == "openviking":
                continue
            started = time.monotonic()
            record = {
                "query": planned_query.text,
                "query_role": planned_query.role,
                "original_query": plan.original_query,
                "trace_id": plan.trace_id,
                "backend": backend_name,
                "filters": plan.filters.to_dict(),
                "status": "started",
            }
            try:
                if backend_name == "hindsight":
                    if hindsight_recall is None:
                        raise RuntimeError("Hindsight backend requested but no recall callable configured")
                    memories = hindsight_recall(
                        planned_query.text,
                        plan.limit,
                        filters=plan.filters.to_dict(),
                    )
                    if isinstance(memories, dict):
                        memory_items = memories.get("results", [])
                        record["result_count"] = len(memory_items)
                    else:
                        memory_items = memories if isinstance(memories, list) else []
                        record["result_count"] = len(memory_items)
                    record["memories"] = memories
                    for item in memory_items:
                        normalized = dict(item) if isinstance(item, dict) else {"value": item}
                        normalized.setdefault("backend", "hindsight")
                        normalized.setdefault("query", planned_query.text)
                        hindsight_results.append(normalized)
                else:
                    raise RuntimeError(f"unsupported execution backend: {backend_name}")
                record["status"] = "completed"
            except Exception as error:
                record["status"] = "failed"
                trace.add_error(f"{backend_name} query failed: {error}")
            record["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
            trace.execution.append(record)

    candidates = sorted(merged.values(), key=_candidate_score, reverse=True)
    if generation_metadata is not None:
        final_sources = [dict(candidate) for candidate in candidates[:20]]
        for index, candidate in enumerate(final_sources, start=1):
            candidate["candidate_id"] = f"C{index:02d}"
        source_cutoff = dict(generation_metadata.get("candidate_selection", {}))
        source_cutoff["selected_count"] = len(final_sources)
        generation_guidance = generation_metadata.get("generation_guidance", "")
        if not generation_guidance:
            generation_guidance = build_source_selection_guidance(len(final_sources))
    elif candidates:
        final_sources, source_cutoff = select_final_sources(
            candidates,
            default_limit=plan.limit,
            max_sources=max_sources,
            reranked=any("rerank_score" in item for item in candidates),
        )
        generation_guidance = ""
    else:
        final_sources, source_cutoff = [], {}
        generation_guidance = ""
    trace.source_uris = [
        item.get("canonical_uri") or item.get("resource_uri") or item.get("uri", "")
        for item in final_sources
        if item.get("canonical_uri") or item.get("resource_uri") or item.get("uri")
    ]
    if trace.status == "planned" and any(item.get("status") == "failed" for item in trace.execution):
        trace.status = "partial_failure"
    return {
        "query": plan.original_query,
        "namespace": PERSONAL_NAMESPACE,
        "plan": plan.to_dict(),
        "trace": trace.to_dict(),
        "source_cutoff": source_cutoff,
        "candidate_selection": generation_metadata.get("candidate_selection", {}) if generation_metadata else {},
        "l2_read_contract": generation_metadata.get("l2_read_contract", {}) if generation_metadata else {},
        "generation_guidance": generation_guidance,
        "openviking_results": final_sources,
        "hindsight_results": hindsight_results,
        "results": final_sources if final_sources else hindsight_results,
    }


def _execute_openviking(
    backend: PersonalOpenVikingBackend,
    query: str,
    plan: QueryPlan,
    *,
    index_path: str | Path,
    dense_query: str | None = None,
    lexical_query: str | None = None,
    blend_rrf_weight: float | None = None,
    blend_reranker_weight: float = 0.30,
    enable_reranker: bool = False,
    generation_mode: bool = True,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if plan.retrieval_mode == "dense":
        raw = backend.search(dense_query or query, limit=plan.limit, filters=filters)
        return {"mode": "dense", "results": raw.get("resources", [])}
    return backend.hybrid_search(
        query,
        index_path=index_path,
        top_k=plan.limit,
        max_sources=8,
        dense_query=dense_query,
        lexical_query=lexical_query,
        blend_rrf_weight=blend_rrf_weight,
        blend_reranker_weight=blend_reranker_weight,
        generation_mode=generation_mode,
        candidate_limit=16,
        use_rrf=False,
        force_rrf=False,
        filters=filters,
        rerank_top=DEFAULT_EXPERIMENTAL_RERANK_TOP
        if enable_reranker and plan.retrieval_mode == "hybrid_rerank"
        else 0,
    )


def _candidate_score(candidate: dict[str, Any]) -> float:
    provenance = candidate.get("retrieval_provenance")
    if isinstance(provenance, dict) and provenance.get("rrf_score") is not None:
        return float(provenance["rrf_score"])
    return float(candidate.get("rerank_score", candidate.get("fused_score", candidate.get("score", 0.0))))


__all__ = ["execute_planned_query"]
