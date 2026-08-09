"""Source-read benchmark for planner-driven Personal retrieval.

The benchmark compares rollback-safe hybrid RRF against an optional normalized
RRF/reranker blend. It evaluates the content returned by source reads rather
than treating short retrieval abstracts as the answer context.
"""

from __future__ import annotations
import sys
import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.openviking_backend import PersonalOpenVikingBackend
from scripts.retrieval.query_orchestrator import execute_planned_query
from scripts.retrieval.query_planner import QueryPlanner

DEFAULT_CONFIG = Path("retrieval_benchmark.json")
DEFAULT_INDEX = Path("data/lexical_index.json")
DEFAULT_URL = "http://127.0.0.1:1934"
DEFAULT_RRF_WEIGHT = 0.70
DEFAULT_RERANKER_WEIGHT = 0.30


def build_plan_response(query: dict[str, Any], retrieval_mode: str) -> str:
    """Build a deterministic planner fixture from the benchmark record."""
    lexical_terms = query.get("required", []) + query.get("source_hints", [])
    lexical = " ".join(dict.fromkeys(str(term) for term in lexical_terms))
    return json.dumps(
        {
            "intent": "document_lookup",
            "backends": ["openviking"],
            "queries": [
                {"text": lexical, "role": "lexical"},
                {"text": query["query"], "role": "semantic"},
            ],
            "filters": {},
            "retrieval_mode": retrieval_mode,
            "limit": 5,
        }
    )


def evaluate_source_reads(
    backend: PersonalOpenVikingBackend,
    result: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    required = [str(term).casefold() for term in query.get("required", [])]
    for rank, item in enumerate(result.get("results", [])[:5], start=1):
        uri = str(item.get("uri", ""))
        try:
            raw = backend.read(uri, limit=50000)
            text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
            error = None
        except Exception as exc:  # source-read failures are benchmark evidence
            text = ""
            error = repr(exc)
        lower = text.casefold()
        covered = [term for term in required if term in lower]
        coverage = len(covered) / len(required) if required else 0.0
        rows.append(
            {
                "rank": rank,
                "uri": uri,
                "coverage": round(coverage, 4),
                "covered_terms": covered,
                "chars": len(text),
                "error": error,
            }
        )

    coverages = [float(row["coverage"]) for row in rows]
    support_rank = next(
        (row["rank"] for row in rows if float(row["coverage"]) >= 0.5),
        None,
    )
    return {
        "top1_coverage": coverages[0] if coverages else 0.0,
        "best_coverage": max(coverages, default=0.0),
        "hit_at_5": max(coverages, default=0.0) >= 0.5,
        "top1_context_hit": bool(coverages and coverages[0] >= 0.5),
        "support_rank": support_rank,
        "read_errors": sum(row["error"] is not None for row in rows),
        "rows": rows,
    }


def aggregate_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {
            "queries": 0,
            "read_hit_at_5_rate": 0.0,
            "read_top1_context_hit_rate": 0.0,
            "mean_best_coverage": 0.0,
            "mean_top1_coverage": 0.0,
            "mrr": 0.0,
            "read_errors": 0,
        }
    reciprocal_ranks = [
        1.0 / evaluation["support_rank"] if evaluation.get("support_rank") else 0.0
        for evaluation in evaluations
    ]
    return {
        "queries": len(evaluations),
        "read_hit_at_5_rate": round(
            sum(bool(item["hit_at_5"]) for item in evaluations) / len(evaluations),
            4,
        ),
        "read_top1_context_hit_rate": round(
            sum(bool(item["top1_context_hit"]) for item in evaluations) / len(evaluations),
            4,
        ),
        "mean_best_coverage": round(
            statistics.mean(float(item["best_coverage"]) for item in evaluations),
            4,
        ),
        "mean_top1_coverage": round(
            statistics.mean(float(item["top1_coverage"]) for item in evaluations),
            4,
        ),
        "mrr": round(statistics.mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "read_errors": sum(int(item["read_errors"]) for item in evaluations),
    }


def probe_openviking_health(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    """Probe the configured service directly, bypassing ambient HTTP proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{url.rstrip('/')}/health", timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError(f"OpenViking health probe failed at {url}: {exc}") from exc
    if not payload.get("healthy", False):
        raise RuntimeError(f"OpenViking reported unhealthy at {url}: {payload}")
    return payload


def _run_mode(
    backend: PersonalOpenVikingBackend,
    query: dict[str, Any],
    *,
    index_path: Path,
    blend_rrf_weight: float | None,
    blend_reranker_weight: float,
) -> dict[str, Any]:
    retrieval_mode = "hybrid_rerank" if blend_rrf_weight is not None else "hybrid"
    planner = QueryPlanner()
    planning = planner.validate_model_response(
        query["query"],
        build_plan_response(query, retrieval_mode),
        trace_id=f"benchmark-{query['id']}-{retrieval_mode}",
    )
    result = execute_planned_query(
        planning,
        backend,
        index_path=index_path,
        blend_rrf_weight=blend_rrf_weight,
        blend_reranker_weight=blend_reranker_weight,
        enable_reranker=blend_rrf_weight is not None,
    )
    return {
        "plan": result["plan"],
        "trace": result["trace"],
        "result": result,
        "read_eval": evaluate_source_reads(backend, result, query),
    }


def run_benchmark(
    config: dict[str, Any],
    backend: PersonalOpenVikingBackend,
    *,
    index_path: Path,
    rrf_weight: float = DEFAULT_RRF_WEIGHT,
    reranker_weight: float = DEFAULT_RERANKER_WEIGHT,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for query in config["queries"]:
        no_rerank = _run_mode(
            backend,
            query,
            index_path=index_path,
            blend_rrf_weight=None,
            blend_reranker_weight=reranker_weight,
        )
        blended = _run_mode(
            backend,
            query,
            index_path=index_path,
            blend_rrf_weight=rrf_weight,
            blend_reranker_weight=reranker_weight,
        )
        rows.append(
            {
                "id": query["id"],
                "query": query,
                "no_rerank": no_rerank,
                "blend": blended,
            }
        )
        print(f"{query['id']} done", flush=True)

    return {
        "benchmark_name": config.get("benchmark_name"),
        "benchmark_version": config.get("version"),
        "gold_uri_labels_verified": bool(config.get("gold_uri_labels_verified", False)),
        "mode": "planner_source_read_rrf_vs_blend",
        "blend": {
            "rrf_weight": rrf_weight,
            "reranker_weight": reranker_weight,
            "normalization": "candidate_pool_minmax",
        },
        "queries": rows,
        "aggregate": {
            "no_rerank": aggregate_evaluations(
                [row["no_rerank"]["read_eval"] for row in rows]
            ),
            "blend": aggregate_evaluations(
                [row["blend"]["read_eval"] for row in rows]
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=Path("test_runs/retrieval_benchmark.json"))
    parser.add_argument("--rrf-weight", type=float, default=DEFAULT_RRF_WEIGHT)
    parser.add_argument("--reranker-weight", type=float, default=DEFAULT_RERANKER_WEIGHT)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    health = probe_openviking_health(args.url)
    backend = PersonalOpenVikingBackend(base_url=args.url, timeout=60)
    report = run_benchmark(
        config,
        backend,
        index_path=args.index_path,
        rrf_weight=args.rrf_weight,
        reranker_weight=args.reranker_weight,
    )
    report["openviking_url"] = args.url
    report["health"] = health
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
