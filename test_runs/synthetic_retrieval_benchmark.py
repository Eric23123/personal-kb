"""Offline retrieval benchmark for the synthetic Personal-KB corpus.

This benchmark is intentionally isolated from OpenViking/Hindsight. It reuses
Personal-KB's BM25 index and RRF implementation, while computing the dense side
from a local SentenceTransformer model over the 18 synthetic text sources.

Example:
    python test_runs/synthetic_retrieval_benchmark.py \
      --output test_runs/synthetic_benchmark_results.json

Use ``--skip-semantic`` only for a lexical smoke test. A normal benchmark must
run the semantic and hybrid strategies so embedding quality is measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieval.hybrid_retrieval import (  # noqa: E402
    _normalize_lexical_result,
    reciprocal_rank_fusion,
)
from scripts.retrieval.lexical_index import LexicalIndex, build_lexical_index  # noqa: E402

DEFAULT_SOURCES = PROJECT_ROOT / "data" / "test_sources"
DEFAULT_QUERIES = PROJECT_ROOT / "test_runs" / "synthetic_benchmark_queries.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "test_runs" / "synthetic_benchmark_results.json"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def _load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("queries")
    if not isinstance(payload, list) or not payload:
        raise ValueError("query file must contain a non-empty JSON list or a queries envelope")
    return payload


def _load_sources(source_dir: Path) -> list[Path]:
    paths = sorted(source_dir.glob("s*.txt"))
    if not paths:
        raise ValueError(f"no synthetic source files found in {source_dir}")
    return paths


def _source_name(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("file_name") or Path(str(item.get("source_path", ""))).name)


def _lexical_rank(index: Any, query: str, limit: int) -> list[dict[str, Any]]:
    return [_normalize_lexical_result(item) for item in index.search(query, top_k=limit)]


def _dense_rank(
    model: Any,
    paths: list[Path],
    query: str,
    limit: int,
    lexical_uri_by_name: dict[str, str],
) -> list[dict[str, Any]]:
    texts = [path.read_text(encoding="utf-8", errors="ignore") for path in paths]
    embeddings = model.encode([query, *texts], normalize_embeddings=True, show_progress_bar=False)
    query_embedding = embeddings[0]
    document_embeddings = embeddings[1:]
    scores = document_embeddings @ query_embedding
    ranked = sorted(range(len(paths)), key=lambda idx: float(scores[idx]), reverse=True)[:limit]
    results: list[dict[str, Any]] = []
    for idx in ranked:
        path = paths[idx]
        uri = lexical_uri_by_name.get(path.name, f"synthetic://{path.name}")
        results.append(
            {
                "uri": uri,
                "canonical_uri": uri.rsplit("/", 1)[0] if "/" in uri else uri,
                "score": round(float(scores[idx]), 6),
                "abstract": texts[idx][:500],
                "metadata": {"file_name": path.name, "source_path": path.name},
                "source": "dense",
            }
        )
    return results


def _required_coverage(path: Path, required: list[str]) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore").casefold()
    if not required:
        return 0.0
    return sum(str(term).casefold() in text for term in required) / len(required)


def _evaluate(ranked: list[dict[str, Any]], query: dict[str, Any], source_by_name: dict[str, Path]) -> dict[str, Any]:
    expected = str(query["expected_source"])
    names = [_source_name(item) for item in ranked]
    rank = names.index(expected) + 1 if expected in names else None
    coverage = _required_coverage(source_by_name[expected], [str(term) for term in query.get("required", [])])
    return {
        "expected_source": expected,
        "rank": rank,
        "recall_at_1": rank == 1,
        "recall_at_5": rank is not None and rank <= 5,
        "mrr": round(1.0 / rank, 6) if rank else 0.0,
        "required_term_coverage": round(coverage, 6),
        "top_sources": names[:5],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    ranks = [row["rank"] for row in rows if row.get("rank")]
    return {
        "queries": count,
        "recall_at_1": round(sum(bool(row["recall_at_1"]) for row in rows) / count, 6) if count else 0.0,
        "recall_at_5": round(sum(bool(row["recall_at_5"]) for row in rows) / count, 6) if count else 0.0,
        "mrr": round(sum(row["mrr"] for row in rows) / count, 6) if count else 0.0,
        "unfound": count - len(ranks),
        "mean_required_term_coverage": round(
            sum(float(row["required_term_coverage"]) for row in rows) / count, 6
        ) if count else 0.0,
    }


def run_benchmark(
    queries: list[dict[str, Any]],
    source_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    include_semantic: bool = True,
) -> dict[str, Any]:
    paths = _load_sources(source_dir)
    source_by_name = {path.name: path for path in paths}
    missing = sorted({str(query["expected_source"]) for query in queries} - set(source_by_name))
    if missing:
        raise ValueError(f"queries reference missing sources: {missing}")

    extra_metadata = {
        str(path.resolve()): {
            "source_type": "synthetic-benchmark",
            "course": "TEST-RETRIEVAL",
            "uri_source_path": path,
        }
        for path in paths
    }
    built_index = build_lexical_index(PROJECT_ROOT, extra_files=paths, extra_metadata=extra_metadata)
    synthetic_indices = [
        index for index, metadata in enumerate(built_index.metadata)
        if metadata.get("source_type") == "synthetic-benchmark"
    ]
    index = LexicalIndex(
        corpus_tokens=[built_index.corpus_tokens[index] for index in synthetic_indices],
        uris=[built_index.uris[index] for index in synthetic_indices],
        abstracts=[built_index.abstracts[index] for index in synthetic_indices],
        metadata=[built_index.metadata[index] for index in synthetic_indices],
    )
    lexical_uri_by_name = {
        str(metadata.get("file_name")): uri
        for uri, metadata in zip(index.uris, index.metadata)
        if metadata.get("file_name")
    }
    model = None
    if include_semantic:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)

    strategies: dict[str, list[dict[str, Any]]] = {
        "lexical_only": [],
        "semantic_only": [],
        "hybrid_rrf": [],
    }
    query_rows: list[dict[str, Any]] = []
    for query in queries:
        lexical_query = str(query.get("lexical") or query["query"])
        semantic_query = str(query.get("semantic") or query["query"])
        lexical = _lexical_rank(index, lexical_query, len(paths))
        lexical_eval = _evaluate(lexical, query, source_by_name)
        strategies["lexical_only"].append(lexical_eval)

        row: dict[str, Any] = {"id": query["id"], "lexical": lexical_eval}
        if model is not None:
            dense = _dense_rank(model, paths, semantic_query, len(paths), lexical_uri_by_name)
            dense_eval = _evaluate(dense, query, source_by_name)
            hybrid = reciprocal_rank_fusion(dense, lexical, k=60)
            hybrid_eval = _evaluate(hybrid, query, source_by_name)
            strategies["semantic_only"].append(dense_eval)
            strategies["hybrid_rrf"].append(hybrid_eval)
            row["semantic"] = dense_eval
            row["hybrid_rrf"] = hybrid_eval
        query_rows.append(row)

    aggregate = {name: _aggregate(rows) for name, rows in strategies.items() if rows}
    return {
        "benchmark_name": "personal-synthetic-retrieval-expansion",
        "version": 1,
        "corpus": {
            "source_dir": str(source_dir),
            "source_count": len(paths),
            "query_count": len(queries),
            "embedding_model": model_name if model is not None else None,
            "production_services_accessed": False,
        },
        "strategies": aggregate,
        "queries": query_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--semantic-model", default=DEFAULT_MODEL)
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    report = run_benchmark(
        _load_queries(args.queries),
        args.sources,
        model_name=args.semantic_model,
        include_semantic=not args.skip_semantic,
    )
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["strategies"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
