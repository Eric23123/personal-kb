"""Offline contract tests for the synthetic retrieval benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.retrieval.lexical_index import build_lexical_index
from test_runs.synthetic_retrieval_benchmark import (
    DEFAULT_QUERIES,
    DEFAULT_SOURCES,
    _load_queries,
    _load_sources,
    run_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_corpus_has_exactly_18_sources() -> None:
    paths = _load_sources(DEFAULT_SOURCES)
    assert [path.name for path in paths] == [f"s{index:02d}_" + path.name.split("_", 1)[1] for index, path in enumerate(paths, 1)]
    assert len(paths) == 18
    assert all(path.read_text(encoding="utf-8").strip() for path in paths)


def test_query_contract_covers_each_source_and_required_terms() -> None:
    queries = _load_queries(DEFAULT_QUERIES)
    paths = _load_sources(DEFAULT_SOURCES)
    source_by_name = {path.name: path for path in paths}

    assert len(queries) == 18
    assert {query["expected_source"] for query in queries} == set(source_by_name)
    assert len({query["id"] for query in queries}) == 18

    for query in queries:
        assert query["query"]
        assert 3 <= len(query["required"]) <= 6
        text = source_by_name[query["expected_source"]].read_text(encoding="utf-8").casefold()
        assert all(str(term).casefold() in text for term in query["required"]), query["id"]
        copied_terms = sum(
            str(term).casefold() in query["query"].casefold()
            for term in query["required"]
        )
        assert copied_terms <= max(1, len(query["required"]) // 2), query["id"]


def test_lexical_index_can_build_synthetic_sources_without_production_writes() -> None:
    paths = _load_sources(DEFAULT_SOURCES)
    metadata = {
        str(path.resolve()): {
            "source_type": "synthetic-benchmark",
            "course": "TEST-RETRIEVAL",
            "uri_source_path": path,
        }
        for path in paths
    }
    index = build_lexical_index(PROJECT_ROOT, extra_files=paths, extra_metadata=metadata)
    synthetic = [item for item in index.metadata if item.get("source_type") == "synthetic-benchmark"]

    assert len(synthetic) == 18
    assert all("viking://resources/personal-kb" in uri for uri in index.uris if uri in {
        item_uri for item_uri, item in zip(index.uris, index.metadata) if item.get("source_type") == "synthetic-benchmark"
    })


def test_lexical_smoke_benchmark_returns_all_queries() -> None:
    report = run_benchmark(_load_queries(DEFAULT_QUERIES), DEFAULT_SOURCES, include_semantic=False)
    assert report["corpus"]["source_count"] == 18
    assert report["corpus"]["query_count"] == 18
    assert report["corpus"]["production_services_accessed"] is False
    assert report["strategies"]["lexical_only"]["queries"] == 18
    assert len(report["queries"]) == 18
