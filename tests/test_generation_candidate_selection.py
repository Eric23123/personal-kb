"""Tests for model-directed source selection candidate preparation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieval import hybrid_retrieval as hr
from scripts.retrieval.query_orchestrator import execute_planned_query
from scripts.retrieval.query_planner import QueryPlanner
from scripts.retrieval.source_reader import read_selected_generation_sources


def _hit(source: str, index: int, *, score: float = 1.0) -> dict:
    return {
        "uri": f"viking://resources/personal-kb/{source}/chunk_{index}.md",
        "canonical_uri": f"viking://resources/personal-kb/{source}",
        "score": score,
        "abstract": f"source {source} candidate {index}",
        "metadata": {"source_type": "lecture", "family": source},
        "source": "dense",
    }


def test_small_candidate_union_is_passed_without_rrf(monkeypatch):
    dense = [_hit("alpha", 1, score=0.9), _hit("beta", 1, score=0.8)]
    lexical = [_hit("beta", 2, score=5.0), _hit("gamma", 1, score=4.0)]

    def fail_rrf(*args, **kwargs):
        raise AssertionError("RRF must not run for a <=20 candidate union")

    monkeypatch.setattr(hr, "reciprocal_rank_fusion", fail_rrf)
    candidates, metadata = hr.build_generation_candidate_pool(
        dense, lexical, candidate_limit=20, k=60,
    )

    assert metadata["policy"] == "union"
    assert metadata["candidate_union_count"] == 3
    assert [item["canonical_uri"] for item in candidates] == [
        "viking://resources/personal-kb/alpha",
        "viking://resources/personal-kb/beta",
        "viking://resources/personal-kb/gamma",
    ]
    assert candidates[1]["retrieval_sources"] == ["dense", "lexical"]


def test_overflow_candidate_union_uses_explicit_rrf_and_caps_at_twenty(monkeypatch):
    dense = [_hit(f"dense_{i}", i, score=1.0 - i / 100) for i in range(15)]
    lexical = [_hit(f"lexical_{i}", i, score=10.0 - i) for i in range(15)]
    calls = []
    original = hr.reciprocal_rank_fusion

    def record_rrf(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(hr, "reciprocal_rank_fusion", record_rrf)
    candidates, metadata = hr.build_generation_candidate_pool(
        dense, lexical, candidate_limit=20, k=60, use_rrf=True,
    )

    assert calls
    assert metadata["policy"] == "rrf_top20_overflow"
    assert metadata["candidate_union_count"] == 30
    assert len(candidates) == 20
    assert all("fused_score" in item for item in candidates)


def test_force_rrf_always_fuses_even_when_union_fits():
    dense = [_hit("dense", 1, score=0.9)]
    lexical = [_hit("lexical", 1, score=4.0)]
    candidates, metadata = hr.build_generation_candidate_pool(
        dense, lexical, candidate_limit=12, force_rrf=True,
    )

    assert metadata["policy"] == "rrf_top12_forced"
    assert metadata["rrf_k"] == 60
    assert len(candidates) == 2
    assert all("fused_score" in item for item in candidates)


def test_no_rrf_union_caps_without_fusion(monkeypatch):
    dense = [_hit(f"dense_{i}", i, score=1.0 - i / 100) for i in range(10)]
    lexical = [_hit(f"lexical_{i}", i, score=10.0 - i) for i in range(10)]

    monkeypatch.setattr(
        hr,
        "reciprocal_rank_fusion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("RRF must be disabled")),
    )
    candidates, metadata = hr.build_generation_candidate_pool(
        dense, lexical, candidate_limit=16, use_rrf=False,
    )

    assert metadata["policy"] == "union_top16_no_rrf"
    assert metadata["rrf_k"] is None
    assert len(candidates) == 16
    assert all("fused_score" not in item for item in candidates)


def test_source_selection_guidance_requires_minimal_complete_evidence():
    guidance = hr.build_source_selection_guidance(candidate_count=7)

    assert "smallest complete evidence set" in guidance
    assert "multiple sources" in guidance
    assert "distractors" in guidance
    assert "cite only" in guidance.lower()
    assert "7 candidate" in guidance
    for field in ("candidate_id", "L0", "L1", "canonical source URI", "retrieval provenance"):
        assert field in guidance
    assert "read_selected_generation_sources" in guidance


def test_orchestrator_requests_generation_candidates_and_returns_guidance(tmp_path):
    import json

    calls = []

    class Backend:
        def hybrid_search(self, query, **kwargs):
            calls.append((query, kwargs))
            return {
                "mode": "generation_candidates",
                "generation_candidate_pool": True,
                "candidate_selection": {
                    "policy": "union",
                    "candidate_union_count": 2,
                },
                "generation_guidance": "select the smallest complete evidence set",
                "l2_read_contract": {
                    "operation": "read_selected_generation_sources",
                    "required_before_detailed_answer": True,
                },
                "results": [
                    {"uri": "viking://resources/personal-kb/doc/a", "score": 0.9},
                    {"uri": "viking://resources/personal-kb/doc/b", "score": 0.8},
                ],
            }

    response = {
        "intent": "document_lookup",
        "backends": ["openviking"],
        "queries": [{"text": "control stability", "role": "semantic"}],
        "filters": {},
        "retrieval_mode": "hybrid",
        "limit": 5,
    }
    planning = QueryPlanner(lambda _prompt: json.dumps(response)).plan("control stability")
    result = execute_planned_query(planning, Backend(), index_path=tmp_path / "index.json")

    assert calls[0][1]["generation_mode"] is True
    assert calls[0][1]["candidate_limit"] == 16
    assert calls[0][1]["use_rrf"] is False
    assert calls[0][1]["force_rrf"] is False
    assert result["candidate_selection"]["policy"] == "union"
    assert "smallest complete evidence set" in result["generation_guidance"]
    assert result["l2_read_contract"]["operation"] == "read_selected_generation_sources"
    assert result["l2_read_contract"]["required_before_detailed_answer"] is True
    assert len(result["results"]) == 2


def test_hybrid_generation_mode_returns_readable_union_without_rrf(tmp_path, monkeypatch):
    class Backend:
        def search(self, query, *, limit, filters=None):
            return {
                "resources": [
                    {
                        "uri": "viking://resources/personal-kb/dense-a/chunk.md",
                        "score": 0.9,
                        "abstract": "dense source",
                    },
                    {
                        "uri": "viking://resources/personal-kb/dense-b/chunk.md",
                        "score": 0.8,
                        "abstract": "second dense source",
                    },
                ]
            }

        def read(self, uri, *, limit):
            return f"readable source text for {uri}"

    lexical = [
        {
            "uri": "viking://resources/personal-kb/lexical-c/chunk.md",
            "canonical_uri": "viking://resources/personal-kb/lexical-c",
            "score": 4.0,
            "abstract": "lexical source",
            "metadata": {},
            "source": "lexical",
        }
    ]
    monkeypatch.setattr(hr, "lexical_search", lambda *args, **kwargs: lexical)
    monkeypatch.setattr(
        hr,
        "reciprocal_rank_fusion",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected RRF")),
    )

    result = hr.hybrid_search(
        Backend(),
        "control stability",
        index_path=tmp_path / "index.json",
        search_limit=3,
        lexical_limit=3,
        generation_mode=True,
        candidate_limit=20,
    )

    assert result["generation_candidate_pool"] is True
    assert result["candidate_selection"]["policy"] == "union"
    assert result["candidate_selection"]["candidate_union_count"] == 3
    assert len(result["results"]) == 3
    assert "smallest complete evidence set" in result["generation_guidance"]
    for index, candidate in enumerate(result["results"], 1):
        assert candidate["candidate_id"] == f"C{index:02d}"
        assert "canonical_uri" in candidate
        assert "l0_abstract" in candidate
        assert "l1_overview" in candidate
        assert "metadata" in candidate
        assert "retrieval_provenance" in candidate
        assert "rerank_context" not in candidate
        assert "source_text_chars" not in candidate


def test_generation_manifest_reads_only_overview_uri_for_l1():
    class Backend:
        def __init__(self):
            self.reads = []

        def read(self, uri, *, limit):
            self.reads.append((uri, limit))
            return "L1 overview content"

    backend = Backend()
    candidates, failures = hr.build_generation_candidate_manifest(
        backend,
        [{
            "uri": "viking://resources/personal-kb/doc/.overview.md",
            "canonical_uri": "viking://resources/personal-kb/doc",
            "abstract": "L0 abstract",
            "metadata": {"source_type": "lecture"},
            "provenance": [{"source": "dense", "rank": 1, "score": 0.9}],
            "retrieval_sources": ["dense"],
        }],
    )

    assert failures == []
    assert backend.reads == [
        ("viking://resources/personal-kb/doc/.overview.md", 8000)
    ]
    assert candidates[0]["l1_overview"] == "L1 overview content"


def test_selected_l2_reader_resolves_ids_and_excludes_l1():
    class Client:
        def ls(self, uri, *, recursive, node_limit):
            return [
                {"name": ".overview.md", "uri": f"{uri}/.overview.md", "isDir": False},
                {"name": "chunk_1.md", "uri": f"{uri}/chunk_1.md", "isDir": False},
                {"name": "chunk_2.md", "uri": f"{uri}/chunk_2.md", "isDir": False},
            ]

    class Backend:
        def __init__(self):
            self.client = Client()
            self.reads = []

        def read(self, uri, *, limit):
            self.reads.append(uri)
            return f"full L2 content for {uri}"

    backend = Backend()
    root = "viking://resources/personal-kb/doc"
    result = read_selected_generation_sources(
        backend,
        [{
            "candidate_id": "C01",
            "canonical_uri": root,
            "resource_uri": f"{root}/.overview.md",
            "metadata": {"source_type": "lecture"},
        }],
        ["C01", "C99"],
    )

    assert result["read_mode"] == "selected_canonical_l2"
    assert len(result["sources"]) == 1
    assert backend.reads == [f"{root}/chunk_1.md", f"{root}/chunk_2.md"]
    assert ".overview.md" not in result["sources"][0]["content"]
    assert "full L2 content" in result["sources"][0]["content"]
    assert result["errors"][0]["candidate_id"] == "C99"


def test_selected_l2_reader_enforces_total_token_budget():
    class Client:
        def ls(self, uri, *, recursive, node_limit):
            return [{"name": "chunk.md", "uri": f"{uri}/chunk.md", "isDir": False}]

    class Backend:
        def __init__(self):
            self.client = Client()

        def read(self, uri, *, limit):
            return "control stability margin lag " * 1000

    root = "viking://resources/personal-kb/doc"
    result = read_selected_generation_sources(
        Backend(),
        [{"candidate_id": "C01", "canonical_uri": root}],
        ["C01"],
        max_total_tokens=25,
    )

    assert result["l2_tokenizer"] == "o200k_base"
    assert result["l2_content_tokens"] <= 25
    assert result["truncated"] is True
    assert result["sources"][0]["content_tokens"] <= 25
    assert result["sources"][0]["truncated"] is True


def test_selected_l2_reader_enforces_aggregate_source_character_budget():
    class Client:
        def __init__(self):
            self.limits = []

        def ls(self, uri, *, recursive, node_limit):
            return [
                {"name": "chunk_1.md", "uri": f"{uri}/chunk_1.md", "isDir": False},
                {"name": "chunk_2.md", "uri": f"{uri}/chunk_2.md", "isDir": False},
            ]

    class Backend:
        def __init__(self):
            self.client = Client()

        def read(self, uri, *, limit):
            self.client.limits.append(limit)
            return "x" * 100

    root = "viking://resources/personal-kb/doc"
    backend = Backend()
    result = read_selected_generation_sources(
        backend,
        [{"candidate_id": "C01", "canonical_uri": root}],
        ["C01"],
        max_chars_per_source=120,
        max_total_tokens=1000,
    )

    assert result["sources"][0]["content_chars"] <= 120
    assert result["sources"][0]["truncated"] is True
    assert backend.client.limits == [120, 20]
