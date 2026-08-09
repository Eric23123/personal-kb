"""Offline tests for hybrid retrieval (RRF fusion and CLI wiring).

Uses fake OpenViking and fake lexical index; no model downloads or live server.
"""

import sys
from pathlib import Path

from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.openviking_backend import PersonalOpenVikingBackend, PERSONAL_NAMESPACE
from scripts.retrieval import hybrid_retrieval as hr
from scripts.retrieval.lexical_index import LexicalIndex, save_index
from scripts.retrieval.search import cmd_hybrid, cmd_hybrid_trace


class FakeSearchClient:
    """Fake OpenViking client returning fixed candidates."""

    def __init__(self, candidates=None):
        self.search_calls = []
        self._candidates = candidates if candidates is not None else [
            {"uri": f"{PERSONAL_NAMESPACE}/doc/alpha/alpha.md", "score": 0.80, "level": 1,
             "abstract": "Alpha: lean manufacturing overview."},
            {"uri": f"{PERSONAL_NAMESPACE}/doc/beta/beta.md", "score": 0.85, "level": 1,
             "abstract": "Beta: kanban calculation formula and examples."},
            {"uri": f"{PERSONAL_NAMESPACE}/doc/gamma/gamma.md", "score": 0.75, "level": 2,
             "abstract": "Gamma: constraint management theory."},
        ]

    def add_resource(self, **kwargs):
        return {"status": "completed"}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"resources": list(self._candidates), "total": len(self._candidates)}

    def read(self, uri, **kwargs):
        return f"Content of {uri}"

    def health(self):
        return True


def _write_lexical_index(tmp_path, query_results: dict[str, list[dict[str, Any]]]) -> Path:
    """Write a fake BM25 index that ignores the query and returns the stored list."""
    index = LexicalIndex(
        corpus_tokens=[["placeholder"]],
        uris=["viking://resources/personal-kb/placeholder"],
        abstracts=["placeholder"],
        metadata=[{}],
    )
    # Monkeypatch the search method for testing.
    def _fake_search(query, top_k=10):
        return query_results.get(query, [])
    index.search = _fake_search
    path = tmp_path / "lexical_index.json"
    save_index(index, path)
    return path


def test_reciprocal_rank_fusion_boosts_documents_in_both_lists():
    dense = [
        {"uri": "viking://resources/personal-kb/a", "score": 0.9, "abstract": "a"},
        {"uri": "viking://resources/personal-kb/b", "score": 0.8, "abstract": "b"},
    ]
    lexical = [
        {"uri": "viking://resources/personal-kb/b", "score": 0.7, "abstract": "b"},
        {"uri": "viking://resources/personal-kb/c", "score": 0.6, "abstract": "c"},
    ]
    merged = hr.reciprocal_rank_fusion(dense, lexical)

    assert len(merged) == 3
    assert merged[0]["uri"] == "viking://resources/personal-kb/b"
    assert len(merged[0]["provenance"]) == 2  # present in both lists


def test_reciprocal_rank_fusion_respects_weights():
    dense = [
        {"uri": "viking://resources/personal-kb/a", "score": 0.9, "abstract": "a"},
    ]
    lexical = [
        {"uri": "viking://resources/personal-kb/b", "score": 0.9, "abstract": "b"},
    ]
    merged = hr.reciprocal_rank_fusion(dense, lexical, dense_weight=2.0, lexical_weight=1.0)
    assert merged[0]["uri"] == "viking://resources/personal-kb/a"
    assert merged[1]["uri"] == "viking://resources/personal-kb/b"


def test_blend_candidate_scores_normalizes_rrf_and_reranker_components():
    candidates = [
        {"uri": "a", "fused_score": 0.0328, "rerank_score": 0.40},
        {"uri": "b", "fused_score": 0.0300, "rerank_score": 0.90},
        {"uri": "c", "fused_score": 0.0270, "rerank_score": 0.20},
    ]

    blended = hr.blend_candidate_scores(
        candidates, rrf_weight=0.70, reranker_weight=0.30,
    )

    assert [item["uri"] for item in blended] == ["a", "b", "c"]
    assert blended[0]["blend_weights"] == {"rrf": 0.7, "reranker": 0.3}
    assert blended[0]["rrf_score_normalized"] == pytest.approx(1.0)
    assert blended[0]["reranker_score_normalized"] == pytest.approx(0.2857, abs=1e-4)
    assert blended[0]["blend_score"] == pytest.approx(0.7857, abs=1e-4)


def test_blend_candidate_scores_rejects_invalid_weights():
    with pytest.raises(ValueError, match="weights must be in the range"):
        hr.blend_candidate_scores(
            [{"uri": "a", "fused_score": 1.0, "rerank_score": 1.0}],
            rrf_weight=1.1,
            reranker_weight=-0.1,
        )


def test_reciprocal_rank_fusion_merges_chunk_resources_by_canonical_uri():
    """Dense URIs from OpenViking chunks and lexical URIs from the BM25 index
    share the same parent directory but differ in the trailing filename.
    RRF must merge them via canonical_uri, not treat them as separate docs."""
    dense = [
        {"uri": "viking://resources/personal-kb/personal-source/data-abc/freq_1.md",
         "canonical_uri": "viking://resources/personal-kb/personal-source/data-abc",
         "score": 0.9, "abstract": "chunk 1"},
        {"uri": "viking://resources/personal-kb/personal-source/data-abc/freq_2.md",
         "canonical_uri": "viking://resources/personal-kb/personal-source/data-abc",
         "score": 0.85, "abstract": "chunk 2"},
    ]
    lexical = [
        {"uri": "viking://resources/personal-kb/personal-source/data-abc/freq_response.json",
         "canonical_uri": "viking://resources/personal-kb/personal-source/data-abc",
         "score": 0.8, "abstract": "whole doc"},
    ]
    merged = hr.reciprocal_rank_fusion(dense, lexical)

    # All three candidates should merge into one entry.
    assert len(merged) == 1
    assert len(merged[0]["provenance"]) == 3  # 2 dense + 1 lexical
    assert merged[0]["canonical_uri"] == "viking://resources/personal-kb/personal-source/data-abc"


def test_canonical_uri_keeps_top_level_resource_filename():
    uri = f"{PERSONAL_NAMESPACE}/step4_note_generator_plan.md"
    assert hr._canonical_uri(uri) == uri


def test_reciprocal_rank_fusion_prefers_richer_abstract_for_reranking():
    dense = [{
        "uri": f"{PERSONAL_NAMESPACE}/doc-abc/chunk.md",
        "canonical_uri": f"{PERSONAL_NAMESPACE}/doc-abc",
        "score": 0.9,
        "abstract": "margin",
    }]
    lexical = [{
        "uri": f"{PERSONAL_NAMESPACE}/doc-abc/source.json",
        "canonical_uri": f"{PERSONAL_NAMESPACE}/doc-abc",
        "score": 0.8,
        "abstract": "Gain margin is measured at the phase crossover frequency.",
    }]
    merged = hr.reciprocal_rank_fusion(dense, lexical)
    assert merged[0]["abstract"] == lexical[0]["abstract"]


def test_dedupe_by_canonical_uri_keeps_highest_scoring_chunk():
    """Multiple chunks from the same source document should collapse to one."""
    results = [
        {"uri": "viking://resources/personal-kb/doc/data-abc/chunk_1.md",
         "canonical_uri": "viking://resources/personal-kb/doc/data-abc",
         "score": 0.9, "abstract": "chunk 1"},
        {"uri": "viking://resources/personal-kb/doc/data-abc/chunk_2.md",
         "canonical_uri": "viking://resources/personal-kb/doc/data-abc",
         "score": 0.7, "abstract": "chunk 2"},
        {"uri": "viking://resources/personal-kb/doc/other-xyz/single.md",
         "canonical_uri": "viking://resources/personal-kb/doc/other-xyz",
         "score": 0.8, "abstract": "other"},
    ]
    deduped = hr._dedupe_by_canonical_uri(results)
    assert len(deduped) == 2
    assert deduped[0]["uri"].endswith("chunk_1.md")  # highest score kept
    assert deduped[1]["uri"].endswith("single.md")


def test_generation_union_cap_preserves_dense_and_lexical_only_candidates():
    dense = [
        {"uri": f"{PERSONAL_NAMESPACE}/dense-{index}", "score": 1.0 - index / 100}
        for index in range(20)
    ]
    lexical = [
        {"uri": f"{PERSONAL_NAMESPACE}/lexical-{index}", "score": 1.0 - index / 100}
        for index in range(20)
    ]

    candidates, cutoff = hr.build_generation_candidate_pool(
        dense,
        lexical,
        candidate_limit=16,
        use_rrf=False,
    )

    assert cutoff["policy"] == "union_top16_no_rrf"
    assert len(candidates) == 16
    assert any(item["uri"].endswith("lexical-0") for item in candidates)
    assert any(item["uri"].endswith("dense-0") for item in candidates)


def test_hybrid_search_routes_distinct_dense_and_lexical_queries(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    index_path = _write_lexical_index(tmp_path, {})
    lexical_calls = []
    rerank_calls = []

    def _fake_lexical_search(query, index_path, *, limit=10):
        lexical_calls.append(query)
        return [{"uri": f"{PERSONAL_NAMESPACE}/doc/beta/beta.md", "score": 1.0, "abstract": "kanban"}]

    def _fake_rerank(query, candidates, **kwargs):
        rerank_calls.append((query, candidates))
        for c in candidates:
            c["rerank_score"] = 1.0
        return candidates

    monkeypatch.setattr(hr, "lexical_search", _fake_lexical_search)
    monkeypatch.setattr(hr.reranker_mod, "rerank_candidates", _fake_rerank)
    result = hr.hybrid_search(
        backend, "original question", index_path=index_path,
        dense_query="semantic explanation", lexical_query="exact kanban formula",
        search_limit=3, lexical_limit=3, top_k=2, rerank_top=2,
    )

    assert backend.client.search_calls[0]["query"] == "semantic explanation"
    assert lexical_calls == ["exact kanban formula"]
    assert rerank_calls and "Exact retrieval terms: exact kanban formula" in rerank_calls[0][0]
    assert result["dense_query"] == "semantic explanation"
    assert result["lexical_query"] == "exact kanban formula"


def test_hybrid_search_drops_unreadable_sources_and_records_failures(tmp_path, monkeypatch):
    class ReadClient(FakeSearchClient):
        def read(self, uri, **kwargs):
            if uri.endswith("/beta/beta.md"):
                raise FileNotFoundError(uri)
            return "readable source with kanban formula"

    backend = PersonalOpenVikingBackend(ReadClient())
    index_path = _write_lexical_index(tmp_path, {})
    monkeypatch.setattr(hr, "lexical_search", lambda query, index_path, *, limit=10: [])
    monkeypatch.setattr(hr.reranker_mod, "rerank_candidates", lambda query, candidates, **kwargs: candidates)
    result = hr.hybrid_search(
        backend, "kanban", index_path=index_path,
        search_limit=3, lexical_limit=3, top_k=2, rerank_top=0,
    )
    assert all(not item["uri"].endswith("/beta/beta.md") for item in result["results"])
    assert any(item["uri"].endswith("/beta/beta.md") for item in result["source_read_failures"])


def test_hybrid_search_passes_filters_to_dense_and_lexical_paths(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    index_path = _write_lexical_index(tmp_path, {})
    lexical_calls = []

    def _fake_lexical_search(query, index_path, *, limit=10, filters=None):
        lexical_calls.append(filters)
        return []

    monkeypatch.setattr(hr, "lexical_search", _fake_lexical_search)
    hr.hybrid_search(
        backend,
        "kanban",
        index_path=index_path,
        search_limit=3,
        lexical_limit=3,
        top_k=2,
        rerank_top=0,
        filters={"course": "KB 1001", "source_type": "lecture"},
    )

    assert backend.client.search_calls[0]["query"] == "kanban"
    assert lexical_calls == [{"course": "KB 1001", "source_type": "lecture"}]


def test_hybrid_search_returns_fused_results(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    lexical_results = [
        {"uri": f"{PERSONAL_NAMESPACE}/doc/alpha", "score": 0.9, "abstract": "alpha"},
    ]
    index_path = _write_lexical_index(tmp_path, {})

    def _fake_lexical_search(query, index_path, *, limit=10):
        return lexical_results

    monkeypatch.setattr(hr, "lexical_search", _fake_lexical_search)

    result = hr.hybrid_search(
        backend, "kanban", index_path=index_path,
        search_limit=3, lexical_limit=3, top_k=3,
        rerank_top=0,
    )

    assert result["query"] == "kanban"
    assert result["mode"] == "dense_plus_lexical"
    assert result["dense_candidates"] == 3
    assert result["lexical_candidates"] == 1
    assert len(result["results"]) <= 3
    # All URIs should be in the Personal namespace.
    for r in result["results"]:
        assert r["uri"].startswith(PERSONAL_NAMESPACE)


def test_hybrid_search_reranker_flag(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    lexical_results = {"kanban": []}
    index_path = _write_lexical_index(tmp_path, lexical_results)

    rerank_called = []

    def _fake_rerank(query, candidates, **kwargs):
        rerank_called.append((query, candidates, kwargs))
        for c in candidates:
            c["rerank_score"] = 1.0
        return candidates

    monkeypatch.setattr(hr.reranker_mod, "rerank_candidates", _fake_rerank)

    result = hr.hybrid_search(
        backend, "kanban", index_path=index_path,
        search_limit=3, lexical_limit=3, top_k=3, rerank_top=2,
    )

    assert result["mode"] == "dense_lexical_reranker"
    assert "rerank_latency_ms" in result
    assert rerank_called


def test_hybrid_search_applies_optional_rrf_reranker_blend(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    index_path = _write_lexical_index(tmp_path, {"kanban": []})

    def _fake_rerank(query, candidates, **kwargs):
        for index, candidate in enumerate(candidates):
            candidate["rerank_score"] = [0.4, 0.9, 0.2][index]
        return candidates

    monkeypatch.setattr(hr.reranker_mod, "rerank_candidates", _fake_rerank)
    result = hr.hybrid_search(
        backend, "kanban", index_path=index_path,
        search_limit=3, lexical_limit=3, top_k=1, rerank_top=3,
        blend_rrf_weight=0.70, blend_reranker_weight=0.30,
    )

    assert result["mode"] == "dense_lexical_blend"
    assert result["blend"]["rrf_weight"] == 0.7
    assert result["blend"]["reranker_weight"] == 0.3
    assert result["results"][0]["blend_score"] >= result["results"][0]["fused_score"]
    assert result["source_cutoff"]["score_key"] == "blend_score"


def test_hybrid_search_defaults_to_rrf_only(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    index_path = _write_lexical_index(tmp_path, {"kanban": []})
    rerank_called = []

    monkeypatch.setattr(
        hr.reranker_mod,
        "rerank_candidates",
        lambda *args, **kwargs: rerank_called.append(True),
    )
    result = hr.hybrid_search(
        backend, "kanban", index_path=index_path,
        search_limit=3, lexical_limit=3, top_k=2,
    )

    assert result["mode"] == "dense_plus_lexical"
    assert rerank_called == []


def test_hybrid_search_defaults_to_reranking_and_records_cutoff(tmp_path, monkeypatch):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    index_path = _write_lexical_index(tmp_path, {"kanban": []})

    def _fake_rerank(query, candidates, **kwargs):
        for index, candidate in enumerate(candidates):
            candidate["rerank_score"] = 1.0 - index * 0.02
        return candidates

    monkeypatch.setattr(hr.reranker_mod, "rerank_candidates", _fake_rerank)
    result = hr.hybrid_search(
        backend, "kanban", index_path=index_path,
        search_limit=3, lexical_limit=3, top_k=2, rerank_top=3,
    )

    assert result["mode"] == "dense_lexical_reranker"
    assert result["rerank_candidates"] == 3
    assert len(result["results"]) == 3  # scores are close, so dynamic cutoff expands to 3
    assert result["source_cutoff"]["score_key"] == "rerank_score"


def test_select_final_sources_respects_score_drop_and_maximum():
    candidates = [
        {"uri": "a", "rerank_score": 1.0},
        {"uri": "b", "rerank_score": 0.98},
        {"uri": "c", "rerank_score": 0.97},
        {"uri": "d", "rerank_score": 0.70},
    ]
    selected, cutoff = hr.select_final_sources(
        candidates, default_limit=2, max_sources=4, score_drop=0.15,
    )
    assert [item["uri"] for item in selected] == ["a", "b", "c"]
    assert cutoff["cutoff_reason"] == "score_drop"


def test_select_final_sources_uses_stricter_default_fused_drop():
    candidates = [
        {"uri": "a", "fused_score": 0.0328},
        {"uri": "b", "fused_score": 0.0300},
        {"uri": "c", "fused_score": 0.0270},
    ]
    selected, cutoff = hr.select_final_sources(
        candidates, default_limit=1, max_sources=3, reranked=False,
    )
    assert [item["uri"] for item in selected] == ["a", "b"]
    assert cutoff["score_drop"] == 0.005


def test_select_final_sources_uses_relative_floor_for_small_reranker_logits():
    candidates = [
        {"uri": "a", "rerank_score": 0.0040},
        {"uri": "b", "rerank_score": 0.0025},
        {"uri": "c", "rerank_score": 0.0002},
    ]
    selected, cutoff = hr.select_final_sources(
        candidates, default_limit=1, max_sources=3,
    )
    assert [item["uri"] for item in selected] == ["a", "b"]
    assert cutoff["relative_floor"] == 0.08
    assert cutoff["cutoff_reason"] == "score_drop"


def test_select_final_sources_can_return_fewer_than_target_when_scores_drop():
    candidates = [
        {"uri": "a", "rerank_score": 0.04},
        {"uri": "b", "rerank_score": 0.001},
    ]
    selected, cutoff = hr.select_final_sources(
        candidates, default_limit=5, max_sources=8,
    )
    assert [item["uri"] for item in selected] == ["a"]
    assert cutoff["cutoff_reason"] == "score_drop"


def test_select_final_sources_never_exceeds_hard_maximum():
    candidates = [{"uri": str(i), "rerank_score": 1.0 - i * 0.01} for i in range(10)]
    selected, cutoff = hr.select_final_sources(
        candidates, default_limit=2, max_sources=4, score_drop=1.0,
    )
    assert len(selected) == 4
    assert cutoff["cutoff_reason"] == "max_sources"


def test_cmd_hybrid_wires_backend_and_index(tmp_path):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    lexical_results = {"kanban": []}
    index_path = _write_lexical_index(tmp_path, lexical_results)

    result = cmd_hybrid(
        backend, "kanban",
        index_path=str(index_path),
        limit=3, lexical_limit=3, top_k=3, k=60,
        dense_weight=1.0, lexical_weight=1.0,
        rerank_top=0, rerank_server="",
    )
    assert result["query"] == "kanban"


def test_cmd_hybrid_trace_reads_top_fused_result(tmp_path):
    backend = PersonalOpenVikingBackend(FakeSearchClient())
    lexical_results = {"kanban": []}
    index_path = _write_lexical_index(tmp_path, lexical_results)

    result = cmd_hybrid_trace(
        backend, "kanban",
        index_path=str(index_path),
        limit=3, lexical_limit=3, top_k=3, k=60,
        dense_weight=1.0, lexical_weight=1.0,
        read_limit=1000,
        rerank_top=0,
    )
    assert "top_source" in result
    assert result["top_source"]["uri"] == result["results"][0]["uri"]
    assert "Content of" in result["top_source"]["content"]
