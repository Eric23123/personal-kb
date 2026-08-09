"""Offline tests for planner validation, visible fallback, and execution trace."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.openviking_backend import PERSONAL_NAMESPACE, PersonalOpenVikingBackend
from scripts.retrieval.query_models import QueryPlan
from scripts.retrieval.query_orchestrator import execute_planned_query
from scripts.retrieval.query_planner import QueryPlanner
from scripts.retrieval.search import cmd_planned_hybrid


def _response(**overrides):
    payload = {
        "intent": "document_lookup",
        "backends": ["openviking"],
        "queries": [
            {"text": "closed loop stability", "role": "lexical"},
        ],
        "filters": {},
        "retrieval_mode": "hybrid_rerank",
        "limit": 5,
    }
    payload.update(overrides)
    return payload


def test_planner_uses_injected_current_model_and_preserves_provenance():
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        import json
        return json.dumps(_response())

    result = QueryPlanner(llm, model="current-main", provider="openai-codex").plan(
        "Explain closed loop stability"
    )

    assert result.trace.status == "planned"
    assert result.trace.planner_attempts == 1
    assert result.plan.planner_model == "current-main"
    assert result.plan.planner_provider == "openai-codex"
    assert result.plan.queries[0].original_query == "Explain closed loop stability"
    assert result.plan.queries[0].trace_id == result.plan.trace_id
    assert prompts and "Return ONLY one JSON object" in prompts[0]


def test_planner_retries_then_surfaces_warning_and_fallback():
    calls = []

    def failing_llm(prompt):
        calls.append(prompt)
        raise TimeoutError("planner timeout")

    result = QueryPlanner(failing_llm, model="current-main", provider="test").plan("kanban formula")

    assert len(calls) == 2
    assert result.trace.status == "planner_failed"
    assert result.trace.fallback_used is True
    assert result.trace.warning is not None
    assert "failed after 2 attempts" in result.trace.warning
    assert result.trace.errors
    assert result.plan.status == "fallback"
    assert result.plan.queries[0].text == "kanban"
    assert result.plan.retrieval_mode == "hybrid"


def test_planner_falls_back_on_malformed_json():
    result = QueryPlanner(lambda prompt: "not json").plan("frequency response")
    assert result.trace.fallback_used is True
    assert result.plan.original_query == "frequency response"
    assert result.plan.filters.to_dict() == {"source_scope": "course"}


def test_planner_accepts_fenced_json_and_explicit_filters():
    import json

    response = _response(
        queries=[
            {"text": "KB 1001 kanban", "role": "lexical"},
        ],
        filters={"course": "KB 1001", "lecture": 3},
    )
    result = QueryPlanner(lambda prompt: "```json\n" + json.dumps(response) + "\n```" ).plan(
        "Compare KB 1001 Lecture 3 kanban"
    )
    assert result.trace.status == "planned"
    assert result.plan.filters.to_dict() == {
        "course": "KB 1001",
        "lecture": 3,
        "source_scope": "course",
    }


@pytest.mark.parametrize(
    "response",
    [
        _response(filters={"course": "PERSONAL-RESEARCH"}),
        _response(filters={"source_type": "diagram"}),
        _response(filters={"semester": "Fall 2026"}),
        _response(filters={"date": "2026-09-01"}),
        _response(filters={"unknown": "x"}),
        _response(limit=0),
        _response(limit=21),
        _response(backends=["foreign_backend"]),
        _response(queries=[{"text": "viking://resources/other/secret"}]),
    ],
)
def test_planner_rejects_unsafe_output_and_falls_back(response):
    import json
    result = QueryPlanner(lambda prompt: json.dumps(response)).plan("KB 1001 Lecture 3 kanban")
    assert result.trace.fallback_used is True
    assert result.plan.queries[0].text == "KB 1001 Lecture 3 kanban"


def test_planner_normalizes_request_language_to_topic_keywords():
    import json
    response = _response(
        queries=[
            {"text": "what's the formula for gain margin", "role": "lexical"},
        ]
    )
    result = QueryPlanner(lambda prompt: json.dumps(response)).plan(
        "What's the formula for gain margin?"
    )
    assert [query.text for query in result.plan.queries] == ["gain margin"]
    assert result.plan.queries[0].role == "lexical"


def test_planner_rejects_semantic_expansion_beyond_original_keywords():
    import json
    response = _response(
        queries=[
            {"text": "gain margin control systems", "role": "lexical"},
        ]
    )
    result = QueryPlanner(lambda prompt: json.dumps(response)).plan(
        "What's the formula for gain margin?"
    )

    assert result.trace.fallback_used is True
    assert result.plan.queries[0].text == "gain margin"


def test_planner_assigns_source_scope_policy():
    import json
    response = _response(queries=[{"text": "gain margin", "role": "lexical"}])

    normal = QueryPlanner(lambda prompt: json.dumps(response)).plan("What's the formula for gain margin?")
    assessment = QueryPlanner(lambda prompt: json.dumps(response)).plan("Find homework questions about gain margin")
    study = QueryPlanner(lambda prompt: json.dumps(response)).plan("Create a study guide for gain margin")
    quiz = QueryPlanner(lambda prompt: json.dumps(response)).plan("Make a quiz on gain margin")

    assert normal.plan.filters.source_scope == "course"
    assert assessment.plan.filters.source_scope == "assessment"
    assert study.plan.filters.source_scope == "all"
    assert quiz.plan.filters.source_scope == "all"


def test_validate_model_response_does_not_call_llm():
    planner = QueryPlanner(llm=lambda _prompt: (_ for _ in ()).throw(AssertionError("must not call")))
    result = planner.validate_model_response(
        "What is a kanban pull system?",
        '{"intent":"document_lookup","backends":["openviking"],"queries":[{"text":"what is a kanban pull system","role":"lexical"}],"filters":{},"retrieval_mode":"hybrid_rerank","limit":5}',
    )
    assert result.trace.status == "planned"
    assert result.plan.planner_model is None


def test_validate_model_response_invalid_plan_is_visible_fallback():
    result = QueryPlanner().validate_model_response("kanban", "not json")
    assert result.trace.fallback_used is True
    assert result.trace.status == "planner_failed"
    assert "invalid query plan" in result.trace.warning


def test_search_layer_uses_active_planner_and_returns_warning(tmp_path):
    backend = PersonalOpenVikingBackend(FakeClient())
    backend.hybrid_search = lambda query, **kwargs: {
        "mode": "dense_lexical_reranker",
        "results": [{"uri": f"{PERSONAL_NAMESPACE}/doc/a", "rerank_score": 0.9}],
    }
    planner = QueryPlanner(lambda prompt: "malformed", model="current-main")

    result = cmd_planned_hybrid(
        backend, "kanban formula",
        planner=planner,
        index_path=tmp_path / "index.json",
    )

    assert result["trace"]["fallback_used"] is True
    assert "failed after 2 attempts" in result["trace"]["warning"]
    assert result["plan"]["planner_model"] == "current-main"
    assert result["results"][0]["uri"].startswith(PERSONAL_NAMESPACE)


def test_orchestrator_executes_lexical_and_semantic_queries(tmp_path):
    backend = PersonalOpenVikingBackend(FakeClient())
    calls = []

    def hybrid(query, **kwargs):
        calls.append((query, kwargs))
        return {
            "mode": "dense_lexical_reranker",
            "results": [
                {"uri": f"{PERSONAL_NAMESPACE}/{len(calls)}", "rerank_score": 0.9},
                {"uri": f"{PERSONAL_NAMESPACE}/low-{len(calls)}", "rerank_score": 0.8},
            ],
        }

    backend.hybrid_search = hybrid
    import json
    response = _response(
        queries=[
            {"text": "what's the formula for gain margin", "role": "lexical"},
        ]
    )
    planning = QueryPlanner(lambda prompt: json.dumps(response)).plan("whats gain margin")
    result = execute_planned_query(planning, backend, index_path=tmp_path / "index.json")

    assert len(calls) == 1
    assert calls[0][0] == "gain margin"
    assert calls[0][1]["dense_query"] == "gain margin"
    assert calls[0][1]["lexical_query"] == "gain margin"
    assert result["trace"]["execution"][0]["query_role"] == "lexical"
    assert result["trace"]["execution"][0]["queries"] == [
        {"text": "gain margin", "role": "lexical"},
    ]
    assert len(result["results"]) == 2


def test_orchestrator_passes_blend_weights_to_hybrid_backend(tmp_path):
    backend = PersonalOpenVikingBackend(FakeClient())
    calls = []

    def hybrid(query, **kwargs):
        calls.append(kwargs)
        return {
            "mode": "dense_lexical_blend",
            "results": [
                {
                    "uri": f"{PERSONAL_NAMESPACE}/doc/a",
                    "fused_score": 0.03,
                    "rerank_score": 0.9,
                    "blend_score": 0.8,
                }
            ],
        }

    backend.hybrid_search = hybrid
    import json
    response = _response(
        queries=[{"text": "kanban", "role": "lexical"}],
    )
    planning = QueryPlanner(lambda prompt: json.dumps(response)).plan("kanban formula")
    execute_planned_query(
        planning,
        backend,
        index_path=tmp_path / "index.json",
        blend_rrf_weight=0.70,
        blend_reranker_weight=0.30,
        enable_reranker=True,
    )

    assert calls[0]["blend_rrf_weight"] == 0.70
    assert calls[0]["blend_reranker_weight"] == 0.30
    assert calls[0]["rerank_top"] == 20


def test_orchestrator_propagates_validated_filters_and_trace(tmp_path):
    backend = PersonalOpenVikingBackend(FakeClient())
    calls = []

    def hybrid(query, **kwargs):
        calls.append(kwargs)
        return {
            "mode": "dense_plus_lexical",
            "results": [{"uri": f"{PERSONAL_NAMESPACE}/doc/a", "fused_score": 0.03}],
        }

    backend.hybrid_search = hybrid
    import json
    response = _response(
        queries=[{"text": "KB 1001 kanban", "role": "lexical"}],
        filters={"course": "KB 1001", "lecture": 3},
    )
    planning = QueryPlanner(lambda prompt: json.dumps(response)).plan(
        "Compare KB 1001 Lecture 3 kanban"
    )
    result = execute_planned_query(planning, backend, index_path=tmp_path / "index.json")

    assert calls[0]["filters"] == {
        "course": "KB 1001",
        "lecture": 3,
        "source_scope": "course",
    }
    assert result["trace"]["execution"][0]["filters"] == {
        "course": "KB 1001",
        "lecture": 3,
        "source_scope": "course",
    }


def test_orchestrator_passes_validated_filters_to_hindsight(tmp_path):
    import json

    backend = PersonalOpenVikingBackend(FakeClient())
    response = _response(
        intent="personal_memory",
        backends=["hindsight"],
        queries=[{"text": "KB 1001 kanban", "role": "lexical"}],
        filters={"course": "KB 1001", "lecture": 3},
    )
    planning = QueryPlanner(lambda prompt: json.dumps(response)).plan(
        "Compare KB 1001 Lecture 3 kanban"
    )
    calls = []

    def recall(query, limit, *, filters):
        calls.append((query, limit, filters))
        return {"results": [{"text": "memory"}]}

    result = execute_planned_query(
        planning,
        backend,
        index_path=tmp_path / "index.json",
        hindsight_recall=recall,
    )

    assert calls == [
        (
            "KB 1001 kanban",
            5,
            {"course": "KB 1001", "lecture": 3, "source_scope": "course"},
        )
    ]
    execution = result["trace"]["execution"][0]
    assert execution["result_count"] == 1
    assert execution["filters"] == calls[0][2]


def test_orchestrator_returns_hindsight_results_at_top_level(tmp_path):
    import json

    backend = PersonalOpenVikingBackend(FakeClient())
    response = _response(
        intent="personal_memory",
        backends=["hindsight"],
        queries=[{"text": "kanban", "role": "lexical"}],
    )
    planning = QueryPlanner(lambda prompt: json.dumps(response)).plan("kanban")

    result = execute_planned_query(
        planning,
        backend,
        index_path=tmp_path / "index.json",
        hindsight_recall=lambda query, limit, *, filters: {"results": [{"text": "memory"}]},
    )

    assert result["results"] == [{"text": "memory", "backend": "hindsight", "query": "kanban"}]
    assert result["hindsight_results"] == result["results"]


def test_orchestrator_defaults_to_rrf_even_when_plan_requests_rerank(tmp_path):
    backend = PersonalOpenVikingBackend(FakeClient())
    calls = []

    def hybrid(query, **kwargs):
        calls.append(kwargs)
        return {
            "mode": "dense_plus_lexical",
            "results": [
                {"uri": f"{PERSONAL_NAMESPACE}/doc/a", "fused_score": 0.03}
            ],
        }

    backend.hybrid_search = hybrid
    import json
    planning = QueryPlanner(
        lambda prompt: json.dumps(_response(retrieval_mode="hybrid_rerank"))
    ).plan("kanban formula")
    execute_planned_query(planning, backend, index_path=tmp_path / "index.json")

    assert calls[0]["rerank_top"] == 0


class FakeClient:
    def search(self, **kwargs):
        return {
            "resources": [
                {"uri": f"{PERSONAL_NAMESPACE}/doc/a", "score": 0.9, "abstract": "kanban"},
            ]
        }

    def read(self, uri, **kwargs):
        return "source"


def test_orchestrator_preserves_fallback_warning_and_source_trace(tmp_path):
    # Avoid real BM25/reranker work while exercising orchestration contract.
    backend = PersonalOpenVikingBackend(FakeClient())
    backend.hybrid_search = lambda query, **kwargs: {
        "mode": "dense_lexical_reranker",
        "results": [{"uri": f"{PERSONAL_NAMESPACE}/doc/a", "rerank_score": 0.9}],
    }
    planning = QueryPlanner(lambda prompt: "broken").plan("kanban formula")
    result = execute_planned_query(planning, backend, index_path=tmp_path / "index.json")

    assert result["trace"]["status"] == "planner_failed"
    assert result["trace"]["fallback_used"] is True
    assert result["trace"]["warning"]
    assert result["trace"]["source_uris"] == [f"{PERSONAL_NAMESPACE}/doc/a"]
    assert result["results"][0]["uri"].startswith(PERSONAL_NAMESPACE)
