"""Offline tests for the source-read benchmark harness."""

from __future__ import annotations

from scripts.retrieval_benchmark import (
    aggregate_evaluations,
    build_plan_response,
    evaluate_source_reads,
)


class FakeBackend:
    def read(self, uri, *, limit):
        if uri.endswith("/unreadable"):
            raise FileNotFoundError(uri)
        return "Closed-loop transfer uses D G and 1 + D G H."


def test_build_plan_response_contains_distinct_lexical_and_semantic_queries():
    response = build_plan_response(
        {
            "query": "Explain closed-loop transfer",
            "required": ["transfer"],
            "source_hints": ["control"],
        },
        "hybrid_rerank",
    )

    assert '"role": "lexical"' in response
    assert '"role": "semantic"' in response
    assert '"retrieval_mode": "hybrid_rerank"' in response


def test_evaluate_source_reads_records_support_and_read_failures():
    result = evaluate_source_reads(
        FakeBackend(),
        {
            "results": [
                {"uri": "viking://resources/personal-kb/readable"},
                {"uri": "viking://resources/personal-kb/unreadable"},
            ]
        },
        {"required": ["transfer", "D G", "missing"]},
    )

    assert result["hit_at_5"] is True
    assert result["top1_context_hit"] is True
    assert result["read_errors"] == 1
    assert result["rows"][0]["covered_terms"] == ["transfer", "d g"]


def test_aggregate_mrr_counts_queries_without_support():
    aggregate = aggregate_evaluations(
        [
            {
                "hit_at_5": True,
                "top1_context_hit": True,
                "best_coverage": 1.0,
                "top1_coverage": 1.0,
                "support_rank": 1,
                "read_errors": 0,
            },
            {
                "hit_at_5": False,
                "top1_context_hit": False,
                "best_coverage": 0.2,
                "top1_coverage": 0.0,
                "support_rank": None,
                "read_errors": 0,
            },
        ]
    )

    assert aggregate["mrr"] == 0.5


def test_aggregate_evaluations_reports_mrr_and_rates():
    aggregate = aggregate_evaluations(
        [
            {
                "hit_at_5": True,
                "top1_context_hit": True,
                "best_coverage": 1.0,
                "top1_coverage": 1.0,
                "support_rank": 1,
                "read_errors": 0,
            },
            {
                "hit_at_5": True,
                "top1_context_hit": False,
                "best_coverage": 0.6,
                "top1_coverage": 0.2,
                "support_rank": 3,
                "read_errors": 1,
            },
        ]
    )

    assert aggregate["read_hit_at_5_rate"] == 1.0
    assert aggregate["read_top1_context_hit_rate"] == 0.5
    assert aggregate["mrr"] == 0.6667
    assert aggregate["read_errors"] == 1
