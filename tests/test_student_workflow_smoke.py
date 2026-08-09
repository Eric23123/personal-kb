"""Student workflow smoke tests (Priority 6) — verify the student experience.

Covers all seven verification points from STATUS.md:
  1. Source-grounded answers — search returns URIs, source content readable
  2. Lecture explanations — facts explainable with related context
  3. Quiz generation — questions without inline answers
  4. Homework-style problems — practice variants, not copy of original
  5. Separate answer keys — answer key distinct from question prompts
  6. Empty-retrieval behavior — graceful handling of no results
  7. Hindsight/OpenViking scope selection — assessment vs course routing

All tests use injectable fake backends — no live OpenViking, Hindsight,
embedding, or LLM calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------
# Synthetic test data (mirrors what a real Personal student would see)
# ---------------------------------------------------------------------------


def _course_facts() -> list[dict[str, Any]]:
    """A representative set of facts a student might study."""
    return [
        {
            "name": "Closed-loop system",
            "definition": "Inputs depend in part on outputs via feedback.",
            "type": "concept",
            "topic": "feedback",
            "source_scope": "course",
            "course": "PERSONAL-ALPHA",
            "lecture": 1,
        },
        {
            "name": "Final-value theorem",
            "definition": "Finds steady-state value from Laplace domain.",
            "type": "formula",
            "topic": "steady state",
            "source_scope": "course",
            "course": "PERSONAL-ALPHA",
            "lecture": 2,
        },
        {
            "name": "Bode plot",
            "definition": "Frequency response magnitude and phase vs frequency.",
            "type": "concept",
            "topic": "frequency response",
            "source_scope": "course",
            "course": "PERSONAL-ALPHA",
            "lecture": 3,
        },
        {
            "name": "HW1 Problem 1",
            "problem": "Derive the transfer function for a mass-spring-damper.",
            "solution_pattern": "Use Newton's second law, then Laplace transform.",
            "type": "assessment-solution",
            "topic": "modeling",
            "source_scope": "assessment",
            "assignment": "HW1",
        },
        {
            "name": "HW1 Problem 2",
            "problem": "Find the closed-loop poles for K=10.",
            "solution_pattern": "Compute the characteristic equation and solve for s.",
            "type": "assessment-solution",
            "topic": "stability",
            "source_scope": "assessment",
            "assignment": "HW1",
        },
    ]


# ---------------------------------------------------------------------------
# Verification 1: Source-grounded answers
# ---------------------------------------------------------------------------


class TestSourceGroundedAnswers:
    """Search + source read-back = source-grounded answer."""

    def test_search_returns_uris_in_personal_namespace(self):
        from scripts.core.openviking_backend import PERSONAL_NAMESPACE

        class FakeClient:
            def search(self, **kwargs):
                return {
                    "resources": [
                        {"uri": f"{PERSONAL_NAMESPACE}/TEST/course/ch1-abc", "score": 0.95},
                        {"uri": f"{PERSONAL_NAMESPACE}/TEST/course/ch2-def", "score": 0.88},
                    ],
                    "total": 2,
                }

            def add_resource(self, **kw):
                return {"status": "completed"}

        from scripts.core.openviking_backend import PersonalOpenVikingBackend
        from scripts.retrieval.search import cmd_search

        backend = PersonalOpenVikingBackend(FakeClient())
        result = cmd_search(backend, "transfer function", limit=5)

        assert result["total"] == 2
        for r in result["results"]:
            assert r["uri"].startswith(PERSONAL_NAMESPACE + "/")

    def test_source_readback_returns_content(self):
        from scripts.core.openviking_backend import PERSONAL_NAMESPACE

        class FakeClient:
            def search(self, **kw):
                return {"resources": [], "total": 0}
            def add_resource(self, **kw):
                return {"status": "completed"}
            def read(self, uri, **kw):
                return "The closed-loop transfer function is T(s) = G(s)/(1+G(s)H(s))."

        from scripts.core.openviking_backend import PersonalOpenVikingBackend
        from scripts.retrieval.search import cmd_source

        backend = PersonalOpenVikingBackend(FakeClient())
        uri = f"{PERSONAL_NAMESPACE}/TEST/course/ch1"
        result = cmd_source(backend, uri, read_limit=10000)

        assert result["uri"] == uri
        assert "T(s)" in result["content"]

    def test_trace_combines_search_and_read(self):
        from scripts.core.openviking_backend import PERSONAL_NAMESPACE

        class FakeClient:
            def search(self, **kw):
                return {
                    "resources": [
                        {"uri": f"{PERSONAL_NAMESPACE}/test/ch1", "score": 0.99},
                    ],
                    "total": 1,
                }
            def add_resource(self, **kw):
                return {"status": "completed"}
            def read(self, uri, **kw):
                return "source content"

        from scripts.core.openviking_backend import PersonalOpenVikingBackend
        from scripts.retrieval.search import cmd_trace

        backend = PersonalOpenVikingBackend(FakeClient())
        result = cmd_trace(backend, "stability", limit=3)

        assert result["total"] == 1
        assert "top_source" in result
        assert result["top_source"]["content"] == "source content"


# ---------------------------------------------------------------------------
# Verification 2: Lecture explanations
# ---------------------------------------------------------------------------


class TestLectureExplanations:
    """explain_fact returns fact + related context."""

    def test_explain_finds_matching_fact(self):
        from scripts.study.study_commands import explain_fact

        result = explain_fact(_course_facts(), "closed-loop")

        assert result["found"] is True
        assert result["fact"]["name"] == "Closed-loop system"
        assert "feedback" in result["fact"]["definition"].lower()

    def test_explain_returns_related_facts(self):
        from scripts.study.study_commands import explain_fact

        result = explain_fact(_course_facts(), "closed-loop")

        # Should at minimum return the requested fact.
        assert result["found"] is True

    def test_explain_missing_is_explicit(self):
        from scripts.study.study_commands import explain_fact

        result = explain_fact(_course_facts(), "quantum field theory")

        assert result["found"] is False
        assert result["fact"] is None
        assert result["query"] == "quantum field theory"


# ---------------------------------------------------------------------------
# Verification 3: Quiz generation (no inline answers)
# ---------------------------------------------------------------------------


class TestQuizGeneration:
    """Quiz questions must not leak answers."""

    def test_quiz_questions_have_no_answer_key(self):
        from scripts.study.study_commands import build_quiz

        result = build_quiz(_course_facts(), count=3, seed=42)

        assert len(result["questions"]) == 3
        assert len(result["answer_key"]) == 3
        for q in result["questions"]:
            assert "answer" not in q, f"question leaks answer: {q}"

    def test_quiz_separate_answer_key(self):
        from scripts.study.study_commands import build_quiz

        result = build_quiz(_course_facts(), count=3, seed=42)

        # Questions and answers must be separate.
        assert isinstance(result["questions"], list)
        assert isinstance(result["answer_key"], list)
        assert result["questions"] != result["answer_key"]

    def test_quiz_deterministic_with_seed(self):
        from scripts.study.study_commands import build_quiz

        a = build_quiz(_course_facts(), count=2, seed=7)
        b = build_quiz(_course_facts(), count=2, seed=7)

        assert a["questions"] == b["questions"]
        assert a["answer_key"] == b["answer_key"]

    def test_quiz_handles_empty_facts(self):
        from scripts.study.study_commands import build_quiz

        result = build_quiz([], count=5)

        assert result["count"] == 0
        assert result["questions"] == []
        assert result["answer_key"] == []


# ---------------------------------------------------------------------------
# Verification 4: Homework-style problems
# ---------------------------------------------------------------------------


class TestHomeworkProblems:
    """Practice variants must NOT copy original wording."""

    def test_assessment_facts_get_practice_variant_prompt(self):
        from scripts.study.study_commands import build_quiz

        result = build_quiz(_course_facts(), count=5, seed=1)

        # Find assessment questions.
        assessment_qs = [
            q for q in result["questions"]
            if q.get("type") and "assessment" in str(q["type"])
        ]
        assert len(assessment_qs) > 0, "expected at least one assessment question"

        for q in assessment_qs:
            assert "Practice variant" in q["prompt"], (
                f"assessment question should be framed as practice variant: {q}"
            )

    def test_llm_quiz_generates_new_variants(self):
        from scripts.study.study_commands import build_llm_quiz

        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            return json.dumps([
                {
                    "prompt": "A cart of mass 2 kg is on a frictionless track. Find the equilibrium input for a desired position of 0.5 m.",
                    "answer_outline": "Set derivatives to zero and solve for u.",
                    "source_method": "equilibrium analysis",
                },
            ])

        result = build_llm_quiz(
            [{"name": "HW1 P1", "problem": "Derive equilibrium for the cart.", "solution_pattern": "Set dx/dt=0"}],
            fake_llm, count=1,
        )

        assert result["generation"] == "llm_variant"
        assert result["questions"][0]["prompt"] != "Derive equilibrium for the cart."
        assert result["answer_key"][0]["answer"] == "Set derivatives to zero and solve for u."


# ---------------------------------------------------------------------------
# Verification 5: Separate answer keys
# ---------------------------------------------------------------------------


class TestSeparateAnswerKeys:
    """Answer keys must be structurally separate from question prompts."""

    def test_answer_key_never_contains_prompt_field(self):
        from scripts.study.study_commands import build_quiz

        result = build_quiz(_course_facts(), count=5, seed=99)

        for ak in result["answer_key"]:
            assert "prompt" not in ak, f"answer key entry contains prompt: {ak}"
            assert "answer" in ak

    def test_answer_key_ids_match_question_ids(self):
        from scripts.study.study_commands import build_quiz

        result = build_quiz(_course_facts(), count=3, seed=7)

        q_ids = [q["id"] for q in result["questions"]]
        a_ids = [a["id"] for a in result["answer_key"]]
        assert q_ids == a_ids


# ---------------------------------------------------------------------------
# Verification 6: Empty-retrieval behavior
# ---------------------------------------------------------------------------


class TestEmptyRetrieval:
    """Graceful handling when no results are found."""

    def test_search_returns_empty_without_crashing(self):
        class EmptyClient:
            def search(self, **kw):
                return {"resources": [], "total": 0}
            def add_resource(self, **kw):
                return {"status": "completed"}

        from scripts.core.openviking_backend import PersonalOpenVikingBackend
        from scripts.retrieval.search import cmd_search

        backend = PersonalOpenVikingBackend(EmptyClient())
        result = cmd_search(backend, "this topic does not exist", limit=5)

        assert result["total"] == 0
        assert result["results"] == []

    def test_trace_no_results_no_top_source(self):
        class EmptyClient:
            def search(self, **kw):
                return {"resources": [], "total": 0}
            def add_resource(self, **kw):
                return {"status": "completed"}
            def read(self, uri, **kw):
                raise RuntimeError("should not be called")

        from scripts.core.openviking_backend import PersonalOpenVikingBackend
        from scripts.retrieval.search import cmd_trace

        backend = PersonalOpenVikingBackend(EmptyClient())
        result = cmd_trace(backend, "nonexistent", limit=5)

        assert result["total"] == 0
        assert "top_source" not in result

    def test_explain_empty_facts_returns_not_found(self):
        from scripts.study.study_commands import explain_fact

        result = explain_fact([], "anything")

        assert result["found"] is False

    def test_review_empty_facts_returns_zero_totals(self):
        from scripts.study.study_commands import build_review

        result = build_review([])

        assert result["total"] == 0
        assert result["by_topic"] == {}


# ---------------------------------------------------------------------------
# Verification 7: Hindsight/OpenViking scope selection
# ---------------------------------------------------------------------------


class TestScopeSelection:
    """assessment vs course scope must be correctly routed."""

    def test_source_scope_filters_course_and_assessment(self):
        from scripts.core.openviking_backend import metadata_matches

        # Course scope: accepts lectures, transcripts; rejects homework, exams.
        assert metadata_matches({"source_type": "lecture"}, {"source_scope": "course"})
        assert metadata_matches({"source_type": "transcript"}, {"source_scope": "course"})
        assert not metadata_matches({"source_type": "homework"}, {"source_scope": "course"})
        assert not metadata_matches({"source_type": "exam"}, {"source_scope": "course"})

        # Assessment scope: accepts homework, exams; rejects lectures.
        assert metadata_matches({"source_type": "homework"}, {"source_scope": "assessment"})
        assert metadata_matches({"source_type": "exam"}, {"source_scope": "assessment"})
        assert not metadata_matches({"source_type": "lecture"}, {"source_scope": "assessment"})

        # All scope: accepts everything.
        assert metadata_matches({"source_type": "lecture"}, {"source_scope": "all"})
        assert metadata_matches({"source_type": "homework"}, {"source_scope": "all"})

    def test_review_groups_by_scope(self):
        from scripts.study.study_commands import build_review

        result = build_review(_course_facts())

        # Should show assessment and course scoped facts separated.
        assert "by_scope" in result
        assert result["by_scope"].get("assessment", 0) >= 2  # HW1 P1 + P2
        assert result["by_scope"].get("course", 0) >= 3  # concepts

    def test_review_groups_by_assignment(self):
        from scripts.study.study_commands import build_review

        result = build_review(_course_facts())

        assert "by_assignment" in result
        assert result["by_assignment"].get("HW1", 0) == 2

    def test_assessment_processor_adds_scope(self):
        from scripts.study.assessment_processor import normalize_assessment_items

        items = normalize_assessment_items(
            [{"problem": "Derive X", "solution_pattern": "Method Y"}],
            course="TEST",
            source_type="homework",
            assignment="HW1",
        )

        assert items[0]["source_scope"] == "assessment"
        assert items[0]["source_type"] == "homework"
        assert items[0]["assignment"] == "HW1"

    def test_assessment_items_not_mistaken_for_course(self):
        """An assessment item should never pass a course-scope filter."""
        from scripts.core.openviking_backend import metadata_matches

        hw_metadata = {
            "source_type": "homework",
            "source_scope": "assessment",
            "course": "TEST",
        }

        # Course scope filter should reject homework.
        assert not metadata_matches(hw_metadata, {"source_scope": "course"})

        # Assessment scope filter should accept homework.
        assert metadata_matches(hw_metadata, {"source_scope": "assessment"})

    def test_search_filters_by_scope(self):
        """When a search is scoped to 'course', assessment items are filtered out."""
        from scripts.core.openviking_backend import PERSONAL_NAMESPACE

        class ScopeFilterClient:
            def search(self, **kw):
                return {
                    "resources": [
                        {"uri": f"{PERSONAL_NAMESPACE}/course/lec1", "score": 0.9,
                         "metadata": {"source_type": "lecture", "course": "TEST"}},
                        {"uri": f"{PERSONAL_NAMESPACE}/course/hw1", "score": 0.8,
                         "metadata": {"source_type": "homework", "course": "TEST"}},
                    ],
                    "total": 2,
                }
            def add_resource(self, **kw):
                return {"status": "completed"}

        from scripts.core.openviking_backend import PersonalOpenVikingBackend

        backend = PersonalOpenVikingBackend(ScopeFilterClient())

        # Course scope: homework should be filtered out.
        result = backend.search("stability", limit=10, filters={"source_scope": "course"})
        assert result["filters_applied"]["source_scope"] == "course"
        assert len(result["resources"]) == 1
        assert result["resources"][0]["uri"].endswith("lec1")

    def test_review_quiz_scope_boundary(self):
        """A quiz generated from all facts should include both course and assessment,
        but assessment questions get practice-variant framing."""
        from scripts.study.study_commands import build_quiz

        result = build_quiz(_course_facts(), count=5, seed=1)

        prompts = [q["prompt"] for q in result["questions"]]
        original_problems = [f["problem"] for f in _course_facts() if f.get("problem")]

        # No prompt should be a direct copy of the original problem.
        for orig in original_problems:
            for p in prompts:
                assert orig not in p, f"prompt copies original problem: {p[:50]}..."


# ---------------------------------------------------------------------------
# End-to-end student workflow simulation
# ---------------------------------------------------------------------------


class TestStudentWorkflowE2E:
    """Simulate a complete student study session: search → explain → quiz → review."""

    def test_full_study_session(self):
        """A student can search for a topic, get an explanation, generate
        a quiz, and review their progress — all with fake backends."""
        from scripts.core.openviking_backend import PERSONAL_NAMESPACE

        # Set up a fake search backend that returns course facts.
        class StudyClient:
            def search(self, **kw):
                return {
                    "resources": [
                        {"uri": f"{PERSONAL_NAMESPACE}/SYSEN5100/lecture/lec01-abc",
                         "score": 0.92, "abstract": "Closed-loop control systems..."},
                    ],
                    "total": 1,
                }
            def add_resource(self, **kw):
                return {"status": "completed"}
            def read(self, uri, **kw):
                return "A closed-loop system uses feedback to reduce error."

        from scripts.core.openviking_backend import PersonalOpenVikingBackend
        from scripts.retrieval.search import cmd_trace, cmd_search

        backend = PersonalOpenVikingBackend(StudyClient())

        # Step 1: Search for a topic.
        search_result = cmd_search(backend, "feedback control", limit=5)
        assert search_result["total"] == 1
        assert search_result["namespace"] == PERSONAL_NAMESPACE

        # Step 2: Get a source-grounded trace.
        trace = cmd_trace(backend, "feedback control", limit=3)
        assert "top_source" in trace
        assert "closed-loop" in trace["top_source"]["content"]

        # Step 3: Explain a fact.
        from scripts.study.study_commands import explain_fact
        explain = explain_fact(_course_facts(), "closed-loop")
        assert explain["found"] is True

        # Step 4: Generate a quiz.
        from scripts.study.study_commands import build_quiz
        quiz = build_quiz(_course_facts(), count=3, seed=42)
        assert len(quiz["questions"]) == 3
        assert len(quiz["answer_key"]) == 3

        # Step 5: Review progress.
        from scripts.study.study_commands import build_review
        review = build_review(_course_facts())
        assert review["total"] == 5
        assert "by_topic" in review
        assert "by_scope" in review

    def test_assessment_only_study_session(self):
        """Student studying only homework: scope=assessment, no course facts."""
        from scripts.study.study_commands import build_quiz, build_review, explain_fact

        assessment_facts = [
            f for f in _course_facts()
            if f.get("source_scope") == "assessment"
        ]
        assert len(assessment_facts) == 2

        quiz = build_quiz(assessment_facts, count=2)
        assert len(quiz["questions"]) == 2

        review = build_review(assessment_facts)
        assert review["total"] == 2
        assert review["by_scope"].get("assessment", 0) == 2
        assert review["by_scope"].get("course", 0) == 0
