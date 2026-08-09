import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def facts():
    return [
        {"name": "Closed-loop system", "definition": "Inputs depend in part on outputs.", "type": "concept", "topic": "feedback"},
        {"name": "Final-value theorem", "definition": "It finds the steady-state value under its conditions.", "type": "formula", "topic": "steady state"},
    ]


def test_quiz_builds_deterministic_problem_prompts_without_inline_answers():
    from scripts.study.study_commands import build_quiz

    result = build_quiz(facts(), count=2, seed=7)

    assert len(result["questions"]) == 2
    assert result["questions"][0]["prompt"]
    assert "answer" not in result["questions"][0]
    assert len(result["answer_key"]) == 2
    assert result["questions"] == build_quiz(facts(), count=2, seed=7)["questions"]
    assert result["answer_key"] == build_quiz(facts(), count=2, seed=7)["answer_key"]


def test_llm_quiz_generates_new_variants_with_injected_callable():
    from scripts.study.study_commands import build_llm_quiz

    seen = []
    def fake_llm(prompt):
        seen.append(prompt)
        return '[{"prompt":"For a cart of mass 2, derive the equilibrium input.","answer_outline":"Set derivatives to zero and solve.","source_method":"equilibrium analysis"}]'

    result = build_llm_quiz([{"name":"HW1 Problem 1","problem":"original","solution_pattern":"method"}], fake_llm, count=1)
    assert result["generation"] == "llm_variant"
    assert result["questions"][0]["prompt"] != "original"
    assert result["answer_key"][0]["answer"] == "Set derivatives to zero and solve."
    assert seen


def test_quiz_uses_problem_statement_for_assessment_facts():
    from scripts.study.study_commands import build_quiz

    result = build_quiz([{
        "name": "HW1 Problem 1",
        "problem": "Derive the cart and pendulum accelerations.",
        "solution_pattern": "Solve the coupled matrix equations.",
        "type": "assessment-solution",
    }], count=1)

    assert result["questions"][0]["prompt"] == "Practice variant: solve the underlying method tested by HW1 Problem 1, without copying the original wording."
    assert result["answer_key"][0]["answer"] == "Solve the coupled matrix equations."


def test_review_groups_assessment_facts_by_assignment_and_type():
    from scripts.study.study_commands import build_review

    result = build_review([{
        "name": "HW1 Problem 1",
        "problem": "Derive the dynamics.",
        "solution_pattern": "Solve the matrix equations.",
        "assignment": "HW1",
        "source_scope": "assessment",
        "type": "assessment-solution",
    }])

    assert result["by_assignment"]["HW1"] == 1
    assert result["by_scope"]["assessment"] == 1


def test_review_groups_facts_by_topic_and_type():
    from scripts.study.study_commands import build_review

    result = build_review(facts())

    assert result["total"] == 2
    assert result["by_topic"]["feedback"] == 1
    assert result["by_type"]["formula"] == 1


def test_explain_returns_requested_fact_with_related_context():
    from scripts.study.study_commands import explain_fact

    result = explain_fact(facts(), "closed-loop")

    assert result["found"] is True
    assert result["fact"]["name"] == "Closed-loop system"
    assert result["related"] == []


def test_explain_missing_fact_is_explicit():
    from scripts.study.study_commands import explain_fact

    result = explain_fact(facts(), "nonexistent")

    assert result == {"found": False, "query": "nonexistent", "fact": None, "related": []}
