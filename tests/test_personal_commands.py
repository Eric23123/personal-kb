import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def facts_file(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps([{"name": "Plant", "definition": "System being controlled.", "topic": "control"}]), encoding="utf-8")
    return path


def test_quiz_wrapper_uses_injected_llm_when_available(tmp_path):
    from scripts.study.personal_commands import quiz
    def fake_llm(_prompt):
        return '[{"prompt":"New variant","answer_outline":"Solve it","source_method":"control"}]'
    result = quiz(facts_file(tmp_path), count=1, active_model=fake_llm)
    assert result["generation"] == "llm_variant"
    assert result["questions"][0]["prompt"] == "New variant"


def test_quiz_wrapper_falls_back_deterministically_without_llm(tmp_path):
    from scripts.study.personal_commands import quiz
    result = quiz(facts_file(tmp_path), count=1)
    assert result["generation"] == "deterministic_fallback"
    assert result["warning"]
    assert result["count"] == 1
    assert "answer" not in result["questions"][0]


def test_quiz_wrapper_reports_llm_failure_and_falls_back(tmp_path):
    from scripts.study.personal_commands import quiz
    def failing_model(_prompt):
        raise RuntimeError("provider unavailable")
    result = quiz(facts_file(tmp_path), count=1, active_model=failing_model)
    assert result["generation"] == "deterministic_fallback"
    assert "provider unavailable" in result["warning"]


def test_review_wrapper_and_connect_do_not_mutate(tmp_path):
    from scripts.study.personal_commands import connect, review
    assert review(facts_file(tmp_path))["total"] == 1
    assert connect() == {"status": "unconfigured", "mutated": False}


def test_ingest_wrapper_is_dry_run(tmp_path):
    from scripts.study.personal_commands import ingest
    source = tmp_path / "assessment.json"
    source.write_text(json.dumps({"items": [{"problem": "Find K", "solution_pattern": "Use root locus"}]}), encoding="utf-8")
    result = ingest(source, course="TEST", assignment="HW1", source_type="homework")
    assert result["dry_run"] is True
    assert result["source_scope"] == "assessment"
