import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_normalize_assessment_items_adds_assessment_scope_and_preserves_solution_fields():
    from scripts.study.assessment_processor import normalize_assessment_items

    items = normalize_assessment_items([
        {
            "problem": "Find the steady-state error.",
            "solution_pattern": "Apply the final-value theorem.",
            "common_mistakes": ["Ignore the stability condition."],
            "source_ref": "p. 4",
        }
    ], course="TEST", source_type="homework", assignment="HW1")

    assert items == [{
        "problem": "Find the steady-state error.",
        "solution_pattern": "Apply the final-value theorem.",
        "common_mistakes": ["Ignore the stability condition."],
        "source_ref": "p. 4",
        "course": "TEST",
        "source_type": "homework",
        "source_scope": "assessment",
        "assignment": "HW1",
    }]


def test_process_assessment_json_dry_run_returns_hindsight_ready_facts(tmp_path):
    from scripts.study.assessment_processor import process_assessment

    source = tmp_path / "hw.json"
    source.write_text(json.dumps({"items": [
        {"problem": "Compute K.", "solution_pattern": "Use root locus.", "common_mistakes": []}
    ]}), encoding="utf-8")

    result = process_assessment(source, course="TEST", source_type="exam", assignment="Midterm", dry_run=True)

    assert result["dry_run"] is True
    assert result["count"] == 1
    assert result["facts"][0]["source_scope"] == "assessment"
    assert result["hindsight_items"][0]["tags"]
    assert "scope:assessment" in result["hindsight_items"][0]["tags"]
    assert result["hindsight_items"][0]["metadata"]["assignment"] == "Midterm"


def test_process_assessment_rejects_non_assessment_source_type(tmp_path):
    from scripts.study.assessment_processor import process_assessment

    source = tmp_path / "lecture.json"
    source.write_text(json.dumps({"items": []}), encoding="utf-8")

    try:
        process_assessment(source, course="TEST", source_type="lecture", assignment="L1", dry_run=True)
    except ValueError as exc:
        assert "homework" in str(exc) and "exam" in str(exc)
    else:
        raise AssertionError("expected source type validation failure")
