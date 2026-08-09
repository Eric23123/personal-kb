"""Tests for media-fixture validation (Priority 4) — all offline, no live services.

Every test uses injectable fake transports/models/loaders. No Whisper, MOSS,
pymupdf (real PDFs only), Qwen-VL, or DeepSeek API calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.ingestion.media_artifact import (
    EXTRACTION_ENGINES,
    ArtifactRecord,
    make_artifact,
    validate_artifact,
)


# ---------------------------------------------------------------------------
# ArtifactRecord / make_artifact tests
# ---------------------------------------------------------------------------


def test_make_artifact_requires_all_four_required_fields():
    """Every make_artifact call must produce all four fields."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "test.txt"
        source.write_text("test content", encoding="utf-8")

        record = make_artifact(
            content="output",
            source_path=str(source),
            extraction_engine="whisperx-turbo",
            source_type="transcript",
        )

    d = record.to_dict()
    assert d["source_hash"] is not None
    assert len(d["source_hash"]) == 64
    assert d["extraction_engine"] == "whisperx-turbo"
    assert d["ingestion_timestamp"] is not None
    assert d["downstream_resource_status"] == "pending_openviking"
    assert d["content"] == "output"
    assert d["source_type"] == "transcript"


def test_make_artifact_rejects_unknown_engine():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "test.txt"
        source.write_text("test", encoding="utf-8")
        with pytest.raises(ValueError, match="Unknown extraction_engine"):
            make_artifact(
                content="x", source_path=str(source),
                extraction_engine="unknown-engine",
                source_type="transcript",
            )


def test_make_artifact_rejects_invalid_status():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "test.txt"
        source.write_text("test", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid downstream_resource_status"):
            make_artifact(
                content="x", source_path=str(source),
                extraction_engine="pymupdf",
                source_type="text",
                downstream_status="invalid_status",
            )


def test_make_artifact_rejects_missing_source():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with pytest.raises(FileNotFoundError):
            make_artifact(
                content="x", source_path=str(root / "nonexistent.txt"),
                extraction_engine="pymupdf",
                source_type="text",
            )


def test_make_artifact_accepts_all_valid_engines():
    """Every engine listed in EXTRACTION_ENGINES must be accepted."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "test.txt"
        source.write_text("test", encoding="utf-8")

        for engine in EXTRACTION_ENGINES:
            record = make_artifact(
                content="x", source_path=str(source),
                extraction_engine=engine,
                source_type="text",
            )
            assert record.extraction_engine == engine


def test_make_artifact_accepts_all_valid_statuses():
    """Every downstream status must be accepted."""
    from scripts.ingestion.media_artifact import DOWNSTREAM_STATUSES
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "test.txt"
        source.write_text("test", encoding="utf-8")

        for status in DOWNSTREAM_STATUSES:
            record = make_artifact(
                content="x", source_path=str(source),
                extraction_engine="pymupdf",
                source_type="text",
                downstream_status=status,
            )
            assert record.downstream_resource_status == status


def test_make_artifact_custom_timestamp():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "test.txt"
        source.write_text("test", encoding="utf-8")

        record = make_artifact(
            content="x", source_path=str(source),
            extraction_engine="pymupdf", source_type="text",
            timestamp="2026-07-21T12:00:00Z",
        )
        assert record.ingestion_timestamp == "2026-07-21T12:00:00Z"


def test_validate_artifact_passes_with_all_fields():
    errors = validate_artifact({
        "source_hash": "a" * 64,
        "extraction_engine": "pymupdf",
        "ingestion_timestamp": "2026-07-21T12:00:00Z",
        "downstream_resource_status": "indexed",
        "content": "test content",
    })
    assert errors == []


def test_validate_artifact_rejects_missing_fields():
    errors = validate_artifact({})
    assert len(errors) >= 4
    assert any("source_hash" in e for e in errors)
    assert any("extraction_engine" in e for e in errors)
    assert any("ingestion_timestamp" in e for e in errors)
    assert any("downstream_resource_status" in e for e in errors)


def test_validate_artifact_rejects_invalid_hash():
    errors = validate_artifact({
        "source_hash": "not-a-hash",
        "extraction_engine": "pymupdf",
        "ingestion_timestamp": "now",
        "downstream_resource_status": "indexed",
        "content": "test",
    })
    assert any("source_hash" in e for e in errors)


def test_artifact_record_to_dict_includes_extra():
    record = ArtifactRecord(
        source_hash="a" * 64,
        extraction_engine="pymupdf",
        ingestion_timestamp="2026-07-21T12:00:00Z",
        downstream_resource_status="indexed",
        content="test",
        source_path="/tmp/test.txt",
        source_type="text",
        extra={"diagram_type": "block_diagram", "page": 1},
    )
    d = record.to_dict()
    assert d["diagram_type"] == "block_diagram"
    assert d["page"] == 1


def test_artifact_record_to_dict_omits_none_course_lecture():
    record = ArtifactRecord(
        source_hash="a" * 64,
        extraction_engine="pymupdf",
        ingestion_timestamp="2026-07-21T12:00:00Z",
        downstream_resource_status="indexed",
        content="test",
        source_path="/tmp/test.txt",
        source_type="text",
    )
    d = record.to_dict()
    assert "course" not in d
    assert "lecture" not in d


# ---------------------------------------------------------------------------
# Individual fixture tests
# ---------------------------------------------------------------------------

from scripts.media.media_fixture_validation import (
    _SILENT_WAV,
    _PNG_1X1,
    _MINIMAL_PDF,
    fixture_whisperx,
    fixture_moss,
    fixture_pymupdf,
    fixture_qwen_ocr,
    fixture_qwen_vl,
    fixture_deepseek_synthesis,
    run_all_fixtures,
)


def test_fixture_whisperx_produces_artifact():
    with tempfile.TemporaryDirectory() as td:
        result = fixture_whisperx(Path(td))
    assert result.passed, result.errors
    assert result.artifact is not None
    errors = validate_artifact(result.artifact)
    assert errors == []
    art = result.artifact
    assert art["extraction_engine"] == "whisperx-turbo"
    assert art["source_type"] == "transcript"
    assert "closed-loop" in art["content"].lower()


def test_fixture_moss_produces_artifact():
    with tempfile.TemporaryDirectory() as td:
        result = fixture_moss(Path(td))
    assert result.passed, result.errors
    assert result.artifact is not None
    errors = validate_artifact(result.artifact)
    assert errors == []
    art = result.artifact
    assert art["extraction_engine"] == "moss-0.9b"
    assert art["source_type"] == "transcript"


def test_fixture_pymupdf_produces_artifact():
    with tempfile.TemporaryDirectory() as td:
        result = fixture_pymupdf(Path(td))
    assert result.passed, result.errors
    assert result.artifact is not None
    errors = validate_artifact(result.artifact)
    assert errors == []
    art = result.artifact
    assert art["extraction_engine"] == "pymupdf"
    assert "TEST PDF" in art["content"]


def test_fixture_qwen_ocr_produces_artifact():
    with tempfile.TemporaryDirectory() as td:
        result = fixture_qwen_ocr(Path(td))
    assert result.passed, result.errors
    assert result.artifact is not None
    errors = validate_artifact(result.artifact)
    assert errors == []
    art = result.artifact
    assert art["extraction_engine"] == "qwen-vl-plus"
    assert art["source_type"] == "ocr_text"
    assert "closed-loop" in art["content"].lower()


def test_fixture_qwen_vl_produces_artifact():
    with tempfile.TemporaryDirectory() as td:
        result = fixture_qwen_vl(Path(td))
    assert result.passed, result.errors
    assert result.artifact is not None
    errors = validate_artifact(result.artifact)
    assert errors == []
    art = result.artifact
    assert art["extraction_engine"] == "qwen-vl-plus"
    assert art["source_type"] == "diagram_description"
    assert art["diagram_type"] == "block_diagram"


def test_fixture_deepseek_synthesis_produces_artifact():
    with tempfile.TemporaryDirectory() as td:
        result = fixture_deepseek_synthesis(Path(td))
    assert result.passed, result.errors
    assert result.artifact is not None
    errors = validate_artifact(result.artifact)
    assert errors == []
    art = result.artifact
    assert art["extraction_engine"] == "deepseek-v4-pro"
    assert art["source_type"] == "synthesis"


# ---------------------------------------------------------------------------
# Full orchestration tests
# ---------------------------------------------------------------------------


def test_run_all_fixtures_passes_all_six():
    report = run_all_fixtures()
    assert report["valid"] is True
    assert report["fixture_count"] == 6
    assert report["passed_count"] == 6
    assert report["failed_count"] == 0


def test_run_all_fixtures_json_output():
    report = run_all_fixtures()
    encoded = json.dumps(report, default=str)
    parsed = json.loads(encoded)
    assert parsed["valid"] is True
    assert len(parsed["fixtures"]) == 6
    for fx in parsed["fixtures"]:
        assert fx["passed"] is True, f"{fx['name']} failed: {fx.get('errors')}"


def test_run_all_fixtures_every_artifact_has_four_fields():
    """Every fixture output must carry the four required provenance fields."""
    report = run_all_fixtures()
    for fx in report["fixtures"]:
        # We validate each one — the fixture_result already includes validation
        assert fx["passed"], f"{fx['name']} failed"
        # But also explicitly check if artifact was present (it always is for
        # passed fixtures because validation checks it).
        # The 'errors' list being empty on a passed fixture means no
        # validation errors.
        assert fx["errors"] == [], f"{fx['name']} has errors: {fx['errors']}"


# ---------------------------------------------------------------------------
# Binary fixture integrity
# ---------------------------------------------------------------------------


def test_silent_wav_is_valid_wav():
    """The synthetic WAV must have a valid RIFF header."""
    assert _SILENT_WAV[:4] == b"RIFF"
    assert _SILENT_WAV[8:12] == b"WAVE"
    assert len(_SILENT_WAV) > 44  # header + some data


def test_minimal_pdf_is_valid_pdf():
    """The synthetic PDF must have a valid PDF header."""
    assert _MINIMAL_PDF[:5] == b"%PDF-"
    assert b"%%EOF" in _MINIMAL_PDF


def test_png_1x1_is_valid_png():
    """The synthetic PNG must have a valid PNG header."""
    assert _PNG_1X1[:8] == b"\x89PNG\r\n\x1a\n"
    # The chunk structure may vary across valid PNGs; just verify the file
    # is recognized by the imaging library.
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(_PNG_1X1))
    assert img.size == (1, 1)
