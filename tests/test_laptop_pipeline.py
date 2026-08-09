from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.ops.laptop_pipeline import detect_input_kind, process_one, run_pipeline


def test_detect_input_kind_routes_supported_inputs():
    assert detect_input_kind(Path("lecture.mp3")) == "audio"
    assert detect_input_kind(Path("slides.pdf")) == "pdf"
    assert detect_input_kind(Path("diagram.png")) == "image"
    assert detect_input_kind(Path("notes.md")) == "text"


def test_process_one_rejects_unknown_binary_input(tmp_path):
    source = tmp_path / "blob.bin"
    source.write_bytes(b"\x00\xff\x00")

    import pytest
    with pytest.raises(ValueError, match="Unsupported input type"):
        process_one(source, tmp_path / "artifacts")


def test_process_one_audio_creates_hash_bound_artifact(tmp_path):
    source = tmp_path / "lecture.wav"
    source.write_bytes(b"synthetic audio")

    def fake_transcribe(path, output, **kwargs):
        Path(output).write_text("[0.00s -> 1.00s] closed loop stability", encoding="utf-8")
        return ["[0.00s -> 1.00s] closed loop stability"]

    result = process_one(
        source,
        output_dir=tmp_path / "artifacts",
        course="TEST-SE424",
        lecture=2,
        transcribe_func=fake_transcribe,
    )

    assert result["kind"] == "audio"
    assert result["artifact"]["extraction_engine"] == "whisperx-turbo"
    assert result["artifact"]["source_hash"] == result["source_hash"]
    assert Path(result["index_path"]).is_file()
    assert "closed loop stability" in Path(result["index_path"]).read_text(encoding="utf-8")


def test_process_one_text_is_idempotent_and_does_not_copy_audio(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("A source-grounded note.", encoding="utf-8")

    first = process_one(source, output_dir=tmp_path / "artifacts", course="TEST-SE424")
    second = process_one(source, output_dir=tmp_path / "artifacts", course="TEST-SE424")

    assert first["source_hash"] == second["source_hash"]
    assert first["index_path"] == second["index_path"]
    assert Path(first["index_path"]).read_text(encoding="utf-8") == "A source-grounded note."


def test_process_one_pdf_uses_injected_ocr_and_returns_artifact(tmp_path):
    source = tmp_path / "slides.pdf"
    source.write_bytes(b"synthetic pdf")

    def fake_ocr(path, output, **kwargs):
        Path(output).write_text("OCR text with a transfer function.", encoding="utf-8")
        return "OCR text with a transfer function."

    result = process_one(source, tmp_path / "artifacts", ocr_pdf_func=fake_ocr)

    assert result["kind"] == "pdf"
    assert result["artifact"]["extraction_engine"] == "pymupdf"
    assert result["artifact"]["source_hash"] == result["source_hash"]
    assert "transfer function" in Path(result["index_path"]).read_text(encoding="utf-8")


def test_process_one_image_uses_injected_diagram_describer(tmp_path):
    source = tmp_path / "diagram.png"
    source.write_bytes(b"synthetic image")

    def fake_diagram(path, **kwargs):
        return "block diagram: input to output", "block_diagram"

    result = process_one(source, tmp_path / "artifacts", diagram_func=fake_diagram)

    assert result["kind"] == "image"
    assert result["artifact"]["extraction_engine"] == "qwen-vl-plus"
    assert result["artifact"]["source_hash"] == result["source_hash"]
    assert "block diagram" in Path(result["index_path"]).read_text(encoding="utf-8")


def test_run_pipeline_persists_index_report_in_manifest(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("A source-grounded note.", encoding="utf-8")

    class FakeBackend:
        def index_file(self, path, **kwargs):
            return SimpleNamespace(
                uri="viking://resources/personal-kb/TEST/source/notes",
                result={"status": "indexed"},
            )

    manifest_path = tmp_path / "manifest.json"
    report = run_pipeline(
        tmp_path,
        tmp_path / "artifacts",
        course="TEST-SE424",
        manifest_path=manifest_path,
        index=True,
        index_backend=FakeBackend(),
    )

    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["index_report"]["counts"]["indexed"] == 1
    assert persisted["index_report"]["counts"]["indexed"] == 1


def test_laptop_pipeline_cli_uses_ascii_safe_json_output(tmp_path, capsys):
    from scripts.ops import laptop_pipeline

    source = tmp_path / "notes.txt"
    source.write_text("x minus y = \u2212z", encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    laptop_pipeline.main(["--input-dir", str(tmp_path), "--output-dir", str(output_dir)])

    captured = capsys.readouterr().out
    captured.encode("gbk")
