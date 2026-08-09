"""Focused, offline regression tests for Personal KB media optimizations."""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.media import diagrams  # noqa: E402
from scripts.media import transcribe  # noqa: E402


# A valid transparent 1x1 PNG; tests never need Pillow, GPU models, or network access.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/"
    "d8Z3qQAAAABJRU5ErkJggg=="
)


class DiagramVisionOptimizationTests(unittest.TestCase):
    def test_analyze_diagram_uses_one_injected_vision_call_for_type_and_description(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "diagram.png"
            image_path.write_bytes(PNG_BYTES)
            calls = []

            def transport(image_b64, prompt, api_url, model, timeout):
                calls.append((image_b64, prompt, api_url, model, timeout))
                return json.dumps({
                    "diagram_type": "block_diagram",
                    "description": "A controller feeds a plant through a feedback loop.",
                })

            description, diagram_type = diagrams.analyze_diagram(
                str(image_path), transport=transport, api_url="http://offline"
            )

        self.assertEqual("block_diagram", diagram_type)
        self.assertEqual("A controller feeds a plant through a feedback loop.", description)
        self.assertEqual(1, len(calls))
        self.assertIn("diagram_type", calls[0][1])
        self.assertEqual("http://offline", calls[0][2])

    def test_deduplicate_extracted_images_keeps_the_first_content_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = temp_path / "first.png"
            duplicate = temp_path / "duplicate.png"
            unique = temp_path / "unique.png"
            first.write_bytes(PNG_BYTES)
            duplicate.write_bytes(PNG_BYTES)
            unique.write_bytes(PNG_BYTES + b"different")

            result = diagrams.deduplicate_extracted_images(
                [(1, 1, str(first)), (2, 1, str(duplicate)), (3, 1, str(unique))]
            )

        self.assertEqual([(1, 1, str(first)), (3, 1, str(unique))], result)

    def test_process_pdf_diagrams_cleans_its_unique_temporary_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)

            def extractor(_pdf_path, output_dir):
                image_path = Path(output_dir) / "page1.png"
                image_path.write_bytes(PNG_BYTES)
                return [(1, 0, str(image_path))]

            def transport(_image_b64, _prompt, _api_url, _model, _timeout):
                return '{"diagram_type": "general", "description": "slide"}'

            facts = diagrams.process_pdf_diagrams(
                "slides.pdf",
                "TEST",
                1,
                extractor=extractor,
                transport=transport,
                temp_root=str(temp_root),
            )

            self.assertEqual("slide", facts[0]["content"])
            self.assertEqual([], list(temp_root.iterdir()))


class TranscriptionOptimizationTests(unittest.TestCase):
    def test_whisperx_uses_cached_runtime_and_alignment(self):
        loads = []
        align_calls = []

        class Model:
            def transcribe(self, audio, batch_size):
                self.last_audio = audio
                self.last_batch_size = batch_size
                return {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": " hello"}]}

        class Runtime:
            @staticmethod
            def load_audio(path):
                return f"audio:{path}"

            @staticmethod
            def load_align_model(language_code, device):
                align_calls.append((language_code, device))
                return object(), {"language": language_code}

            @staticmethod
            def align(segments, _model, _metadata, _audio, _device):
                return {"segments": segments}

        def loader(model_size, device, compute_type, vad_method):
            loads.append((model_size, device, compute_type, vad_method))
            return Runtime, Model()

        with tempfile.TemporaryDirectory() as temp_dir:
            transcribe.clear_model_cache()
            first = transcribe.transcribe_whisperx("first.wav", str(Path(temp_dir) / "first.txt"), runtime_loader=loader)
            second = transcribe.transcribe_whisperx("second.wav", str(Path(temp_dir) / "second.txt"), runtime_loader=loader)

        self.assertEqual(["[0.00s -> 1.00s] hello"], first)
        self.assertEqual(first, second)
        self.assertEqual([("turbo", "cuda", "float16", "silero")], loads)
        self.assertEqual([("en", "cuda"), ("en", "cuda")], align_calls)

    def test_whisper_preflight_loads_and_reuses_the_runtime(self):
        loads = []

        def loader(model_size, device, compute_type):
            loads.append((model_size, device, compute_type))
            return object()

        transcribe.clear_model_cache()
        first = transcribe.ensure_whisper_available(
            "large-v3-turbo", "cuda", "float16", model_loader=loader
        )
        second = transcribe.ensure_whisper_available(
            "large-v3-turbo", "cuda", "float16", model_loader=loader
        )

        self.assertTrue(first["available"])
        self.assertTrue(second["available"])
        self.assertEqual(1, len(loads))

    def test_whisper_reuses_cached_model_with_an_injected_loader(self):
        class Segment:
            start, end, text = 0.0, 1.0, " hello"

        class Info:
            language, language_probability, duration = "en", 1.0, 1.0

        class Model:
            def transcribe(self, *_args, **_kwargs):
                return iter([Segment()]), Info()

        loads = []

        def loader(model_size, device, compute_type):
            loads.append((model_size, device, compute_type))
            return Model()

        with tempfile.TemporaryDirectory() as temp_dir:
            transcribe.clear_model_cache()
            transcribe.transcribe_whisper(
                "first.mp3", str(Path(temp_dir) / "first.txt"),
                model_loader=loader, device="cpu", compute_type="int8",
            )
            transcribe.transcribe_whisper(
                "second.mp3", str(Path(temp_dir) / "second.txt"),
                model_loader=loader, device="cpu", compute_type="int8",
            )

        self.assertEqual([("large-v3-turbo", "cpu", "int8")], loads)

    def test_ocr_pdf_uses_150_dpi_and_cleans_a_run_specific_temp_directory(self):
        rendered_dpis = []

        class Pixmap:
            def save(self, path):
                Path(path).write_bytes(PNG_BYTES)

        class Page:
            def get_text(self, mode=None):
                if mode == "dict":
                    return {"blocks": []}
                return ""

            def get_images(self, full=True):
                return [(1,)]

            def get_pixmap(self, dpi):
                rendered_dpis.append(dpi)
                return Pixmap()

        class Document:
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return Page()

            def close(self):
                pass

        ocr_calls = []

        def ocr_func(image_path, engine, **_kwargs):
            self.assertTrue(Path(image_path).exists())
            ocr_calls.append(engine)
            return "recognized"

        fake_pymupdf = SimpleNamespace(open=lambda _path: Document())
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(sys.modules, {"pymupdf": fake_pymupdf}):
            output_path = Path(temp_dir) / "ocr.txt"
            text = transcribe.ocr_pdf(
                "scanned.pdf", str(output_path), ocr_func=ocr_func, temp_root=temp_dir
            )
            self.assertEqual([], [p for p in Path(temp_dir).iterdir() if p.is_dir()])

        self.assertEqual([150], rendered_dpis)
        self.assertEqual(["qwen-vl-plus"], ocr_calls)
        self.assertIn("recognized", text)

    def test_ocr_pdf_preserves_selectable_text_without_requiring_images(self):
        class Page:
            def get_text(self, _mode=None):
                return "The geometry of linear equations"

            def get_images(self, full=True):
                return []

        class Document:
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return Page()

            def close(self):
                pass

        fake_pymupdf = SimpleNamespace(open=lambda _path: Document())
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(sys.modules, {"pymupdf": fake_pymupdf}):
            output_path = Path(temp_dir) / "text.pdf.txt"
            text = transcribe.ocr_pdf("digital.pdf", str(output_path), temp_root=temp_dir)

        self.assertIn("The geometry of linear equations", text)

    def test_cli_exposes_model_reuse_and_ocr_rendering_configuration(self):
        parser = transcribe.build_parser()
        transcription = parser.parse_args(
            ["transcribe", "lecture.mp3", "--device", "cpu", "--compute-type", "int8", "--no-model-cache"]
        )
        ocr = parser.parse_args(
            ["ocr", "scan.pdf", "--dpi", "150", "--fallback-engine", "qwen-vl-max", "--api-url", "http://offline"]
        )
        moss = parser.parse_args(["transcribe", "group.wav", "--engine", "moss", "--moss-device", "cpu"])

        self.assertEqual(("cpu", "int8", False), (transcription.device, transcription.compute_type, transcription.reuse_model))
        self.assertEqual((150, "qwen-vl-max", "http://offline"), (ocr.dpi, ocr.fallback_engine, ocr.api_url))
        self.assertEqual("cpu", moss.moss_device)

    def test_moss_reuses_an_injected_runtime_bundle(self):
        class Segment:
            start, end, speaker, text = 0.0, 1.0, "SPEAKER_00", "hello"

        loads = []

        def loader(model_id, device_preference):
            loads.append((model_id, device_preference))
            return (
                object(), object(), "cpu", "float32",
                lambda _text: [Segment()], lambda _audio: ["message"],
                lambda *_args, **_kwargs: {"text": "raw"},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            transcribe.clear_model_cache()
            transcribe.transcribe_moss("first.wav", str(Path(temp_dir) / "one.txt"), model_loader=loader, device_preference="cpu")
            transcribe.transcribe_moss("second.wav", str(Path(temp_dir) / "two.txt"), model_loader=loader, device_preference="cpu")

        self.assertEqual([("OpenMOSS-Team/MOSS-Transcribe-Diarize", "cpu")], loads)


if __name__ == "__main__":
    unittest.main()
