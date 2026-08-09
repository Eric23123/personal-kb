"""Laptop media-processing pipeline — local ingestion without OpenViking.

Contract
--------
- ``detect_input_kind(Path) -> str``  returns ``"audio"``, ``"pdf"``, ``"image"``,
  or ``"text"`` for common extensions.
- ``process_one(source, output_dir, *, course, lecture, transcribe_func,
  ocr_pdf_func, diagram_func) -> dict``  returns a deterministic result with
  ``kind``, ``source_hash``, ``index_path``, and (for audio) a validated
  ``artifact`` dict whose ``extraction_engine`` is ``whisperx-turbo``.
- The CLI discovers inputs, processes media locally on the laptop, emits a
  machine-readable pipeline manifest, and indexes only when ``--index`` is
  explicitly provided.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pipeline_laptop.yaml"


def load_pipeline_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the non-secret YAML pipeline configuration."""
    import yaml

    config_path = Path(path).expanduser()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline config must contain a mapping: {config_path}")
    return data

# ---------------------------------------------------------------------------
# Extension maps
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".aiff", ".aif", ".wv", ".amr",
})

PDF_EXTENSIONS: frozenset[str] = frozenset({".pdf"})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp",
    ".svg", ".ico", ".heic", ".heif",
})

TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".log",
    ".csv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".c", ".cpp", ".h", ".hpp", ".rs",
    ".go", ".java", ".rb", ".sh", ".bat", ".ps1", ".toml", ".cfg",
    ".ini", ".sql", ".r", ".rmd", ".mm", ".m",
})


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def detect_input_kind(path: Path) -> str:
    """Return the media kind for *path* based on its file extension.

    Returns one of ``"audio"``, ``"pdf"``, ``"image"``, ``"text"``.
    """
    suffix = path.suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        return "audio"
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    # Fallback: try reading as UTF-8 text (binary files will fail).
    try:
        path.read_text(encoding="utf-8")
        return "text"
    except (UnicodeDecodeError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return the hex-encoded SHA-256 digest of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deterministic_output_path(
    source: Path,
    output_dir: Path,
    kind: str,
    source_hash: str,
) -> Path:
    """Produce a deterministic, idempotent output path for a processed file.

    The path is ``<output_dir>/<source_hash>_<kind_label>.txt`` so two calls
    with the same source produce the same path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kind_labels: dict[str, str] = {
        "audio": "transcript",
        "pdf": "ocr",
        "image": "diagram",
        "text": "text",
    }
    label = kind_labels.get(kind, "output")
    return output_dir / f"{source_hash}_{label}.txt"


# ---------------------------------------------------------------------------
# Core: process_one
# ---------------------------------------------------------------------------

def process_one(
    source: Path,
    output_dir: Path,
    course: str | None = None,
    lecture: int | None = None,
    transcribe_func: Callable[..., Any] | None = None,
    ocr_pdf_func: Callable[..., Any] | None = None,
    diagram_func: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Process a single media source and produce a deterministic artifact.

    Parameters
    ----------
    source:
        Path to the input file.
    output_dir:
        Directory for deterministic output files.
    course:
        Optional course code (e.g. ``"TEST-SE424"``).
    lecture:
        Optional lecture number.
    transcribe_func:
        Injected transcription function.  Called as
        ``transcribe_func(path, output)``.  When ``None``, defaults to
        ``scripts.media.transcribe.transcribe_whisperx`` (**imported lazily**
        to avoid live-network side-effects at import time).
    ocr_pdf_func:
        Reserved for PDF OCR (not exercised by current RED tests).
    diagram_func:
        Reserved for image/diagram processing (not exercised by current RED
        tests).

    Returns
    -------
    dict
        Keys: ``kind``, ``source_hash``, ``index_path``.
        For audio: additionally ``artifact`` (a validated dict with
        ``extraction_engine`` and ``source_hash``).
        For text: the content is written to ``index_path``.
    """
    source = Path(source)

    kind = detect_input_kind(source)
    if kind == "unknown":
        raise ValueError(f"Unsupported input type: {source}")
    source_hash = _sha256_file(source)
    index_path = _deterministic_output_path(source, output_dir, kind, source_hash)

    result: dict[str, Any] = {
        "kind": kind,
        "source_path": str(source.resolve()),
        "source_hash": source_hash,
        "index_path": str(index_path),
    }

    if kind == "audio":
        # ── resolve transcribe_func ──────────────────────────────────────
        if transcribe_func is None:
            # Lazy import — avoids live network calls on module import
            from scripts.media.transcribe import transcribe_whisperx
            transcribe_func = transcribe_whisperx

        transcribe_func(str(source), str(index_path))

        # ── build & validate artifact ────────────────────────────────────
        from scripts.ingestion.media_artifact import make_artifact, validate_artifact

        content = index_path.read_text(encoding="utf-8")
        artifact = make_artifact(
            content=content,
            source_path=str(source),
            extraction_engine="whisperx-turbo",
            source_type="transcript",
            course=course,
            lecture=lecture,
            output_path=str(index_path),
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(
                f"Artifact validation failed for {source}: {errors}"
            )

        result["artifact"] = artifact.to_dict()

    elif kind == "text":
        # Copy source content to deterministic output (idempotent).
        content = source.read_text(encoding="utf-8")
        index_path.write_text(content, encoding="utf-8")

    elif kind == "pdf":
        if ocr_pdf_func is None:
            from scripts.media.transcribe import ocr_pdf
            ocr_pdf_func = ocr_pdf
        ocr_pdf_func(str(source), str(index_path))
        from scripts.ingestion.media_artifact import make_artifact, validate_artifact
        artifact = make_artifact(
            content=index_path.read_text(encoding="utf-8"),
            source_path=str(source),
            extraction_engine="pymupdf",
            source_type="ocr_text",
            course=course,
            lecture=lecture,
            output_path=str(index_path),
            extra={"vision_model": "qwen-vl-plus"},
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"Artifact validation failed for {source}: {errors}")
        result["artifact"] = artifact.to_dict()

    elif kind == "image":
        if diagram_func is None:
            from scripts.media.diagrams import describe_diagram
            diagram_func = describe_diagram
        described = diagram_func(str(source))
        if isinstance(described, tuple):
            description, diagram_type = described
        else:
            description, diagram_type = str(described), "general"
        index_path.write_text(str(description), encoding="utf-8")
        from scripts.ingestion.media_artifact import make_artifact, validate_artifact
        artifact = make_artifact(
            content=str(description),
            source_path=str(source),
            extraction_engine="qwen-vl-plus",
            source_type="diagram_description",
            course=course,
            lecture=lecture,
            output_path=str(index_path),
            extra={"diagram_type": str(diagram_type)},
        )
        errors = validate_artifact(artifact)
        if errors:
            raise ValueError(f"Artifact validation failed for {source}: {errors}")
        result["artifact"] = artifact.to_dict()

    return result


# ---------------------------------------------------------------------------
# CLI / Orchestration
# ---------------------------------------------------------------------------

def discover_inputs(
    input_dir: Path,
    extensions: frozenset[str] | None = None,
) -> list[Path]:
    """Walk *input_dir* and return all files matching known media extensions.

    Parameters
    ----------
    input_dir:
        Directory to scan recursively.
    extensions:
        Optional set of lowercase extensions (with leading dot).  When
        ``None``, all known extensions are used.

    Returns
    -------
    list[Path]
        Absolute paths sorted for reproducibility.
    """
    if extensions is None:
        extensions = AUDIO_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS

    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        return []

    paths: list[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            paths.append(p)
    paths.sort()
    return paths


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    *,
    course: str | None = None,
    lecture: int | None = None,
    transcribe_func: Callable[..., Any] | None = None,
    ocr_pdf_func: Callable[..., Any] | None = None,
    diagram_func: Callable[..., Any] | None = None,
    manifest_path: Path | None = None,
    index: bool = False,
    index_backend: Any = None,
) -> dict[str, Any]:
    """Discover and process all media in *input_dir*, returning a manifest.

    This is the batch-capable entry-point that the CLI delegates to.
    """
    inputs = discover_inputs(input_dir)
    manifest_entries: list[dict[str, Any]] = []

    for src in inputs:
        entry = process_one(
            src,
            output_dir,
            course=course,
            lecture=lecture,
            transcribe_func=transcribe_func,
            ocr_pdf_func=ocr_pdf_func,
            diagram_func=diagram_func,
        )
        manifest_entries.append(entry)

    manifest: dict[str, Any] = {
        "pipeline": "laptop_pipeline",
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "course": course,
        "lecture": lecture,
        "items": manifest_entries,
    }

    # Write manifest
    if manifest_path is None:
        manifest_path = Path(output_dir) / "pipeline_manifest.json"
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)

    if index:
        # Indexing is deliberately opt-in.  Construct the live backend only
        # here, after all local processing and manifest emission completed.
        if index_backend is None:
            from scripts.core.openviking_backend import PersonalOpenVikingBackend
            import os
            index_backend = PersonalOpenVikingBackend(
                base_url=os.environ.get("PERSONAL_KB_OPENVIKING_URL", "http://127.0.0.1:1934"),
                root=Path.cwd(),
            )
        from scripts.ops.batch_ingest import BatchItem, run_batch
        batch_items: list[BatchItem] = []
        source_types = {
            "audio": "transcript",
            "pdf": "ocr_text",
            "image": "diagram_description",
            "text": "source",
        }
        for entry in manifest_entries:
            index_path = Path(entry["index_path"])
            if index_path.exists():
                batch_items.append(
                    BatchItem(
                        index_path=index_path,
                        source_path=Path(entry["source_path"]),
                        source_hash=entry["source_hash"],
                        source_type=source_types.get(entry["kind"], "source"),
                        course=course,
                        lecture=lecture,
                    )
                )
        if batch_items:
            manifest["index_report"] = run_batch(
                batch_items,
                backend=index_backend,
                log_path=Path(output_dir) / "index_ingestion.jsonl",
            )

    # Rewrite after optional indexing so the on-disk manifest is complete.
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the stable CLI parser for testing and offline inspection."""
    parser = argparse.ArgumentParser(
        description="Laptop media-processing pipeline",
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH),
        help="Pipeline YAML configuration (default: config/pipeline_laptop.yaml).",
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing media files to process.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for artifacts (default: output.base_dir from config).",
    )
    parser.add_argument(
        "--course", default=None,
        help="Course code (default: courses.default from config).",
    )
    parser.add_argument(
        "--lecture", type=int, default=None,
        help="Lecture number.",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Path for the pipeline manifest JSON (default: <output-dir>/pipeline_manifest.json).",
    )
    parser.add_argument(
        "--index", action="store_true", default=False,
        help="Index processed artifacts into OpenViking via batch_ingest.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point for ``python -m scripts.ops.laptop_pipeline``."""
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config)
    output_defaults = config.get("output", {})
    output_dir = Path(args.output_dir or output_defaults.get("base_dir", "artifacts/laptop_pipeline"))
    course = args.course if args.course is not None else config.get("courses", {}).get("default")
    manifest_name = output_defaults.get("manifest_file", "pipeline_manifest.json")
    manifest_path = Path(args.manifest) if args.manifest else output_dir / manifest_name

    manifest = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=output_dir,
        course=course,
        lecture=args.lecture,
        manifest_path=manifest_path,
        index=args.index,
    )

    # Keep the saved manifest UTF-8, but make the Windows CLI response safe
    # for legacy GBK consoles after processing source material with symbols.
    print(json.dumps(manifest, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
