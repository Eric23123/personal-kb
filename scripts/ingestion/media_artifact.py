"""Unified artifact metadata for Personal KB media processing outputs.

Every media processor output (transcription, OCR, diagram description,
synthesis note) must carry these four fields so downstream consumers can
trace provenance, verify freshness, and track ingestion status without
depending on an external manifest.

Usage:
    from scripts.ingestion.media_artifact import ArtifactRecord, make_artifact

    record = make_artifact(
        content="transcribed text...",
        source_path="/data/lecture.mp3",
        extraction_engine="whisperx-turbo",
        source_type="transcript",
        downstream_status="pending_openviking",
        course="PERSONAL-ALPHA",
        lecture=1,
    )
    # record is a dict with all required fields already set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Valid status values for downstream_resource_status
# ---------------------------------------------------------------------------

DOWNSTREAM_STATUSES = frozenset({
    "pending_openviking",    # artifact produced but not yet indexed
    "indexed",               # successfully indexed in OpenViking
    "pending_hindsight",     # indexed in OpenViking, awaiting Hindsight
    "complete",              # fully ingested in both systems
    "skipped",               # intentionally not ingested (e.g. duplicate)
    "failed",                # ingestion attempted but failed
})

EXTRACTION_ENGINES = frozenset({
    "whisperx-turbo",
    "moss-0.9b",
    "pymupdf",
    "qwen-vl-plus",
    "deepseek-v4-pro",
})


# ---------------------------------------------------------------------------
# Artifact record
# ---------------------------------------------------------------------------


@dataclass
class ArtifactRecord:
    """Structured media processing artifact carrying the four required fields.

    This is the contract: every media pipeline output MUST produce an
    ArtifactRecord or a dict with all required keys before entering the
    OpenViking or Hindsight ingestion path.
    """

    # ── Required provenance fields ──
    source_hash: str
    """SHA-256 of the original input file (source file, not derived artifact)."""

    extraction_engine: str
    """The tool/model used, e.g. 'whisperx-turbo', 'qwen-vl-plus'."""

    ingestion_timestamp: str
    """ISO 8601 timestamp of when this artifact was produced."""

    downstream_resource_status: str
    """One of the valid ``DOWNSTREAM_STATUSES`` values."""

    # ── Content ──
    content: str
    """The extracted/processed text content."""

    # ── Identification ──
    source_path: str
    """Absolute path to the original source file."""

    source_type: str
    """Semantic type: 'transcript', 'ocr_text', 'diagram_description', 'synthesis', etc."""

    # ── Optional metadata ──
    course: str | None = None
    lecture: int | None = None
    output_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output or fact ingestion."""
        d: dict[str, Any] = {
            "source_hash": self.source_hash,
            "extraction_engine": self.extraction_engine,
            "ingestion_timestamp": self.ingestion_timestamp,
            "downstream_resource_status": self.downstream_resource_status,
            "content": self.content,
            "source_path": self.source_path,
            "source_type": self.source_type,
        }
        if self.course is not None:
            d["course"] = self.course
        if self.lecture is not None:
            d["lecture"] = self.lecture
        if self.output_path is not None:
            d["output_path"] = self.output_path
        d.update(self.extra)
        return d


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_artifact(
    content: str,
    source_path: str | Path,
    extraction_engine: str,
    source_type: str,
    *,
    downstream_status: str = "pending_openviking",
    course: str | None = None,
    lecture: int | None = None,
    output_path: str | None = None,
    timestamp: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ArtifactRecord:
    """Create an ArtifactRecord with validated fields.

    ``source_hash`` is computed as SHA-256 of the source file content.
    The caller must ensure the file exists at ``source_path``.
    """
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"Source file not found: {source}")

    if extraction_engine not in EXTRACTION_ENGINES:
        raise ValueError(
            f"Unknown extraction_engine {extraction_engine!r}; "
            f"valid: {sorted(EXTRACTION_ENGINES)}"
        )

    if downstream_status not in DOWNSTREAM_STATUSES:
        raise ValueError(
            f"Invalid downstream_resource_status {downstream_status!r}; "
            f"valid: {sorted(DOWNSTREAM_STATUSES)}"
        )

    source_hash = _sha256_file(source)
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    return ArtifactRecord(
        source_hash=source_hash,
        extraction_engine=extraction_engine,
        ingestion_timestamp=ts,
        downstream_resource_status=downstream_status,
        content=content,
        source_path=str(source.resolve()),
        source_type=source_type,
        course=course,
        lecture=lecture,
        output_path=output_path,
        extra=extra or {},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(record: dict[str, Any] | ArtifactRecord) -> list[str]:
    """Validate that an artifact carries all four required fields.

    Returns a list of error messages; empty list means valid.
    """
    errors: list[str] = []
    if isinstance(record, ArtifactRecord):
        d = record.to_dict()
    else:
        d = record

    required = [
        "source_hash",
        "extraction_engine",
        "ingestion_timestamp",
        "downstream_resource_status",
    ]
    for key in required:
        if key not in d or not d[key]:
            errors.append(f"Missing required field: {key}")
        elif key == "source_hash" and not _is_sha256(d[key]):
            errors.append(f"source_hash is not a valid SHA-256: {d[key][:40]}...")
        elif key == "extraction_engine" and d[key] not in EXTRACTION_ENGINES:
            errors.append(
                f"Unknown extraction_engine: {d[key]!r}; "
                f"valid: {sorted(EXTRACTION_ENGINES)}"
            )
        elif key == "downstream_resource_status" and d[key] not in DOWNSTREAM_STATUSES:
            errors.append(
                f"Invalid downstream_resource_status: {d[key]!r}; "
                f"valid: {sorted(DOWNSTREAM_STATUSES)}"
            )

    if "content" not in d or not d.get("content"):
        errors.append("Missing required field: content")

    return errors


def _is_sha256(value: str) -> bool:
    import re
    return bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(value)))
