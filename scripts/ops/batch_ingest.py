"""Batch ingestion with resume via JSONL log.

Reads a previously written JSONL log to skip already-indexed items (matched by
source_hash + source_path).  Continues after individual errors and records
every outcome so the log is always resumable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BatchItem:
    """One item to ingest.

    ``index_path`` is the actual file path passed to the backend (may differ
    from ``source_path`` in manifest-augmented workflows).
    """

    index_path: str | Path
    source_path: str | Path
    source_hash: str
    source_type: str
    course: str | None = None
    lecture: int | None = None
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_indexed_keys(log_path: str | Path) -> set[tuple[str, str]]:
    """Return {(source_hash, source_path)} for every ``"indexed"`` entry."""
    path = Path(log_path)
    if not path.is_file():
        return set()

    keys: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("status") in {"indexed", "skipped"} and not entry.get("dry_run"):
                sh = entry.get("source_hash", "")
                sp = entry.get("source_path", "")
                if sh and sp:
                    keys.add((sh, str(Path(sp).expanduser().resolve())))
    return keys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_batch(
    items: list[BatchItem],
    *,
    backend: Any,
    log_path: str | Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Index every *item*, writing a resumable JSONL log.

    Parameters
    ----------
    items:
        Batch items to process.
    backend:
        Anything with an ``index_file(path, **kwargs)`` method that returns a
        resource descriptor or raises on failure.
    log_path:
        Path to the JSONL log file (created if it does not exist; appended
        otherwise).
    dry_run:
        When *True* the backend receives ``dry_run=True`` so it can skip
        persistent writes, but the item is still logged as ``"indexed"`` so
        callers can inspect the plan.

    Returns
    -------
    dict with:
    * ``counts`` — ``{"indexed": N, "skipped": N, "failed": N}``
    * ``failed`` — list of dicts describing each failed item
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    indexed_keys = _read_indexed_keys(log_path)

    counts: dict[str, int] = {"indexed": 0, "skipped": 0, "failed": 0}
    failed: list[dict[str, Any]] = []

    with log_path.open("a", encoding="utf-8") as log_fh:
        for item in items:
            sp = str(Path(item.source_path).expanduser().resolve())
            key = (item.source_hash, sp)

            # --- Resume: skip already-indexed items ---
            if key in indexed_keys:
                counts["skipped"] += 1
                continue

            # --- Attempt ingestion ---
            try:
                kwargs: dict[str, Any] = {
                    "source_type": item.source_type,
                    "source_hash": item.source_hash,
                    "provenance_source_path": sp,
                    "dry_run": dry_run,
                }
                if item.course is not None:
                    kwargs["course"] = item.course
                if item.lecture is not None:
                    kwargs["lecture"] = item.lecture
                if item.metadata is not None:
                    kwargs["metadata"] = item.metadata

                resource = backend.index_file(str(item.index_path), **kwargs)
                resource_result = getattr(resource, "result", None)
                if isinstance(resource_result, dict) and resource_result.get("status") == "skipped":
                    status = "skipped"
                    counts["skipped"] += 1
                else:
                    status = "indexed"
                    counts["indexed"] += 1
                entry: dict[str, Any] = {
                    "status": status,
                    "source_path": sp,
                    "source_hash": item.source_hash,
                    "source_type": item.source_type,
                    "course": item.course,
                    "lecture": item.lecture,
                    "uri": getattr(resource, "uri", None),
                    "reason": resource_result.get("reason") if isinstance(resource_result, dict) else None,
                    "dry_run": dry_run,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_fh.flush()
                indexed_keys.add(key)

            except Exception as exc:
                # Log the error so it is visible in the file; continue.
                error_entry: dict[str, Any] = {
                    "status": "error",
                    "source_path": sp,
                    "source_hash": item.source_hash,
                    "source_type": item.source_type,
                    "course": item.course,
                    "lecture": item.lecture,
                    "error": str(exc),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                log_fh.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
                log_fh.flush()
                counts["failed"] += 1
                failed.append(
                    {
                        "source_path": sp,
                        "source_hash": item.source_hash,
                        "source_type": item.source_type,
                        "course": item.course,
                        "lecture": item.lecture,
                        "error": str(exc),
                    }
                )

    return {"counts": counts, "failed": failed}
