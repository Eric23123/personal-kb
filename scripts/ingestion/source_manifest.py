"""Validate and execute the Personal KB source manifest.

The manifest is the auditable boundary between local source files and the
OpenViking/lexical indexes. It requires a full SHA-256 content hash and keeps
lecture metadata optional for homework, exams, and multi-lecture materials.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

try:
    from ..retrieval.lexical_index import LexicalIndex, build_lexical_index, save_index
    from ..core.openviking_backend import (
        AUDIO_EXTENSIONS,
        PersonalOpenVikingBackend,
        is_approved_personal_path,
        resource_uri,
    )
except ImportError:  # pragma: no cover - direct CLI use
    from scripts.retrieval.lexical_index import LexicalIndex, build_lexical_index, save_index
    from scripts.core.openviking_backend import (
        AUDIO_EXTENSIONS,
        PersonalOpenVikingBackend,
        is_approved_personal_path,
        resource_uri,
    )

HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REQUIRED_FIELDS = ("source_path", "source_hash", "source_type")
OPTIONAL_FIELDS = (
    "course",
    "semester",
    "date",
    "lecture",
    "corpus",
    "file_name",
    "derived_text_path",
    "derived_hash",
)


def load_manifest(manifest: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, dict):
        return manifest
    return json.loads(Path(manifest).expanduser().read_text(encoding="utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(source_path: str, root: Path) -> Path:
    path = Path(source_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _metadata_for_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in entry.items()
        if key not in {"source_path", "file_name", "derived_text_path", "derived_hash"}
        and value is not None
    }


def _entries(manifest: dict[str, Any], root: Path) -> list[tuple[Path, dict[str, Any]]]:
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ValueError("manifest.sources must be a list")
    return [(_resolve_path(entry["source_path"], root), entry) for entry in sources]


def validate_manifest(
    manifest: str | Path | dict[str, Any],
    root: str | Path,
) -> list[str]:
    """Return validation errors; an empty list means the manifest is valid."""
    data = load_manifest(manifest)
    base = Path(root).expanduser().resolve()
    errors: list[str] = []
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        return ["manifest.sources must be a non-empty list"]

    seen_paths: set[Path] = set()
    for index, entry in enumerate(sources):
        label = f"sources[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label} must be an object")
            continue
        missing = [key for key in REQUIRED_FIELDS if key not in entry]
        if missing:
            errors.append(f"{label} missing required fields: {', '.join(missing)}")
            continue
        source_path = _resolve_path(str(entry["source_path"]), base)
        if source_path in seen_paths:
            errors.append(f"{label} duplicates source_path: {source_path}")
        seen_paths.add(source_path)
        if not source_path.is_file():
            errors.append(f"{label} source does not exist: {source_path}")
            continue
        derived_path_value = entry.get("derived_text_path")
        derived_hash = entry.get("derived_hash")
        if derived_path_value is not None:
            derived_path = _resolve_path(str(derived_path_value), base)
            if not derived_path.is_file():
                errors.append(f"{label} derived text does not exist: {derived_path}")
            elif derived_hash is not None:
                if not isinstance(derived_hash, str) or not HASH_RE.fullmatch(derived_hash):
                    errors.append(f"{label}.derived_hash must be a full 64-character SHA-256")
                elif sha256_file(derived_path) != derived_hash.casefold():
                    errors.append(f"{label} derived hash mismatch: {derived_path}")
        elif derived_hash is not None:
            errors.append(f"{label}.derived_hash requires derived_text_path")
        if source_path.suffix.lower() in AUDIO_EXTENSIONS:
            errors.append(f"{label} selects forbidden audio: {source_path}")
        if not isinstance(entry["source_type"], str) or not entry["source_type"].strip():
            errors.append(f"{label}.source_type must be non-empty text")
        source_hash = entry["source_hash"]
        if not isinstance(source_hash, str) or not HASH_RE.fullmatch(source_hash):
            errors.append(f"{label}.source_hash must be a full 64-character SHA-256")
        elif sha256_file(source_path) != source_hash.casefold():
            errors.append(f"{label} hash mismatch: {source_path}")
        if entry.get("file_name") is not None and entry["file_name"] != source_path.name:
            errors.append(f"{label}.file_name does not match source_path")
        lecture = entry.get("lecture")
        if lecture is not None and (
            not isinstance(lecture, int) or isinstance(lecture, bool) or not 1 <= lecture <= 1000
        ):
            errors.append(f"{label}.lecture must be an integer 1..1000 when present")
        for key in ("course", "semester", "date", "corpus"):
            value = entry.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"{label}.{key} must be non-empty text when present")
        if not _is_within(source_path, base) or not is_approved_personal_path(source_path, base):
            if entry.get("corpus") != "test":
                errors.append(
                    f"{label} is outside the curated root and must be corpus=test: {source_path}"
                )
    return errors


def _validated_entries(
    manifest: str | Path | dict[str, Any], root: str | Path
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    data = load_manifest(manifest)
    base = Path(root).expanduser().resolve()
    errors = validate_manifest(data, base)
    if errors:
        raise ValueError("Invalid source manifest:\n- " + "\n- ".join(errors))
    return data, _entries(data, base)


def dry_run_manifest(
    manifest: str | Path | dict[str, Any], root: str | Path
) -> list[Any]:
    """Validate and describe every manifest entry without contacting OpenViking."""
    _, entries = _validated_entries(manifest, root)
    backend = PersonalOpenVikingBackend(root=root)
    return [
        backend.index_file(
            path,
            source_type=entry["source_type"],
            course=entry.get("course"),
            lecture=entry.get("lecture"),
            metadata=_metadata_for_entry(entry),
            source_hash=entry.get("source_hash"),
            dry_run=True,
        )
        for path, entry in entries
    ]


def build_lexical_index_from_manifest(
    manifest: str | Path | dict[str, Any], root: str | Path
) -> LexicalIndex:
    """Build the curated index plus external manifest sources."""
    _, entries = _validated_entries(manifest, root)
    base = Path(root).expanduser().resolve()
    external_paths: list[Path] = []
    external_metadata: dict[str, dict[str, Any]] = {}
    for source_path, entry in entries:
        index_path = (
            _resolve_path(str(entry["derived_text_path"]), base)
            if entry.get("derived_text_path")
            else source_path
        )
        if not is_approved_personal_path(index_path, base):
            external_paths.append(index_path)
            external_metadata[str(index_path)] = {
                "uri_source_path": str(source_path),
                **_metadata_for_entry(entry),
            }
    index = build_lexical_index(
        base,
        extra_files=external_paths,
        extra_metadata=external_metadata,
    )
    by_source = {entry["source_path"]: entry for _, entry in entries}
    by_name = {path.name: entry for path, entry in entries}
    by_derived_name = {
        Path(entry["derived_text_path"]).name: entry
        for _, entry in entries
        if entry.get("derived_text_path")
    }
    for metadata in index.metadata:
        entry = (
            by_source.get(metadata.get("source_path"))
            or by_name.get(metadata.get("file_name"))
            or by_derived_name.get(metadata.get("file_name"))
        )
        if entry:
            metadata.update(_metadata_for_entry(entry))
            metadata["source_path"] = entry["source_path"]
            if entry.get("file_name"):
                metadata["file_name"] = entry["file_name"]
        metadata.pop("uri_source_path", None)
    return index


def _canonical_root(uri: str) -> str:
    return "/".join(uri.split("/")[:7])


def manifest_canonical_roots(
    manifest: str | Path | dict[str, Any], root: str | Path
) -> set[str]:
    _, entries = _validated_entries(manifest, root)
    base = Path(root).expanduser().resolve()
    return {
        _canonical_root(_canonical_resource_uri(path, entry, base))
        for path, entry in entries
    }


def _canonical_resource_uri(path: Path, entry: dict[str, Any], root: Path) -> str:
    return resource_uri(
        path,
        course=entry.get("course"),
        source_type=entry.get("source_type", "source"),
        root=root,
        source_hash=entry.get("source_hash"),
    )

def ingest_manifest(
    manifest: str | Path | dict[str, Any],
    root: str | Path,
    *,
    base_url: str = "http://127.0.0.1:1934",
    only_missing: bool = True,
) -> dict[str, Any]:
    """Ingest manifest sources, optionally skipping existing canonical roots.

    With strict source-change rejection, any entry whose logical source
    (course + source_type + stem) already exists with a different content
    hash will raise ``SourceHashMismatch`` rather than silently creating a
    duplicate resource.
    """
    _, entries = _validated_entries(manifest, root)
    base = Path(root).expanduser().resolve()
    backend = PersonalOpenVikingBackend(base_url=base_url, root=base)
    existing = set()
    if only_missing:
        existing = {_canonical_root(item["uri"]) for item in backend._resource_files() if item.get("uri")}
    ingested: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    idempotent: list[dict[str, Any]] = []
    for path, entry in entries:
        expected_root = _canonical_root(_canonical_resource_uri(path, entry, base))
        record = {"source_path": str(path), "canonical_root": expected_root}
        if expected_root in existing:
            record["status"] = "skipped_existing"
            skipped.append(record)
            continue
        resource = backend.index_file(
            path,
            source_type=entry["source_type"],
            course=entry.get("course"),
            lecture=entry.get("lecture"),
            metadata=_metadata_for_entry(entry),
            source_hash=entry.get("source_hash"),
            dry_run=False,
        )
        if isinstance(resource.result, dict) and resource.result.get("status") == "skipped":
            record["status"] = "skipped_idempotent"
            idempotent.append(record)
        else:
            record.update({"status": "ingested", "uri": resource.uri, "result": resource.result})
            ingested.append(record)
        existing.add(expected_root)
    return {
        "manifest_sources": len(entries),
        "ingested": ingested,
        "skipped": skipped,
        "idempotent": idempotent,
        "ingested_count": len(ingested),
        "skipped_count": len(skipped),
        "idempotent_count": len(idempotent),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and ingest the Personal source manifest")
    parser.add_argument("command", choices=("validate", "dry-run", "rebuild-index"))
    parser.add_argument("--manifest", default="config/source_manifest.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--index-path", default="data/lexical_index.json")
    args = parser.parse_args()
    if args.command == "validate":
        errors = validate_manifest(args.manifest, args.root)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    if args.command == "dry-run":
        resources = dry_run_manifest(args.manifest, args.root)
        print(json.dumps({
            "valid": True,
            "resources": [
                {
                    "source_path": resource.source_path,
                    "uri": resource.uri,
                    "metadata": resource.metadata,
                }
                for resource in resources
            ],
        }, indent=2, default=str))
        return 0
    errors = validate_manifest(args.manifest, args.root)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return 1
    index = build_lexical_index_from_manifest(args.manifest, args.root)
    save_index(index, args.index_path)
    print(json.dumps({
        "valid": True,
        "index_path": str(args.index_path),
        "documents": len(index.uris),
        "manifest": str(args.manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
