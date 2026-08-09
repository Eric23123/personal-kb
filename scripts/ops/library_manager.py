"""Filesystem-first management for a Personal KB vault.

The user-selected vault is the source of truth.  This module stores only
derived artifacts and per-document records under ``.personal-kb``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUPPORTED = {
    ".pdf", ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif", ".webp",
    ".txt", ".md", ".markdown", ".rst", ".org", ".tex", ".csv", ".json",
    ".yaml", ".yml", ".xml", ".html", ".htm",
}
KIND_BY_SUFFIX = {
    ".pdf": "pdf", ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".flac": "audio",
    ".ogg": "audio", ".opus": "audio", ".aac": "audio", ".wma": "audio",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image", ".tiff": "image",
    ".tif": "image", ".gif": "image", ".webp": "image",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paths(vault: Path) -> dict[str, Path]:
    hidden = vault / ".personal-kb"
    return {
        "hidden": hidden,
        "artifacts": hidden / "artifacts",
        "records": hidden / "records",
        "recycle": hidden / "recycle",
        "logs": hidden / "logs",
        "index": hidden / "lexical-index.json",
    }


def ensure_vault(vault: str | Path) -> Path:
    root = Path(vault).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"知识库目录不存在：{root}")
    for directory in paths(root).values():
        if directory.suffix:
            directory.parent.mkdir(parents=True, exist_ok=True)
        else:
            directory.mkdir(parents=True, exist_ok=True)
    (root / "收件箱").mkdir(exist_ok=True)
    return root


def relative(path: Path, vault: Path) -> str:
    return path.resolve().relative_to(vault).as_posix()


def is_source(path: Path, vault: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in SUPPORTED:
        return False
    try:
        parts = path.resolve().relative_to(vault).parts
    except ValueError:
        return False
    if any(part.startswith(".") for part in parts):
        return False
    if path.name in {"pipeline_manifest.json", "index_ingestion.jsonl"}:
        return False
    return not (len(parts) == 1 and path.name.endswith(("_ocr.txt", "_transcript.txt", "_diagram.txt")))


def source_files(vault: Path) -> list[Path]:
    return sorted(path for path in vault.rglob("*") if is_source(path, vault))


def record_path(vault: Path, document_id: str) -> Path:
    return paths(vault)["records"] / f"{document_id}.json"


def load_records(vault: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths(vault)["records"].glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict) and record.get("id"):
                result[str(record["id"])] = record
        except (OSError, ValueError):
            continue
    return result


def write_record(vault: Path, record: dict[str, Any]) -> None:
    target = record_path(vault, str(record["id"]))
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def kind_for(path: Path) -> str:
    return KIND_BY_SUFFIX.get(path.suffix.lower(), "text")


def sync(vault: str | Path) -> dict[str, Any]:
    root = ensure_vault(vault)
    records = load_records(root)
    active_by_path = {str(item.get("relative_path")): item for item in records.values() if item.get("state") == "active"}
    active_by_hash: dict[str, list[dict[str, Any]]] = {}
    for item in records.values():
        if item.get("state") == "active" and item.get("source_hash"):
            active_by_hash.setdefault(str(item["source_hash"]), []).append(item)

    discovered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in source_files(root):
        rel = relative(source, root)
        digest = sha256(source)
        record = active_by_path.get(rel)
        if record is None:
            candidates = active_by_hash.get(digest, [])
            record = candidates[0] if len(candidates) == 1 else None
        if record is None:
            document_id = uuid.uuid4().hex
            stat = source.stat()
            record = {
                "id": document_id, "relative_path": rel, "name": source.name,
                "kind": kind_for(source), "source_hash": digest, "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "imported_at": now(), "tags": [], "state": "active", "processing_status": "pending",
                "index_status": "not_indexed", "artifact_path": None, "openviking_uri": None,
            }
        else:
            stat = source.stat()
            changed = record.get("source_hash") != digest
            record.update({"relative_path": rel, "name": source.name, "kind": kind_for(source), "size": stat.st_size,
                           "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
            if changed:
                record.update({"source_hash": digest, "processing_status": "needs_processing", "index_status": "needs_index"})
        record["state"] = "active"
        write_record(root, record)
        seen_ids.add(str(record["id"]))
        discovered.append(public_record(record))

    missing = 0
    for document_id, record in records.items():
        if record.get("state") == "active" and document_id not in seen_ids:
            record["state"] = "missing"
            record["updated_at"] = now()
            write_record(root, record)
            missing += 1
    return {"vault": str(root), "items": discovered, "missing": missing, "folders": folder_tree(root)}


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    keys = ("id", "relative_path", "name", "kind", "size", "modified_at", "imported_at", "tags", "processing_status", "index_status", "artifact_path")
    return {key: record.get(key) for key in keys}


def folder_tree(vault: Path) -> list[str]:
    folders = ["/"]
    for path in sorted(vault.rglob("*")):
        if path.is_dir() and not any(part.startswith(".") for part in path.resolve().relative_to(vault).parts):
            folders.append(relative(path, vault))
    return folders


def update_tags(vault: str | Path, ids: list[str], tags: list[str]) -> dict[str, Any]:
    root = ensure_vault(vault)
    clean = sorted({tag.strip() for tag in tags if tag.strip()})
    changed = 0
    for document_id in ids:
        target = record_path(root, document_id)
        if not target.is_file():
            continue
        record = json.loads(target.read_text(encoding="utf-8"))
        record["tags"] = clean
        record["updated_at"] = now()
        write_record(root, record)
        changed += 1
    return {"changed": changed}


def move_to_recycle(vault: str | Path, ids: list[str]) -> dict[str, Any]:
    root = ensure_vault(vault)
    moved: list[str] = []
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for document_id in ids:
        target = record_path(root, document_id)
        if not target.is_file():
            continue
        record = json.loads(target.read_text(encoding="utf-8"))
        source = root / str(record.get("relative_path", ""))
        if not source.is_file():
            continue
        recycled = paths(root)["recycle"] / stamp / str(record["relative_path"])
        recycled.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(recycled))
        record.update({"state": "recycled", "recycle_path": relative(recycled, root), "deleted_at": now()})
        write_record(root, record)
        moved.append(document_id)
    return {"moved": moved, "warning": "向量索引将由下一次索引维护任务清理。"}


def restore(vault: str | Path, ids: list[str]) -> dict[str, Any]:
    root = ensure_vault(vault)
    restored: list[str] = []
    for document_id in ids:
        target = record_path(root, document_id)
        if not target.is_file():
            continue
        record = json.loads(target.read_text(encoding="utf-8"))
        recycled = root / str(record.get("recycle_path", ""))
        destination = root / str(record.get("relative_path", ""))
        if not recycled.is_file() or destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(recycled), str(destination))
        record.update({"state": "active", "processing_status": "needs_processing", "index_status": "needs_index"})
        record.pop("recycle_path", None)
        write_record(root, record)
        restored.append(document_id)
    return {"restored": restored}


def create_note(vault: str | Path, document_id: str) -> dict[str, Any]:
    root = ensure_vault(vault)
    record = json.loads(record_path(root, document_id).read_text(encoding="utf-8"))
    notes = root / "notes"
    notes.mkdir(exist_ok=True)
    target = notes / f"{Path(str(record['name'])).stem} - 笔记.md"
    count = 2
    while target.exists():
        target = notes / f"{Path(str(record['name'])).stem} - 笔记 {count}.md"
        count += 1
    source_link = str(record["relative_path"]).replace("\\", "/")
    tags = "[" + ", ".join(json.dumps(tag, ensure_ascii=False) for tag in record.get("tags", [])) + "]"
    text = f"---\npersonal_kb_source: {json.dumps(source_link, ensure_ascii=False)}\ntags: {tags}\n---\n\n# {Path(str(record['name'])).stem}\n\n[[{source_link}|打开原资料]]\n\n## 笔记\n\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return {"note_path": relative(target, root)}


RELATION_LABELS = {
    "related": "相关知识",
    "same-lecture": "同一课堂",
    "sequence": "前后关系",
}


def link_records(
    vault: str | Path,
    ids: list[str],
    relation: str = "related",
    before_id: str | None = None,
    after_id: str | None = None,
) -> dict[str, Any]:
    """Persist a user-confirmed relation without creating note files."""
    root = ensure_vault(vault)
    clean_ids = list(dict.fromkeys(ids))
    if len(clean_ids) < 2:
        raise ValueError("请至少选择两份资料")
    label = RELATION_LABELS.get(relation, RELATION_LABELS["related"])
    records = load_records(root)
    selected = [records[item_id] for item_id in clean_ids if item_id in records and records[item_id].get("state") == "active"]
    if len(selected) < 2:
        raise ValueError("选中的资料不足两份")
    if relation == "sequence":
        if not before_id or not after_id or before_id == after_id:
            raise ValueError("请分别选择前面的资料和后面的资料")
        selected_ids = {str(record["id"]) for record in selected}
        if before_id not in selected_ids or after_id not in selected_ids:
            raise ValueError("前后关系只能在本次选中的资料中建立")
        selected = [records[before_id], records[after_id]]
    relations_path = paths(root)["hidden"] / "relations.json"
    try:
        data = json.loads(relations_path.read_text(encoding="utf-8")) if relations_path.is_file() else {"relations": []}
    except (OSError, ValueError):
        data = {"relations": []}
    relation_entry = {"ids": [str(record["id"]) for record in selected], "type": relation, "label": label, "created_at": now()}
    if relation == "sequence":
        relation_entry.update({"before_id": before_id, "after_id": after_id})
    existing = data.setdefault("relations", [])
    is_duplicate = any(
        set(item.get("ids", [])) == set(relation_entry["ids"])
        and item.get("type") == relation
        and item.get("before_id") == relation_entry.get("before_id")
        and item.get("after_id") == relation_entry.get("after_id")
        for item in existing
    )
    if not is_duplicate:
        existing.append(relation_entry)
    relations_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return {"linked": [str(record["id"]) for record in selected], "relation": relation, "relations_path": relative(relations_path, root)}


def process_records(vault: str | Path, ids: list[str]) -> dict[str, Any]:
    """Process selected active records into vault-local artifacts and index them."""
    from scripts.ops.laptop_pipeline import process_one
    from scripts.core.openviking_backend import PersonalOpenVikingBackend
    root = ensure_vault(vault)
    records = load_records(root)
    # A one-click update is the recovery path after a transient provider or
    # connectivity failure, so it must retry failed active records as well.
    selected = ids or [key for key, value in records.items() if value.get("state") == "active" and value.get("processing_status") in {"pending", "needs_processing", "error"}]
    processed: list[str] = []
    failed: list[dict[str, str]] = []
    backend = PersonalOpenVikingBackend(root=root)
    for document_id in selected:
        record = records.get(document_id)
        if not record or record.get("state") != "active":
            continue
        source = root / str(record["relative_path"])
        if not source.is_file():
            continue
        artifact_dir = paths(root)["artifacts"] / document_id
        try:
            result = process_one(source, artifact_dir)
            artifact = Path(str(result["index_path"]))
            indexed = backend.index_file(
                artifact,
                source_type={"pdf": "ocr_text", "audio": "transcript", "image": "diagram_description", "text": "source"}.get(str(record["kind"]), "source"),
                source_hash=str(record["source_hash"]),
                provenance_source_path=source,
                metadata={"relative_path": str(record["relative_path"]), "tags": ",".join(record.get("tags", []))},
            )
            record.update({"artifact_path": relative(artifact, root), "processing_status": "processed", "index_status": "indexed", "openviking_uri": indexed.uri, "processed_at": now()})
            write_record(root, record)
            processed.append(document_id)
        except Exception as error:
            record.update({"processing_status": "error", "processing_error": str(error), "index_status": "error"})
            write_record(root, record)
            failed.append({"id": document_id, "error": str(error)})
    # The lexical index is disposable, but must be rebuilt from active artifacts
    # after every successful processing pass so hybrid search stays in sync.
    try:
        from scripts.retrieval.lexical_index import build_lexical_index, save_index
        refreshed = load_records(root)
        extra_files: list[Path] = []
        extra_metadata: dict[str, dict[str, Any]] = {}
        for record in refreshed.values():
            artifact_rel = record.get("artifact_path")
            if record.get("state") != "active" or not artifact_rel:
                continue
            artifact = root / str(artifact_rel)
            if artifact.is_file():
                extra_files.append(artifact)
                extra_metadata[str(artifact.resolve())] = {
                    "source_type": {"pdf": "ocr_text", "audio": "transcript", "image": "diagram_description", "text": "source"}.get(str(record.get("kind")), "source"),
                    "source_hash": record.get("source_hash"),
                    "uri_source_path": str(artifact),
                    "relative_path": record.get("relative_path"),
                    "tags": record.get("tags", []),
                }
        save_index(build_lexical_index(root, extra_files=extra_files, extra_metadata=extra_metadata), paths(root)["index"])
    except Exception as error:
        failed.append({"id": "lexical-index", "error": str(error)})
    return {"processed": processed, "failed": failed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage a Personal KB vault")
    parser.add_argument("command", choices=["sync", "process", "tags", "recycle", "restore", "create-note", "link"])
    parser.add_argument("--vault", required=True)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--id")
    parser.add_argument("--relation", default="related")
    parser.add_argument("--before-id")
    parser.add_argument("--after-id")
    args = parser.parse_args()
    try:
        if args.command == "sync": result = sync(args.vault)
        elif args.command == "process": result = process_records(args.vault, args.ids)
        elif args.command == "tags": result = update_tags(args.vault, args.ids, args.tags)
        elif args.command == "recycle": result = move_to_recycle(args.vault, args.ids)
        elif args.command == "restore": result = restore(args.vault, args.ids)
        elif args.command == "link": result = link_records(args.vault, args.ids, args.relation, args.before_id, args.after_id)
        else:
            if not args.id: raise ValueError("--id is required")
            result = create_note(args.vault, args.id)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
