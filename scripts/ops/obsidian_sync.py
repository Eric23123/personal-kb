"""Synchronize generated Personal-KB notes into the Obsidian vault.

The sync boundary is deliberately filesystem-only: it reads generated Markdown
from a staging/source tree and writes only managed copies plus managed course
index/link sections into the vault. It never calls OpenViking, Hindsight, or an
LLM.

Typical usage from the Personal-KB project root::

    python scripts/ops/obsidian_sync.py --once
    python scripts/ops/obsidian_sync.py --source-root courses --watch

Generated source notes should contain frontmatter with at least ``course``.
``source_type`` and ``lecture`` are recommended; the sync layer infers sensible
values when they are absent. The course registry in
``config/course_namespaces.yaml`` supplies the destination vault path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by CLI environments
    raise RuntimeError("PyYAML is required for Personal-KB Obsidian sync") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "courses"
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "course_namespaces.yaml"
DEFAULT_VAULT = Path(os.environ.get("OBSIDIAN_VAULT_PATH", Path.home() / "obsidian-vault")).expanduser()

SYNC_KEY = "personal_kb_sync"
SOURCE_KEY = "personal_kb_source"
HASH_KEY = "personal_kb_sync_hash"
RELATED_START = "<!-- personal-kb:auto-related:start -->"
RELATED_END = "<!-- personal-kb:auto-related:end -->"
INDEX_START = "<!-- personal-kb:auto-index:start -->"
INDEX_END = "<!-- personal-kb:auto-index:end -->"


class SyncError(ValueError):
    """Raised for invalid source metadata or unsafe sync configuration."""


@dataclass(frozen=True)
class SourceNote:
    source_path: Path
    relative_source: str
    text: str
    metadata: dict[str, Any]
    body: str
    course: str
    source_type: str
    lecture: int | None
    title: str
    destination: Path


@dataclass(frozen=True)
class SyncResult:
    scanned: int = 0
    synced: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    errors: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "synced": self.synced,
            "unchanged": self.unchanged,
            "removed": self.removed,
            "skipped": self.skipped,
            "errors": list(self.errors),
        }


def _require_within(path: Path, root: Path) -> Path:
    """Resolve ``path`` and reject traversal outside ``root``."""
    root = root.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SyncError(f"Path escapes configured root: {path}") from exc
    return resolved


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load the course registry and return its course mapping."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise SyncError(f"Course registry not found: {path}") from exc
    courses = data.get("courses") if isinstance(data, dict) else None
    if not isinstance(courses, dict):
        raise SyncError(f"Registry has no valid 'courses' mapping: {path}")
    return {str(key): value for key, value in courses.items() if isinstance(value, dict)}


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return YAML frontmatter and the Markdown body.

    Notes without frontmatter are accepted so a caller can report/skip them;
    the sync command requires a course field before it writes anything.
    """
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", text, re.DOTALL)
    if not match:
        raise SyncError("frontmatter starts with '---' but has no closing delimiter")
    raw, body = match.groups()
    metadata = yaml.safe_load(raw) or {}
    if not isinstance(metadata, dict):
        raise SyncError("frontmatter must decode to a mapping")
    return dict(metadata), body


def _slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "note"


def _safe_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]', "-", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value or "Untitled"


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _title_from_body(body: str, fallback: str) -> str:
    for line in body.splitlines():
        match = re.match(r"^#\s+.*?:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
        match = re.match(r"^#\s+(.+?)\s*$", line)
        if match:
            return match.group(1).strip()
    return fallback


def _source_type(metadata: Mapping[str, Any], lecture: int | None) -> str:
    value = str(metadata.get("source_type", "")).strip().lower().replace("-", "_")
    if value:
        return value
    if metadata.get("assignment"):
        return "homework"
    if metadata.get("exam"):
        return "exam"
    if lecture is not None:
        return "synthesis"
    return "note"


def _folder_for(source_type: str, lecture: int | None) -> str:
    if source_type in {"homework", "assignment", "assessment"}:
        return "Homework"
    if source_type in {"exam", "quiz", "midterm", "final"}:
        return "Exams"
    if source_type in {"diagram", "diagrams", "diagram_description"}:
        return "Diagrams"
    if lecture is not None or source_type in {"lecture", "transcript", "slides", "synthesis", "ocr_text"}:
        return "Lectures"
    return "Notes"


def _filename(metadata: Mapping[str, Any], source_type: str, lecture: int | None, title: str, source_path: Path) -> str:
    if source_type in {"homework", "assignment", "assessment"}:
        assignment = str(metadata.get("assignment", "")).strip()
        stem = assignment or title or source_path.stem
        if not stem.lower().endswith("review"):
            stem += " Review"
    elif source_type in {"exam", "quiz", "midterm", "final"}:
        exam = str(metadata.get("exam", "")).strip() or title or source_path.stem
        stem = exam if exam.lower().endswith("review") else f"{exam} Review"
    elif lecture is not None:
        stem = f"Lecture {lecture:02d} - {title or source_path.stem}"
    else:
        stem = title or source_path.stem
    return _safe_filename(stem) + ".md"


def _vault_relative(path: Path, vault_root: Path) -> str:
    return path.resolve().relative_to(vault_root.resolve()).with_suffix("").as_posix()


def _wikilink(path: Path, vault_root: Path, display: str | None = None) -> str:
    target = _vault_relative(path, vault_root)
    return f"[[{target}|{display or path.stem}]]"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dump_frontmatter(metadata: Mapping[str, Any]) -> str:
    rendered = yaml.safe_dump(
        dict(metadata),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    return f"---\n{rendered}\n---\n"


def _replace_managed_block(text: str, start: str, end: str, content: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    block = f"{start}\n{content.rstrip()}\n{end}"
    if pattern.search(text):
        return pattern.sub(block, text, count=1).rstrip() + "\n"
    separator = "\n\n" if text.rstrip() else ""
    return text.rstrip() + separator + block + "\n"


def _related_tokens(note: SourceNote) -> set[str]:
    values = [note.title, str(note.metadata.get("topic", ""))]
    tags = note.metadata.get("tags", [])
    if isinstance(tags, list):
        values.extend(str(tag) for tag in tags)
    elif tags:
        values.append(str(tags))
    return {token for value in values for token in re.findall(r"[a-z0-9]{4,}", value.lower())}


def _related_links(note: SourceNote, all_notes: Iterable[SourceNote], vault_root: Path) -> list[str]:
    candidates: list[tuple[int, str, SourceNote]] = []
    note_tokens = _related_tokens(note)
    for other in all_notes:
        if other.destination == note.destination:
            continue
        score = len(note_tokens & _related_tokens(other))
        if note.lecture is not None and other.lecture is not None and abs(note.lecture - other.lecture) == 1:
            score += 2
        if score:
            candidates.append((score, other.title.lower(), other))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [f"- {_wikilink(item[2].destination, vault_root)}" for item in candidates[:8]]


def _managed_note_text(note: SourceNote, all_notes: Iterable[SourceNote], vault_root: Path) -> str:
    metadata = dict(note.metadata)
    metadata[SYNC_KEY] = True
    metadata[SOURCE_KEY] = note.relative_source
    metadata[HASH_KEY] = _hash_text(note.text)
    related = _related_links(note, all_notes, vault_root)
    related_content = "## Related Course Notes\n" + ("\n".join(related) if related else "- None yet")
    return _replace_managed_block(_dump_frontmatter(metadata) + note.body, RELATED_START, RELATED_END, related_content)


def _course_index_text(existing: str, notes: Iterable[SourceNote], vault_root: Path, course: str) -> str:
    ordered = sorted(notes, key=lambda item: (item.lecture is None, item.lecture or 0, item.title.lower()))
    lines = ["## Generated Notes", ""]
    for note in ordered:
        lines.append(f"- {_wikilink(note.destination, vault_root)}")
    if not ordered:
        lines.append("- None yet")
    return _replace_managed_block(existing, INDEX_START, INDEX_END, "\n".join(lines))


def _load_source(path: Path, source_root: Path, registry: Mapping[str, Mapping[str, Any]], vault_root: Path) -> SourceNote:
    text = path.read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(text)
    course = str(metadata.get("course", "")).strip()
    if not course:
        raise SyncError("missing frontmatter course")
    registry_entry = registry.get(course)
    if registry_entry is None:
        raise SyncError(f"course is not registered: {course}")
    obsidian_path = str(registry_entry.get("obsidian", "")).strip()
    if not obsidian_path:
        raise SyncError(f"course has no Obsidian path: {course}")
    course_root = _require_within(vault_root / obsidian_path, vault_root)
    lecture = _as_int(metadata.get("lecture"))
    source_type = _source_type(metadata, lecture)
    fallback = path.stem.replace("_", " ").replace("-", " ").strip()
    title = str(metadata.get("topic", metadata.get("title", ""))).strip() or _title_from_body(body, fallback)
    destination = course_root / _folder_for(source_type, lecture) / _filename(metadata, source_type, lecture, title, path)
    destination = _require_within(destination, vault_root)
    relative_source = path.resolve().relative_to(source_root.resolve()).as_posix()
    return SourceNote(path, relative_source, text, metadata, body, course, source_type, lecture, title, destination)


def _iter_sources(source_root: Path, pattern: str = "**/*.md") -> list[Path]:
    if not source_root.exists():
        return []
    return sorted(path for path in source_root.glob(pattern) if path.is_file() and not any(part.startswith(".") for part in path.parts))


def _prune_managed(vault_root: Path, source_root: Path, active_sources: set[str]) -> int:
    removed = 0
    for path in vault_root.glob("Personal/**/*.md"):
        try:
            metadata, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except (OSError, SyncError):
            continue
        if metadata.get(SYNC_KEY) is not True:
            continue
        source = str(metadata.get(SOURCE_KEY, ""))
        if source and source not in active_sources:
            path.unlink()
            removed += 1
    return removed


def sync_once(
    source_root: Path = DEFAULT_SOURCE_ROOT,
    vault_root: Path = DEFAULT_VAULT,
    registry_path: Path = DEFAULT_REGISTRY,
    *,
    pattern: str = "**/*.md",
    dry_run: bool = False,
    prune: bool = False,
) -> SyncResult:
    """Synchronize generated notes once and update managed index/link blocks."""
    source_root = source_root.expanduser().resolve()
    vault_root = vault_root.expanduser().resolve()
    registry = load_registry(registry_path.expanduser().resolve())
    paths = _iter_sources(source_root, pattern)
    notes: list[SourceNote] = []
    errors: list[str] = []
    for path in paths:
        try:
            notes.append(_load_source(path, source_root, registry, vault_root))
        except (OSError, UnicodeError, SyncError) as exc:
            errors.append(f"{path}: {exc}")

    by_course: dict[str, list[SourceNote]] = {}
    for note in notes:
        by_course.setdefault(note.course, []).append(note)

    synced = unchanged = 0
    if not dry_run:
        vault_root.mkdir(parents=True, exist_ok=True)
    for note in notes:
        rendered = _managed_note_text(note, by_course[note.course], vault_root)
        if dry_run:
            synced += 1
            continue
        note.destination.parent.mkdir(parents=True, exist_ok=True)
        if note.destination.exists() and note.destination.read_text(encoding="utf-8") == rendered:
            unchanged += 1
        else:
            temp = note.destination.with_suffix(note.destination.suffix + ".tmp")
            temp.write_text(rendered, encoding="utf-8", newline="\n")
            temp.replace(note.destination)
            synced += 1

    if not dry_run:
        for course, course_notes in by_course.items():
            registry_entry = registry[course]
            course_root = _require_within(vault_root / str(registry_entry["obsidian"]), vault_root)
            index_path = course_root / "Course Index.md"
            existing = index_path.read_text(encoding="utf-8") if index_path.exists() else f"# {course}\n"
            updated = _course_index_text(existing, course_notes, vault_root, course)
            if existing != updated:
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(updated, encoding="utf-8", newline="\n")
        removed = _prune_managed(vault_root, source_root, {note.relative_source for note in notes}) if prune else 0
    else:
        removed = 0
    skipped = len(paths) - len(notes)
    return SyncResult(len(paths), synced, unchanged, removed, skipped, tuple(errors))


def watch(
    source_root: Path,
    vault_root: Path,
    registry_path: Path,
    *,
    pattern: str,
    interval: float,
    prune: bool,
) -> None:
    """Poll the source tree and sync changed notes until interrupted."""
    previous: tuple[tuple[str, int, int], ...] | None = None
    while True:
        paths = _iter_sources(source_root, pattern)
        fingerprint = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
        if fingerprint != previous:
            result = sync_once(source_root, vault_root, registry_path, pattern=pattern, prune=prune)
            print(json.dumps(result.as_dict(), ensure_ascii=False), flush=True)
            previous = fingerprint
        time.sleep(max(0.2, interval))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync generated Personal-KB notes into Obsidian")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Generated-note staging root (default: project/courses)")
    parser.add_argument("--vault-root", type=Path, default=DEFAULT_VAULT, help="Obsidian vault root")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="Course namespace YAML")
    parser.add_argument("--pattern", default="**/*.md", help="Source glob relative to --source-root")
    parser.add_argument("--once", action="store_true", help="Run one sync pass (the default)")
    parser.add_argument("--watch", action="store_true", help="Poll for new/changed source notes")
    parser.add_argument("--interval", type=float, default=2.0, help="Watch polling interval in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without writing the vault")
    parser.add_argument("--prune", action="store_true", help="Delete only managed vault notes whose source disappeared")
    parser.add_argument("--json", action="store_true", help="Print one JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.watch and args.dry_run:
        print("--dry-run cannot be combined with --watch", file=sys.stderr)
        return 2
    try:
        if args.watch:
            watch(args.source_root, args.vault_root, args.registry, pattern=args.pattern, interval=args.interval, prune=args.prune)
            return 0
        result = sync_once(
            args.source_root,
            args.vault_root,
            args.registry,
            pattern=args.pattern,
            dry_run=args.dry_run,
            prune=args.prune,
        )
    except (OSError, SyncError, RuntimeError) as exc:
        print(f"obsidian-sync error: {exc}", file=sys.stderr)
        return 1
    payload = result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if result.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
