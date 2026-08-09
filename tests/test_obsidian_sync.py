from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ops.obsidian_sync import sync_once


def _write_registry(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "courses": {
                    "PERSONAL-ALPHA": {
                        "obsidian": "Personal/PERSONAL-ALPHA",
                        "slug": "personal-alpha",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_note(path: Path, lecture: int, topic: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""---
course: PERSONAL-ALPHA
lecture: {lecture}
topic: {topic}
source_type: synthesis
tags:
  - personal
  - control
---

# Lecture {lecture}: {topic}

## Key Concepts
- Feedback and control.
""",
        encoding="utf-8",
    )


def test_sync_normalizes_destination_and_updates_course_index(tmp_path):
    source = tmp_path / "courses"
    vault = tmp_path / "vault"
    registry = tmp_path / "course_namespaces.yaml"
    _write_registry(registry)
    _write_note(source / "personal-alpha" / "derived" / "notes" / "lecture_01.md", 1, "Feedback Basics")

    result = sync_once(source, vault, registry)

    destination = vault / "Personal" / "PERSONAL-ALPHA" / "Lectures" / "Lecture 01 - Feedback Basics.md"
    index = vault / "Personal" / "PERSONAL-ALPHA" / "Course Index.md"
    assert result.scanned == 1
    assert result.synced == 1
    assert destination.exists()
    content = destination.read_text(encoding="utf-8")
    assert "personal_kb_sync: true" in content
    assert "personal_kb_source: personal-alpha/derived/notes/lecture_01.md" in content
    assert "<!-- personal-kb:auto-related:start -->" in content
    assert "Generated Notes" in index.read_text(encoding="utf-8")
    assert "Lecture 01 - Feedback Basics" in index.read_text(encoding="utf-8")


def test_sync_is_idempotent_and_adds_adjacent_lecture_link(tmp_path):
    source = tmp_path / "courses"
    vault = tmp_path / "vault"
    registry = tmp_path / "course_namespaces.yaml"
    _write_registry(registry)
    first = source / "personal-alpha" / "derived" / "notes" / "lecture_01.md"
    second = source / "personal-alpha" / "derived" / "notes" / "lecture_02.md"
    _write_note(first, 1, "Feedback Basics")

    first_result = sync_once(source, vault, registry)
    second_result = sync_once(source, vault, registry)
    assert first_result.synced == 1
    assert second_result.unchanged == 1
    assert second_result.synced == 0

    _write_note(second, 2, "Feedback Stability")
    result = sync_once(source, vault, registry)
    assert result.synced == 2  # new note plus Lecture 01's generated related-link block
    first_destination = vault / "Personal" / "PERSONAL-ALPHA" / "Lectures" / "Lecture 01 - Feedback Basics.md"
    assert "Lecture 02 - Feedback Stability" in first_destination.read_text(encoding="utf-8")


def test_sync_preserves_user_index_content_and_dry_run_writes_nothing(tmp_path):
    source = tmp_path / "courses"
    vault = tmp_path / "vault"
    registry = tmp_path / "course_namespaces.yaml"
    _write_registry(registry)
    _write_note(source / "personal-alpha" / "derived" / "notes" / "lecture_01.md", 1, "Feedback Basics")
    index = vault / "Personal" / "PERSONAL-ALPHA" / "Course Index.md"
    index.parent.mkdir(parents=True)
    index.write_text("# My Course Index\n\nDo not remove this.\n", encoding="utf-8")

    dry = sync_once(source, vault, registry, dry_run=True)
    assert dry.synced == 1
    assert not (vault / "Personal" / "PERSONAL-ALPHA" / "Lectures" / "Lecture 01 - Feedback Basics.md").exists()
    assert index.read_text(encoding="utf-8") == "# My Course Index\n\nDo not remove this.\n"

    sync_once(source, vault, registry)
    updated = index.read_text(encoding="utf-8")
    assert "Do not remove this." in updated
    assert "<!-- personal-kb:auto-index:start -->" in updated


def test_sync_prune_removes_only_managed_missing_source(tmp_path):
    source = tmp_path / "courses"
    vault = tmp_path / "vault"
    registry = tmp_path / "course_namespaces.yaml"
    _write_registry(registry)
    note = source / "personal-alpha" / "derived" / "notes" / "lecture_01.md"
    _write_note(note, 1, "Feedback Basics")
    sync_once(source, vault, registry)
    destination = vault / "Personal" / "PERSONAL-ALPHA" / "Lectures" / "Lecture 01 - Feedback Basics.md"
    assert destination.exists()

    note.unlink()
    result = sync_once(source, vault, registry, prune=True)
    assert result.removed == 1
    assert not destination.exists()
