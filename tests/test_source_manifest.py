import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingestion.source_manifest import (  # noqa: E402
    build_lexical_index_from_manifest,
    dry_run_manifest,
    validate_manifest,
)


def _manifest_for(path: Path, **metadata):
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "required_fields": ["source_path", "source_hash", "source_type"],
        "optional_fields": ["course", "semester", "date", "lecture", "corpus"],
        "sources": [
            {
                "source_path": str(path),
                "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_type": "homework",
                **metadata,
            }
        ],
    }


def test_manifest_allows_external_test_source_without_lecture(tmp_path):
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"test pdf bytes")
    manifest = _manifest_for(pdf, course="TEST", semester="TEST", corpus="test")

    assert validate_manifest(manifest, tmp_path) == []


def test_manifest_rejects_hash_mismatch(tmp_path):
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"test pdf bytes")
    manifest = _manifest_for(pdf, course="TEST", corpus="test")
    manifest["sources"][0]["source_hash"] = "0" * 64

    errors = validate_manifest(manifest, tmp_path)

    assert any("hash mismatch" in error for error in errors)


def test_manifest_dry_run_returns_optional_lecture_metadata(tmp_path):
    pdf = tmp_path / "homework.pdf"
    pdf.write_bytes(b"test pdf bytes")
    manifest = _manifest_for(pdf, course="TEST", semester="TEST", corpus="test")

    resources = dry_run_manifest(manifest, tmp_path)

    assert len(resources) == 1
    assert resources[0].lecture is None
    assert resources[0].metadata["course"] == "TEST"
    assert resources[0].metadata["source_hash"] == manifest["sources"][0]["source_hash"]


def test_build_lexical_index_from_manifest_adds_external_test_source(tmp_path):
    pdf = tmp_path / "homework.pdf"
    pdf.write_bytes(b"test pdf bytes")
    derived = tmp_path / "homework.json"
    derived.write_text(
        json.dumps({"content": "kanban homework test content"}),
        encoding="utf-8",
    )
    manifest = _manifest_for(
        pdf,
        course="TEST",
        semester="TEST",
        corpus="test",
        derived_text_path=str(derived),
    )

    index = build_lexical_index_from_manifest(manifest, tmp_path)
    results = index.search("kanban", filters={"course": "TEST", "source_type": "homework"})

    assert len(results) == 1
    assert results[0]["metadata"]["course"] == "TEST"
    assert results[0]["metadata"]["source_hash"] == manifest["sources"][0]["source_hash"]


def test_rebuild_index_direct_cli_imports_its_lexical_dependencies(tmp_path):
    source = tmp_path / "homework.txt"
    source.write_text("kanban homework test content", encoding="utf-8")
    manifest = _manifest_for(source, course="TEST", corpus="test")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    index_path = tmp_path / "index.json"

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ingestion" / "source_manifest.py"),
            "rebuild-index",
            "--manifest",
            str(manifest_path),
            "--root",
            str(tmp_path),
            "--index-path",
            str(index_path),
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert result.returncode == 0, result.stderr
    assert index_path.is_file()


def test_manifest_canonical_root_changes_with_authoritative_source_hash(tmp_path):
    """Changed content at the same path must yield a distinct canonical root.

    The manifest's authoritative ``source_hash`` is folded into canonical
    resource identity, so ``ingest_manifest`` with ``only_missing=True`` will
    ingest revised content rather than silently skipping it as an existing
    canonical root.  This is the manifest-level identity safety guarantee:
    a stale URI can never alias changed content.
    """
    from scripts.ingestion.source_manifest import manifest_canonical_roots

    pdf = tmp_path / "homework.pdf"

    # ``manifest_canonical_roots`` validates the manifest against live file
    # content, so each roots computation must run while the file holds the
    # content whose sha256 is recorded in that manifest.
    pdf.write_bytes(b"version one content")
    hash_one = hashlib.sha256(pdf.read_bytes()).hexdigest()
    manifest_one = _manifest_for(pdf, course="TEST", corpus="test")
    manifest_one["sources"][0]["source_hash"] = hash_one
    roots_one = manifest_canonical_roots(manifest_one, tmp_path)

    pdf.write_bytes(b"version two content, revised submission")
    hash_two = hashlib.sha256(pdf.read_bytes()).hexdigest()
    manifest_two = _manifest_for(pdf, course="TEST", corpus="test")
    manifest_two["sources"][0]["source_hash"] = hash_two
    roots_two = manifest_canonical_roots(manifest_two, tmp_path)

    assert len(roots_one) == 1
    assert len(roots_two) == 1
    assert roots_one != roots_two, (
        "changed source_hash at the same path must produce a different "
        "canonical root so ingestion cannot silently reuse the old URI"
    )


def test_manifest_identical_hash_yields_same_canonical_root(tmp_path):
    """Identical content (identical hash) at the same path is idempotent.

    This preserves deterministic idempotency: re-running manifest validation or
    ingestion for unchanged content must address the same canonical root.
    """
    from scripts.ingestion.source_manifest import manifest_canonical_roots

    pdf = tmp_path / "homework.pdf"
    pdf.write_bytes(b"stable content")
    manifest = _manifest_for(pdf, course="TEST", corpus="test")

    first = manifest_canonical_roots(manifest, tmp_path)
    second = manifest_canonical_roots(manifest, tmp_path)

    assert first == second
    assert len(first) == 1
