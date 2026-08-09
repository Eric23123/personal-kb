"""Tests for recovery/checkpoint/cleanup tooling (Priority 5) — all offline.

No live OpenViking, Hindsight, or embedding calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.core.openviking_backend import PERSONAL_NAMESPACE, canonical_resource_root


# ---------------------------------------------------------------------------
# IngestionLogger tests
# ---------------------------------------------------------------------------


class TestIngestionLogger:
    def test_record_writes_jsonl_line(self):
        from scripts.ingestion.ingestion_logger import IngestionLogger
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.jsonl"
            log = IngestionLogger(log_path)
            log.record("indexed", uri="viking://test/uri", source_path="/tmp/test.pdf",
                       source_hash="a" * 64)
            log.close()

            content = log_path.read_text(encoding="utf-8")
            assert "viking://test/uri" in content
            assert '"status": "indexed"' in content

    def test_summary_counts_correctly(self):
        from scripts.ingestion.ingestion_logger import IngestionLogger
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.jsonl"
            log = IngestionLogger(log_path)
            log.record("indexed", uri="viking://a")
            log.record("indexed", uri="viking://b")
            log.record("skipped", uri="viking://c", reason="duplicate")
            log.record("error", source_path="/tmp/bad.pdf", error="SourceHashMismatch")
            log.close()

            s = log.summary()
            assert s["entries"]["total"] == 4
            assert s["entries"]["indexed"] == 2
            assert s["entries"]["skipped"] == 1
            assert s["entries"]["error"] == 1
            assert s["error_count"] == 1

    def test_record_indexed_convenience(self):
        from scripts.ingestion.ingestion_logger import IngestionLogger
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.jsonl"
            log = IngestionLogger(log_path)

            # simulate an IndexedResource
            class FakeResource:
                source_path = "/tmp/ch1.pdf"
                uri = "viking://personal-kb/ch1"
                source_type = "textbook"
                course = "PERSONAL-ALPHA"
                lecture = 1

            log.record_indexed(FakeResource())
            log.close()

            content = log_path.read_text(encoding="utf-8")
            assert "PERSONAL-ALPHA" in content
            assert '"status": "indexed"' in content

    def test_record_skipped_convenience(self):
        from scripts.ingestion.ingestion_logger import IngestionLogger
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.jsonl"
            with IngestionLogger(log_path) as log:
                log.record_skipped("viking://x", "identical_source_hash")
            assert log.summary()["entries"]["skipped"] == 1

    def test_record_rejects_invalid_status(self):
        from scripts.ingestion.ingestion_logger import IngestionLogger
        with tempfile.TemporaryDirectory() as td:
            log = IngestionLogger(Path(td) / "test.jsonl")
            with pytest.raises(ValueError, match="Invalid status"):
                log.record("unknown", uri="x")
            log.close()

    def test_context_manager_closes_handle(self):
        from scripts.ingestion.ingestion_logger import IngestionLogger
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.jsonl"
            with IngestionLogger(log_path) as log:
                log.record("indexed", uri="viking://x")
            # After __exit__, the file should have content.
            assert log_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint / resume tests
# ---------------------------------------------------------------------------


class TestCheckpointResume:
    def test_read_logged_uris_extracts_indexed_only(self):
        from scripts.ingestion.ingestion_logger import read_logged_uris
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.jsonl"
            log_path.write_text(
                json.dumps({"status": "indexed", "uri": "viking://a"}) + "\n" +
                json.dumps({"status": "skipped", "uri": "viking://b"}) + "\n" +
                json.dumps({"status": "indexed", "uri": "viking://c"}) + "\n" +
                json.dumps({"status": "error", "source_path": "/tmp/x", "error": "fail"}) + "\n",
                encoding="utf-8",
            )
            uris = read_logged_uris(log_path)
            assert uris == {"viking://a", "viking://c"}

    def test_read_logged_uris_missing_file_returns_empty(self):
        from scripts.ingestion.ingestion_logger import read_logged_uris
        assert read_logged_uris("/nonexistent/log.jsonl") == set()

    def test_build_resume_manifest_filters_out_logged(self):
        from scripts.ingestion.ingestion_logger import build_resume_manifest
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            # Create a source file.
            src = td_path / "ch1.pdf"
            src.write_bytes(b"test pdf content")

            # Write a manifest.
            manifest = {
                "schema_version": 1,
                "hash_algorithm": "sha256",
                "sources": [
                    {
                        "source_path": str(src),
                        "source_hash": "a" * 64,
                        "source_type": "textbook",
                        "course": "TEST",
                    }
                ],
            }
            manifest_path = td_path / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            # Write a log with the same source_hash already indexed.
            log_path = td_path / "log.jsonl"
            log_path.write_text(
                json.dumps({
                    "status": "indexed",
                    "uri": "viking://resources/personal-kb/TEST/textbook/ch1-" + "a" * 12,
                    "source_path": str(src),
                    "source_hash": "a" * 64,
                }) + "\n",
                encoding="utf-8",
            )

            result = build_resume_manifest(manifest, log_path)
            assert len(result["sources"]) == 0
            assert result["_skipped_from_log"] == 1

    def test_build_resume_manifest_does_not_skip_same_stem_with_different_path(self):
        from scripts.ingestion.ingestion_logger import build_resume_manifest

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            first = td_path / "course_a" / "lecture.md"
            second = td_path / "course_b" / "lecture.md"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            manifest = {
                "sources": [
                    {"source_path": str(first), "source_hash": "a" * 64, "source_type": "lecture"},
                    {"source_path": str(second), "source_hash": "b" * 64, "source_type": "lecture"},
                ]
            }
            log_path = td_path / "log.jsonl"
            log_path.write_text(
                json.dumps({
                    "status": "indexed",
                    "uri": "viking://resources/personal-kb/course-a/lecture/one",
                    "source_path": str(first),
                    "source_hash": "a" * 64,
                }) + "\n",
                encoding="utf-8",
            )

            result = build_resume_manifest(manifest, log_path)

            assert [item["source_path"] for item in result["sources"]] == [str(second)]
            assert result["_skipped_from_log"] == 1


# ---------------------------------------------------------------------------
# Resource inventory tests
# ---------------------------------------------------------------------------


class TestResourceInventory:
    def test_snapshot_with_fake_client(self):
        from scripts.ops.resource_inventory import snapshot, Inventory

        class FakeClient:
            def ls(self, uri, **kwargs):
                return [
                    {"isDir": False, "uri": f"{PERSONAL_NAMESPACE}/TEST/textbook/ch1-abc123"},
                    {"isDir": False, "uri": f"{PERSONAL_NAMESPACE}/TEST/textbook/ch1-xyz789/.overview.md"},
                ]

        client = FakeClient()
        inv = snapshot(client)
        assert isinstance(inv, Inventory)
        assert inv.resource_count == 2
        assert inv.namespace == PERSONAL_NAMESPACE

    def test_snapshot_skips_dirs(self):
        from scripts.ops.resource_inventory import snapshot

        class FakeClient:
            def ls(self, uri, **kwargs):
                return [
                    {"isDir": True, "uri": f"{PERSONAL_NAMESPACE}/TEST"},
                    {"isDir": False, "uri": f"{PERSONAL_NAMESPACE}/TEST/ch1"},
                ]

        client = FakeClient()
        inv = snapshot(client)
        assert inv.resource_count == 1

    def test_diff_computes_correctly(self):
        from scripts.ops.resource_inventory import (
            Inventory, ResourceEntry, diff, InventoryDiff,
        )

        before = Inventory(
            namespace=PERSONAL_NAMESPACE, timestamp="t1", resource_count=2,
            resources=[
                ResourceEntry(uri="viking://a", canonical_root="viking://a"),
                ResourceEntry(uri="viking://b", canonical_root="viking://b"),
            ],
        )
        after = Inventory(
            namespace=PERSONAL_NAMESPACE, timestamp="t2", resource_count=3,
            resources=[
                ResourceEntry(uri="viking://a", canonical_root="viking://a"),
                ResourceEntry(uri="viking://c", canonical_root="viking://c"),
                ResourceEntry(uri="viking://d", canonical_root="viking://d"),
            ],
        )

        d = diff(before, after)
        assert d.before_count == 2
        assert d.after_count == 3
        assert d.net_change == 1
        assert d.added == ["viking://c", "viking://d"]
        assert d.removed == ["viking://b"]
        assert d.unchanged == 1

    def test_diff_no_change(self):
        from scripts.ops.resource_inventory import Inventory, ResourceEntry, diff

        inv = Inventory(
            namespace=PERSONAL_NAMESPACE, timestamp="t", resource_count=1,
            resources=[ResourceEntry(uri="viking://a", canonical_root="viking://a")],
        )
        d = diff(inv, inv)
        assert d.added == []
        assert d.removed == []
        assert d.net_change == 0

    def test_save_and_load_inventory_roundtrip(self):
        from scripts.ops.resource_inventory import (
            Inventory, ResourceEntry, save_inventory, load_inventory,
        )
        with tempfile.TemporaryDirectory() as td:
            inv = Inventory(
                namespace=PERSONAL_NAMESPACE, timestamp="2026-07-21T12:00:00Z",
                resource_count=2,
                resources=[
                    ResourceEntry(uri="viking://a", canonical_root="viking://a", size_hint=1024),
                    ResourceEntry(uri="viking://b", canonical_root="viking://b"),
                ],
                metadata={"note": "test"},
            )
            path = save_inventory(inv, Path(td) / "inv.json")
            loaded = load_inventory(path)
            assert loaded.resource_count == 2
            assert loaded.namespace == PERSONAL_NAMESPACE
            assert loaded.resources[0].size_hint == 1024
            assert loaded.metadata["note"] == "test"


# ---------------------------------------------------------------------------
# Test cleanup tests
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_run_cleanup_dry_run_finds_resources(self):
        from scripts.ops.test_cleanup import run_cleanup, CleanupReport

        class FakeCleanupClient:
            def rm(self, uri, **kwargs):
                pass
            def ls(self, uri, **kwargs):
                return [
                    {"isDir": False, "uri": f"{PERSONAL_NAMESPACE}/E2E-TEST/textbook/ch1-abc"},
                    {"isDir": False, "uri": f"{PERSONAL_NAMESPACE}/E2E-TEST/transcript/lec1-def"},
                ]

        client = FakeCleanupClient()
        report = run_cleanup(client, pattern="E2E-TEST", dry_run=True)
        assert report.dry_run is True
        assert report.total_found == 2
        assert report.removed == 0
        assert report.failed == 0

    def test_run_cleanup_live_removes(self):
        from scripts.ops.test_cleanup import run_cleanup

        removed = []

        class FakeCleanupClient:
            def rm(self, uri, **kwargs):
                removed.append(uri)
            def ls(self, uri, **kwargs):
                return [
                    {"isDir": False, "uri": f"{PERSONAL_NAMESPACE}/TEST/ch1"},
                ]

        client = FakeCleanupClient()
        report = run_cleanup(client, pattern="TEST", dry_run=False)
        assert report.removed == 1
        assert len(removed) == 1

    def test_cleanup_with_explicit_uris(self):
        from scripts.ops.test_cleanup import run_cleanup

        class FakeCleanupClient:
            def rm(self, uri, **kwargs):
                pass
            def ls(self, uri, **kwargs):
                return []

        client = FakeCleanupClient()
        report = run_cleanup(
            client,
            uris=["viking://resources/personal-kb/TEST/ch1", "viking://resources/personal-kb/TEST/ch2"],
            dry_run=True,
        )
        assert report.total_found == 2

    def test_cleanup_hindsight_bank_dry_run(self):
        from scripts.ops.test_cleanup import cleanup_hindsight_bank

        result = cleanup_hindsight_bank(None, "test-bank", dry_run=True)
        assert result["dry_run"] is True
        assert result["removed"] is False

    def test_cleanup_hindsight_bank_live(self):
        from scripts.ops.test_cleanup import cleanup_hindsight_bank

        class FakeHS:
            def delete_bank(self, bank):
                assert bank == "test-bank"

        client = FakeHS()
        result = cleanup_hindsight_bank(client, "test-bank", dry_run=False)
        assert result["removed"] is True
        assert result["error"] is None

    def test_cleanup_hindsight_bank_live_error(self):
        from scripts.ops.test_cleanup import cleanup_hindsight_bank

        class FailingHS:
            def delete_bank(self, bank):
                raise RuntimeError("bank not found")

        client = FailingHS()
        result = cleanup_hindsight_bank(client, "bad-bank", dry_run=False)
        assert result["removed"] is False
        assert "bank not found" in result["error"]


# ---------------------------------------------------------------------------
# Manifest template test
# ---------------------------------------------------------------------------


class TestManifestTemplate:
    def test_template_is_valid_json(self):
        path = PROJECT_ROOT / "config" / "course_manifest_template.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "schema_version" in data
        assert "sources" in data
        assert len(data["sources"]) >= 3  # at least textbook, transcript, homework

    def test_template_has_required_fields(self):
        path = PROJECT_ROOT / "config" / "course_manifest_template.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        required = data["required_fields"]
        for source in data["sources"]:
            # Skip _comment markers.
            if "_comment" in source:
                continue
            for field in required:
                assert field in source, f"Missing {field} in {source}"
