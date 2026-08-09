import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingestion.preflight import (  # noqa: E402
    CheckResult,
    check_dry_run,
    check_git_tree,
    check_logical_source_conflicts,
    check_manifest,
    check_routing,
    check_services,
    run_preflight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_manifest(path: Path, **metadata) -> dict[str, Any]:
    """Build a minimal valid manifest with one source."""
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "sources": [
            {
                "source_path": str(path),
                "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_type": "homework",
                **metadata,
            }
        ],
    }


# ---------------------------------------------------------------------------
# check_manifest
# ---------------------------------------------------------------------------


def test_check_manifest_passes_for_valid_manifest(tmp_path):
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"test content")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest(pdf, course="TEST", corpus="test")), encoding="utf-8")

    result = check_manifest(manifest_path, tmp_path)

    assert result.passed
    assert result.details["source_count"] == 1


def test_check_manifest_fails_when_file_missing(tmp_path):
    result = check_manifest(tmp_path / "nonexistent.json", tmp_path)
    assert not result.passed
    assert any("not found" in err for err in result.errors)


def test_check_manifest_fails_when_hash_mismatch(tmp_path):
    pdf = tmp_path / "exam.pdf"
    pdf.write_bytes(b"test content")
    manifest = _valid_manifest(pdf, course="TEST", corpus="test")
    manifest["sources"][0]["source_hash"] = "0" * 64
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_manifest(manifest_path, tmp_path)

    assert not result.passed
    assert any("hash mismatch" in err for err in result.errors)


# ---------------------------------------------------------------------------
# check_services
# ---------------------------------------------------------------------------


def test_check_services_all_healthy_with_injectable_probes():
    result = check_services(
        ov_client=_FakeHealthyOVClient(),
        embedding_probe=lambda: (True, "ok"),
        hindsight_probe=lambda: (True, "ok"),
    )
    assert result.passed
    for svc in ("openviking", "embedding", "hindsight"):
        assert result.details["services"][svc]["status"] == "healthy"


def test_check_services_flags_unreachable_services():
    result = check_services(
        ov_client=_FakeHealthyOVClient(),
        embedding_probe=lambda: (False, "connection refused"),
        hindsight_probe=lambda: (True, "ok"),
    )
    assert not result.passed
    assert result.details["services"]["embedding"]["status"] == "unreachable"


class _FakeHealthyOVClient:
    @staticmethod
    def health():
        return {"healthy": True}


class _FakeUnhealthyOVClient:
    @staticmethod
    def health():
        raise ConnectionError("unreachable")


def test_check_services_flags_unhealthy_ov_client():
    result = check_services(
        ov_client=_FakeUnhealthyOVClient(),
        embedding_probe=lambda: (True, "ok"),
        hindsight_probe=lambda: (True, "ok"),
    )
    assert not result.passed
    assert result.details["services"]["openviking"]["status"] == "unreachable"


# ---------------------------------------------------------------------------
# check_routing
# ---------------------------------------------------------------------------


def test_check_routing_passes_with_correct_config():
    result = check_routing(
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-fake",
    )
    assert result.passed
    assert result.details["api_key"] == "configured"


def test_check_routing_rejects_wrong_model():
    result = check_routing(model="gpt-4o")
    assert not result.passed
    assert any("Model" in err for err in result.errors)


def test_check_routing_rejects_wrong_base_url():
    result = check_routing(base_url="https://api.openai.com/v1")
    assert not result.passed
    assert any("Base URL" in err for err in result.errors)


def test_check_routing_rejects_flash_override():
    result = check_routing(model="deepseek-v4-pro-flash")
    assert not result.passed
    assert any("Flash" in err for err in result.errors)


def test_check_routing_warns_missing_api_key():
    """When no key provided, warn but don't fail."""
    result = check_routing(api_key="")
    assert result.passed
    # Warning lands in details (from _ok helper), not on result.warnings.
    assert any("not set" in w for w in result.details.get("warnings", []))


# ---------------------------------------------------------------------------
# check_logical_source_conflicts
# ---------------------------------------------------------------------------


def test_check_logical_source_conflicts_no_conflicts(tmp_path):
    pdf = tmp_path / "hw1.pdf"
    pdf.write_bytes(b"unique content")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest(pdf, course="TEST", corpus="test")), encoding="utf-8")

    result = check_logical_source_conflicts(manifest_path, tmp_path)

    assert result.passed
    assert result.details["unique_sources"] == 1


def test_check_logical_source_conflicts_no_conflict_with_parent_disambiguation(tmp_path):
    """Files with same stem in different dirs are now disambiguated — no false conflict."""
    dir_a = tmp_path / "a"
    dir_a.mkdir()
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_a / "notes.md").write_text("alpha content")
    (dir_b / "notes.md").write_text("beta content")

    manifest = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "sources": [
            {
                "source_path": str(dir_a / "notes.md"),
                "source_hash": hashlib.sha256((dir_a / "notes.md").read_bytes()).hexdigest(),
                "source_type": "source",
                "corpus": "test",
            },
            {
                "source_path": str(dir_b / "notes.md"),
                "source_hash": hashlib.sha256((dir_b / "notes.md").read_bytes()).hexdigest(),
                "source_type": "source",
                "corpus": "test",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_logical_source_conflicts(manifest_path, tmp_path)

    # Parent-directory disambiguation → two distinct logical sources, no conflict.
    assert result.passed
    assert result.details["unique_sources"] == 2
    assert result.details["conflicts"] == []


def test_check_logical_source_conflicts_same_dir_same_stem_different_hash(tmp_path):
    """Same file re-ingested with different hash in manifest → still a conflict."""
    pdf = tmp_path / "hw1.pdf"
    pdf.write_bytes(b"version one")
    hash_one = hashlib.sha256(pdf.read_bytes()).hexdigest()

    # Build a manifest with two entries for the SAME file path but different hashes.
    manifest = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "sources": [
            {
                "source_path": str(pdf),
                "source_hash": hash_one,
                "source_type": "homework",
                "corpus": "test",
            },
            {
                "source_path": str(pdf),
                "source_hash": "b" * 64,
                "source_type": "homework",
                "corpus": "test",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = check_logical_source_conflicts(manifest_path, tmp_path)

    # Same logical source + different declared hashes = conflict.
    assert not result.passed
    assert len(result.details["conflicts"]) == 1


# ---------------------------------------------------------------------------
# check_dry_run
# ---------------------------------------------------------------------------


def test_check_dry_run_produces_preview_for_every_entry(tmp_path):
    pdf = tmp_path / "hw1.pdf"
    pdf.write_bytes(b"homework content")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest(pdf, course="TEST", corpus="test")), encoding="utf-8")

    result = check_dry_run(manifest_path, tmp_path)

    assert result.passed
    assert result.details["count"] == 1
    preview = result.details["previews"][0]
    assert preview["source_path"] == str(pdf)
    assert preview["uri"].startswith("viking://resources/personal-kb/")
    assert preview["source_hash"] is not None


def test_check_dry_run_fails_on_missing_manifest(tmp_path):
    result = check_dry_run(tmp_path / "missing.json", tmp_path)
    assert not result.passed


# ---------------------------------------------------------------------------
# check_git_tree
# ---------------------------------------------------------------------------


def test_check_git_tree_skips_when_git_not_found(tmp_path, monkeypatch):
    """Simulate git not available."""
    import subprocess

    original_run = subprocess.run
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(1)
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = check_git_tree(tmp_path)
    assert result.passed
    assert result.details["status"] == "no-git"


# ---------------------------------------------------------------------------
# run_preflight (integration)
# ---------------------------------------------------------------------------


def test_run_preflight_full_passes_with_valid_manifest_and_injectables(tmp_path):
    pdf = tmp_path / "hw1.pdf"
    pdf.write_bytes(b"test homework")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest(pdf, course="TEST", corpus="test")), encoding="utf-8")

    report = run_preflight(
        manifest_path=manifest_path,
        root=tmp_path,
        ov_client=_FakeHealthyOVClient(),
        embedding_probe=lambda: (True, "ok"),
        hindsight_probe=lambda: (True, "ok"),
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-test",
        skip_git=True,
    )

    assert report["valid"]
    assert report["passed_count"] == report["check_count"]


def test_run_preflight_fails_on_bad_routing(tmp_path):
    pdf = tmp_path / "hw1.pdf"
    pdf.write_bytes(b"test homework")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest(pdf, course="TEST", corpus="test")), encoding="utf-8")

    report = run_preflight(
        manifest_path=manifest_path,
        root=tmp_path,
        ov_client=_FakeHealthyOVClient(),
        embedding_probe=lambda: (True, "ok"),
        hindsight_probe=lambda: (True, "ok"),
        model="gpt-4o",  # wrong model
        skip_git=True,
    )

    assert not report["valid"]
    assert any(c["name"] == "routing" and not c["passed"] for c in report["checks"])


def test_run_preflight_skips_services_when_requested(tmp_path):
    pdf = tmp_path / "hw1.pdf"
    pdf.write_bytes(b"test homework")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_valid_manifest(pdf, course="TEST", corpus="test")), encoding="utf-8")

    report = run_preflight(
        manifest_path=manifest_path,
        root=tmp_path,
        skip_services=True,
        skip_git=True,
    )

    assert report["valid"]
    assert not any(c["name"] == "services" for c in report["checks"])
    assert not any(c["name"] == "git-tree" for c in report["checks"])
