"""Tests for the isolated end-to-end gate (offline, no live services).

Every test uses injectable fake clients — no OpenViking, Hindsight,
embedding, or LLM calls.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "test_runs"))

from scripts.core.openviking_backend import (
    PERSONAL_NAMESPACE,
    PersonalOpenVikingBackend,
    SourceHashMismatch,
    canonical_resource_root,
    resource_uri,
)
from e2e_gate import (
    FakeHindsightClient,
    FakeLiveClient,
    GateReport,
    GateStep,
    _load_manifest,
    _make_source,
    _step_cleanup,
    _step_duplicate_idempotent,
    _step_first_write,
    _step_hash_mismatch,
    _step_hindsight_isolated,
    _step_provenance,
    _step_rollback_inventory,
    _step_source_readback,
    run_e2e_gate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gate_tmp():
    """Temporary directory with a valid test manifest."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        gate_root = root / "test_runs" / "e2e_gate_workdir"
        gate_root.mkdir(parents=True)
        manifest_path = gate_root / "manifest.json"
        source_txt = gate_root / "chapter1_test.txt"
        content = (
            "This is an end-to-end gate source for the Personal KB pipeline.\n"
            "A closed-loop control system uses feedback to reduce error.\n"
            "Token: TEST\n"
        )
        source_path_str, source_hash = _make_source(source_txt, content)
        manifest = {
            "schema_version": 1,
            "hash_algorithm": "sha256",
            "required_fields": ["source_path", "source_hash", "source_type"],
            "sources": [
                {
                    "source_path": str(source_txt),
                    "source_hash": source_hash,
                    "source_type": "textbook",
                    "course": "E2E-TEST",
                    "corpus": "test",
                    "semester": "TEST",
                    "date": "2026-07-21",
                }
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        yield {
            "root": root,
            "gate_root": gate_root,
            "manifest_path": manifest_path,
            "source_txt": source_txt,
            "source_hash": source_hash,
            "source_path_str": source_path_str,
        }


@pytest.fixture
def shared_client():
    return FakeLiveClient()


@pytest.fixture
def hs_client():
    return FakeHindsightClient()


def _factory(root: Path, client: FakeLiveClient):
    def f():
        return PersonalOpenVikingBackend(client, root=root)
    return f


# ---------------------------------------------------------------------------
# Individual step tests
# ---------------------------------------------------------------------------


def test_first_write_passes(gate_tmp, shared_client):
    result = _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    assert result.passed, result.errors
    assert result.detail["uri"].startswith(PERSONAL_NAMESPACE + "/")
    assert result.detail["source_type"] == "textbook"
    assert result.detail["course"] == "E2E-TEST"
    # add_resource was called
    assert len(shared_client.add_calls) == 1
    # reason is empty (no memory linking)
    assert shared_client.add_calls[0]["reason"] == ""


def test_duplicate_idempotent_passes(gate_tmp, shared_client):
    # First write to populate the tree.
    _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    add_before = len(shared_client.add_calls)

    result = _step_duplicate_idempotent(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    assert result.passed, result.errors
    assert result.detail["reason"] == "identical_source_hash"
    # No additional add_resource call
    assert len(shared_client.add_calls) == add_before


def test_hash_mismatch_raises(gate_tmp, shared_client):
    # First write to populate the tree.
    _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    add_before = len(shared_client.add_calls)

    result = _step_hash_mismatch(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    assert result.passed, result.errors
    assert "was changed" in result.detail["error_message"]
    # No additional add_resource call
    assert len(shared_client.add_calls) == add_before


def test_hash_mismatch_requires_prior_write(gate_tmp, shared_client):
    """Without a prior write, hash mismatch should NOT fire."""
    result = _step_hash_mismatch(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    # Without prior write, the logical-source index is empty,
    # so no mismatch is detected. The step expects a mismatch raise
    # but gets normal ingestion instead.
    assert not result.passed
    assert "Expected SourceHashMismatch" in result.errors[0]


def test_provenance_passes(gate_tmp, shared_client):
    # First write so there are add_calls to inspect.
    _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )

    result = _step_provenance(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    assert result.passed, result.errors
    assert result.detail["reason_is_empty"] is True
    assert "personal_kb_processor_version" in result.detail["provenance_keys"]
    assert "source_hash" in result.detail["provenance_keys"]


def test_source_readback_passes(gate_tmp, shared_client):
    _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )

    result = _step_source_readback(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )
    assert result.passed, result.errors
    assert result.detail["content_length"] > 0
    assert "end-to-end gate source" in result.detail["expected_fragment"]


def test_source_readback_uses_leaf_when_real_sdk_rejects_directory(gate_tmp):
    class DirectoryRejectingClient(FakeLiveClient):
        def ls(self, uri, **kwargs):
            if uri != PERSONAL_NAMESPACE:
                return [{"isDir": False, "uri": uri.rstrip("/") + "/content.txt"}]
            return super().ls(uri, **kwargs)

        def read(self, uri, **kwargs):
            if uri == canonical_resource_root(uri):
                raise ValueError("Cannot read directory as file")
            return super().read(uri, **kwargs)

    client = DirectoryRejectingClient()
    _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], client),
    )
    result = _step_source_readback(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], client),
    )
    assert result.passed, result.errors
    assert result.detail["read_uri"] != result.detail["canonical_uri"]


def test_hindsight_isolated_passes(hs_client):
    token = "TEST-TOKEN-12345"
    bank = "hermes-e2e-gate-test"

    result = _step_hindsight_isolated(hs_client, bank, token)
    assert result.passed, result.errors
    assert result.detail["recall_count"] == 1
    assert result.detail["bank_count"] == 1


def test_hindsight_isolated_skipped_when_none():
    result = _step_hindsight_isolated(None, "bank", "token")
    assert result.passed
    assert result.detail["skipped"] == "no-hindsight-client"


def test_rollback_inventory_passes():
    pre = {"namespace": PERSONAL_NAMESPACE, "resource_count": 5, "resources": ["a", "b"]}
    post = {"namespace": PERSONAL_NAMESPACE, "resource_count": 6}

    result = _step_rollback_inventory(pre, post)
    assert result.passed
    assert result.detail["pre_resources"] == 5
    assert result.detail["post_resources"] == 6


def test_rollback_inventory_no_data():
    result = _step_rollback_inventory(None, None)
    assert result.passed


def test_cleanup_passes(gate_tmp, shared_client, hs_client):
    _step_first_write(
        gate_tmp["manifest_path"],
        gate_tmp["root"],
        _factory(gate_tmp["root"], shared_client),
    )

    bank = "hermes-e2e-cleanup-test"
    hs_client.create_bank(bank, "test")

    result = _step_cleanup(
        gate_tmp["root"],
        gate_tmp["manifest_path"],
        _factory(gate_tmp["root"], shared_client),
        hs_client,
        bank,
    )
    assert result.passed, result.errors
    assert result.detail["ov_errors"] == []
    assert result.detail["hs_errors"] == []


# ---------------------------------------------------------------------------
# Full gate tests
# ---------------------------------------------------------------------------


def test_run_e2e_gate_offline_passes():
    """Full offline gate must pass all 8 steps."""
    with tempfile.TemporaryDirectory() as td:
        report = run_e2e_gate(
            workdir=td,
            live=False,
            hindsight_client=FakeHindsightClient(),
        )
    assert report.passed, [s.errors for s in report.steps if not s.passed]
    assert report.step_count == 8
    assert report.passed_count == 8


def test_run_e2e_gate_is_repeatable():
    """The gate must be idempotent — running twice with the same workdir
    must pass both times."""
    with tempfile.TemporaryDirectory() as td:
        report1 = run_e2e_gate(
            workdir=td,
            live=False,
            hindsight_client=FakeHindsightClient(),
        )
        assert report1.passed

        report2 = run_e2e_gate(
            workdir=td,
            live=False,
            hindsight_client=FakeHindsightClient(),
        )
        assert report2.passed, [s.errors for s in report2.steps if not s.passed]


def test_run_e2e_gate_json_output():
    """Gate must produce valid JSON output."""
    with tempfile.TemporaryDirectory() as td:
        report = run_e2e_gate(
            workdir=td,
            live=False,
            hindsight_client=FakeHindsightClient(),
        )
    data = {
        "passed": report.passed,
        "step_count": report.step_count,
        "passed_count": report.passed_count,
        "steps": [
            {"name": s.name, "passed": s.passed, "errors": s.errors, **s.detail}
            for s in report.steps
        ],
    }
    encoded = json.dumps(data, default=str)
    assert json.loads(encoded)["passed"] is True
    assert json.loads(encoded)["step_count"] == 8


def test_e2e_gate_fake_client_records_add_calls():
    client = FakeLiveClient()
    client.add_resource(to="test-uri", path="/tmp/test.txt", reason="", args={})
    assert len(client.add_calls) == 1
    assert client.add_calls[0]["reason"] == ""


def test_e2e_gate_fake_hindsight_fake_is_isolated():
    hs = FakeHindsightClient()
    hs.create_bank("bank-a", "Bank A")
    hs.retain("bank-a", items=[{"content": "fact A", "context": "test", "document_id": "1", "tags": ["a"]}])
    hs.retain("bank-b", items=[{"content": "fact B", "context": "test", "document_id": "2", "tags": ["b"]}])

    # Recall from bank-a must not see bank-b items.
    result = hs.recall("bank-a", query="fact", top_k=5)
    assert len(result["results"]) == 1
    assert "fact A" in result["results"][0]["content"]

    # Delete bank-a, verify bank-b is untouched.
    hs.delete_bank("bank-a")
    assert hs.count_bank("bank-a") == 0
    assert hs.count_bank("bank-b") == 1
