from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.ops.batch_ingest import BatchItem, run_batch


class FakeBackend:
    def __init__(self, *, fail_hash: str | None = None):
        self.calls = []
        self.fail_hash = fail_hash

    def index_file(self, path, **kwargs):
        self.calls.append((Path(path), kwargs))
        if kwargs.get("source_hash") == self.fail_hash:
            raise RuntimeError("simulated index failure")
        return SimpleNamespace(
            source_path=str(path),
            uri=f"viking://resources/personal-kb/test/{Path(path).stem}",
            source_type=kwargs["source_type"],
            course=kwargs.get("course"),
            lecture=kwargs.get("lecture"),
            result={"status": "dry_run" if kwargs.get("dry_run") else "indexed"},
        )


class IdempotentBackend(FakeBackend):
    def index_file(self, path, **kwargs):
        self.calls.append((Path(path), kwargs))
        return SimpleNamespace(
            source_path=str(path),
            uri=f"viking://resources/personal-kb/test/{Path(path).stem}",
            result={"status": "skipped", "reason": "identical_source_hash"},
        )


def _item(root: Path, name: str, digest: str) -> BatchItem:
    path = root / name
    path.write_text(f"content for {name}", encoding="utf-8")
    return BatchItem(
        index_path=path,
        source_path=path,
        source_hash=digest,
        source_type="transcript",
        course="TEST-SE424",
        lecture=1,
    )


def test_run_batch_indexes_all_items_and_writes_a_resume_log(tmp_path):
    backend = FakeBackend()
    log_path = tmp_path / "ingestion.jsonl"
    items = [_item(tmp_path, "a.txt", "a" * 64), _item(tmp_path, "b.txt", "b" * 64)]

    report = run_batch(items, backend=backend, log_path=log_path)

    assert report["counts"] == {"indexed": 2, "skipped": 0, "failed": 0}
    assert len(backend.calls) == 2
    assert backend.calls[0][1]["provenance_source_path"] == str(items[0].source_path.resolve())
    assert log_path.read_text(encoding="utf-8").count("\n") == 2

    second = run_batch(items, backend=backend, log_path=log_path)
    assert second["counts"] == {"indexed": 0, "skipped": 2, "failed": 0}
    assert len(backend.calls) == 2


def test_run_batch_continues_after_one_failure_and_reports_exact_item(tmp_path):
    bad_hash = "b" * 64
    backend = FakeBackend(fail_hash=bad_hash)
    items = [_item(tmp_path, "a.txt", "a" * 64), _item(tmp_path, "b.txt", bad_hash), _item(tmp_path, "c.txt", "c" * 64)]

    report = run_batch(items, backend=backend, log_path=tmp_path / "errors.jsonl")

    assert report["counts"] == {"indexed": 2, "skipped": 0, "failed": 1}
    assert report["failed"][0]["source_hash"] == bad_hash
    assert [call[0].name for call in backend.calls] == ["a.txt", "b.txt", "c.txt"]


def test_run_batch_dry_run_never_marks_items_as_indexed(tmp_path):
    backend = FakeBackend()
    item = _item(tmp_path, "dry.txt", "d" * 64)

    report = run_batch([item], backend=backend, log_path=tmp_path / "dry.jsonl", dry_run=True)

    assert report["counts"] == {"indexed": 1, "skipped": 0, "failed": 0}
    assert backend.calls[0][1]["dry_run"] is True
    assert '"status": "indexed"' in (tmp_path / "dry.jsonl").read_text(encoding="utf-8")


def test_dry_run_log_does_not_skip_following_real_run(tmp_path):
    item = _item(tmp_path, "planned.txt", "p" * 64)
    log_path = tmp_path / "resume.jsonl"
    dry_backend = FakeBackend()
    run_batch([item], backend=dry_backend, log_path=log_path, dry_run=True)

    live_backend = FakeBackend()
    report = run_batch([item], backend=live_backend, log_path=log_path)

    assert report["counts"] == {"indexed": 1, "skipped": 0, "failed": 0}
    assert len(live_backend.calls) == 1


def test_run_batch_preserves_backend_idempotent_skip_status(tmp_path):
    backend = IdempotentBackend()
    item = _item(tmp_path, "same.txt", "s" * 64)

    report = run_batch([item], backend=backend, log_path=tmp_path / "skip.jsonl")

    assert report["counts"] == {"indexed": 0, "skipped": 1, "failed": 0}
    assert '"status": "skipped"' in (tmp_path / "skip.jsonl").read_text(encoding="utf-8")
