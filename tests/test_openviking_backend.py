import hashlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.openviking_backend import (  # noqa: E402
    PERSONAL_NAMESPACE,
    AudioSourceRejected,
    PersonalOpenVikingBackend,
    is_approved_personal_path,
    metadata_matches,
    resource_uri,
    ResourceInventoryUnavailable,
    UnsupportedSource,
)


class FakeOpenVikingClient:
    def __init__(self, search_response=None):
        self.add_calls = []
        self.search_calls = []
        self.read_calls = []
        self.search_response = search_response or {"resources": []}

    def add_resource(self, **kwargs):
        self.add_calls.append(kwargs)
        return {"uri": kwargs["to"], "status": "completed"}

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_response

    def read(self, uri, **kwargs):
        self.read_calls.append((uri, kwargs))
        return "transcript context"


def test_audio_is_rejected_before_client_call(tmp_path):
    audio = tmp_path / "lecture.m4a"
    audio.write_bytes(b"audio bytes")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    with pytest.raises(AudioSourceRejected, match="local-only"):
        backend.index_file(audio, source_type="lecture-audio", course="PERSONAL-ALPHA")

    assert client.add_calls == []


def test_transcript_is_indexed_without_audio_metadata(tmp_path):
    transcript = tmp_path / "lecture_01.txt"
    transcript.write_text("The closed-loop transfer function is important.", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    resource = backend.index_file(
        transcript,
        source_type="transcript",
        course="PERSONAL-ALPHA",
        lecture=1,
        metadata={"engine": "whisper", "audio_path": str(tmp_path / "lecture_01.m4a")},
    )

    assert resource.uri.startswith(PERSONAL_NAMESPACE)
    assert len(client.add_calls) == 1
    call = client.add_calls[0]
    assert call["path"] == str(transcript.resolve())
    assert call["to"] == resource.uri
    assert "m4a" not in repr(call).lower()
    assert resource.metadata == {
        "engine": "whisper",
        "course": "PERSONAL-ALPHA",
        "lecture": 1,
        "source_type": "transcript",
    }


def test_index_file_carries_provenance_in_args_instead_of_reason(tmp_path):
    source = tmp_path / "homework.txt"
    source.write_text("test source", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    backend.index_file(
        source,
        source_type="homework",
        course="TEST",
        source_hash="a" * 64,
        metadata={
            "semester": "TEST",
            "corpus": "test",
        },
    )

    call = client.add_calls[0]
    # reason must be empty to prevent server-side memory linking.
    assert call["reason"] == "", f"expected empty reason, got {call['reason']!r}"

    args = call.get("args", {})
    assert args["source_type"] == "homework"
    assert args["course"] == "TEST"
    assert args["source_hash"] == "a" * 64
    assert args["semester"] == "TEST"
    assert args["corpus"] == "test"
    assert "personal_kb_processor_version" in args
    assert "ingestion_timestamp" in args
    assert "provenance_source_path" in args


def test_dry_run_validates_transcript_without_uploading(tmp_path):
    transcript = tmp_path / "lecture.md"
    transcript.write_text("A transcript with enough content for validation.", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    resource = backend.index_file(transcript, source_type="transcript", dry_run=True)

    assert resource.result is None
    assert resource.uri.startswith(PERSONAL_NAMESPACE)
    assert client.add_calls == []


def test_tree_applies_curated_policy_and_skips_unapproved_files(tmp_path):
    """Only CURATED_SOURCE_PATHS entries are approved; everything else skipped."""
    config = tmp_path / "config"
    config.mkdir()
    (config / "course_catalog.yaml").write_text("courses: {}", encoding="utf-8")
    (tmp_path / "lecture.txt").write_text("lecture transcript", encoding="utf-8")
    (tmp_path / "test_models").mkdir()
    (tmp_path / "test_models" / "fixture.txt").write_text("fixture", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    resources = backend.index_tree(tmp_path, source_type="personal-source", dry_run=True)

    assert {Path(item.source_path).name for item in resources} == {"course_catalog.yaml"}
    assert len(resources) == 1
    assert client.add_calls == []
    assert is_approved_personal_path(config / "course_catalog.yaml", tmp_path)
    assert not is_approved_personal_path(tmp_path / "lecture.txt", tmp_path)

def test_resource_uri_is_deterministic_for_idempotent_reingestion(tmp_path):
    source = tmp_path / "lecture.txt"

    first = resource_uri(source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path)
    second = resource_uri(source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path)

    assert first == second
    assert first.startswith(PERSONAL_NAMESPACE + "/")


def test_search_and_read_are_limited_to_personal_namespace():
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client)

    backend.search("feedback", limit=3)
    backend.read(PERSONAL_NAMESPACE + "/transcript/lecture-abc")

    with pytest.raises(ValueError, match="outside the Personal KB namespace"):
        backend.read("viking://resources/other")


def test_source_scope_filters_course_and_assessment_material():
    assert metadata_matches({"source_type": "lecture"}, {"source_scope": "course"})
    assert metadata_matches({"source_type": "transcript"}, {"source_scope": "course"})
    assert not metadata_matches({"source_type": "homework"}, {"source_scope": "course"})
    assert not metadata_matches({"source_type": "exam"}, {"source_scope": "course"})
    assert metadata_matches({"source_type": "homework"}, {"source_scope": "assessment"})
    assert metadata_matches({"source_type": "exam"}, {"source_scope": "assessment"})
    assert not metadata_matches({"source_type": "lecture"}, {"source_scope": "assessment"})
    assert metadata_matches({"source_type": "exam"}, {"source_scope": "all"})


def test_search_applies_metadata_filters_after_dense_retrieval():
    client = FakeOpenVikingClient(
        search_response={
            "resources": [
                {
                    "uri": f"{PERSONAL_NAMESPACE}/lecture-one",
                    "score": 0.9,
                    "metadata": {"course": "KB 1001", "lecture": 1},
                },
                {
                    "uri": f"{PERSONAL_NAMESPACE}/lecture-two",
                    "score": 0.8,
                    "metadata": {"course": "KB 1001", "lecture": 2},
                },
            ]
        }
    )
    backend = PersonalOpenVikingBackend(client)

    result = backend.search(
        "kanban",
        limit=10,
        filters={"course": "KB 1001", "lecture": 1},
    )

    assert [item["uri"] for item in result["resources"]] == [
        f"{PERSONAL_NAMESPACE}/lecture-one"
    ]
    assert client.search_calls[0]["limit"] == 10
    assert "filters" not in client.search_calls[0]


def test_search_merges_manifest_metadata_for_filters_not_in_uri(tmp_path):
    source = tmp_path / "homework.pdf"
    source.write_bytes(b"test pdf")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "config" / "source_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_path": str(source),
                        "source_hash": source_hash,
                        "source_type": "homework",
                        "course": "TEST",
                        "semester": "TEST",
                        "corpus": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    canonical = resource_uri(
        source, course="TEST", source_type="homework", root=tmp_path,
        source_hash=source_hash,
    )
    client = FakeOpenVikingClient(
        search_response={
            "resources": [
                {"uri": f"{canonical}/.overview.md", "score": 0.9},
            ]
        }
    )
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    result = backend.search("homework", filters={"semester": "TEST"})

    assert [item["uri"] for item in result["resources"]] == [
        f"{canonical}/.overview.md"
    ]
    assert result["filters_applied"] == {"semester": "TEST"}


def test_read_repairs_stale_filename_to_canonical_tree_uri():
    canonical = f"{PERSONAL_NAMESPACE}/uncategorized/personal-source/course_catalog-abc/course_catalog.md"
    stale = f"{PERSONAL_NAMESPACE}/course_catalog.yaml"

    class RepairClient(FakeOpenVikingClient):
        def read(self, uri, **kwargs):
            self.read_calls.append((uri, kwargs))
            if uri == stale:
                raise FileNotFoundError(uri)
            return "canonical source"

        def ls(self, uri, **kwargs):
            if uri == PERSONAL_NAMESPACE:
                return [{"isDir": True, "uri": f"{PERSONAL_NAMESPACE}/uncategorized/personal-source/course_catalog-abc"}]
            return [{"isDir": False, "name": "course_catalog.md", "uri": canonical}]

    backend = PersonalOpenVikingBackend(RepairClient())
    assert backend.read(stale) == "canonical source"
    assert backend.resolve_uri(stale) == canonical


def test_ambiguous_uri_repair_fails_instead_of_picking_first():
    class AmbiguousClient(FakeOpenVikingClient):
        def ls(self, uri, **kwargs):
            return [
                {"isDir": True, "uri": f"{PERSONAL_NAMESPACE}/course-a/source/a"},
                {"isDir": True, "uri": f"{PERSONAL_NAMESPACE}/course-b/source/b"},
            ] if uri == PERSONAL_NAMESPACE else [
                {"isDir": False, "name": "notes.md", "uri": f"{uri}/notes.md"},
            ]

    backend = PersonalOpenVikingBackend(AmbiguousClient())

    with pytest.raises(ValueError, match="ambiguous"):
        backend.resolve_uri(f"{PERSONAL_NAMESPACE}/notes.md")


def test_inventory_failure_is_explicit_in_strict_mode():
    class BrokenInventoryClient(FakeOpenVikingClient):
        def ls(self, uri, **kwargs):
            raise ConnectionError("inventory unavailable")

    backend = PersonalOpenVikingBackend(BrokenInventoryClient(), strict_inventory=True)

    with pytest.raises(ResourceInventoryUnavailable, match="inventory unavailable"):
        backend._resource_files()


def test_resource_inventory_cache_is_invalidated_after_write(tmp_path):
    source = tmp_path / "lecture.txt"
    source.write_text("content", encoding="utf-8")

    class InventoryClient(FakeOpenVikingClient):
        def __init__(self):
            super().__init__()
            self.resource_root = None

        def ls(self, uri, **kwargs):
            if uri == PERSONAL_NAMESPACE:
                return (
                    [{"isDir": True, "uri": self.resource_root}]
                    if self.resource_root
                    else []
                )
            return (
                [{"isDir": False, "uri": f"{self.resource_root}/chunk.md"}]
                if self.resource_root
                else []
            )

        def add_resource(self, **kwargs):
            result = super().add_resource(**kwargs)
            self.resource_root = kwargs["to"]
            return result

    client = InventoryClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)
    assert backend._resource_files() == []
    backend.index_file(source, source_type="lecture", source_hash="a" * 64)

    assert backend._resource_files()


def test_index_file_rejects_source_outside_configured_root(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    backend = PersonalOpenVikingBackend(FakeOpenVikingClient(), root=tmp_path)

    with pytest.raises(UnsupportedSource, match="outside configured root"):
        backend.index_file(outside, source_type="lecture")


def test_unrepairable_uri_remains_missing():
    backend = PersonalOpenVikingBackend(FakeOpenVikingClient())
    assert backend.resolve_uri(f"{PERSONAL_NAMESPACE}/README.md") is None


def test_namespace_prefix_collision_is_rejected():
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client)

    with pytest.raises(ValueError, match="outside the Personal KB namespace"):
        backend.read(PERSONAL_NAMESPACE + "-unauthorized/resource")


def test_provenance_args_includes_processor_version():
    from scripts.core.openviking_backend import _provenance_args

    args = _provenance_args(source_type="homework", course="TEST", source_hash="a" * 64)

    assert args["personal_kb_processor_version"] == "1.0.0"
    assert args["source_type"] == "homework"
    assert args["course"] == "TEST"
    assert args["source_hash"] == "a" * 64
    assert "ingestion_timestamp" in args


def test_provenance_args_includes_optional_metadata():
    from scripts.core.openviking_backend import _provenance_args

    args = _provenance_args(
        {"semester": "FA26", "date": "2026-09-01", "corpus": "test"},
        source_type="exam",
        lecture=1,
        source_path="/tmp/test.pdf",
    )

    assert args["semester"] == "FA26"
    assert args["date"] == "2026-09-01"
    assert args["corpus"] == "test"
    assert args["lecture"] == 1
    assert args["provenance_source_path"] == "/tmp/test.pdf"


def test_memory_linking_disabled_by_empty_reason(tmp_path):
    source = tmp_path / "chapter.txt"
    source.write_text("test content", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    backend.index_file(source, source_type="textbook")

    call = client.add_calls[0]
    assert call["reason"] == "", "reason must be empty to prevent memory linking"


def test_index_file_passes_source_hash_to_provenance(tmp_path):
    source = tmp_path / "paper.txt"
    source.write_text("test content", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    backend.index_file(
        source, source_type="source", course="TEST",
        source_hash="abcd1234" * 8,
    )

    args = client.add_calls[0].get("args", {})
    assert args.get("source_hash") == "abcd1234" * 8


def test_provenance_args_omits_audio_keys():
    from scripts.core.openviking_backend import _provenance_args

    args = _provenance_args(
        {"audio_path": "/tmp/lecture.m4a"},
        source_type="transcript",
    )

    assert "audio_path" not in args


def test_resource_uri_binds_authoritative_source_hash(tmp_path):
    source = tmp_path / "lecture.txt"
    source.write_text("version one", encoding="utf-8")

    first = resource_uri(
        source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path,
        source_hash="a" * 64,
    )
    same = resource_uri(
        source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path,
        source_hash="A" * 64,
    )
    changed = resource_uri(
        source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path,
        source_hash="b" * 64,
    )

    assert first == same
    assert changed != first


def test_resource_uri_rejects_invalid_source_hash(tmp_path):
    source = tmp_path / "lecture.txt"
    source.write_text("version one", encoding="utf-8")

    with pytest.raises(ValueError, match="64-character SHA-256"):
        resource_uri(source, root=tmp_path, source_hash="not-a-hash")


def test_index_file_uri_changes_when_source_hash_changes(tmp_path):
    """Changed content at the same path must not silently reuse the old URI.

    This is the end-to-end identity safety guarantee: when an authoritative
    source hash is supplied to ``index_file``, the canonical URI becomes
    content-addressed, so re-ingesting edited content at the same path produces
    a distinct URI instead of aliasing the stale resource.
    """
    source = tmp_path / "homework.txt"
    source.write_text("original submission", encoding="utf-8")
    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    original = backend.index_file(
        source, source_type="homework", course="TEST",
        source_hash="a" * 64, dry_run=True,
    )
    revised = backend.index_file(
        source, source_type="homework", course="TEST",
        source_hash="b" * 64, dry_run=True,
    )

    assert original.uri != revised.uri, (
        "changed source_hash must produce a different canonical URI"
    )
    assert original.uri.startswith(PERSONAL_NAMESPACE + "/")
    assert revised.uri.startswith(PERSONAL_NAMESPACE + "/")
    # Both URIs keep the same 3-part root structure so canonical_resource_root
    # and namespace parsing remain valid.
    assert original.uri.count("/") == revised.uri.count("/")


def test_load_manifest_metadata_skips_malformed_hash_without_crashing(tmp_path):
    """A malformed manifest hash must not break backend construction.

    ``_load_manifest_metadata`` is a tolerant loader used by search/filter
    callers; ``validate_manifest`` remains the authoritative ingestion gate.
    A single bad entry must be skipped, not crash the whole backend.
    """
    good = tmp_path / "good.txt"
    good.write_text("good content", encoding="utf-8")
    good_hash = hashlib.sha256(good.read_bytes()).hexdigest()
    manifest_path = tmp_path / "config" / "source_manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_path": str(good),
                        "source_hash": good_hash,
                        "source_type": "homework",
                        "course": "TEST",
                        "corpus": "test",
                    },
                    {
                        "source_path": str(tmp_path / "bad.txt"),
                        "source_hash": "not-a-real-hash",
                        "source_type": "homework",
                        "course": "TEST",
                        "corpus": "test",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "bad.txt").write_text("bad content", encoding="utf-8")

    # Construction must not raise despite the malformed hash entry.
    backend = PersonalOpenVikingBackend(FakeOpenVikingClient(), root=tmp_path)

    # The good entry is still indexed by its canonical root; the bad one is
    # simply absent rather than crashing the loader.
    good_uri = resource_uri(
        good, course="TEST", source_type="homework", root=tmp_path,
        source_hash=good_hash,
    )
    from scripts.core.openviking_backend import canonical_resource_root

    assert canonical_resource_root(good_uri) in backend._manifest_metadata
    assert len(backend._manifest_metadata) == 1


# ---------------------------------------------------------------------------
# Strict source-change rejection tests
# ---------------------------------------------------------------------------


def test_logical_source_stem_strips_digest():
    from scripts.core.openviking_backend import _logical_source_stem

    assert _logical_source_stem("lecture-01-a1b2c3d4e5f6") == "lecture-01"
    assert _logical_source_stem("homework_1-0123456789ab") == "homework_1"
    # No digest (less than 13 chars): returned as-is.
    assert _logical_source_stem("plain-stem") == "plain-stem"
    # Trailing hyphen but not enough hex: returned as-is.
    assert _logical_source_stem("short-ab12") == "short-ab12"


def test_logical_source_key_matches_same_source_different_hash(tmp_path):
    from scripts.core.openviking_backend import _logical_source_key, resource_uri

    source = tmp_path / "lecture.txt"
    source.write_text("content v1", encoding="utf-8")

    uri_one = resource_uri(
        source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path,
        source_hash="a" * 64,
    )
    uri_two = resource_uri(
        source, course="PERSONAL-ALPHA", source_type="transcript", root=tmp_path,
        source_hash="b" * 64,
    )

    assert uri_one != uri_two  # different hashes → different URIs
    assert _logical_source_key(uri_one) == _logical_source_key(uri_two)  # same logical source


def test_index_file_raises_mismatch_when_same_logical_source_different_hash(tmp_path):
    """Strict rejection: same logical source + different hash → SourceHashMismatch."""
    from scripts.core.openviking_backend import SourceHashMismatch, resource_uri

    source = tmp_path / "homework.txt"
    source.write_text("original submission", encoding="utf-8")

    existing_uri = resource_uri(
        source, course="TEST", source_type="homework", root=tmp_path,
        source_hash="a" * 64,
    )

    class StrictFakeClient(FakeOpenVikingClient):
        def ls(self, uri, **kwargs):
            # Return a fake tree so _logical_source_index finds the existing resource.
            return [
                {"isDir": True, "uri": existing_uri.rsplit("/", 1)[0]},
            ]

    # Pre-populate resource files cache with the existing URI so the
    # logical-source index picks it up.
    client = StrictFakeClient()
    client._fake_files = [{"uri": existing_uri}]
    backend = PersonalOpenVikingBackend(client, root=tmp_path)
    backend._resource_files_cache = client._fake_files

    # Now try to ingest the same logical source with a DIFFERENT hash.
    with pytest.raises(SourceHashMismatch, match="was changed"):
        backend.index_file(
            source, source_type="homework", course="TEST",
            source_hash="b" * 64,
        )

    # Verify we never called add_resource.
    assert client.add_calls == []


def test_index_file_is_idempotent_when_same_logical_source_same_hash(tmp_path):
    """Same logical source + same hash → idempotent skip, no add_resource call."""
    from scripts.core.openviking_backend import resource_uri

    source = tmp_path / "homework.txt"
    source.write_text("stable content", encoding="utf-8")

    existing_uri = resource_uri(
        source, course="TEST", source_type="homework", root=tmp_path,
        source_hash="a" * 64,
    )

    class IdempotentFakeClient(FakeOpenVikingClient):
        def ls(self, uri, **kwargs):
            return [{"isDir": True, "uri": existing_uri.rsplit("/", 1)[0]}]

    client = IdempotentFakeClient()
    client._fake_files = [{"uri": existing_uri}]
    backend = PersonalOpenVikingBackend(client, root=tmp_path)
    backend._resource_files_cache = client._fake_files

    resource = backend.index_file(
        source, source_type="homework", course="TEST",
        source_hash="a" * 64,
    )

    assert client.add_calls == []
    assert resource.result == {"status": "skipped", "reason": "identical_source_hash"}
    assert resource.uri == existing_uri


def test_index_file_indexes_normally_when_no_existing_logical_source(tmp_path):
    """No existing logical source → normal ingestion proceeds."""
    source = tmp_path / "new_file.txt"
    source.write_text("brand new content", encoding="utf-8")

    class EmptyFakeClient(FakeOpenVikingClient):
        def ls(self, uri, **kwargs):
            return []

    client = EmptyFakeClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    resource = backend.index_file(
        source, source_type="homework", course="TEST",
        source_hash="c" * 64,
    )

    assert len(client.add_calls) == 1
    assert resource.uri.startswith(PERSONAL_NAMESPACE + "/")
    assert resource.result == {"uri": resource.uri, "status": "completed"}


def test_index_file_without_hash_skips_strict_check(tmp_path):
    """When no source_hash is provided, skip the strict rejection check."""
    source = tmp_path / "legacy_file.txt"
    source.write_text("legacy content", encoding="utf-8")

    client = FakeOpenVikingClient()
    backend = PersonalOpenVikingBackend(client, root=tmp_path)

    resource = backend.index_file(
        source, source_type="homework", course="TEST",
        # No source_hash — legacy path, no strict check.
    )

    assert len(client.add_calls) == 1
    assert resource.uri.startswith(PERSONAL_NAMESPACE + "/")
