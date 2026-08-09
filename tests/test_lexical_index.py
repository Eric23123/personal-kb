"""Offline tests for the BM25 lexical index.

No live OpenViking service or model download required.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.retrieval import lexical_index as li
from scripts.core.openviking_backend import resource_uri
from scripts.retrieval.lexical_index import LexicalIndex, _tokenize, build_lexical_index


def test_tokenize_preserves_course_codes():
    tokens = _tokenize("KB 1001 kanban Jidoka")
    assert "kb1001" in tokens
    assert "kanban" in tokens
    assert "jidoka" in tokens


def test_tokenize_lowercases_and_filters_short_punctuation():
    tokens = _tokenize("DG/(1+DGH)")
    # We expect formula fragments to survive tokenization.
    assert "dg" in tokens
    assert "1" in tokens
    assert "dgh" in tokens


def test_tokenize_preserves_single_letter_formula_variables():
    """Single uppercase letters are common variable names in engineering
    formulas (e.g. N = Z - P in control theory) and must survive tokenization."""
    tokens = _tokenize("N = Z - P")
    assert "n" in tokens
    assert "z" in tokens
    assert "p" in tokens


def test_lexical_index_search_ranks_exact_terms(tmp_path):
    doc_a = tmp_path / "kanban.md"
    doc_a.write_text("# Kanban\nKanban is a pull system.", encoding="utf-8")
    doc_b = tmp_path / "house.md"
    doc_b.write_text("# House of Lean\nJidoka and JIT are pillars.", encoding="utf-8")

    index = build_lexical_index(tmp_path, extra_files=[doc_a, doc_b])
    results = index.search("kanban", top_k=5)

    assert len(results) == 2
    top = results[0]
    assert top["source_path"] == doc_a.name
    assert top["score"] > 0


def test_build_lexical_index_uses_openviking_uris(tmp_path):
    doc = tmp_path / "kanban.md"
    doc.write_text("# Kanban\nKanban calculation formula.", encoding="utf-8")

    index = build_lexical_index(tmp_path, extra_files=[doc])
    assert len(index.uris) == 1
    assert index.uris[0].startswith("viking://resources/personal-kb/")


def test_build_lexical_index_derives_filter_metadata_from_json(tmp_path):
    doc = tmp_path / "lecture_facts.json"
    doc.write_text(
        '[{"content": "Kanban pull system", "course": "TEST", '
        '"lecture": 2, "source_type": "diagram"}]',
        encoding="utf-8",
    )

    index = build_lexical_index(tmp_path, extra_files=[doc])

    assert index.metadata[0]["course"] == "TEST"
    assert index.metadata[0]["lecture"] == 2
    assert index.metadata[0]["source_type"] == "diagram"
    assert "/uncategorized/personal-source/" in index.uris[0]


def test_manifest_source_hash_is_preserved_in_lexical_uri(tmp_path):
    doc = tmp_path / "lecture.txt"
    doc.write_text("Kanban pull system", encoding="utf-8")
    source_hash = "a" * 64

    index = build_lexical_index(
        tmp_path,
        extra_files=[doc],
        extra_metadata={
            str(doc.resolve()): {
                "uri_source_path": str(doc),
                "course": "TEST",
                "source_type": "lecture",
                "source_hash": source_hash,
            }
        },
    )

    expected_root = resource_uri(
        doc,
        course="TEST",
        source_type="lecture",
        root=tmp_path,
        source_hash=source_hash,
    )
    assert index.uris == [f"{expected_root}/{doc.name}"]


def test_lexical_index_search_filters_by_metadata(tmp_path):
    first = tmp_path / "lecture_one.json"
    first.write_text(
        '[{"content": "kanban pull system", "course": "KB 1001", '
        '"lecture": 1, "source_type": "lecture"}]',
        encoding="utf-8",
    )
    second = tmp_path / "lecture_two.json"
    second.write_text(
        '[{"content": "kanban pull system", "course": "KB 1001", '
        '"lecture": 2, "source_type": "lecture"}]',
        encoding="utf-8",
    )

    index = build_lexical_index(tmp_path, extra_files=[first, second])
    results = index.search(
        "kanban",
        top_k=5,
        filters={"course": "KB 1001", "lecture": 1},
    )

    assert len(results) == 1
    assert results[0]["metadata"]["lecture"] == 1


def test_lexical_index_serialisation_roundtrip(tmp_path):
    doc = tmp_path / "test.md"
    doc.write_text("# Test\nThis is a test document for BM25.", encoding="utf-8")

    index = build_lexical_index(tmp_path, extra_files=[doc])
    path = tmp_path / "index.json"
    li.save_index(index, path)
    loaded = li.load_index(path)

    assert loaded.uris == index.uris
    assert loaded.corpus_tokens == index.corpus_tokens
    assert loaded.search("bm25") == index.search("bm25")


def test_lexical_index_search_empty_query_returns_empty():
    index = LexicalIndex(corpus_tokens=[["a", "b"]], uris=["u"], abstracts=["ab"], metadata=[{}])
    assert index.search("", top_k=5) == []


def test_lexical_index_rejects_non_positive_top_k():
    index = LexicalIndex(corpus_tokens=[["kanban"]], uris=["u"], abstracts=["kanban"], metadata=[{}])

    with pytest.raises(ValueError, match="top_k must be positive"):
        index.search("kanban", top_k=0)


def test_lexical_index_returns_zero_for_unmatched_query(tmp_path):
    doc = tmp_path / "lean.md"
    doc.write_text("# Lean\nManufacturing principles.", encoding="utf-8")
    index = build_lexical_index(tmp_path, extra_files=[doc])
    assert index.search("quantum physics", top_k=5) == []
