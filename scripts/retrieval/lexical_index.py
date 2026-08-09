"""Build and query a local BM25 lexical index for Personal KB.

The index is built from approved source and derived files. It tokenizes text,
formulas, course codes, acronyms, and headings as searchable terms. The index
is persisted to disk as a JSON file so it can be reused across queries.

Usage:
    python scripts/retrieval/lexical_index.py build --root .
    python scripts/retrieval/lexical_index.py search "kanban" --top 5
    python scripts/retrieval/lexical_index.py search "DG/(1+DGH)" --index-path data/lexical_index.json --top 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from rank_bm25 import BM25Okapi, BM25Plus

# Re-use the corpus policy and deterministic URI builder from the OpenViking backend.
try:
    from ..core.openviking_backend import (
        CURATED_SOURCE_PATHS,
        metadata_matches,
        resource_uri,
        validate_metadata_filters,
    )
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from scripts.core.openviking_backend import (
        CURATED_SOURCE_PATHS,
        metadata_matches,
        resource_uri,
        validate_metadata_filters,
    )

DEFAULT_INDEX_PATH = Path("data/lexical_index.json")
DEFAULT_ROOT = Path.home() / "hermes_related" / "Personal_KB"

# Course codes and technical acronyms are kept as single tokens.
_COURSE_CODE_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Z]{2,5}\s?\d{4}[A-Z]?(?![A-Za-z0-9_])")


def _tokenize(text: str) -> list[str]:
    """Tokenize text, preserving formulas, course codes, and acronyms."""
    if not text:
        return []

    # Protect course codes by removing the space, so "KB 1001" is one token.
    text = _COURSE_CODE_RE.sub(lambda m: m.group(0).replace(" ", "").lower(), text)

    # Split into alphanumeric/underscore fragments; keep LaTeX commands too.
    tokens = re.findall(r"[A-Za-z0-9_]+|\\[a-zA-Z]+", text)
    # Keep tokens longer than 1 char, digits, and single uppercase letters
    # (which are common variable names in engineering formulas like N = Z - P).
    tokens = [t.lower() for t in tokens if len(t) > 1 or t.isdigit() or (len(t) == 1 and t.isalpha())]
    return tokens


def _read_document_text(path: Path) -> str:
    """Read text from a supported source/derived file."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, ensure_ascii=False)
        except (json.JSONDecodeError, OSError):
            return ""
    if suffix in (".md", ".markdown", ".txt", ".tex"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in (".yaml", ".yml"):
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _derive_json_metadata(path: Path) -> dict[str, Any]:
    """Extract uniform filter metadata from JSON fact/diagram records."""
    if path.suffix.lower() != ".json":
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    records = data if isinstance(data, list) else [data]
    records = [record for record in records if isinstance(record, dict)]
    if not records:
        return {}

    metadata: dict[str, Any] = {}
    for key in ("course", "course_name", "lecture", "source_type", "semester", "date"):
        values: list[Any] = []
        for record in records:
            value = record.get(key)
            if value is None or value in values:
                continue
            if isinstance(value, (str, int, float, bool)):
                values.append(value)
        # A file-level filter is safe only when every populated record agrees.
        if len(values) == 1 and all(record.get(key) == values[0] for record in records):
            metadata[key] = values[0]
    return metadata


def _build_field_text(
    path: Path,
    root: Path,
    text: str,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """Combine metadata fields into a single searchable text blob."""
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    parts = [relative, path.stem]

    # Extract headings from markdown.
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)
    parts.extend(headings)

    # Extract table cells (often contain acronyms and formula terms).
    for line in text.splitlines():
        if "|" in line:
            parts.append(line)

    # Include the full body text.
    parts.append(text)

    if extra_fields:
        for value in extra_fields.values():
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (int, float)):
                parts.append(str(value))

    return "\n".join(part for part in parts if part)


@dataclass
class LexicalIndex:
    """In-memory BM25 index plus metadata for serialisation."""

    corpus_tokens: list[list[str]]
    uris: list[str]
    abstracts: list[str]
    metadata: list[dict[str, Any]]
    bm25: BM25Okapi | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.bm25 is None and self.corpus_tokens:
            # BM25Plus avoids zero scores for rare terms in very small corpora
            # (where the standard BM25Okapi IDF term can collapse to zero).
            self.bm25 = BM25Plus(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top lexical matches with BM25 scores and safe metadata filters."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.bm25 or not self.corpus_tokens:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        normalized_filters = validate_metadata_filters(filters)
        scores = self.bm25.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            if not metadata_matches(self.metadata[idx], normalized_filters):
                continue
            results.append({
                "uri": self.uris[idx],
                "source_path": self.metadata[idx].get("source_path", ""),
                "score": round(float(score), 4),
                "abstract": self.abstracts[idx],
                "metadata": self.metadata[idx],
            })
            if len(results) >= top_k:
                break
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "uris": self.uris,
            "corpus_tokens": self.corpus_tokens,
            "abstracts": self.abstracts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LexicalIndex":
        return cls(
            corpus_tokens=data.get("corpus_tokens", []),
            uris=data.get("uris", []),
            abstracts=data.get("abstracts", []),
            metadata=data.get("metadata", []),
        )


def build_lexical_index(
    root: Path,
    *,
    extra_files: Iterable[Path] | None = None,
    extra_metadata: dict[str, dict[str, Any]] | None = None,
) -> LexicalIndex:
    """Build a BM25 index over all approved Personal KB files.

    The returned results use the same deterministic viking:// URIs as OpenViking
    so that lexical and dense candidates can be merged by URI.
    """
    root = Path(root).expanduser().resolve()
    approved_paths: list[Path] = []

    # Add the curated source files (even if not under a tree).
    for relative in CURATED_SOURCE_PATHS:
        path = root / relative
        if path.is_file():
            approved_paths.append(path)

    # Caller-supplied extra files.
    if extra_files:
        for path in extra_files:
            p = Path(path).expanduser().resolve()
            if p.is_file():
                approved_paths.append(p)

    # Deduplicate by absolute path.
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in approved_paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)

    corpus_tokens: list[list[str]] = []
    uris: list[str] = []
    abstracts: list[str] = []
    metadata: list[dict[str, Any]] = []

    for path in unique_paths:
        text = _read_document_text(path)
        if not text:
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        provided_fields = (extra_metadata or {}).get(str(path), {})
        extra_fields = {
            **_derive_json_metadata(path),
            **provided_fields,
        }
        extra_fields.setdefault("source_type", "personal-source")
        field_text = _build_field_text(path, root, text, extra_fields)
        tokens = _tokenize(field_text)
        if not tokens:
            continue
        # Use the same deterministic URI OpenViking uses for this source.
        # The ingest script uses source_type="personal-source", and OpenViking
        # appends the filename to the directory URI returned by resource_uri().
        # URI identity must match the ingestion metadata supplied by the caller.
        # Inferred JSON metadata is semantic filter data, not a new OpenViking URI.
        uri_source_type = provided_fields.get("source_type", "personal-source")
        uri_source_path = provided_fields.get("uri_source_path", path)
        uri_course = provided_fields.get("course")
        uri = resource_uri(
            uri_source_path,
            course=uri_course,
            source_type=uri_source_type,
            root=root,
            source_hash=provided_fields.get("source_hash"),
        )
        uri = f"{uri}/{Path(uri_source_path).name}"
        corpus_tokens.append(tokens)
        uris.append(uri)
        # Abstract is the first non-empty lines plus a few headings.
        abstract_lines: list[str] = []
        for line in text.splitlines()[:80]:
            stripped = line.strip()
            if stripped:
                abstract_lines.append(stripped)
            if len(" ".join(abstract_lines)) > 300:
                break
        abstracts.append(" ".join(abstract_lines)[:500])
        metadata.append({
            "source_path": relative,
            "file_name": path.name,
            "extension": path.suffix.lower(),
            **extra_fields,
        })

    return LexicalIndex(corpus_tokens=corpus_tokens, uris=uris, abstracts=abstracts, metadata=metadata)


def save_index(index: LexicalIndex, path: Path) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(path: Path) -> LexicalIndex:
    path = Path(path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    return LexicalIndex.from_dict(data)


def cmd_build(root: Path, index_path: Path) -> dict[str, Any]:
    """Build the lexical index and save it to disk."""
    index = build_lexical_index(root)
    save_index(index, index_path)
    return {
        "root": str(root),
        "index_path": str(index_path),
        "documents": len(index.uris),
        "tokens": sum(len(t) for t in index.corpus_tokens),
    }


def cmd_search(index_path: Path, query: str, top_k: int) -> dict[str, Any]:
    """Search a previously built lexical index."""
    index = load_index(index_path)
    results = index.search(query, top_k=top_k)
    return {
        "query": query,
        "index_path": str(index_path),
        "documents": len(index.uris),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="lexical_index",
        description="Build and query a BM25 lexical index for Personal KB",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build the lexical index from approved files")
    build_parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Personal KB root directory")
    build_parser.add_argument(
        "--index-path", default=str(DEFAULT_INDEX_PATH),
        help="Output JSON path for the BM25 index",
    )

    search_parser = subparsers.add_parser("search", help="Search the lexical index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--index-path", default=str(DEFAULT_INDEX_PATH),
        help="Path to the BM25 index JSON",
    )
    search_parser.add_argument("--top", type=int, default=10, help="Max results")

    args = parser.parse_args()

    if args.command == "build":
        result = cmd_build(Path(args.root), Path(args.index_path))
    elif args.command == "search":
        result = cmd_search(Path(args.index_path), args.query, args.top)
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
