"""Homework/exam processor with offline-first structured assessment extraction.

The processor accepts JSON containing an ``items`` array. Each item describes a
problem, solution pattern, common mistakes, and optional source reference. It
normalizes assessment metadata and can prepare Hindsight payloads without any
network call. LLM extraction is intentionally deferred until actual course
files are available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from ..ingestion.ingestion.ingest import format_fact_tags
except ImportError:  # pragma: no cover
    from scripts.ingestion.ingest import format_fact_tags

ASSESSMENT_TYPES = {"homework", "exam"}
REQUIRED_ITEM_FIELDS = ("problem", "solution_pattern")


def normalize_assessment_items(
    items: list[dict[str, Any]], *, course: str, source_type: str, assignment: str
) -> list[dict[str, Any]]:
    if source_type not in ASSESSMENT_TYPES:
        raise ValueError("source_type must be homework or exam")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError(f"items[{index}] must be an object")
        missing = [field for field in REQUIRED_ITEM_FIELDS if not str(raw.get(field, "")).strip()]
        if missing:
            raise ValueError(f"items[{index}] missing required fields: {', '.join(missing)}")
        item = dict(raw)
        item.setdefault("common_mistakes", [])
        item.update({
            "course": course,
            "source_type": source_type,
            "source_scope": "assessment",
            "assignment": assignment,
        })
        normalized.append(item)
    return normalized


def _fact_content(item: dict[str, Any]) -> str:
    mistakes = item.get("common_mistakes") or []
    mistake_text = "; ".join(str(x) for x in mistakes) or "none recorded"
    return (
        f"Problem: {item['problem']}\n"
        f"Solution pattern: {item['solution_pattern']}\n"
        f"Common mistakes: {mistake_text}"
    )[:2000]


def assessment_hindsight_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        fact = {
            **item,
            "type": "assessment-problem",
            "topic": item.get("assignment", "assessment"),
        }
        result.append({
            "content": _fact_content(item),
            "context": f"{item['course']} {item['assignment']} [{item['source_type']}]",
            "tags": format_fact_tags(fact),
            "metadata": {
                "course": item["course"],
                "assignment": item["assignment"],
                "source_type": item["source_type"],
                "source_scope": "assessment",
            },
        })
    return result


def process_assessment(
    source: str | Path, *, course: str, source_type: str, assignment: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    raw_items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(raw_items, list):
        raise ValueError("assessment source must contain an items array")
    items = normalize_assessment_items(raw_items, course=course, source_type=source_type, assignment=assignment)
    hindsight_items = assessment_hindsight_items(items)
    return {
        "source": str(source), "course": course, "assignment": assignment,
        "source_type": source_type, "source_scope": "assessment", "count": len(items),
        "dry_run": dry_run, "facts": items, "hindsight_items": hindsight_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize homework/exam assessment items")
    parser.add_argument("source")
    parser.add_argument("--course", required=True)
    parser.add_argument("--assignment", required=True)
    parser.add_argument("--source-type", choices=sorted(ASSESSMENT_TYPES), required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = process_assessment(args.source, course=args.course, source_type=args.source_type,
                                assignment=args.assignment, dry_run=args.dry_run or True)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
