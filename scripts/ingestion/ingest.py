"""Personal KB Hindsight ingestion with batched, retryable retain requests.

Usage:
    python -m scripts.ingestion.ingest transcript_facts.json --batch-size 20
    python -m scripts.ingestion.ingest transcript_facts.json --dry-run
    python -m scripts.ingestion.ingest --from-transcript transcript.txt --course "PERSONAL-ALPHA" --lecture 1
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from typing import Any, Iterable

try:  # Supports both `python scripts/ingest.py` and `import scripts.ingestion.ingest`.
    from ..core.common_client import JsonHttpClient
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from scripts.core.common_client import JsonHttpClient

HINDSIGHT_URL = "http://localhost:8888"
DEFAULT_BANK_ID = "hermes-history"
DEFAULT_BATCH_SIZE = 20
PRODUCTION_BANK_IDS = frozenset({"hermes-history"})


def _require_non_production(bank_id: str, *, allow_production: bool = False) -> None:
    """Guard against accidental writes to production Hindsight banks.

    Callers targeting ``DEFAULT_BANK_ID`` or another production bank MUST
    pass ``allow_production=True`` to confirm the intent.  Test and dry-run
    paths are unaffected.
    """
    if not allow_production and bank_id in PRODUCTION_BANK_IDS:
        raise ValueError(
            f"Bank {bank_id!r} is a production bank. "
            "Set allow_production=True only after explicit user authorization."
        )


def _memory_url(url: str, bank_id: str) -> str:
    return f"{url.rstrip('/')}/v1/default/banks/{bank_id}/memories"


def _make_client(timeout: float, retries: int) -> JsonHttpClient:
    return JsonHttpClient(timeout=timeout, retries=retries)


def hindsight_retain_items(
    items: list[dict[str, Any]],
    *,
    bank_id: str = DEFAULT_BANK_ID,
    url: str = HINDSIGHT_URL,
    timeout: float = 30,
    retries: int = 2,
    allow_production: bool = False,
    client: JsonHttpClient | Any | None = None,
) -> Any:
    """Retain one or more Hindsight memory items in the documented API shape."""
    _require_non_production(bank_id, allow_production=allow_production)
    if not items:
        raise ValueError("items cannot be empty")
    active_client = client or _make_client(timeout, retries)
    response = active_client.post_json(
        _memory_url(url, bank_id),
        {"items": items},
        timeout=timeout,
        retries=retries,
    )
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Hindsight rejected batch: {response['error']}")
    return response


def hindsight_retain(
    content: str,
    context: str = "",
    tags: list[str] | None = None,
    bank_id: str = DEFAULT_BANK_ID,
    url: str = HINDSIGHT_URL,
    *,
    allow_production: bool = False,
    timeout: float = 30,
    retries: int = 2,
    client: JsonHttpClient | Any | None = None,
) -> Any:
    """Compatibility wrapper retaining one fact through the batched endpoint."""
    _require_non_production(bank_id, allow_production=allow_production)
    item = {"content": content, "context": context}
    if tags:
        item["tags"] = tags
    try:
        return hindsight_retain_items(
            [item], bank_id=bank_id, url=url, timeout=timeout, retries=retries,
            allow_production=allow_production, client=client,
        )
    except Exception as error:  # Preserve the historical non-throwing wrapper contract.
        return {"error": str(error)}


def hindsight_recall(
    query: str,
    top_k: int = 10,
    tags: list[str] | None = None,
    bank_id: str = DEFAULT_BANK_ID,
    url: str = HINDSIGHT_URL,
    *,
    filters: Mapping[str, Any] | None = None,
    tags_match: str = "all_strict",
    timeout: float = 30,
    retries: int = 2,
    client: JsonHttpClient | Any | None = None,
) -> Any:
    """Search Hindsight memory and return an error mapping rather than raising."""
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if filters:
        filter_tags = hindsight_tags_for_filters(filters)
        if filter_tags:
            payload["tags"] = filter_tags
            payload["tags_match"] = tags_match
    if tags:
        payload["tags"] = list(dict.fromkeys([*(payload.get("tags", [])), *tags]))
        payload["tags_match"] = tags_match
    try:
        active_client = client or _make_client(timeout, retries)
        return active_client.post_json(
            f"{_memory_url(url, bank_id)}/recall", payload, timeout=timeout, retries=retries
        )
    except Exception as error:
        return {"error": str(error)}


def format_fact_context(fact: dict[str, Any]) -> str:
    """Build the context string for a fact's metadata."""
    parts = []
    if fact.get("course"):
        parts.append(f"{fact['course']}")
    if fact.get("course_name"):
        parts.append(f"({fact['course_name']})")
    if fact.get("lecture"):
        parts.append(f"Lecture #{fact['lecture']}")
    if fact.get("topic"):
        parts.append(f"— {fact['topic']}")
    if fact.get("source_type"):
        parts.append(f"[{fact['source_type']}]")
    if fact.get("timestamp_start") is not None and fact.get("timestamp_end") is not None:
        parts.append(f"@ {fact['timestamp_start']:.1f}s-{fact['timestamp_end']:.1f}s")
    return " ".join(parts)


def format_fact_tags(fact: dict[str, Any]) -> list[str]:
    """Build backward-compatible and namespaced metadata tags for a fact."""
    tags = ["personal-kb"]
    course = fact.get("course")
    source_type = fact.get("source_type")
    if course:
        course_tag = _tag_value(course)
        tags.extend([course_tag, f"course:{course_tag}"])
    if fact.get("topic"):
        topic_tag = _tag_value(fact["topic"])
        tags.extend([topic_tag, f"topic:{topic_tag}"])
    if fact.get("type"):
        fact_type = _tag_value(fact["type"])
        tags.extend([fact_type, f"fact-type:{fact_type}"])
    if source_type:
        source_tag = _tag_value(source_type)
        tags.extend([source_tag, f"source-type:{source_tag}"])
    if fact.get("engine"):
        engine_tag = _tag_value(fact["engine"])
        tags.extend([engine_tag, f"engine:{engine_tag}"])
    if fact.get("lecture") is not None:
        tags.append(f"lecture:{fact['lecture']}")
    if fact.get("semester"):
        tags.append(f"semester:{_tag_value(fact['semester'])}")
    if fact.get("date"):
        tags.append(f"date:{_tag_value(fact['date'])}")

    scope = fact.get("source_scope")
    if scope not in {"course", "assessment"}:
        scope = "assessment" if source_type in {"homework", "exam"} else "course"
    tags.append(f"scope:{scope}")
    return list(dict.fromkeys(tags))


def _tag_value(value: Any) -> str:
    """Normalize a metadata value into a stable Hindsight tag component."""
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().casefold()).strip("-")


def hindsight_tags_for_filters(filters: Mapping[str, Any]) -> list[str]:
    """Translate validated Personal query filters into Hindsight tag filters."""
    tags = ["personal-kb"]
    if filters.get("course"):
        tags.append(f"course:{_tag_value(filters['course'])}")
    if filters.get("lecture") is not None:
        tags.append(f"lecture:{filters['lecture']}")
    if filters.get("source_type"):
        tags.append(f"source-type:{_tag_value(filters['source_type'])}")
    source_scope = filters.get("source_scope")
    if source_scope in {"course", "assessment"}:
        tags.append(f"scope:{source_scope}")
    if filters.get("semester"):
        tags.append(f"semester:{_tag_value(filters['semester'])}")
    if filters.get("date"):
        tags.append(f"date:{_tag_value(filters['date'])}")
    return list(dict.fromkeys(tags))


def _batched(items: list[tuple[int, dict[str, Any]]], batch_size: int) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def ingest_facts(
    facts: list[dict[str, Any]],
    *,
    dry_run: bool = False,
    url: str = HINDSIGHT_URL,
    bank_id: str = DEFAULT_BANK_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    allow_production: bool = False,
    timeout: float = 30,
    retries: int = 2,
    client: JsonHttpClient | Any | None = None,
) -> dict[str, Any]:
    """Validate facts then send retained items in bounded batches.

    A failed HTTP batch is reported with every affected input index as
    *unconfirmed*.  This avoids claiming success when a network failure makes
    server-side persistence ambiguous, while continuing safely with later
    batches so callers can retry only the reported inputs.
    """
    _require_non_production(bank_id, allow_production=allow_production)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    results: dict[str, Any] = {
        "success": 0,
        "failed": 0,
        "errors": [],
        "failed_items": [],
        "batches_total": 0,
        "batches_succeeded": 0,
        "batches_failed": 0,
    }
    valid: list[tuple[int, dict[str, Any]]] = []
    for index, fact in enumerate(facts, start=1):
        content = fact.get("content", "")
        if not isinstance(content, str) or len(content) < 20:
            error = f"Fact {index}: content is too short ({len(content) if isinstance(content, str) else 0} chars)"
            print(f"  Skipping {error}")
            results["failed"] += 1
            results["errors"].append(error)
            results["failed_items"].append({"index": index, "error": error, "status": "invalid"})
            continue
        item = {"content": content, "context": format_fact_context(fact), "tags": format_fact_tags(fact)}
        valid.append((index, item))

    if dry_run:
        for index, item in valid:
            print(f"\n[DRY RUN] Fact {index}/{len(facts)}:")
            print(f"  Context: {item['context']}")
            print(f"  Tags: {item['tags']}")
            print(f"  Content: {item['content'][:200]}...")
        results["success"] = len(valid)
        results["batches_total"] = (len(valid) + batch_size - 1) // batch_size
        results["batches_succeeded"] = results["batches_total"]
        return results

    active_client = client or _make_client(timeout, retries)
    for batch_number, batch in enumerate(_batched(valid, batch_size), start=1):
        results["batches_total"] += 1
        indexes = [index for index, _item in batch]
        items = [item for _index, item in batch]
        try:
            response = hindsight_retain_items(
                items,
                bank_id=bank_id,
                url=url,
                timeout=timeout,
                retries=retries,
                allow_production=allow_production,
                client=active_client,
            )
            if isinstance(response, dict) and response.get("error"):
                raise RuntimeError(str(response["error"]))
        except Exception as error:
            message = f"Batch {batch_number} (facts {indexes}) was unconfirmed: {error}"
            print(f"  Error: {message}")
            results["batches_failed"] += 1
            results["failed"] += len(batch)
            results["errors"].append(message)
            results["failed_items"].extend(
                {"index": index, "error": str(error), "status": "unconfirmed"} for index in indexes
            )
            continue
        results["batches_succeeded"] += 1
        results["success"] += len(batch)
        print(f"  Ingested batch {batch_number}: {len(batch)} facts ({results['success']}/{len(valid)})")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest extracted facts into Hindsight")
    parser.add_argument("facts_json", nargs="?", help="Path to facts JSON file (from chunker.py)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview facts without making requests")
    parser.add_argument("--hindsight-url", default=HINDSIGHT_URL, help="Hindsight API base URL")
    parser.add_argument("--bank-id", default=DEFAULT_BANK_ID, help="Hindsight bank ID")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Facts per retain request (default: 20)")
    parser.add_argument("--timeout", type=float, default=30, help="Request timeout in seconds (default: 30)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed request (default: 2)")
    parser.add_argument("--from-transcript", help="Run chunker + ingest in one step")
    parser.add_argument("--course", help="Course code (for --from-transcript)")
    parser.add_argument("--lecture", type=int, help="Lecture number (for --from-transcript)")
    parser.add_argument("--topic", help="Topic (for --from-transcript)")
    args = parser.parse_args()

    if args.batch_size < 1 or args.retries < 0 or args.timeout <= 0:
        parser.error("--batch-size must be >= 1, --retries >= 0, and --timeout > 0")

    if args.from_transcript:
        if not args.course or args.lecture is None:
            parser.error("--course and --lecture are required with --from-transcript")
        try:
            from ..notes.chunker import chunk_transcript
        except ImportError:  # pragma: no cover - direct CLI import
            from scripts.notes.chunker import chunk_transcript
        facts = chunk_transcript(args.from_transcript, args.course, args.lecture, topic=args.topic)
        print(f"Chunked {len(facts)} facts from transcript")
    elif args.facts_json:
        with open(args.facts_json, "r", encoding="utf-8") as file:
            facts = json.load(file)
    else:
        parser.error("provide facts_json or --from-transcript")

    print(f"\nIngesting {len(facts)} facts into Hindsight...")
    results = ingest_facts(
        facts,
        dry_run=args.dry_run,
        url=args.hindsight_url,
        bank_id=args.bank_id,
        batch_size=args.batch_size,
        timeout=args.timeout,
        retries=args.retries,
    )
    print("\n--- Results ---")
    print(f"Success: {results['success']}")
    print(f"Failed/unconfirmed: {results['failed']}")
    print(f"Batches succeeded: {results['batches_succeeded']}/{results['batches_total']}")
    if results["errors"]:
        print("Errors (rerun only these reported items after checking Hindsight):")
        for error in results["errors"][:3]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()
