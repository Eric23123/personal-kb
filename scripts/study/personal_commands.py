"""Thin Hermes-facing command helpers for the Personal-KB edge layer.

These helpers are intentionally free of provider/model creation. They route
local TEST data to the deterministic offline functions and keep live retrieval
as a future injected dependency.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from scripts.study.assessment_processor import process_assessment
from scripts.study.study_commands import build_llm_quiz, build_quiz, build_review, explain_fact, load_facts
from scripts.core.active_model import call_active_model


def quiz(
    facts_path: str | Path,
    *,
    count: int = 5,
    seed: int = 0,
    active_model: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Generate quiz content using the active Hermes model when supplied.

    The caller injects the current-session model callable. This function never
    creates a provider client. If generation fails, it returns a visible
    deterministic fallback with an explicit warning.
    """
    facts = load_facts(facts_path)
    if active_model is None:
        result = build_quiz(facts, count=count, seed=seed)
        result.update({"generation": "deterministic_fallback", "warning": "No active model callable was supplied."})
        return result
    try:
        return build_llm_quiz(facts, active_model, count=count)
    except Exception as exc:
        result = build_quiz(facts, count=count, seed=seed)
        result.update({"generation": "deterministic_fallback", "warning": f"LLM quiz generation failed: {exc}"})
        return result


def quiz_with_llm(facts_path: str | Path, llm: Callable[[str], Any], *, count: int = 5) -> dict[str, Any]:
    return build_llm_quiz(load_facts(facts_path), llm, count=count)


def quiz_with_active_agent(agent: Any, facts_path: str | Path, *, count: int = 5, seed: int = 0) -> dict[str, Any]:
    """Generate with the current Hermes agent client, with visible fallback."""
    return quiz(
        facts_path,
        count=count,
        seed=seed,
        active_model=lambda prompt: call_active_model(agent, prompt),
    )


def review(facts_path: str | Path) -> dict[str, Any]:
    return build_review(load_facts(facts_path))


def explain(facts_path: str | Path, query: str) -> dict[str, Any]:
    return explain_fact(load_facts(facts_path), query)


def ingest(path: str | Path, *, course: str, assignment: str, source_type: str) -> dict[str, Any]:
    """Validate and prepare assessment input; never performs a live write."""
    return process_assessment(path, course=course, assignment=assignment, source_type=source_type, dry_run=True)


def connect(health_probe: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Report injected service health without changing configuration."""
    if health_probe is None:
        return {"status": "unconfigured", "mutated": False}
    return {"status": "ok", "mutated": False, "health": health_probe()}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="personal_commands")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("quiz"); p.add_argument("facts"); p.add_argument("--count", type=int, default=5); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("review"); p.add_argument("facts")
    p = sub.add_parser("explain"); p.add_argument("facts"); p.add_argument("query")
    p = sub.add_parser("ingest"); p.add_argument("source"); p.add_argument("--course", required=True); p.add_argument("--assignment", required=True); p.add_argument("--source-type", choices=("homework", "exam"), required=True)
    sub.add_parser("connect")
    args = parser.parse_args()
    if args.command == "quiz": result = quiz(args.facts, count=args.count, seed=args.seed)
    elif args.command == "review": result = review(args.facts)
    elif args.command == "explain": result = explain(args.facts, args.query)
    elif args.command == "ingest": result = ingest(args.source, course=args.course, assignment=args.assignment, source_type=args.source_type)
    else: result = connect()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
