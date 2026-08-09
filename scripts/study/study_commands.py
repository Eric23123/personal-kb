"""LLM-enhanced quiz, review, and explain commands with deterministic fallback."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def build_quiz(facts: list[dict[str, Any]], count: int = 5, seed: int = 0) -> dict[str, Any]:
    candidates = [
        f for f in facts
        if f.get("name") and (f.get("definition") or f.get("content") or f.get("problem"))
    ]
    random.Random(seed).shuffle(candidates)
    questions = []
    answer_key = []
    for index, fact in enumerate(candidates[:max(0, count)], 1):
        is_assessment = str(fact.get("type", "")).startswith("assessment") or fact.get("problem")
        if is_assessment:
            prompt = _practice_variant(fact)
            answer = fact.get("solution_pattern") or fact.get("definition") or fact.get("content", "")
        else:
            prompt = f"What is {fact['name']}?"
            answer = fact.get("definition") or fact.get("content", "")
        questions.append({"id": index, "prompt": prompt, "topic": fact.get("topic", ""), "type": fact.get("type", "concept")})
        answer_key.append({"id": index, "answer": answer, "source": fact.get("name", "")})
    return {"count": len(questions), "seed": seed, "questions": questions, "answer_key": answer_key}


def build_llm_quiz(facts: list[dict[str, Any]], llm: Any, *, count: int = 5) -> dict[str, Any]:
    """Generate new practice variants through an injected LLM callable."""
    source = [{"name": f.get("name", ""), "problem": f.get("problem", ""), "solution_pattern": f.get("solution_pattern", ""), "topic": f.get("topic", "")} for f in facts[:max(0, count)]]
    prompt = ("Create genuinely new practice problems from these solution methods. "
              "Do not copy wording, numbers, symbols, or narrative. Change parameters "
              "or initial conditions where safe. Return a JSON array with prompt, "
              "answer_outline, source_method.\n" + json.dumps(source, ensure_ascii=False))
    raw = llm(prompt)
    generated = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(generated, list):
        raise ValueError("LLM quiz response must be a JSON array")
    generated = generated[:max(0, count)]
    return {
        "count": len(generated),
        "questions": [{"id": i, "prompt": item["prompt"], "source_method": item.get("source_method", "")} for i, item in enumerate(generated, 1)],
        "answer_key": [{"id": i, "answer": item["answer_outline"]} for i, item in enumerate(generated, 1)],
        "generation": "llm_variant",
    }


def _practice_variant(fact: dict[str, Any]) -> str:
    problem = str(fact.get("problem", "")).strip()
    # Assessment quiz prompts deliberately use a generated practice framing,
    # not the original homework/exam wording.
    if problem:
        topic = str(fact.get("name", "the problem")).strip()
        return f"Practice variant: solve the underlying method tested by {topic}, without copying the original wording."
    return "Practice the underlying solution method without copying the source wording."


def build_review(facts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(facts),
        "by_topic": dict(sorted(Counter(str(f.get("topic", "uncategorized")) for f in facts).items())),
        "by_type": dict(sorted(Counter(str(f.get("type", "unknown")) for f in facts).items())),
        "by_assignment": dict(sorted(Counter(str(f.get("assignment", "uncategorized")) for f in facts).items())),
        "by_scope": dict(sorted(Counter(str(f.get("source_scope", "unknown")) for f in facts).items())),
        "facts": facts,
    }


def explain_fact(facts: list[dict[str, Any]], query: str) -> dict[str, Any]:
    needle = query.casefold().strip()
    matches = []
    for fact in facts:
        text = " ".join(str(fact.get(k, "")) for k in ("name", "definition", "content", "topic"))
        if needle and needle in text.casefold():
            matches.append((0 if needle in str(fact.get("name", "")).casefold() else 1, fact))
    matches.sort(key=lambda pair: pair[0])
    if not matches:
        return {"found": False, "query": query, "fact": None, "related": []}
    fact = matches[0][1]
    related = [f for f in facts if f is not fact and f.get("topic") == fact.get("topic")]
    return {"found": True, "query": query, "fact": fact, "related": related[:5]}


def load_facts(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("facts", "verified_items", "items"):
            if isinstance(data.get(key), list):
                return data[key]
    raise ValueError("input must be a JSON list or contain facts/verified_items/items")


def main() -> None:
    parser = argparse.ArgumentParser(prog="study_commands")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("quiz", "review", "explain"):
        p = sub.add_parser(name)
        p.add_argument("facts")
        if name == "quiz":
            p.add_argument("--count", type=int, default=5)
            p.add_argument("--seed", type=int, default=0)
        elif name == "explain":
            p.add_argument("query")
    args = parser.parse_args()
    facts = load_facts(args.facts)
    if args.command == "quiz":
        result = build_quiz(facts, args.count, args.seed)
    elif args.command == "review":
        result = build_review(facts)
    else:
        result = explain_fact(facts, args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

__all__ = ["build_quiz", "build_llm_quiz", "build_review", "explain_fact", "load_facts"]
