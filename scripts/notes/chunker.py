"""Extract detailed study facts from a lecture transcript with a bounded LLM pool.

By default chunks are processed sequentially and no checkpoint is written.  Opt
into parallelism with ``--workers`` and resumability with ``--checkpoint``;
``--resume`` never runs unless an explicit checkpoint path is supplied.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

try:  # Supports both direct execution and `import scripts.notes.chunker`.
    from ..core.common_client import DeepSeekClient, JsonHttpClient, load_api_key
except ImportError:  # pragma: no cover - direct CLI use
    from scripts.core.common_client import DeepSeekClient, JsonHttpClient, load_api_key

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"

CHUNKING_PROMPT = """You are extracting DETAILED study facts from a lecture transcript for {course_name} ({course_code}).
Lecture #{lecture}: {topic}

## Instructions

Extract key study facts from this transcript segment. Each fact must be DETAILED and COMPLETE — not a summary.

### For DEFINITIONS:
- Include the FULL definition as stated by the professor
- Include all qualifying conditions and context
- Include any examples or analogies used to explain the term
- BAD: "Closed-loop control uses feedback"
- GOOD: "Closed-loop control (also called feedback or negative feedback control) dynamically adjusts inputs based on output measurements to maintain desired states. It involves three key components: a sensor measuring the actual output, a reference signal defining the target state (e.g., 'desired cleanliness'), and a controller that converts the error between measured and reference values into corrective input adjustments."

### For FORMULAS:
- Include the COMPLETE formula in LaTeX notation
- Define what EACH variable means
- Include how the formula is derived or used
- Include any simplifications or algebraic steps shown
- BAD: "E = V - H*Y"
- GOOD: "Error equation: E = V - H·Y, where V is the reference signal, H is the sensor gain, Y is the system output. The error E is the difference between what we want (V) and what we measure (H·Y). Substituting into Y = D·G·E gives Y = D·G·(V - H·Y), which simplifies to Y = DG/(1+DGH)·V."

### For EXAMPLES:
- Include the COMPLETE example with all details
- Include what the system is, how it works, what happens in each scenario
- Include the professor's exact reasoning
- BAD: "Car cruise control demonstrates feedback"
- GOOD: "Car cruise control: The speedometer acts as the sensor measuring actual speed. The reference speed is set by the driver (e.g., 100 mph). On flat ground at steady state, measured speed equals reference speed, error is zero, and throttle stays constant. When going uphill, gravity reduces speed below 100 mph, creating positive error (reference > measured), which triggers the controller to increase throttle. When going downhill, speed exceeds reference, creating negative error, which reduces throttle."

### For CONCEPTS:
- Include a thorough explanation with the "why" and "how"
- Include any analogies or intuitive explanations
- Include the professor's key points and emphasis

Each fact should be:
- Self-contained (understandable without surrounding context)
- 100-400 words (detailed, not summarized)
- Include technical terms for searchability
- Preserve the professor's exact examples and explanations

Output EXACTLY in this format (one block per fact):
---FACT---
TOPIC: <topic tag>
TYPE: <definition|formula|example|concept>
CONTENT: <the DETAILED fact, 100-400 words>
---END---

If the segment contains no meaningful facts (filler, transitions, etc.), output:
---SKIP---

Transcript segment:
{segment}
"""


def split_transcript(text: str, max_chars: int = 3000) -> list[str]:
    """Split a transcript by timestamp lines while targeting a segment size."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1")
    segments, current_segment, current_len = [], [], 0
    for raw_line in text.strip().split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if current_len + len(line) > max_chars and current_segment:
            segments.append("\n".join(current_segment))
            current_segment, current_len = [], 0
        current_segment.append(line)
        current_len += len(line)
    if current_segment:
        segments.append("\n".join(current_segment))
    return segments


def extract_timestamp_range(segment_text: str) -> tuple[float | None, float | None]:
    """Extract the first start and final end timestamp from a segment."""
    timestamps = re.findall(r"\[(\d+\.?\d*)s\s*->\s*(\d+\.?\d*)s\]", segment_text)
    if timestamps:
        return float(timestamps[0][0]), float(timestamps[-1][1])
    return None, None


def parse_facts(llm_response: str) -> list[dict[str, str]]:
    """Parse the expected fact blocks from an LLM response."""
    facts: list[dict[str, str]] = []
    for block in llm_response.split("---FACT---"):
        block = block.strip()
        if not block or "---SKIP---" in block:
            continue
        block = re.sub(r"---END---.*", "", block, flags=re.DOTALL).strip()
        topic_match = re.search(r"TOPIC:\s*(.+?)(?:\n|$)", block)
        type_match = re.search(r"TYPE:\s*(.+?)(?:\n|$)", block)
        content_match = re.search(r"CONTENT:\s*(.+?)(?=TOPIC:|TYPE:|$)", block, re.DOTALL)
        if content_match:
            facts.append(
                {
                    "topic": topic_match.group(1).strip() if topic_match else "general",
                    "type": type_match.group(1).strip() if type_match else "text",
                    "content": content_match.group(1).strip(),
                }
            )
    return facts


def _checkpoint_identity(
    transcript: str, course_code: str, lecture: int, topic: str, model: str, max_chars: int
) -> dict[str, Any]:
    return {
        "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "course": course_code,
        "lecture": lecture,
        "topic": topic,
        "model": model,
        "max_chars": max_chars,
    }


def _load_checkpoint(path: Path, identity: dict[str, Any], resume: bool) -> dict[int, list[dict[str, Any]]]:
    if not resume:
        return {}
    if not path.exists():
        print(f"No checkpoint found at {path}; starting from the first segment.")
        return {}
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read checkpoint {path}: {error}") from error
    if checkpoint.get("identity") != identity:
        raise ValueError("Checkpoint does not match this transcript or chunking options; refuse to resume it.")
    completed = checkpoint.get("completed", {})
    if not isinstance(completed, dict):
        raise ValueError(f"Checkpoint {path} has invalid completed data")
    return {int(index): facts for index, facts in completed.items() if isinstance(facts, list)}


def _save_checkpoint(path: Path, identity: dict[str, Any], completed: dict[int, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "identity": identity, "completed": {str(key): value for key, value in completed.items()}}
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _enrich_facts(
    facts: list[dict[str, str]],
    *,
    segment: str,
    transcript_path: str,
    course_code: str,
    course_name: str,
    lecture: int,
) -> list[dict[str, Any]]:
    start_time, end_time = extract_timestamp_range(segment)
    for fact in facts:
        fact.update(
            {
                "course": course_code,
                "course_name": course_name,
                "lecture": lecture,
                "source_type": "lecture",
                "timestamp_start": start_time,
                "timestamp_end": end_time,
                "source_file": os.path.basename(transcript_path),
                "engine": "whisper" if "whisper" in transcript_path.lower() else "moss",
            }
        )
    return facts


def _process_segment(
    index: int,
    segment: str,
    *,
    prompt: str,
    call_llm: Callable[..., str],
    model: str,
    api_url: str | None,
    api_key: str | None,
    retries: int,
    retry_backoff_seconds: float,
    sleep: Callable[[float], None],
    transcript_path: str,
    course_code: str,
    course_name: str,
    lecture: int,
) -> tuple[int, list[dict[str, Any]], str | None]:
    for attempt in range(retries + 1):
        try:
            response = call_llm(prompt, model=model, api_url=api_url, api_key=api_key)
            facts = _enrich_facts(
                parse_facts(response),
                segment=segment,
                transcript_path=transcript_path,
                course_code=course_code,
                course_name=course_name,
                lecture=lecture,
            )
            return index, facts, None
        except Exception as error:
            if attempt == retries:
                return index, [], str(error)
            sleep(retry_backoff_seconds * (2**attempt))
    raise AssertionError("unreachable")


def chunk_transcript(
    transcript_path: str,
    course_code: str,
    lecture: int,
    topic: str | None = None,
    model: str = DEFAULT_MODEL,
    course_catalog: dict[str, Any] | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    *,
    max_chars: int = 3000,
    max_workers: int = 1,
    retries: int = 2,
    retry_backoff_seconds: float = 1.0,
    timeout: float = 600,
    checkpoint_path: str | None = None,
    resume: bool = False,
    call_llm: Callable[..., str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Chunk a transcript with ordered output, bounded concurrency, and opt-in resume."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if retries < 0 or retry_backoff_seconds < 0 or timeout <= 0:
        raise ValueError("retries/backoff/timeout must be non-negative and timeout positive")
    if resume and not checkpoint_path:
        raise ValueError("resume requires an explicit checkpoint_path")

    transcript = Path(transcript_path).read_text(encoding="utf-8")
    topic = topic or "lecture content"
    course_name = course_code
    if course_catalog:
        for code, info in course_catalog.items():
            if code.upper() == course_code.upper():
                course_name = info.get("name", course_code)
                break
    segments = split_transcript(transcript, max_chars=max_chars)
    print(f"Split transcript into {len(segments)} segments (workers={max_workers})")

    identity = _checkpoint_identity(transcript, course_code, lecture, topic, model, max_chars)
    checkpoint = Path(checkpoint_path) if checkpoint_path else None
    completed = _load_checkpoint(checkpoint, identity, resume) if checkpoint else {}
    if completed:
        print(f"Resuming {len(completed)} completed segment(s) from {checkpoint}.")

    if call_llm is None:
        # Segment-level retries are authoritative here. Disable lower-level
        # retries to avoid multiplying attempts while still sharing one opener.
        shared_client = DeepSeekClient(
            api_key=api_key,
            api_url=api_url or DEEPSEEK_URL,
            timeout=timeout,
            http_client=JsonHttpClient(timeout=timeout, retries=0),
        )

        def call_llm(prompt: str, **kwargs: Any) -> str:
            return shared_client.complete(prompt, model=kwargs["model"])

    work: list[tuple[int, str, str]] = []
    for index, segment in enumerate(segments):
        if index in completed:
            continue
        prompt = CHUNKING_PROMPT.format(
            course_name=course_name, course_code=course_code, lecture=lecture, topic=topic, segment=segment
        )
        work.append((index, segment, prompt))

    def submit(executor: concurrent.futures.Executor, index: int, segment: str, prompt: str):
        return executor.submit(
            _process_segment,
            index,
            segment,
            prompt=prompt,
            call_llm=call_llm,
            model=model,
            api_url=api_url,
            api_key=api_key,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            sleep=sleep,
            transcript_path=transcript_path,
            course_code=course_code,
            course_name=course_name,
            lecture=lecture,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [submit(executor, *entry) for entry in work]
        for future in concurrent.futures.as_completed(futures):
            index, facts, error = future.result()
            if error:
                print(f"  Segment {index + 1}/{len(segments)} failed after {retries + 1} attempt(s): {error}")
                continue
            completed[index] = facts
            print(f"  Segment {index + 1}/{len(segments)}: extracted {len(facts)} facts")
            if checkpoint:
                _save_checkpoint(checkpoint, identity, completed)

    return [fact for index in range(len(segments)) for fact in completed.get(index, [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk a transcript into detailed study facts")
    parser.add_argument("transcript", help="Path to transcript file")
    parser.add_argument("--course", required=True, help="Course code (e.g., PERSONAL-ALPHA)")
    parser.add_argument("--lecture", type=int, required=True, help="Lecture number")
    parser.add_argument("--topic", default=None, help="Topic name")
    parser.add_argument("--model", choices=(DEFAULT_MODEL,), default=DEFAULT_MODEL, help="DeepSeek model (fixed to deepseek-v4-pro)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--api-url", default=DEEPSEEK_URL, help="OpenAI-compatible API URL")
    parser.add_argument("--api-key", help="API key (otherwise DEEPSEEK_API_KEY is loaded safely)")
    parser.add_argument("--workers", type=int, default=1, help="Bounded concurrent LLM requests (default: 1)")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed segment (default: 2)")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="Initial retry delay seconds (default: 1)")
    parser.add_argument("--timeout", type=float, default=600, help="LLM request timeout seconds (default: 600)")
    parser.add_argument("--max-chars", type=int, default=3000, help="Target characters per segment (default: 3000)")
    parser.add_argument("--checkpoint", help="Write completed segments to this checkpoint file")
    parser.add_argument("--resume", action="store_true", help="Resume only an explicit matching --checkpoint")
    args = parser.parse_args()
    if args.workers < 1 or args.retries < 0 or args.retry_backoff < 0 or args.timeout <= 0 or args.max_chars < 1:
        parser.error("workers/max-chars must be >= 1; retries/backoff >= 0; timeout > 0")
    if args.resume and not args.checkpoint:
        parser.error("--resume requires --checkpoint PATH")

    catalog = None
    catalog_path = Path(__file__).resolve().parent.parent / "config" / "course_catalog.yaml"
    if catalog_path.exists():
        try:
            import yaml
            catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8")).get("courses", {})
        except ImportError:
            print("Note: install pyyaml for course catalog support")

    facts = chunk_transcript(
        args.transcript,
        args.course,
        args.lecture,
        topic=args.topic,
        model=args.model,
        course_catalog=catalog,
        api_url=args.api_url,
        api_key=args.api_key or load_api_key(),
        max_chars=args.max_chars,
        max_workers=args.workers,
        retries=args.retries,
        retry_backoff_seconds=args.retry_backoff,
        timeout=args.timeout,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    print(f"\nTotal facts extracted: {len(facts)}")
    output_path = args.output or str(Path(args.transcript).with_suffix("")) + "_facts.json"
    Path(output_path).write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
