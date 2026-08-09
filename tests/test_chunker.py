import threading
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_chunk_transcript_bounds_concurrency_preserves_order_and_resumes(tmp_path):
    from scripts.notes.chunker import chunk_transcript

    transcript = tmp_path / "lecture.txt"
    transcript.write_text(
        "[0.0s -> 1.0s] Alpha segment\n"
        "[1.0s -> 2.0s] Beta segment\n"
        "[2.0s -> 3.0s] Gamma segment\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "chunker.checkpoint.json"
    active = 0
    peak_active = 0
    attempts = {"Alpha": 0, "Beta": 0, "Gamma": 0}
    lock = threading.Lock()

    def call_llm(prompt, **_kwargs):
        nonlocal active, peak_active
        name = next(name for name in attempts if f"{name} segment" in prompt)
        with lock:
            active += 1
            peak_active = max(peak_active, active)
            attempts[name] += 1
        try:
            if name == "Alpha" and attempts[name] == 1:
                raise RuntimeError("transient LLM error")
            time.sleep({"Alpha": 0.02, "Beta": 0.01, "Gamma": 0.0}[name])
            return f"---FACT---\nTOPIC: {name}\nTYPE: concept\nCONTENT: {name} detailed fact\n---END---"
        finally:
            with lock:
                active -= 1

    facts = chunk_transcript(
        str(transcript),
        "TEST",
        1,
        max_chars=25,
        max_workers=2,
        retries=1,
        retry_backoff_seconds=0,
        checkpoint_path=str(checkpoint),
        call_llm=call_llm,
    )

    assert peak_active <= 2
    assert attempts["Alpha"] == 2
    assert [fact["topic"] for fact in facts] == ["Alpha", "Beta", "Gamma"]
    assert checkpoint.exists()

    resumed = chunk_transcript(
        str(transcript),
        "TEST",
        1,
        max_chars=25,
        checkpoint_path=str(checkpoint),
        resume=True,
        call_llm=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not call LLM")),
    )
    assert resumed == facts
