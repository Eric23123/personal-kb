"""Focused TDD coverage for note-generation performance optimizations.

These tests inject deterministic LLM/search callables.  They never invoke a real
LLM or web transport.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.notes.note_generator as ng


NOTE_WITH_FIVE_SECTIONS = """# Lecture 1: Feedback

## Key Concepts
- Feedback compares a measured output to a reference.
- The error drives the controller.

## Definitions
| Term | Definition |
| --- | --- |
| Feedback | Comparing output to a reference. |

## Formulas
- Closed-loop: $T = G/(1+GH)$.
- Unity feedback uses $H = 1$.

## Examples
### Unity feedback
- Set $H = 1$.

## Diagrams
### Feedback loop
- Summing junction feeds plant G and sensor H.
"""


def test_fact_check_parallelism_keeps_section_output_order_and_uses_compact_context():
    """Independent checks may run concurrently, but the report remains canonical."""
    prompts = []

    def fake_llm(prompt, **_kwargs):
        prompts.append(prompt)
        # Make completion order differ from submission order.
        if "Formulas Section" in prompt:
            time.sleep(0.03)
            return "✅ formula — correct"
        return "✅ item — correct"

    report, summary = ng.fact_check_batched(
        NOTE_WITH_FIVE_SECTIONS,
        transcript_text="TRANSCRIPT-CONTEXT " * 1000,
        facts_text="FACT-CONTEXT " * 1000,
        diagrams_text="DIAGRAM-CONTEXT " * 1000,
        max_workers=2,
        llm_call=fake_llm,
    )

    headers = [line for line in report.splitlines() if line.startswith("### ")]
    assert headers == [
        "### FORMULAS",
        "### DEFINITIONS",
        "### CONCEPTS",
        "### EXAMPLES",
        "### DIAGRAMS",
    ]
    assert summary == {"checked": 5, "verified": 5, "needs_check": 0, "issues": 0}
    formula_prompt = next(prompt for prompt in prompts if "Formulas Section" in prompt)
    diagram_prompt = next(prompt for prompt in prompts if "Diagrams Section" in prompt)
    assert "TRANSCRIPT-CONTEXT" not in formula_prompt
    assert "TRANSCRIPT-CONTEXT" not in diagram_prompt
    assert len(formula_prompt) < 9000
    assert len(diagram_prompt) < 7000


def test_web_verification_deduplicates_claims_caches_normalized_queries_and_bounds_workers():
    verification = """### FORMULAS
⚠️ Gain margin formula is uncertain — needs an authoritative source
⚠️ Gain margin formula is uncertain — needs an authoritative source
### DEFINITIONS
⚠️ Nyquist stability definition is uncertain — needs an authoritative source
"""
    active = 0
    peak_active = 0
    searches = []

    def fake_search(query, num_results):
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        searches.append((query, num_results))
        time.sleep(0.02)
        active -= 1
        return "authoritative source result"

    def fake_llm(prompt, **_kwargs):
        return "✅ claim — verified by injected source"

    cache = {}
    _report, results = ng.web_verify_items(
        verification,
        max_items=10,
        max_workers=2,
        cache=cache,
        web_search_fn=fake_search,
        llm_call=fake_llm,
    )

    assert len(results) == 2
    assert len(searches) == 2
    assert peak_active <= 2
    assert set(cache) == {ng.normalize_web_query(query) for query, _ in searches}


def test_generate_notes_can_omit_or_externalize_raw_sources_without_network_calls():
    def fake_synthesis(_prompt, **_kwargs):
        return "# Generated note\n\n## Key Concepts\n- Detail"

    common = dict(
        transcript_text="private transcript detail",
        facts=[{"topic": "feedback", "content": "fact detail"}],
        diagrams=[],
        run_fact_check=False,
        llm_call=fake_synthesis,
    )
    omitted = ng.generate_notes(source_mode="none", **common)
    externalized = ng.generate_notes(
        source_mode="external", source_reference="Lecture01.sources.md", **common
    )

    assert "## Source Verification" not in omitted
    assert "private transcript detail" not in omitted
    assert "[Lecture01.sources.md](Lecture01.sources.md)" in externalized
    assert "private transcript detail" not in externalized


def test_cli_exposes_bounded_worker_and_source_output_controls():
    import subprocess

    result = subprocess.run(
        [sys.executable, str(Path(ng.__file__)), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--fact-check-workers" in result.stdout
    assert "--web-workers" in result.stdout
    assert "--source-mode" in result.stdout
    assert "--source-output" in result.stdout


def test_fact_check_does_not_skip_a_populated_two_line_section():
    report, summary = ng.fact_check_batched(
        "## Formulas\n- $T = G/(1+GH)$.",
        transcript_text="",
        facts_text="",
        diagrams_text="",
        llm_call=lambda _prompt, **_kwargs: "✅ formula — correct",
    )

    assert report.startswith("### FORMULAS")
    assert summary["checked"] == 1
