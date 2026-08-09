import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

"""
Personal KB — Lecture Note Generator (v2)

Takes pre-processed inputs (transcript, chunked facts, diagram descriptions)
and generates structured, study-ready Obsidian lecture notes via LLM synthesis.

Includes:
  - Source verification: raw source text in collapsible sections for transparency
  - Fact-check pass: second LLM call that verifies formulas, definitions, examples
    against raw sources and flags potential OCR issues

Usage:
    python note_generator.py --facts data/Test_facts.json --course "PERSONAL-ALPHA" --lecture 1 --topic "Intro to Control Systems"
    python note_generator.py --transcript data/Test_whisper.txt --facts data/Test_facts.json --course "PERSONAL-ALPHA" --lecture 1
    python note_generator.py --facts data/Test_facts.json --course "PERSONAL-ALPHA" --lecture 1 --no-fact-check
"""

import argparse
import json
import os
import sys
import datetime
import re
from concurrent.futures import ThreadPoolExecutor

try:  # Supports both direct script execution and package imports.
    from ..core.common_client import DeepSeekClient, JsonHttpClient, load_api_key as _load_api_key
except ImportError:  # pragma: no cover - direct CLI use
    from scripts.core.common_client import DeepSeekClient, JsonHttpClient, load_api_key as _load_api_key

_HTTP_CLIENTS = {}


def _shared_http_client(timeout):
    """Reuse one configured JSON client per timeout profile."""
    key = float(timeout)
    if key not in _HTTP_CLIENTS:
        _HTTP_CLIENTS[key] = JsonHttpClient(timeout=key, retries=2)
    return _HTTP_CLIENTS[key]

# ── LLM Backend Configuration ──────────────────────────────────────────────
#
# Personal-KB note generation is hardened to a single text LLM route: DeepSeek
# V4 Pro through the official DeepSeek API. Alternate cloud (OpenAI) and local
# (Ollama text-generation) backends have been removed from the generation path
# to keep the Personal script model consistent with the Hermes delegation route
# (see references/deepseek-v4-pro-routing.md). Local OCR/vision/STT engines
# (Qwen3-VL, GLM-OCR, Whisper, MOSS) remain separate media components and are
# NOT affected by this routing — they live in diagrams.py and transcribe.py.

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BACKEND = "deepseek"  # retained for backward-compatible call signatures

# Single allowed backend. ``call_llm`` rejects any other value so a stale
# caller cannot silently fall back to a removed cloud/local route.
ALLOWED_BACKENDS = {"deepseek"}

HINDSIGHT_URL = "http://localhost:8888"
PROJECT_ROOT = _Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "courses")


# ── API Key Loading ────────────────────────────────────────────────────────

def load_api_key(key_name="DEEPSEEK_API_KEY"):
    """Load an API key through the shared, non-sourcing .env parser."""
    return _load_api_key(key_name)


# ── LLM Call ───────────────────────────────────────────────────────────────

def call_llm(prompt, backend=DEFAULT_BACKEND, model=None, api_key=None, api_url=None,
             ollama_url=None, http_client=None, deepseek_client=None):
    """Call the DeepSeek V4 Pro chat completion API and return the response text.

    Personal-KB note generation uses a single hardened text LLM route: DeepSeek
    V4 Pro via the official DeepSeek API (endpoint ``DEEPSEEK_API_URL``, model
    ``DEFAULT_MODEL``, credential ``DEEPSEEK_API_KEY`` from the shared loader).
    The ``ollama_url`` parameter is accepted for backward-compatible call sites
    but is intentionally unused — local Ollama text generation is no longer an
    allowed Personal generation backend. Local OCR/vision engines in
    ``diagrams.py`` and ``transcribe.py`` are unaffected.

    ``deepseek_client`` and ``http_client`` are injection points for tests so
    they never need a real API key or network call. Callers may still override
    ``model``/``api_key``/``api_url`` for custom DeepSeek-compatible endpoints,
    but the defaults are always the canonical DeepSeek route.
    """
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(
            f" Personal-KB note generation only permits the 'deepseek' backend "
            f"(got {backend!r}). OpenAI and Ollama text-generation backends have "
            f"been removed; use DeepSeek V4 Pro via the DeepSeek API."
        )

    model = model or DEFAULT_MODEL
    if model != DEFAULT_MODEL:
        raise ValueError(f"Personal-KB requires {DEFAULT_MODEL}; got {model!r}")
    url = api_url or DEEPSEEK_API_URL
    resolved_key = api_key or load_api_key("DEEPSEEK_API_KEY")
    if not resolved_key:
        print("Error: DEEPSEEK_API_KEY is not set")
        print("Set it via the environment or the configured Personal-KB env file")
        sys.exit(1)

    client = deepseek_client or DeepSeekClient(
        api_key=resolved_key,
        api_url=url,
        http_client=http_client or _shared_http_client(600),
        timeout=600,
    )
    try:
        return client.complete(prompt, model=model, max_tokens=8192, temperature=0.3)
    except Exception as error:
        print(f"DeepSeek API error: {error}")
        sys.exit(1)


# ── Input Loading ──────────────────────────────────────────────────────────

def load_transcript(path):
    """Load a timestamped transcript file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_facts(path):
    """Load chunked facts JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_diagrams(path):
    """Load diagram descriptions JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_course_catalog():
    """Load course catalog YAML if available."""
    catalog_path = os.path.join(os.path.dirname(__file__), "..", "config", "course_catalog.yaml")
    if os.path.exists(catalog_path):
        try:
            import yaml
            with open(catalog_path, "r") as f:
                return yaml.safe_load(f).get("courses", {})
        except ImportError:
            pass
    return {}


# ── Prompt Construction ────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """You are generating structured lecture notes for a Personal MEng Systems Engineering student.

## Course Info
- Course: {course_code} — {course_name}
- Lecture: #{lecture}
- Topic: {topic}

## Source Material

### Transcript (from {engine}):
{transcript_text}

### Chunked Facts (extracted earlier):
{facts_text}

### Diagram Descriptions (OCR'd from slides):
{diagrams_text}

## Instructions

Generate comprehensive lecture notes in this EXACT format (Obsidian-compatible Markdown):

```markdown
---
date: {date}
course: {course_code}
lecture: {lecture}
topic: {topic}
source_type: synthesis
extraction_engine: deepseek-v4-pro
generated: {date}
tags: [lecture-notes, {tags}]
sources: [{sources}]
---

# Lecture {lecture}: {topic}
**Course:** {course_code} — {course_name}

## Key Concepts
- **Concept name**: detailed explanation (3-7 items)

## Definitions
| Term | Definition |
|------|-----------|
| Term1 | Full detailed definition as stated in the source |

## Formulas
- Formula name: $formula$ — complete derivation with variable definitions (all equations in LaTeX)

## Examples (from lecture)
### Example Title
- Complete description of the professor's example with all details

## Diagrams
### Diagram Title
> Detailed description of visual content from slides

## Professor Notes
- Interesting quotes, hints, or questions from the professor

## Action Items
- [ ] Study task 1
- [ ] Study task 2

## Related Topics
- [[Related Lecture or Concept]]
```

Rules:
- Be DETAILED, not summarized — include full definitions, complete formulas with all variable meanings, and thorough examples
- Use Obsidian markdown syntax ([[wikilinks]], tables, code blocks)
- Preserve professor examples exactly as stated with ALL details
- Include LaTeX for all formulas: $formula$ for inline, $$formula$$ for display
- Don't invent content — only use what's in the source material
- Tag action items with checkboxes: - [ ]
- Keep Key Concepts to 3-7 items but make each one detailed
- Keep Definitions table — include the COMPLETE definition, not a summary
- Include ALL formulas with variable definitions
- For Diagrams, describe what the visual shows in detail
- Generate appropriate tags based on the content topics
- Generate wikilinks to related concepts
"""

def format_facts_for_prompt(facts):
    """Format chunked facts into a readable string for the prompt."""
    if not facts:
        return "(no facts provided)"
    lines = []
    for i, fact in enumerate(facts, 1):
        topic = fact.get("topic", "general")
        ftype = fact.get("type", "text")
        content = fact.get("content", "")
        ts = ""
        if fact.get("timestamp_start") and fact.get("timestamp_end"):
            ts = f" [{fact['timestamp_start']:.0f}s-{fact['timestamp_end']:.0f}s]"
        source = fact.get("source_file", "")
        lines.append(f"{i}. [{ftype.upper()}] {topic}{ts} ({source}):\n{content}")
    return "\n\n".join(lines)


def format_diagrams_for_prompt(diagrams):
    """Format diagram descriptions into a readable string for the prompt."""
    if not diagrams:
        return "(no diagrams provided)"
    lines = []
    for i, diag in enumerate(diagrams, 1):
        dtype = diag.get("diagram_type", "general")
        page = diag.get("page", "?")
        content = diag.get("content", "")
        source = diag.get("source_file", "")
        engine = diag.get("engine", "unknown")
        lines.append(f"{i}. [{dtype}] Page {page} ({source}, engine: {engine}):\n{content}")
    return "\n\n".join(lines)


def detect_engine(facts, diagrams):
    """Detect the transcription/OCR engine used."""
    engines = set()
    for f in facts:
        if f.get("engine"):
            engines.add(f["engine"])
    for d in diagrams:
        if d.get("engine"):
            engines.add(d["engine"])
    return ", ".join(sorted(engines)) if engines else "mixed"


def extract_topics(facts, diagrams):
    """Extract topic tags from facts and diagrams."""
    topics = set()
    for f in facts:
        if f.get("topic"):
            t = f["topic"].lower().replace(" ", "-").replace("_", "-")
            topics.add(t)
    for d in diagrams:
        if d.get("topic"):
            t = d["topic"].lower().replace(" ", "-").replace("_", "-")
            topics.add(t)
    return ", ".join(sorted(topics)) if topics else "systems-engineering"


def extract_sources(facts, diagrams):
    """Extract source filenames."""
    sources = set()
    for f in facts:
        if f.get("source_file"):
            sources.add(f["source_file"])
    for d in diagrams:
        if d.get("source_file"):
            sources.add(d["source_file"])
    return ", ".join(sorted(sources)) if sources else "unknown"


# ── Fact Check ─────────────────────────────────────────────────────────────

# Per-section fact-check prompts — verify against MODEL'S KNOWLEDGE, not just raw sources
# Raw sources are secondary context for catching transcription-specific mismatches
SECTION_FACT_CHECK_PROMPTS = {
    "formulas": """You are a subject matter expert verifying lecture notes for accuracy.

## Formulas Section to Verify
{section_text}

## Context: Raw Sources (for reference only — these may contain OCR errors)
### Facts:
{facts_text}
### Diagrams:
{diagrams_text}
### Transcript:
{transcript_text}

## Verification Approach
For EACH formula, verify against YOUR OWN KNOWLEDGE of the subject:
- Is the formula mathematically correct?
- Are variables and notation standard?
- Is LaTeX notation accurate?
- Any OCR misreads from the source (1 vs l, 0 vs O, θ vs 0)?
- Does it match the standard form taught in control systems / engineering courses?

IMPORTANT: If the raw source has an OCR error but the note corrected it, mark as ✅.
If the note has the SAME OCR error as the source, flag it as ❌ with the correct formula.

Output one line per formula:
✅ [formula] — verified, mathematically correct
⚠️ [formula] — [specific concern]
❌ [formula] — [specific problem, with correct version]

End with: Total: N | Verified: N | Check: N | Issues: N""",

    "definitions": """You are a subject matter expert verifying lecture notes for accuracy.

## Definitions Section to Verify
{section_text}

## Context: Raw Sources (for reference only — these may contain OCR errors)
### Facts:
{facts_text}
### Transcript:
{transcript_text}

## Verification Approach
For EACH definition, verify against YOUR OWN KNOWLEDGE:
- Is the definition technically correct and complete?
- Are important qualifiers or conditions missing?
- Does it match standard textbook definitions?
- Any OCR-induced inaccuracies from the source?

IMPORTANT: If the raw source has an OCR error but the note corrected it, mark as ✅.
If the note propagated an OCR error from the source, flag it.

Output one line per definition:
✅ [term] — verified, technically correct
⚠️ [term] — [specific concern]
❌ [term] — [specific problem, with correct definition]

End with: Total: N | Verified: N | Check: N | Issues: N""",

    "concepts": """You are a subject matter expert verifying lecture notes for accuracy.

## Key Concepts Section to Verify
{section_text}

## Context: Raw Sources (for reference only — these may contain OCR errors)
### Facts:
{facts_text}
### Transcript:
{transcript_text}

## Verification Approach
For EACH concept, verify against YOUR OWN KNOWLEDGE:
- Is the explanation technically correct?
- Are any important aspects missing or misrepresented?
- Does it match how this concept is taught in standard courses?

Output one line per concept:
✅ [concept] — verified, technically correct
⚠️ [concept] — [specific concern]
❌ [concept] — [specific problem]

End with: Total: N | Verified: N | Check: N | Issues: N""",

    "examples": """You are a subject matter expert verifying lecture notes for accuracy.

## Examples Section to Verify
{section_text}

## Context: Raw Sources (for reference only — these may contain OCR errors)
### Facts:
{facts_text}
### Transcript:
{transcript_text}

## Verification Approach
For EACH example, verify against YOUR OWN KNOWLEDGE:
- Are the technical details correct?
- Is the example well-explained?
- Any errors introduced during transcription or note generation?

Output one line per example:
✅ [example title] — verified
⚠️ [example title] — [specific concern]
❌ [example title] — [specific problem]

End with: Total: N | Verified: N | Check: N | Issues: N""",

    "diagrams": """You are a subject matter expert verifying lecture notes for accuracy.

## Diagrams Section to Verify
{section_text}

## Context: Raw Sources (OCR'd, may contain errors)
{diagrams_text}

## Verification Approach
For EACH diagram description, verify against YOUR OWN KNOWLEDGE:
- Are the components and relationships described correctly?
- Are labels, equations, and values accurate?
- Any OCR misreads that the note may have propagated?

IMPORTANT: OCR often garbles LaTeX, subscripts, and Greek letters.
If the note has the same garbled text as the OCR source, flag it.

Output one line per diagram:
✅ [diagram title] — verified
⚠️ [diagram title] — [specific concern]
❌ [diagram title] — [specific problem]

End with: Total: N | Verified: N | Check: N | Issues: N""",
}


def extract_section(note_text, section_name):
    """Extract a section from the note by its ## header.

    Returns the section text (including header) up to the next ## or end of note.
    """
    # Map section names to possible headers
    header_map = {
        "formulas": "## Formulas",
        "definitions": "## Definitions",
        "concepts": "## Key Concepts",
        "examples": "## Examples",
        "diagrams": "## Diagrams",
    }
    header = header_map.get(section_name, f"## {section_name}")

    # Find the section
    idx = note_text.find(header)
    if idx == -1:
        return None

    # Find the next ## section (skip the header line itself)
    rest = note_text[idx + len(header):]
    next_section = rest.find("\n## ")
    if next_section == -1:
        # Last section — take everything until source verification or fact check
        for marker in ["\n## Source Verification", "\n## Fact Check", "\n---\n"]:
            end_idx = rest.find(marker)
            if end_idx != -1:
                return header + rest[:end_idx]
        return header + rest
    return header + rest[:next_section]


def _compact_source_context(source_text, section_text, max_chars):
    """Return a bounded, relevant raw-source excerpt for one section check.

    Model knowledge remains the primary standard; raw source is only secondary
    context for OCR/transcription mismatches. Keeping it bounded avoids sending
    the complete lecture to all five independent checks.
    """
    if not source_text:
        return "(no source provided)"
    if len(source_text) <= max_chars:
        return source_text

    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", section_text.lower()))
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", source_text) if chunk.strip()]
    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (-sum(term in item[1].lower() for term in terms), item[0]),
    )
    selected, used = [], 0
    for _, chunk in ranked:
        remaining = max_chars - used
        if remaining <= 0:
            break
        selected.append(chunk[:remaining])
        used += len(selected[-1]) + 2
    return "\n\n".join(selected)[:max_chars] + "\n[raw source excerpt truncated]"


def _section_source_context(section_name, section_text, transcript_text, facts_text, diagrams_text):
    """Select the smallest appropriate secondary source context per section."""
    context = {
        "transcript_text": "(not needed for this section)",
        "facts_text": "(not needed for this section)",
        "diagrams_text": "(not needed for this section)",
    }
    if section_name == "formulas":
        context["facts_text"] = _compact_source_context(facts_text, section_text, 3500)
        context["diagrams_text"] = _compact_source_context(diagrams_text, section_text, 2000)
    elif section_name in {"definitions", "concepts", "examples"}:
        context["facts_text"] = _compact_source_context(facts_text, section_text, 3500)
        # Transcript detail matters for these lecturer-specific sections.
        context["transcript_text"] = _compact_source_context(transcript_text, section_text, 3500)
    elif section_name == "diagrams":
        context["diagrams_text"] = _compact_source_context(diagrams_text, section_text, 3500)
    return context


def fact_check_batched(note_text, transcript_text, facts_text, diagrams_text,
                       backend=DEFAULT_BACKEND, model=None, api_key=None, api_url=None,
                       max_workers=2, llm_call=call_llm):
    """Check all five note sections using bounded, independent LLM calls.

    ``max_workers`` defaults to 2 (a conservative change from sequential calls)
    to reduce elapsed time without overloading a remote API or local model. The
    returned section order is always formulas, definitions, concepts, examples,
    diagrams regardless of completion order.
    """
    max_workers = max(1, int(max_workers))
    jobs = []
    total_summary = {"checked": 0, "verified": 0, "needs_check": 0, "issues": 0}

    for section_name, prompt_template in SECTION_FACT_CHECK_PROMPTS.items():
        section_text = extract_section(note_text, section_name)
        if not section_text:
            continue
        # A header plus one real item is still a populated section and must be
        # covered; only skip headers with no body at all.
        body_text = section_text.split("\n", 1)[1].strip() if "\n" in section_text else ""
        if not body_text:
            continue
        source_context = _section_source_context(
            section_name, section_text, transcript_text, facts_text, diagrams_text
        )
        jobs.append((section_name, prompt_template.format(section_text=section_text, **source_context)))

    def check_one(section_name, prompt):
        try:
            result = llm_call(prompt, backend=backend, model=model, api_key=api_key, api_url=api_url)
            if result.startswith("```"):
                result = result.split("\n", 1)[1] if "\n" in result else result[3:]
            if result.endswith("```"):
                result = result[:-3]
            return result.strip(), None
        except Exception as error:
            return None, error

    # Futures are resolved in canonical job order, making the user-facing report
    # deterministic even when the independent API requests complete out of order.
    all_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(check_one, *job) for job in jobs]
        for (section_name, _), future in zip(jobs, futures):
            result, error = future.result()
            if error:
                print(f"    Checking {section_name}... failed: {error}")
                all_results.append(f"### {section_name.upper()}\n\nFact-check failed: {error}")
                continue
            batch_summary = parse_verification_summary(result)
            for key in total_summary:
                total_summary[key] += batch_summary[key]
            print(f"    Checking {section_name}... {batch_summary['verified']} verified, "
                  f"{batch_summary['needs_check']} check, {batch_summary['issues']} issues")
            all_results.append(f"### {section_name.upper()}\n\n{result}")

    return "\n\n".join(all_results), total_summary


# ── Web Verification (for ⚠️ items) ─────────────────────────────────────

WEB_VERIFY_PROMPT = """You are verifying a claim from lecture notes using web search results.

## Original Claim (marked as ⚠️ needs review)
{claim}

## Web Search Results for: "{query}"
{search_results}

## Task
Based on the search results above, determine if the original claim is:
- ✅ VERIFIED: The search results confirm the claim is correct
- ❌ INCORRECT: The search results show the claim is wrong (provide correction)
- ⚠️ INCONCLUSIVE: Search results don't clearly confirm or deny

Be concise. Output ONE line:
✅ [claim summary] — verified by web search
❌ [claim summary] — [correction with source]
⚠️ [claim summary] — inconclusive, needs manual check"""


def web_search_ddg(query, num_results=5):
    """Search DuckDuckGo Instant Answer API + Wikipedia. No API key needed, no CAPTCHA."""
    from urllib import parse, request
    results = []

    # Source 1: DuckDuckGo Instant Answer API (returns abstract + related topics)
    params = parse.urlencode({
        "q": query, "format": "json", "no_redirect": "1", "no_html": "1"
    })
    url = f"https://api.duckduckgo.com/?{params}"
    req = request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        # Abstract (usually from Wikipedia)
        abstract = data.get("Abstract", "").strip()
        if abstract:
            results.append(f"[{data.get('AbstractSource', 'Wikipedia')}] {abstract}")

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and "Text" in topic:
                results.append(f"[Related] {topic['Text']}")
            if len(results) >= num_results:
                break
    except Exception:
        pass

    # Source 2: Wikipedia API (more detailed, good for technical topics)
    if len(results) < num_results:
        # Extract key terms for Wikipedia search
        wiki_query = re.sub(r'\$[^$]*\$', '', query)  # Remove LaTeX
        wiki_query = re.sub(r'\\[a-zA-Z]+', '', wiki_query)
        wiki_query = re.sub(r'[^a-zA-Z0-9 ]', ' ', wiki_query).strip()
        if len(wiki_query) > 5:
            wiki_params = parse.urlencode({
                "action": "query", "list": "search", "srsearch": wiki_query,
                "srlimit": str(num_results - len(results)), "format": "json"
            })
            wiki_url = f"https://en.wikipedia.org/w/api.php?{wiki_params}"
            wiki_req = request.Request(wiki_url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with request.urlopen(wiki_req, timeout=15) as resp:
                    wiki_data = json.loads(resp.read())
                for item in wiki_data.get("query", {}).get("search", []):
                    # Clean HTML from snippet
                    snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                    if snippet:
                        results.append(f"[Wikipedia: {item.get('title', '')}] {snippet}")
                    if len(results) >= num_results:
                        break
            except Exception:
                pass

    return "\n".join(results) if results else "(no search results found)"


def extract_warning_items(verification_text):
    """Extract ⚠️ items from verification text. Returns list of (section, claim_text) tuples."""
    items = []
    current_section = "unknown"
    for line in verification_text.split("\n"):
        stripped = line.strip()
        # Track current section
        if stripped.startswith("### "):
            current_section = stripped[4:].strip().lower()
        # Find warning items
        if "⚠️" in stripped or "⚠" in stripped:
            # Clean up the claim text
            claim = stripped.replace("⚠️", "").replace("⚠", "").strip()
            if claim.startswith("- "):
                claim = claim[2:]
            items.append((current_section, claim))
        # Also check for text label
        elif stripped.upper().startswith("CHECK:") or stripped.upper().startswith("WARN:"):
            claim = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
            items.append((current_section, claim))
    return items


def normalize_web_query(query):
    """Normalize search text for stable per-run web-result cache keys."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", query.lower())).strip()


def _web_query_for_claim(section, claim):
    """Build a short, focused search query from a fact-check warning."""
    query = claim
    query = re.sub(r'\$[^$]*\$', '', query)
    query = re.sub(r'\\[a-zA-Z]+', '', query)
    query = re.sub(r'[\*_\[\](){}]', '', query)
    query = re.sub(r'—.*$', '', query)
    query = re.sub(r'✅|⚠️|❌|⚠', '', query)
    query = re.sub(r'\s+', ' ', query).strip()
    section_terms = {"formulas": "formula", "definitions": "definition", "concepts": "",
                     "examples": "example", "diagrams": ""}
    if section_terms.get(section):
        query = f"{section_terms[section]} {query}"
    if len(query) > 60:
        query = query[:60].rsplit(' ', 1)[0]
    return query if len(query) >= 10 else f"{section} control systems"


def web_verify_items(verification_text, backend=DEFAULT_BACKEND, model=None,
                     api_key=None, api_url=None, max_items=5, max_workers=2,
                     cache=None, web_search_fn=web_search_ddg, llm_call=call_llm):
    """Web-verify unique warnings with a normalized-query cache and bounded work.

    ``cache`` is injectable (and may be retained by the caller) and maps a
    normalized search query to the raw web result text. Duplicate claims are
    removed before either web or LLM calls. Output ordering follows the original
    verification report, even when web work runs concurrently.
    """
    warning_items = extract_warning_items(verification_text)
    if not warning_items:
        return verification_text, []
    max_workers = max(1, int(max_workers))
    cache = cache if cache is not None else {}

    # Same warning can be emitted by retrying a batch or by parser variants.
    unique_items, seen_claims = [], set()
    for section, claim in warning_items:
        claim_key = normalize_web_query(claim)
        if claim_key and claim_key not in seen_claims:
            seen_claims.add(claim_key)
            unique_items.append((section, claim))
    items_to_verify = unique_items[:max_items]
    if len(unique_items) > max_items:
        print(f"    Found {len(unique_items)} unique ⚠️ items, verifying top {max_items}")

    jobs = [(section, claim, _web_query_for_claim(section, claim)) for section, claim in items_to_verify]

    def verify_one(section, claim, query):
        cache_key = normalize_web_query(query)
        try:
            search_results = cache.get(cache_key)
            if search_results is None:
                search_results = web_search_fn(query, num_results=3)
                cache[cache_key] = search_results
            if search_results.startswith("(no search") or search_results.startswith("(search failed"):
                return "⚠️ INCONCLUSIVE — no web results"
            prompt = WEB_VERIFY_PROMPT.format(claim=claim, query=query, search_results=search_results)
            return llm_call(prompt, backend=backend, model=model, api_key=api_key, api_url=api_url).strip().split("\n")[0]
        except Exception as error:
            return f"⚠️ verification failed: {error}"

    web_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(verify_one, *job) for job in jobs]
        for (section, claim, query), future in zip(jobs, futures):
            result = future.result()
            print(f"    🔍 Searching: {query}... → {result[:80]}")
            web_results.append((section, claim, result))

    if web_results:
        verification_text += "\n\n### WEB VERIFICATION (for ⚠️ items)\n\n"
        verification_text += "*The following items were flagged for review and verified via web search:*\n\n"
        for section, claim, result in web_results:
            verification_text += f"**[{section}]** {claim}\n→ {result}\n\n"
    return verification_text, web_results


def parse_verification_summary(verification_text):
    """Extract summary counts from verification text.

    Handles multiple output formats:
    - Emoji markers: ✅, ⚠️, ❌
    - Text labels: VERIFIED, CHECK, ISSUE, PASS, WARN, FAIL
    - Summary line parsing: 'Total items checked: N', 'Verified: N', etc.
    - Markdown bold: **Verified: N**
    """
    summary = {"checked": 0, "verified": 0, "needs_check": 0, "issues": 0}

    # Count emoji and text markers per line
    for line in verification_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Emoji detection (handle both raw and escaped unicode)
        has_verified = "✅" in stripped or "✔" in stripped
        has_warning = "⚠️" in stripped or "⚠" in stripped
        has_issue = "❌" in stripped or "✖" in stripped

        # Text label fallback
        if not (has_verified or has_warning or has_issue):
            upper = stripped.upper()
            has_verified = upper.startswith(("VERIFIED", "PASS", "CORRECT"))
            has_warning = upper.startswith(("CHECK", "WARN", "REVIEW", "NEEDS CHECK"))
            has_issue = upper.startswith(("ISSUE", "FAIL", "ERROR", "INCORRECT", "MISREAD"))

        # Only count lines that look like verification entries:
        # - Start with a marker emoji
        # - Start with a bullet (-, *, •)
        # - Contain → or -> as separator
        # - Short lines with colon separator (not prose sentences)
        has_marker = has_verified or has_warning or has_issue
        starts_with_marker = has_marker and (
            stripped[0] in "✅⚠❌✔✖" or  # emoji start
            stripped.startswith(("-", "*", "•")) or
            upper.startswith(("VERIFIED", "PASS", "CORRECT", "CHECK", "WARN",
                              "REVIEW", "ISSUE", "FAIL", "ERROR", "INCORRECT", "MISREAD"))
        )
        has_separator = "→" in stripped or "->" in stripped
        is_short_entry = (":" in stripped and len(stripped) < 200 and
                          not stripped.endswith(".") and
                          not stripped.endswith("found."))
        is_entry = starts_with_marker or has_separator or (has_marker and is_short_entry)

        # Exclude summary lines (they contain counts like "Verified: 25")
        is_summary_line = bool(re.search(r'[:\s]+\d+\s*$', stripped)) and any(
            kw in stripped.upper() for kw in ["TOTAL", "VERIFIED", "CHECK", "ISSUE", "NEEDS", "SUMMARY"]
        )

        if has_verified and is_entry and not is_summary_line:
            summary["verified"] += 1
            summary["checked"] += 1
        elif has_issue and is_entry and not is_summary_line:
            summary["issues"] += 1
            summary["checked"] += 1
        elif has_warning and is_entry and not is_summary_line:
            summary["needs_check"] += 1
            summary["checked"] += 1

    # If no per-line entries found, try parsing summary lines
    if summary["checked"] == 0:
        for line in verification_text.split("\n"):
            lower = line.lower().strip()
            # Match patterns like "Total items checked: 15" or "Total checked: 15"
            m = re.search(r'total\s+(?:items\s+)?checked[:\s]+(\d+)', lower)
            if m:
                summary["checked"] = int(m.group(1))
            m = re.search(r'(?:^|\s)verified[:\s]+(\d+)', lower)
            if m:
                summary["verified"] = int(m.group(1))
            m = re.search(r'(?:needs?\s+check|needs?\s+review|warnings?)[:\s]+(\d+)', lower)
            if m:
                summary["needs_check"] = int(m.group(1))
            m = re.search(r'(?:issues?|errors?|problems?)[:\s]+(\d+)', lower)
            if m:
                summary["issues"] = int(m.group(1))

    return summary


# ── Correction Pass ───────────────────────────────────────────────────────

CORRECTION_PROMPT = """You are correcting errors in lecture notes based on a fact-check report.

## Original Lecture Notes
{note_text}

## Fact-Check Report (with errors identified)
{verification_text}

## Instructions
Fix ALL items marked with ❌ (issues) and ⚠️ (needs review) in the note.
- For ❌ items: replace the incorrect content with the correct version provided in the report
- For ⚠️ items: if the report provides a correction, apply it; if not, add a [NEEDS VERIFICATION] tag
- Do NOT change any ✅ (verified) content
- Preserve the exact markdown structure, LaTeX formatting, and Obsidian syntax
- Output the COMPLETE corrected note (not just the changed parts)

Output the full corrected note:"""


def correction_pass(note_text, verification_text,
                    backend=DEFAULT_BACKEND, model=None, api_key=None, api_url=None,
                    llm_call=call_llm):
    """Apply corrections from fact-check report to the note.

    Takes the original note and the verification report, returns the corrected note.
    Only rewrites sections that have ❌ or ⚠️ items.
    """
    # Extract only the problematic items
    has_corrections = False
    for line in verification_text.split("\n"):
        if "❌" in line or ("⚠️" in line and "INCONCLUSIVE" not in line):
            has_corrections = True
            break

    if not has_corrections:
        return note_text

    prompt = CORRECTION_PROMPT.format(
        note_text=note_text,
        verification_text=verification_text,
    )

    try:
        corrected = llm_call(prompt, backend=backend, model=model,
                             api_key=api_key, api_url=api_url)
        # Strip code block wrapper if present
        if corrected.startswith("```markdown\n"):
            corrected = corrected[len("```markdown\n"):]
        if corrected.startswith("```\n"):
            corrected = corrected[len("```\n"):]
        if corrected.endswith("\n```"):
            corrected = corrected[:-len("\n```")]

        # Sanity check: corrected note should be roughly similar length
        if len(corrected) < len(note_text) * 0.5:
            print("  Warning: corrected note much shorter than original, keeping original")
            return note_text

        print(f"  Corrections applied ({len(note_text)} → {len(corrected)} chars)")
        return corrected.strip()
    except Exception as e:
        print(f"  Correction pass failed: {e}, keeping original")
        return note_text


# ── Source Verification ────────────────────────────────────────────────────

def build_source_verification(transcript_text, facts, diagrams, mode="inline", external_path=None):
    """Build source transparency content.

    ``inline`` preserves the historic default. ``none`` omits raw material;
    ``external`` leaves a compact link for a separately-written source sidecar.
    """
    if mode not in {"inline", "none", "external"}:
        raise ValueError(f"Unknown source mode: {mode}")
    if mode == "none":
        return ""
    if mode == "external":
        reference = external_path or "lecture.sources.md"
        return ("---\n\n## Source Verification\n\n"
                f"*Raw transcript, extracted facts, and diagram descriptions are externalized in "
                f"[{reference}]({reference}).*\n")

    sections = []
    sections.append("---\n")
    sections.append("## Source Verification\n")

    # Summary of sources used
    source_count = 0
    if transcript_text:
        source_count += 1
    if facts:
        source_count += 1
    if diagrams:
        source_count += 1
    sections.append(f"*This note was generated from {source_count} source(s). Expand the sections below to verify against the original material.*\n")

    # Transcript
    if transcript_text:
        line_count = len(transcript_text.strip().split("\n"))
        sections.append(f"<details>\n<summary>📝 Transcript ({line_count} lines)</summary>\n")
        sections.append("```")
        sections.append(transcript_text.strip())
        sections.append("```\n")
        sections.append("</details>\n")

    # Extracted Facts
    if facts:
        sections.append(f"<details>\n<summary>📋 Extracted Facts ({len(facts)} facts)</summary>\n")
        for i, fact in enumerate(facts, 1):
            topic = fact.get("topic", "general")
            ftype = fact.get("type", "text")
            content = fact.get("content", "")
            source = fact.get("source_file", "unknown")
            engine = fact.get("engine", "unknown")
            ts = ""
            if fact.get("timestamp_start") and fact.get("timestamp_end"):
                ts = f" [{fact['timestamp_start']:.1f}s - {fact['timestamp_end']:.1f}s]"

            sections.append(f"#### Fact {i}: {topic}")
            sections.append(f"- **Type:** {ftype}")
            sections.append(f"- **Source:** {source} (engine: {engine}){ts}")
            sections.append(f"- **Content:**\n")
            sections.append(content)
            sections.append("")
        sections.append("</details>\n")

    # Diagram Descriptions
    if diagrams:
        sections.append(f"<details>\n<summary>🖼️ Diagram Descriptions ({len(diagrams)} diagrams)</summary>\n")
        for i, diag in enumerate(diagrams, 1):
            dtype = diag.get("diagram_type", "general")
            page = diag.get("page", "?")
            content = diag.get("content", "")
            source = diag.get("source_file", "unknown")
            engine = diag.get("engine", "unknown")

            sections.append(f"#### Diagram {i}: {dtype} (Page {page})")
            sections.append(f"- **Source:** {source} (engine: {engine})")
            sections.append(f"- **Content:**\n")
            sections.append(content)
            sections.append("")
        sections.append("</details>\n")

    return "\n".join(sections)


# ── Output ─────────────────────────────────────────────────────────────────

def default_note_output_path(course_code, lecture):
    """Return the generated-note staging path consumed by obsidian_sync."""
    course_slug = re.sub(r"[^a-z0-9]+", "-", str(course_code).lower()).strip("-")
    return os.path.join(
        DEFAULT_OUTPUT_DIR,
        course_slug,
        "derived",
        "notes",
        f"Lecture{int(lecture):02d}.md",
    )


def get_course_name(course_code, catalog):
    """Look up course name from catalog."""
    for code, info in catalog.items():
        if code.upper() == course_code.upper():
            return info.get("name", course_code)
    return course_code


def save_note(note_text, output_path):
    """Save the generated note to a file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(note_text)
    print(f"Saved note to: {output_path}")


def retain_to_hindsight(note_text, course_code, lecture, topic, url=HINDSIGHT_URL):
    """Retain a summary of the generated notes into Hindsight."""
    summary_match = re.search(r'## Key Concepts\n(.*?)(?=\n## |\Z)', note_text, re.DOTALL)
    summary = summary_match.group(1).strip() if summary_match else note_text[:500]

    course_tag = course_code.lower().replace(" ", "-")
    topic_tag = topic.lower().replace(" ", "-")
    item = {
        "content": f"Lecture {lecture} ({topic}) notes generated. Key concepts:\n{summary}",
        "context": f"{course_code} Lecture {lecture} — {topic} [lecture-notes]",
        "tags": [
            "personal-kb", course_tag, f"course:{course_tag}", "lecture-notes",
            "source-type:lecture", f"lecture:{lecture}", f"topic:{topic_tag}",
            "scope:course", topic_tag,
        ],
    }
    payload = {"items": [item]}
    try:
        result = _shared_http_client(30).post_json(
            f"{url}/v1/default/banks/hermes-history/memories",
            payload,
            timeout=30,
        )
        print("Retained summary into Hindsight")
        return result
    except Exception as error:
        print(f"Hindsight retention failed: {error}")
        return {"error": str(error)}


# ── Main ───────────────────────────────────────────────────────────────────

def generate_notes(transcript_text=None, facts=None, diagrams=None,
                   course_code="UNKNOWN", lecture=1, topic="Lecture",
                   backend=DEFAULT_BACKEND, model=None, api_key=None,
                   api_url=None, ollama_url=None, catalog=None,
                   run_fact_check=True, web_verify=False, fact_check_workers=2,
                   web_workers=2, web_max_items=5, source_mode="inline",
                   source_reference=None, web_cache=None, llm_call=call_llm,
                   web_search_fn=web_search_ddg):
    """Generate lecture notes from available inputs. Returns markdown string."""
    if catalog is None:
        catalog = load_course_catalog()

    course_name = get_course_name(course_code, catalog)
    engine = detect_engine(facts or [], diagrams or [])
    topic_tags = extract_topics(facts or [], diagrams or [])
    sources = extract_sources(facts or [], diagrams or [])
    date = datetime.date.today().isoformat()

    # Build prompt
    prompt = SYNTHESIS_PROMPT.format(
        course_code=course_code,
        course_name=course_name,
        lecture=lecture,
        topic=topic,
        engine=engine,
        transcript_text=transcript_text or "(no transcript provided)",
        facts_text=format_facts_for_prompt(facts),
        diagrams_text=format_diagrams_for_prompt(diagrams),
        date=date,
        tags=topic_tags,
        sources=sources,
    )

    print(f"Generating notes for {course_code} Lecture {lecture}: {topic}")
    print(f"  Backend: {backend}, Model: {model or DEFAULT_MODEL}")
    print(f"  Inputs: {'transcript, ' if transcript_text else ''}"
          f"{'facts (' + str(len(facts)) + '), ' if facts else ''}"
          f"{'diagrams (' + str(len(diagrams)) + ')' if diagrams else ''}")

    # Step 1: Generate the note
    note_text = llm_call(prompt, backend=backend, model=model,
                         api_key=api_key, api_url=api_url, ollama_url=ollama_url)

    # Strip markdown code block wrapper if LLM wrapped output
    if note_text.startswith("```markdown\n"):
        note_text = note_text[len("```markdown\n"):]
    if note_text.startswith("```\n"):
        note_text = note_text[len("```\n"):]
    if note_text.endswith("\n```"):
        note_text = note_text[:-len("\n```")]

    # Step 2: Fact-check pass (batched by section)
    if run_fact_check:
        facts_text = format_facts_for_prompt(facts)
        diagrams_text = format_diagrams_for_prompt(diagrams)

        print("  Running batched fact-check...")
        verification, summary = fact_check_batched(
            note_text,
            transcript_text,
            facts_text,
            diagrams_text,
            backend=backend,
            model=model,
            api_key=api_key,
            api_url=api_url,
            max_workers=fact_check_workers,
            llm_call=llm_call,
        )

        print(f"  Fact-check total: {summary['verified']} verified, "
              f"{summary['needs_check']} need review, {summary['issues']} issues")

        # Step 2b: Web verification for ⚠️ items
        if summary['needs_check'] > 0 and web_verify:
            print("  Running web verification for ⚠️ items...")
            verification, web_results = web_verify_items(
                verification,
                backend=backend, model=model,
                api_key=api_key, api_url=api_url,
                max_items=web_max_items,
                max_workers=web_workers,
                cache=web_cache,
                web_search_fn=web_search_fn,
                llm_call=llm_call,
            )
            if web_results:
                print(f"  Web verified: {len(web_results)} items")

        # Step 2c: Correction pass — fix ❌ and resolved ⚠️ items in the note
        has_issues = summary['issues'] > 0 or summary['needs_check'] > 0
        if has_issues:
            print("  Running correction pass...")
            note_text = correction_pass(
                note_text, verification,
                backend=backend, model=model,
                api_key=api_key, api_url=api_url,
                llm_call=llm_call,
            )

        # Append verification section with header
        note_text += "\n\n---\n\n## Fact Check Report\n\n" + verification

    # Step 3: Source verification is opt-in or externalized to keep notes compact.
    source_section = build_source_verification(
        transcript_text,
        facts,
        diagrams,
        mode=source_mode,
        external_path=source_reference,
    )
    if source_section:
        note_text += "\n\n" + source_section

    return note_text


def main():
    parser = argparse.ArgumentParser(description="Generate structured lecture notes")
    # Inputs
    parser.add_argument("--transcript", help="Path to transcript text file")
    parser.add_argument("--facts", help="Path to chunked facts JSON")
    parser.add_argument("--diagrams", help="Path to diagram descriptions JSON")

    # Metadata
    parser.add_argument("--course", required=True, help="Course code (e.g., PERSONAL-ALPHA)")
    parser.add_argument("--lecture", type=int, required=True, help="Lecture number")
    parser.add_argument("--topic", default="Lecture Content", help="Lecture topic name")

    # LLM backend (hardened to DeepSeek V4 Pro via the official DeepSeek API)
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=sorted(ALLOWED_BACKENDS),
                        help="LLM backend (only 'deepseek' is supported; default: deepseek)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Override model name (default: deepseek-v4-pro)")
    parser.add_argument("--api-key", help="Override DeepSeek API key (default: DEEPSEEK_API_KEY)")
    parser.add_argument("--api-url", default=DEEPSEEK_API_URL,
                        help="Override DeepSeek chat completion endpoint (default: https://api.deepseek.com/v1/chat/completions)")
    # ``--ollama-url`` retained as a no-op for backward CLI compatibility; local
    # Ollama text generation is no longer an allowed Personal generation backend.
    parser.add_argument("--ollama-url", default=None,
                        help="(deprecated, ignored) Ollama text generation is no longer used; use diagrams.py/transcribe.py for local vision/OCR.")

    # Output
    parser.add_argument("--output", help="Output file path (default: project/courses/<course_slug>/derived/notes/LectureNN.md; watcher syncs it to Obsidian)")
    parser.add_argument("--retain", action="store_true", help="Retain summary into Hindsight")
    parser.add_argument("--hindsight-url", default=HINDSIGHT_URL, help="Hindsight API URL")
    parser.add_argument("--no-fact-check", action="store_true", help="Skip fact-check pass")
    parser.add_argument("--fact-check-workers", type=int, default=2,
                        help="Independent fact-check worker limit (default: 2; formerly sequential)")
    parser.add_argument("--web-verify", action="store_true", help="Web search for flagged fact-check items")
    parser.add_argument("--web-workers", type=int, default=2,
                        help="Concurrent web-verification worker limit (default: 2)")
    parser.add_argument("--web-max-items", type=int, default=5,
                        help="Maximum unique flagged claims to web-verify (default: 5)")
    parser.add_argument("--source-mode", choices=("inline", "external", "none"), default="inline",
                        help="Raw source output: inline (legacy default), external sidecar, or none")
    parser.add_argument("--source-output", help="Sidecar path for --source-mode external (default: note path + .sources.md)")

    args = parser.parse_args()

    # Load inputs
    transcript_text = None
    if args.transcript:
        transcript_text = load_transcript(args.transcript)
        print(f"Loaded transcript: {args.transcript} ({len(transcript_text)} chars)")

    facts = None
    if args.facts:
        facts = load_facts(args.facts)
        print(f"Loaded facts: {args.facts} ({len(facts)} facts)")

    diagrams = None
    if args.diagrams:
        diagrams = load_diagrams(args.diagrams)
        print(f"Loaded diagrams: {args.diagrams} ({len(diagrams)} diagrams)")

    if not any([transcript_text, facts, diagrams]):
        print("Error: provide at least one of --transcript, --facts, or --diagrams")
        sys.exit(1)

    # Resolve output before generation so an external source link can be stable.
    if args.output:
        output_path = args.output
    else:
        output_path = default_note_output_path(args.course, args.lecture)
    source_output = args.source_output or f"{output_path}.sources.md"
    source_reference = os.path.relpath(
        source_output, os.path.dirname(os.path.abspath(output_path))
    ).replace(os.sep, "/")

    # Generate
    catalog = load_course_catalog()
    note_text = generate_notes(
        transcript_text=transcript_text,
        facts=facts,
        diagrams=diagrams,
        course_code=args.course,
        lecture=args.lecture,
        topic=args.topic,
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        api_url=args.api_url,
        ollama_url=args.ollama_url,
        catalog=catalog,
        run_fact_check=not args.no_fact_check,
        web_verify=args.web_verify,
        fact_check_workers=args.fact_check_workers,
        web_workers=args.web_workers,
        web_max_items=args.web_max_items,
        source_mode=args.source_mode,
        source_reference=source_reference,
        web_cache={},
    )

    save_note(note_text, output_path)
    if args.source_mode == "external":
        save_note(build_source_verification(transcript_text, facts, diagrams), source_output)

    # Retain to Hindsight
    if args.retain:
        retain_to_hindsight(note_text, args.course, args.lecture, args.topic,
                            url=args.hindsight_url)

    print(f"\nDone! Note generated ({len(note_text)} chars)")
    print(f"Path: {output_path}")


if __name__ == "__main__":
    main()
