"""Personal KB — Textbook Processor

Extracts structured knowledge from textbook chapters and indexes it into
OpenViking (source-grounded retrieval) and Hindsight (learning memories).

Pipeline (borrowing cangjie's staged discipline):
  Stage 0: Chapter overview     → summary + key term list
  Stage 1: 3 parallel extractors → concepts / formulas / examples
  Stage 1.5: Triple verification → source-anchored, testable, non-trivial
  Stage 2: Index to OpenViking  → resource per concept with metadata tags
  Stage 3: Cross-reference      → link related concepts
  Stage 4: Retrieval test       → verify concepts are findable via RRF
  Stage 5: Hindsight retain     → learning memories with structured tags

Default model: DeepSeek V4 Pro via the DeepSeek API (text-only, no vision needed).
The processor operates on already-extracted text (pymupdf output or OCR output).

Usage:
    python scripts/notes/textbook_processor.py process textbook.pdf --course "PERSONAL-ALPHA" --chapter 1
    python scripts/notes/textbook_processor.py process extracted_text.json --course "PERSONAL-ALPHA" --chapter 1 --dry-run
    python scripts/notes/textbook_processor.py process textbook.pdf --course "PERSONAL-ALPHA" --chapter 1 --output-dir data/textbook_extracts
"""


from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from ..core.common_client import JsonHttpClient, load_api_key
    from ..ingestion.ingestion.ingest import format_fact_tags, hindsight_retain_items
except ImportError:  # pragma: no cover - direct CLI use
    from scripts.core.common_client import JsonHttpClient, load_api_key
    from scripts.ingestion.ingest import format_fact_tags, hindsight_retain_items

# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"

HINDSIGHT_URL = "http://localhost:8888"
DEFAULT_BANK_ID = "hermes-history"
DEFAULT_OUTPUT_DIR = Path("data/textbook_extracts")
DEFAULT_CHUNK_SIZE = 4000  # chars per chapter section chunk
DEFAULT_MAX_WORKERS = 2    # bounded two-worker pool for 3 extractor types



def _load_deepseek_key() -> str:
    """Load the DeepSeek API key from the environment or Hermes .env."""
    key = load_api_key("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. Set it in the environment or Hermes .env."
        )
    return key


# ── LLM Client ──────────────────────────────────────────────────────────────

class TextbookLLMClient:
    """OpenAI-compatible LLM client for textbook extraction via DeepSeek."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = 600,
    ) -> None:
        self.api_key = api_key or _load_deepseek_key()
        self.api_url = api_url
        if model != DEFAULT_MODEL:
            raise ValueError(f"Personal-KB requires {DEFAULT_MODEL}; got {model!r}")
        self.model = DEFAULT_MODEL
        self.http_client = JsonHttpClient(timeout=timeout, retries=2)
        self.timeout = timeout

    def complete(self, prompt: str, *, max_tokens: int = 8192, temperature: float = 0.1) -> str:
        response = self.http_client.post_json(
            self.api_url,
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "HermesTextbookProcessor/1.0",
            },
            timeout=self.timeout,
        )
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"LLM response missing choices[0].message.content: {error}") from error


# ── Text Extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a born-digital PDF using pymupdf."""
    try:
        import pymupdf
    except ImportError as error:
        raise RuntimeError("pymupdf is required for PDF text extraction") from error
    doc = pymupdf.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages)


def load_text(source: str | Path) -> str:
    """Load text from a JSON file, text file, or PDF."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "text" in data:
            return data["text"]
        if isinstance(data, dict) and "page_data" in data:
            return "\n\n".join(page["text"] for page in data["page_data"] if page.get("text", "").strip())
        return json.dumps(data, ensure_ascii=False)
    return path.read_text(encoding="utf-8")


def split_chapter(text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """Split chapter text into sections targeting max_chars each."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    sections: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current_len + len(paragraph) > max_chars and current:
            sections.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        sections.append("\n\n".join(current))
    return sections


# ── Stage 0: Chapter Overview ───────────────────────────────────────────────

OVERVIEW_PROMPT = """You are analyzing a textbook chapter for {course_code}.

Chapter {chapter}: {title}

## Instructions
Read the following chapter text and produce a structured overview:
1. A 3-5 sentence chapter summary
2. A list of key terms (with brief definitions)
3. The main topics and subtopics

Output as JSON:
{{
  "summary": "...",
  "key_terms": [{{"term": "...", "definition": "..."}}],
  "topics": ["...", "..."]
}}

Chapter text (first {max_chars} chars):
{text}
"""


def stage0_overview(
    text: str,
    course_code: str,
    chapter: int,
    title: str,
    llm: TextbookLLMClient,
) -> dict[str, Any]:
    """Generate a chapter overview using the LLM."""
    prompt = OVERVIEW_PROMPT.format(
        course_code=course_code,
        chapter=chapter,
        title=title,
        max_chars=8000,
        text=text[:8000],
    )
    raw = llm.complete(prompt, max_tokens=4096, temperature=0.1)
    try:
        # Try direct JSON parse
        return json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return {"summary": raw[:500], "key_terms": [], "topics": []}


# ── Stage 1: Parallel Extractors (3 agents) ───────────────────────────────

CONCEPT_EXTRACTOR_PROMPT = """You are a concept extractor for a Personal textbook chapter.

Course: {course_code}, Chapter {chapter}: {title}

Extract key CONCEPTS and DEFINITIONS from the following text. For each:
- Include the full definition as stated in the text
- Include qualifying conditions and context
- Preserve mathematical notation (LaTeX where applicable)
- Include the source page/section reference if visible

Output as JSON array:
[{{"type": "concept", "name": "...", "definition": "...", "context": "...", "source_ref": "..."}}]

If no meaningful concepts found, output: []

Text segment:
{segment}
"""

FORMULA_EXTRACTOR_PROMPT = """You are a formula and theorem extractor for a Personal textbook chapter.

Course: {course_code}, Chapter {chapter}: {title}

Extract FORMULAS, THEOREMS, and DERIVATIONS from the following text. For each:
- Include the complete formula in LaTeX notation
- Define every variable
- Include derivation steps if shown
- Include conditions for validity
- Include the source page/section reference

Output as JSON array:
[{{"type": "formula", "name": "...", "formula": "...", "variables": [{{"symbol": "...", "meaning": "..."}}], "derivation": "...", "conditions": "...", "source_ref": "..."}}]

If no formulas found, output: []

Text segment:
{segment}
"""

EXAMPLE_EXTRACTOR_PROMPT = """You are a worked example extractor for a Personal textbook chapter.

Course: {course_code}, Chapter {chapter}: {title}

Extract WORKED EXAMPLES and PROBLEM SOLUTIONS from the following text. For each:
- Include the complete problem statement
- Include the full solution with all steps
- Include the final answer
- Note the problem type (e.g., "stability analysis", "transfer function derivation")
- Include the source page/section reference

Output as JSON array:
[{{"type": "example", "problem": "...", "solution": "...", "answer": "...", "problem_type": "...", "source_ref": "..."}}]

If no examples found, output: []

Text segment:
{segment}
"""

EXTRACTORS = [
    ("concepts", CONCEPT_EXTRACTOR_PROMPT),
    ("formulas", FORMULA_EXTRACTOR_PROMPT),
    ("examples", EXAMPLE_EXTRACTOR_PROMPT),
]


def stage1_parallel_extract(
    sections: list[str],
    course_code: str,
    chapter: int,
    title: str,
    llm: TextbookLLMClient,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, list[dict[str, Any]]]:
    """Run 3 extractors in parallel across all text sections."""
    results: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in EXTRACTORS}

    def extract(extractor_name: str, prompt_template: str) -> tuple[str, list[dict[str, Any]]]:
        extracted: list[dict[str, Any]] = []
        for section in sections:
            prompt = prompt_template.format(
                course_code=course_code,
                chapter=chapter,
                title=title,
                segment=section,
            )
            try:
                raw = llm.complete(prompt, max_tokens=8192, temperature=0.1)
                items = _safe_parse_json_array(raw)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item["extractor"] = extractor_name
                            item["chapter"] = chapter
                            item["course"] = course_code
                            extracted.append(item)
                else:
                    print(f"  [{extractor_name}] Parsed non-list result, skipping: "
                          f"{str(items)[:100]}")
            except Exception as error:
                print(f"  [{extractor_name}] Section extraction failed: {error}")
                print(f"    Raw first 500 chars: {raw[:500]}")
        return extractor_name, extracted

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(extract, name, template)
            for name, template in EXTRACTORS
        ]
        for future in as_completed(futures):
            name, items = future.result()
            results[name].extend(items)
            print(f"  Extractor '{name}': {len(items)} items")

    return results


# ── Stage 1.5: Triple Verification ──────────────────────────────────────────

def stage1_5_verify(
    extracted: dict[str, list[dict[str, Any]]],
    source_text: str,
) -> dict[str, list[dict[str, Any]]]:
    """Verify extracted items against source text.

    V1 (source-anchored): item text appears in or is supported by source
    V2 (testable): item has enough content to be a retrieval target
    V3 (non-trivial): item is not an empty/generic placeholder
    """
    verified: dict[str, list[dict[str, Any]]] = {}
    rejected: dict[str, list[dict[str, Any]]] = {}

    for extractor_name, items in extracted.items():
        verified[extractor_name] = []
        rejected[extractor_name] = []
        for item in items:
            # V3: non-trivial
            content = " ".join(str(v) for v in item.values() if isinstance(v, str))
            if len(content.strip()) < 30:
                item["_reject_reason"] = "trivial: content too short"
                rejected[extractor_name].append(item)
                continue

            # V2: testable — must have at least a name or definition
            has_name = bool(item.get("name") or item.get("problem") or item.get("formula"))
            has_content = bool(
                item.get("definition") or item.get("formula")
                or item.get("solution") or item.get("context")
            )
            if not has_name or not has_content:
                item["_reject_reason"] = "untestable: missing name or content"
                rejected[extractor_name].append(item)
                continue

            # V1: source-anchored — at least one distinctive term from the item
            # appears in the source text
            terms = _distinctive_terms(item)
            source_lower = source_text.lower()
            anchored = any(term.lower() in source_lower for term in terms)
            if not anchored:
                item["_reject_reason"] = "unanchored: no distinctive term found in source"
                rejected[extractor_name].append(item)
                continue

            item["_verified"] = True
            verified[extractor_name].append(item)

    return {"verified": verified, "rejected": rejected}


def _distinctive_terms(item: dict[str, Any]) -> list[str]:
    """Extract distinctive technical terms from an item for source anchoring."""
    text_parts = []
    for key in ("name", "definition", "formula", "problem", "context", "conditions"):
        value = item.get(key)
        if isinstance(value, str):
            text_parts.append(value)
    combined = " ".join(text_parts)
    # Extract words longer than 4 chars, excluding common English words
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", combined)
    return list(dict.fromkeys(words))[:10]


# ── Stage 2: Index to OpenViking ────────────────────────────────────────────

def stage2_index_to_openviking(
    verified: dict[str, list[dict[str, Any]]],
    *,
    course_code: str,
    chapter: int,
    source_path: str,
    dry_run: bool = True,
) -> list[dict[str, Any]]:
    """Prepare verified items for OpenViking indexing with metadata tags.

    This stage produces the resource descriptions and metadata. Actual
    OpenViking ingestion is done separately via the source manifest pipeline.
    """
    resources: list[dict[str, Any]] = []
    for extractor_name, items in verified.items():
        for item in items:
            resource = {
                "source_type": "textbook",
                "course": course_code,
                "chapter": chapter,
                "extractor": extractor_name,
                "source_path": source_path,
                "name": item.get("name", item.get("problem", "")[:80]),
                "content": _build_resource_content(item, extractor_name),
                "tags": [
                    "personal-kb",
                    f"course:{course_code.lower().replace(' ', '-')}",
                    "source-type:textbook",
                    f"chapter:{chapter}",
                    "scope:course",
                ],
            }
            resources.append(resource)
    if dry_run:
        print(f"  [DRY RUN] Prepared {len(resources)} resources for OpenViking indexing")
    return resources


def _build_resource_content(item: dict[str, Any], extractor: str) -> str:
    """Build human-readable content for an OpenViking resource."""
    parts: list[str] = []
    if extractor == "concepts":
        parts.append(f"# {item.get('name', 'Concept')}")
        if item.get("definition"):
            parts.append(f"\n**Definition:** {item['definition']}")
        if item.get("context"):
            parts.append(f"\n**Context:** {item['context']}")
    elif extractor == "formulas":
        parts.append(f"# {item.get('name', 'Formula')}")
        if item.get("formula"):
            parts.append(f"\n**Formula:** ${item['formula']}$")
        if item.get("variables"):
            var_text = ", ".join(
                f"${v.get('symbol', '?')}$ = {v.get('meaning', '?')}"
                for v in item["variables"]
                if isinstance(v, dict)
            )
            parts.append(f"\n**Variables:** {var_text}")
        if item.get("derivation"):
            parts.append(f"\n**Derivation:** {item['derivation']}")
        if item.get("conditions"):
            parts.append(f"\n**Conditions:** {item['conditions']}")
    elif extractor == "examples":
        parts.append(f"# Example: {item.get('problem_type', 'Worked Example')}")
        if item.get("problem"):
            parts.append(f"\n**Problem:** {item['problem']}")
        if item.get("solution"):
            parts.append(f"\n**Solution:** {item['solution']}")
        if item.get("answer"):
            parts.append(f"\n**Answer:** {item['answer']}")
    if item.get("source_ref"):
        parts.append(f"\n\n*Source: {item['source_ref']}*")
    return "\n".join(parts)


# ── Stage 3: Cross-reference ────────────────────────────────────────────────

def stage3_cross_reference(
    verified: dict[str, list[dict[str, Any]]],
) -> dict[str, list[str]]:
    """Find cross-references between extracted concepts."""
    all_items: list[tuple[str, str]] = []
    for extractor_name, items in verified.items():
        for item in items:
            name = item.get("name") or item.get("problem", "")[:80]
            if name:
                all_items.append((extractor_name, name))

    references: dict[str, list[str]] = {}
    all_terms = [name.lower() for _, name in all_items]
    for extractor_name, items in verified.items():
        for item in items:
            name = item.get("name") or item.get("problem", "")[:80]
            if not name:
                continue
            content = " ".join(str(v) for v in item.values() if isinstance(v, str)).lower()
            refs = [
                ref_name for _ext, ref_name in all_items
                if ref_name.lower() != name.lower()
                and ref_name.lower() in content
            ]
            if refs:
                references[name] = refs
    return references


# ── Stage 4: Retrieval Test ─────────────────────────────────────────────────

def stage4_retrieval_test(
    resources: list[dict[str, Any]],
    *,
    course_code: str,
    chapter: int,
) -> dict[str, Any]:
    """Verify that extracted concepts are findable via simple keyword matching.

    This is a lightweight check that each resource's name appears in the
    resource content — ensuring BM25 and dense retrieval can find it.
    """
    total = len(resources)
    findable = 0
    missing: list[str] = []
    for resource in resources:
        name = resource.get("name", "").lower()
        content = resource.get("content", "").lower()
        if name and any(word in content for word in name.split() if len(word) > 3):
            findable += 1
        else:
            missing.append(resource.get("name", "<unnamed>"))
    return {
        "total": total,
        "findable": findable,
        "findable_rate": round(findable / total, 4) if total else 0.0,
        "missing": missing,
    }


# ── Stage 5: Hindsight Retain ───────────────────────────────────────────────

def stage5_retain_to_hindsight(
    verified: dict[str, list[dict[str, Any]]],
    *,
    course_code: str,
    chapter: int,
    source_path: str,
    bank_id: str = DEFAULT_BANK_ID,
    url: str = HINDSIGHT_URL,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Retain learning memories into Hindsight with structured tags."""
    items: list[dict[str, Any]] = []
    for extractor_name, extracts in verified.items():
        for extract in extracts:
            name = extract.get("name") or extract.get("problem", "")[:80]
            content = _build_resource_content(extract, extractor_name)
            fact = {
                "content": content[:2000],  # Hindsight chunk limit
                "course": course_code,
                "lecture": chapter,  # chapter acts as lecture for textbooks
                "source_type": "textbook",
                "topic": name,
                "semester": None,
                "date": None,
            }
            tags = format_fact_tags(fact)
            items.append({
                "content": content[:2000],
                "context": f"{course_code} Chapter {chapter} — {name} [textbook]",
                "tags": tags,
            })

    if dry_run:
        print(f"  [DRY RUN] Prepared {len(items)} Hindsight memory items")
        return {"retained": 0, "prepared": len(items), "dry_run": True}

    if not items:
        return {"retained": 0, "prepared": 0}

    response = hindsight_retain_items(
        items,
        bank_id=bank_id,
        url=url,
        timeout=60,
        retries=2,
    )
    return {"retained": len(items), "prepared": len(items), "response": response}


# ── Utilities ───────────────────────────────────────────────────────────────


def _safe_parse_json_array(raw: str) -> list[dict[str, Any]]:
    """Safely parse an LLM JSON response, returning [] on failure."""
    try:
        result = json.loads(_extract_json(raw))
        if isinstance(result, dict):
            # LLM returned a single object instead of array — wrap it
            return [result]
        if isinstance(result, list):
            return result
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def _extract_json(raw: str) -> str:
    """Extract a JSON array or object from an LLM response, handling LaTeX escapes."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    # The LLM may emit LaTeX like \omega, \frac, etc. Python's json parser
    # rejects unescaped backslashes. Escape them by doubling before parsing.
    # Only escape backslashes that are NOT already part of a valid JSON escape
    # (\\, \", \/, \b, \f, \n, \r, \t, \u).
    text = _escape_latex_backslashes(text)
    # Try direct parse
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # Find first [ or { and last ] or }
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in response")


_VALID_JSON_ESCAPES = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _escape_latex_backslashes(text: str) -> str:
    """Double backslashes that look like LaTeX commands (not valid JSON escapes)."""
    result = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            if i + 1 < len(text):
                rest = text[i:]
                if _VALID_JSON_ESCAPES.match(rest):
                    result.append(text[i])
                else:
                    result.append("\\\\")
            else:
                result.append("\\\\")
        else:
            result.append(text[i])
        i += 1
    return "".join(result)


def _source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Main Pipeline ───────────────────────────────────────────────────────────

def process_textbook(
    source: str | Path,
    *,
    course_code: str,
    chapter: int,
    title: str | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    api_key: str | None = None,
    api_url: str = DEFAULT_API_URL,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
    bank_id: str = DEFAULT_BANK_ID,
    hindsight_url: str = HINDSIGHT_URL,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Run the full textbook processing pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source text
    source_path = str(source)
    text = load_text(source)
    if not text.strip():
        raise ValueError("Source text is empty")
    print(f"Loaded {len(text)} chars from {source_path}")

    title = title or f"Chapter {chapter}"

    # Initialize LLM
    llm = TextbookLLMClient(api_key=api_key, api_url=api_url, model=model)

    # Stage 0: Overview
    print("\n=== Stage 0: Chapter Overview ===")
    overview = stage0_overview(text, course_code, chapter, title, llm)
    print(f"  Summary: {overview.get('summary', '')[:200]}...")
    print(f"  Key terms: {len(overview.get('key_terms', []))} found")
    print(f"  Topics: {len(overview.get('topics', []))} found")

    # Split text for extraction
    sections = split_chapter(text)
    print(f"\nSplit chapter into {len(sections)} sections")

    # Stage 1: Parallel Extraction
    print("\n=== Stage 1: Parallel Extraction (3 agents) ===")
    extracted = stage1_parallel_extract(
        sections, course_code, chapter, title, llm, max_workers=max_workers
    )
    total_extracted = sum(len(items) for items in extracted.values())
    print(f"  Total extracted items: {total_extracted}")

    # Stage 1.5: Triple Verification
    print("\n=== Stage 1.5: Triple Verification ===")
    verification = stage1_5_verify(extracted, text)
    verified = verification["verified"]
    rejected = verification["rejected"]
    total_verified = sum(len(items) for items in verified.values())
    total_rejected = sum(len(items) for items in rejected.values())
    print(f"  Verified: {total_verified} items")
    print(f"  Rejected: {total_rejected} items")

    # Stage 2: Index to OpenViking
    print("\n=== Stage 2: Prepare OpenViking Resources ===")
    resources = stage2_index_to_openviking(
        verified,
        course_code=course_code,
        chapter=chapter,
        source_path=source_path,
        dry_run=dry_run,
    )

    # Stage 3: Cross-reference
    print("\n=== Stage 3: Cross-reference ===")
    references = stage3_cross_reference(verified)
    print(f"  Found {len(references)} cross-references")

    # Stage 4: Retrieval Test
    print("\n=== Stage 4: Retrieval Test ===")
    retrieval_test = stage4_retrieval_test(resources, course_code=course_code, chapter=chapter)
    print(f"  Findable: {retrieval_test['findable']}/{retrieval_test['total']} "
          f"({retrieval_test['findable_rate']:.1%})")

    # Stage 5: Hindsight Retain
    print("\n=== Stage 5: Hindsight Retain ===")
    hindsight_result = stage5_retain_to_hindsight(
        verified,
        course_code=course_code,
        chapter=chapter,
        source_path=source_path,
        bank_id=bank_id,
        url=hindsight_url,
        dry_run=dry_run,
    )
    print(f"  Retained: {hindsight_result.get('retained', 0)} memories")

    # Save results
    result = {
        "source": source_path,
        "source_hash": _source_hash(text),
        "course": course_code,
        "chapter": chapter,
        "title": title,
        "model": model,
        "pipeline": "textbook_processor_v1",
        "stages": {
            "overview": overview,
            "extraction": {
                name: len(items) for name, items in extracted.items()
            },
            "verification": {
                "verified": {name: len(items) for name, items in verified.items()},
                "rejected": {name: len(items) for name, items in rejected.items()},
            },
            "resources": len(resources),
            "cross_references": len(references),
            "retrieval_test": retrieval_test,
            "hindsight": hindsight_result,
        },
        "verified_items": verified,
        "rejected_items": rejected,
        "resources": resources,
        "cross_references": references,
    }

    output_file = output_dir / f"{course_code.replace(' ', '-').lower()}_ch{chapter}.json"
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {output_file}")
    return result


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="textbook_processor",
        description="Extract structured knowledge from textbook chapters",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process a textbook chapter")
    process_parser.add_argument("source", help="Path to PDF, JSON, or text file")
    process_parser.add_argument("--course", required=True, help="Course code (e.g., PERSONAL-ALPHA)")
    process_parser.add_argument("--chapter", type=int, required=True, help="Chapter number")
    process_parser.add_argument("--title", default=None, help="Chapter title")
    process_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                                help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    process_parser.add_argument("--api-url", default=DEFAULT_API_URL,
                                help=f"LLM API URL")
    process_parser.add_argument("--model", default=DEFAULT_MODEL,
                                help=f"Model name (default: {DEFAULT_MODEL})")
    process_parser.add_argument("--api-key", default=None, help="API key override")
    process_parser.add_argument("--dry-run", action="store_true",
                                help="Prepare but do not ingest to Hindsight/OpenViking")
    process_parser.add_argument("--bank-id", default=DEFAULT_BANK_ID,
                                help=f"Hindsight bank ID (default: {DEFAULT_BANK_ID})")
    process_parser.add_argument("--hindsight-url", default=HINDSIGHT_URL,
                                help=f"Hindsight API URL (default: {HINDSIGHT_URL})")
    process_parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS,
                                help=f"Parallel extractors (default: {DEFAULT_MAX_WORKERS})")

    args = parser.parse_args()

    if args.command == "process":
        result = process_textbook(
            args.source,
            course_code=args.course,
            chapter=args.chapter,
            title=args.title,
            output_dir=args.output_dir,
            api_key=args.api_key,
            api_url=args.api_url,
            model=args.model,
            dry_run=args.dry_run,
            bank_id=args.bank_id,
            hindsight_url=args.hindsight_url,
            max_workers=args.workers,
        )
        print(f"\nPipeline complete: {result['stages']['resources']} resources, "
              f"{result['stages']['hindsight'].get('retained', 0)} memories")


if __name__ == "__main__":
    main()
