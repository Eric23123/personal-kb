"""Turn Personal KB retrieval results into a readable, source-grounded answer."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.core.common_client import DeepSeekClient, HttpClientError, load_api_key
from scripts.retrieval.search import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_SOURCES,
    DEFAULT_RERANK_TOP,
    _get_backend,
    cmd_hybrid,
)


_DOCUMENT_ID = re.compile(r"(?:^|/)([0-9a-f]{32})-[0-9a-f]{64}_[a-z_]+-", re.IGNORECASE)
_SOURCE_HASH = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
_INTERNAL_SOURCE_WORDS = re.compile(
    r"^(?:json|md|txt|ocr|transcript|diagram|description|unknown|untitled)$",
    re.IGNORECASE,
)


def record_source_lookup(index_path: str | Path) -> dict[str, str]:
    """Map an OpenViking artifact URI back to the user's original file.

    OpenViking turns artifacts such as OCR text and image descriptions into
    Markdown chunks, and its search response does not retain our metadata.
    The vault record is therefore the durable provenance authority.
    """
    index = Path(index_path).expanduser().resolve()
    records_dir = index.parent / "records"
    lookup: dict[str, str] = {}
    if not records_dir.is_dir():
        return lookup
    for record_path in records_dir.glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        document_id = str(record.get("id", "")).strip().lower()
        relative_path = str(record.get("relative_path", "")).strip()
        if re.fullmatch(r"[0-9a-f]{32}", document_id) and relative_path:
            lookup[document_id] = relative_path.replace("\\", "/")
        source_hash = str(record.get("source_hash", "")).strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", source_hash) and relative_path:
            # Legacy OpenViking chunks can outlive a moved or replaced record
            # ID, but still retain the deterministic artifact/source hash.
            lookup.setdefault(f"hash:{source_hash}", relative_path.replace("\\", "/"))
    return lookup


def source_name(result: dict, record_sources: dict[str, str]) -> str:
    metadata = result.get("metadata") or {}
    relative_path = str(metadata.get("relative_path", "")).strip()
    if relative_path:
        return relative_path.replace("\\", "/")
    uri = str(result.get("uri", ""))
    match = _DOCUMENT_ID.search(uri)
    if match:
        original = record_sources.get(match.group(1).lower())
        if original:
            return original
    for source_hash in _SOURCE_HASH.findall(uri):
        original = record_sources.get(f"hash:{source_hash.lower()}")
        if original:
            return original
    context = str(result.get("rerank_context", ""))
    for line in context.splitlines():
        if line.lower().endswith((".pdf", ".md", ".txt")):
            return Path(line.strip()).name
    for line in context.splitlines():
        candidate = line.strip().strip('"').strip("#").strip()
        if (
            4 <= len(candidate) <= 100
            and "Page " not in candidate
            and not candidate.startswith("[")
            and any(char.isalpha() for char in candidate)
            and not _INTERNAL_SOURCE_WORDS.fullmatch(candidate)
        ):
            return candidate
    # Never expose an OpenViking chunk filename (often a hash or .json/.md)
    # as though it were a user-visible source document.
    return "知识库相关资料"


def source_citation(sources: list[str]) -> str:
    return "来源：" + summarize_sources(sources) if sources else ""


def summarize_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    first = sources[0].replace("\\", "/")
    parts = [part for part in first.split("/") if part]
    course = parts[0] if parts else "知识库资料"
    lectures = []
    for source in sources:
        match = re.search(r"(?:Lec|Lecture)[ _-]*(\d+)", source, re.IGNORECASE)
        if match and match.group(1) not in lectures:
            lectures.append(match.group(1))
    if lectures:
        lesson = "、".join(lectures)
        return f"{course}，第 {lesson} 讲相关资料"
    return f"{course}相关资料"

def append_sources(text: str, sources: list[str]) -> str:
    if not sources:
        return text
    # The LLM may echo a detailed citation. Keep that detail only in the
    # expandable metadata list rendered by the desktop UI.
    text = re.sub(r"\n?来源：[^\n]*(?:\n?来源：[^\n]*)*", "", text).rstrip()
    marker = json.dumps({"summary": summarize_sources(sources), "files": sources}, ensure_ascii=False, separators=(",", ":"))
    return text.rstrip() + "\n\n来源：" + summarize_sources(sources) + f"\n<!--PERSONAL_KB_SOURCES:{marker}-->"


def build_context(results: list[dict], record_sources: dict[str, str]) -> tuple[str, list[str]]:
    excerpts: list[str] = []
    sources: list[str] = []
    used = 0
    for item in results[:3]:
        text = str(item.get("rerank_context", "")).strip()
        if not text:
            continue
        name = source_name(item, record_sources)
        if name not in sources:
            sources.append(name)
        room = max(0, 7000 - used)
        if not room:
            break
        excerpt = text[:room]
        excerpts.append(f"【资料：{name}】\n{excerpt}")
        used += len(excerpt)
    return "\n\n".join(excerpts), sources


def fallback_answer(question: str, context: str, sources: list[str]) -> str:
    if not context:
        return "没有在当前知识库中找到足够相关的已处理资料。"
    lines = []
    for line in context.splitlines():
        cleaned = line.strip().strip('"').strip("#").strip()
        if not cleaned or cleaned.startswith("[") or cleaned.startswith("【资料：") or cleaned.startswith("--- Page"):
            continue
        lines.append(cleaned)
    excerpt = " ".join(lines)
    answer = f"根据已检索到的资料，和“{question}”最相关的内容是：\n\n{excerpt[:900]}"
    return append_sources(answer, sources)


def answer(question: str, index_path: str) -> str:
    backend = _get_backend("http://127.0.0.1:1934", 30)
    retrieval = cmd_hybrid(
        backend,
        question,
        index_path=index_path,
        limit=DEFAULT_LIMIT,
        lexical_limit=20,
        top_k=5,
        k=60,
        dense_weight=1.0,
        lexical_weight=1.0,
        rerank_top=DEFAULT_RERANK_TOP,
        max_sources=DEFAULT_MAX_SOURCES,
    )
    context, sources = build_context(
        list(retrieval.get("results", [])), record_source_lookup(index_path)
    )
    key = load_api_key("DEEPSEEK_API_KEY")
    if not key:
        return fallback_answer(question, context, sources)
    prompt = f"""你是个人知识库的检索助手。只根据下方资料回答问题，不得补充资料中没有的事实。

问题：{question}

资料：
{context}

请用自然、简洁的中文回答。资料不足时明确说明。不要使用 Markdown 标记（例如 **、反引号或代码块）。不要提及 JSON、向量、模型、分数或内部 URI。回答末尾单独给出“来源：”并列出使用到的资料名称。"""
    try:
        text = DeepSeekClient(api_key=key, timeout=90).complete(
            prompt, model="deepseek-chat", max_tokens=1200, temperature=0.1
        ).strip()
        generated = text or fallback_answer(question, context, sources)
        return append_sources(generated, sources)
    except HttpClientError:
        return fallback_answer(question, context, sources)


def main() -> int:
    parser = argparse.ArgumentParser(description="Answer a question from Personal KB sources")
    parser.add_argument("query")
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()
    try:
        print(answer(args.query, args.index_path))
        return 0
    except Exception as error:
        print(f"检索失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
