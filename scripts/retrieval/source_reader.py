"""Manifest-bound L2 reads for model-selected Personal sources."""

from __future__ import annotations

from typing import Any, Iterable

import tiktoken

from scripts.core.openviking_backend import PERSONAL_NAMESPACE, PersonalOpenVikingBackend

L2_TOKENIZER_NAME = "o200k_base"
L2_TOKENIZER = tiktoken.get_encoding(L2_TOKENIZER_NAME)
DEFAULT_MAX_L2_TOKENS = 150_000
DEFAULT_MAX_CHUNKS_PER_SOURCE = 50
DEFAULT_MAX_CHARS_PER_SOURCE = 600_000


def _token_ids(text: str) -> list[int]:
    return L2_TOKENIZER.encode(text, disallowed_special=())


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, int, bool]:
    if max_tokens <= 0:
        return "", 0, bool(text)
    ids = _token_ids(text)
    if len(ids) <= max_tokens:
        return text, len(ids), False
    truncated = L2_TOKENIZER.decode(ids[:max_tokens])
    return truncated, max_tokens, True


def read_selected_generation_sources(
    backend: PersonalOpenVikingBackend,
    candidates: Iterable[dict[str, Any]],
    selected_candidate_ids: Iterable[str],
    *,
    max_chunks_per_source: int = DEFAULT_MAX_CHUNKS_PER_SOURCE,
    max_chars_per_source: int = DEFAULT_MAX_CHARS_PER_SOURCE,
    max_total_tokens: int = DEFAULT_MAX_L2_TOKENS,
) -> dict[str, Any]:
    """Read L2 chunks selected by manifest IDs under a total token budget.

    The model supplies IDs, never arbitrary URIs. Canonical URIs are resolved
    from the already validated manifest, overview resources are excluded from
    the L2 payload, and the combined selected-source content is capped at
    ``max_total_tokens`` using the configured tokenizer.
    """
    if max_chunks_per_source < 1 or max_chars_per_source < 1 or max_total_tokens < 1:
        raise ValueError("L2 read limits must be positive")
    candidate_map = {
        str(candidate.get("candidate_id")): candidate
        for candidate in candidates
        if candidate.get("candidate_id")
    }
    selected_ids = list(dict.fromkeys(str(item) for item in selected_candidate_ids))
    result: dict[str, Any] = {
        "selected_candidate_ids": selected_ids,
        "sources": [],
        "errors": [],
        "warnings": [],
        "read_mode": "selected_canonical_l2",
        "l2_tokenizer": L2_TOKENIZER_NAME,
        "max_total_tokens": max_total_tokens,
        "l2_content_tokens": 0,
        "truncated": False,
    }

    for candidate_id in selected_ids:
        candidate = candidate_map.get(candidate_id)
        if candidate is None:
            result["errors"].append({
                "candidate_id": candidate_id,
                "error": "candidate_id was not present in the validated manifest",
            })
            continue
        canonical_uri = str(candidate.get("canonical_uri", ""))
        if not canonical_uri.startswith(PERSONAL_NAMESPACE):
            result["errors"].append({
                "candidate_id": candidate_id,
                "error": "candidate canonical URI is outside the Personal namespace",
            })
            continue
        if result["l2_content_tokens"] >= max_total_tokens:
            result["truncated"] = True
            result["warnings"].append({
                "candidate_id": candidate_id,
                "canonical_uri": canonical_uri,
                "warning": "L2 token budget exhausted before this selected source",
            })
            continue
        try:
            nodes = backend.client.ls(
                canonical_uri,
                recursive=True,
                node_limit=max_chunks_per_source * 2,
            ) or []
            l2_nodes = [
                node for node in nodes
                if not node.get("isDir")
                and str(node.get("uri", ""))
                and not str(node.get("uri", "")).endswith("/.overview.md")
            ][:max_chunks_per_source]
            if not l2_nodes:
                raise RuntimeError("no L2 child resources were found under canonical URI")

            chunks: list[str] = []
            l2_uris: list[str] = []
            used_chars = 0
            source_tokens = 0
            source_truncated = False
            for node in l2_nodes:
                remaining_chars = max_chars_per_source - used_chars
                if remaining_chars <= 0:
                    source_truncated = True
                    break
                uri = str(node["uri"])
                raw = backend.read(uri, limit=remaining_chars)
                text = raw if isinstance(raw, str) else str(raw)
                if len(text) > remaining_chars:
                    text = text[:remaining_chars]
                    source_truncated = True
                remaining_tokens = max_total_tokens - result["l2_content_tokens"]
                text, token_count, was_truncated = _truncate_to_tokens(text, remaining_tokens)
                if not text:
                    source_truncated = source_truncated or bool(raw)
                    break
                chunks.append(f"### {node.get('name', uri.rsplit('/', 1)[-1])}\n{text}")
                l2_uris.append(uri)
                used_chars += len(text)
                source_tokens += token_count
                result["l2_content_tokens"] += token_count
                source_truncated = source_truncated or was_truncated
                if was_truncated or result["l2_content_tokens"] >= max_total_tokens:
                    result["truncated"] = True
                    break

            result["sources"].append({
                "candidate_id": candidate_id,
                "canonical_uri": canonical_uri,
                "resource_uri": candidate.get("resource_uri", ""),
                "metadata": candidate.get("metadata", {}),
                "l2_uris": l2_uris,
                "content": "\n\n".join(chunks),
                "content_chars": used_chars,
                "content_tokens": source_tokens,
                "truncated": source_truncated,
            })
        except Exception as error:
            result["errors"].append({
                "candidate_id": candidate_id,
                "canonical_uri": canonical_uri,
                "error": repr(error),
            })

    return result


__all__ = [
    "DEFAULT_MAX_L2_TOKENS",
    "L2_TOKENIZER_NAME",
    "read_selected_generation_sources",
]
