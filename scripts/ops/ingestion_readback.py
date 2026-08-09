"""OpenViking source-leaf selection and normalization-tolerant readback checks."""

from __future__ import annotations

import re
from typing import Any


def select_readable_source_leaf(nodes: list[dict[str, Any]]) -> str | None:
    """Choose an indexed source leaf, excluding generated metadata leaves."""
    generated_names = {".overview.md", ".abstract.md", ".resource.md"}
    candidates: list[str] = []
    for node in nodes or []:
        if node.get("isDir") or not node.get("uri"):
            continue
        uri = str(node["uri"])
        name = str(node.get("name") or uri.rsplit("/", 1)[-1])
        if name.casefold() in generated_names:
            continue
        candidates.append(uri)
    return candidates[0] if candidates else None


def source_readback_matches(
    expected: str,
    actual: str,
    *,
    window_chars: int = 2000,
) -> bool:
    """Verify source support while tolerating parser escaping and line endings.

    OpenViking may normalize JSON escaping and CRLF/LF boundaries while
    materializing a readable source leaf. This checks several substantive
    tokens from the original artifact instead of requiring byte-identical text.
    """
    token_pattern = re.compile(r"[a-z0-9][a-z0-9_.-]{5,}", re.IGNORECASE)
    expected_tokens = set(token_pattern.findall(expected[:window_chars].casefold()))
    actual_tokens = set(token_pattern.findall(actual[:window_chars].casefold()))
    if not expected_tokens:
        return bool(expected.strip()) and bool(actual.strip())
    overlap = expected_tokens & actual_tokens
    required = min(3, len(expected_tokens))
    return len(overlap) >= required
