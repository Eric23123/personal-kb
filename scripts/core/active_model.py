"""Provider-neutral adapter for the currently active Hermes agent model."""
from __future__ import annotations

from typing import Any


def _content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if content:
            return str(content)
    output = getattr(response, "output_text", None)
    if output:
        return str(output)
    raise RuntimeError("Active model response contained no text content")


def call_active_model(agent: Any, prompt: str, *, max_tokens: int = 4096) -> str:
    """Call the existing agent client; never create credentials or a client.

    Supports the normal OpenAI-compatible client used by API-key and OAuth
    routes. Codex app-server and other transports fail explicitly so the
    caller can show a visible fallback rather than silently using another model.
    """
    client = getattr(agent, "client", None)
    if client is None:
        raise RuntimeError("Active Hermes transport does not expose a reusable client")
    model = getattr(agent, "model", None)
    chat = getattr(getattr(client, "chat", None), "completions", None)
    create = getattr(chat, "create", None)
    if create is not None:
        response = create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        return _content(response)
    responses = getattr(client, "responses", None)
    create = getattr(responses, "create", None)
    if create is not None:
        response = create(model=model, input=prompt, max_output_tokens=max_tokens)
        return _content(response)
    raise RuntimeError("Active Hermes transport has no supported text-completion method")


__all__ = ["call_active_model"]
