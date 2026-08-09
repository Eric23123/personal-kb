from __future__ import annotations

import pytest

from scripts.core.model_availability import _ollama_command


def test_ollama_command_defaults_to_local_executable():
    assert _ollama_command(None, "ollama") == ["ollama", "serve"]


def test_ollama_command_builds_explicit_ssh_command():
    assert _ollama_command("study@worker", "/opt/ollama") == [
        "ssh",
        "-tt",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
        "study@worker",
        'set OLLAMA_HOST=0.0.0.0&&"/opt/ollama" serve',
    ]


def test_ollama_command_rejects_empty_executable():
    with pytest.raises(ValueError, match="non-empty"):
        _ollama_command(None, "   ")
