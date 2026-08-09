"""Tests for the DeepSeek V4 Pro routing hardening in note_generator.

Personal-KB note generation must use a single text LLM route: DeepSeek V4 Pro
through the official DeepSeek API. These tests verify the openai/ollama
text-generation backends have been removed and that ``call_llm`` routes through
``common_client.DeepSeekClient`` with the canonical model/endpoint/credential.

No test contacts a real provider: a fake ``deepseek_client`` is injected so the
DeepSeek API key and network are never required.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import scripts.notes.note_generator as ng


# ── Backend allowlist ───────────────────────────────────────────────────────

def test_only_deepseek_backend_is_allowed():
    """The multi-backend dict is gone; only 'deepseek' remains permitted."""
    assert ng.ALLOWED_BACKENDS == {"deepseek"}
    assert ng.DEFAULT_BACKEND == "deepseek"
    # The old BACKENDS dict (deepseek/openai/ollama) must no longer exist.
    assert not hasattr(ng, "BACKENDS")


def test_removed_helpers_are_gone():
    """The OpenAI-compat and Ollama helper functions have been deleted."""
    assert not hasattr(ng, "_call_openai_compat")
    assert not hasattr(ng, "_call_ollama")


def test_canonical_deepseek_constants():
    assert ng.DEEPSEEK_API_URL == "https://api.deepseek.com/v1/chat/completions"
    assert ng.DEFAULT_MODEL == "deepseek-v4-pro"


# ── call_llm routing ────────────────────────────────────────────────────────

class _FakeDeepSeekClient:
    """Records the call and returns canned text without any network/key."""

    def __init__(self, *args, **kwargs):
        self.calls = []

    def complete(self, prompt, *, model, max_tokens, temperature):
        self.calls.append({
            "prompt": prompt, "model": model,
            "max_tokens": max_tokens, "temperature": temperature,
        })
        return "synthesized note body"


def test_call_llm_uses_deepseek_v4_pro_defaults_through_deepseek_client():
    fake = _FakeDeepSeekClient()
    # api_key is provided so the loader is never consulted.
    result = ng.call_llm(
        "generate notes", api_key="test-key", deepseek_client=fake,
    )

    assert result == "synthesized note body"
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["model"] == "deepseek-v4-pro"
    assert call["max_tokens"] == 8192
    assert call["temperature"] == 0.3
    assert call["prompt"] == "generate notes"


def test_call_llm_injects_canonical_endpoint_and_key_into_deepseek_client(monkeypatch):
    """When no client is injected, a real DeepSeekClient is built with the
    canonical URL and key, but we stub the class to avoid network/exit."""
    constructed = {}

    class StubClient:
        def __init__(self, *, api_key, api_url, http_client, timeout):
            constructed["api_key"] = api_key
            constructed["api_url"] = api_url
            constructed["timeout"] = timeout

        def complete(self, prompt, *, model, max_tokens, temperature):
            constructed["model"] = model
            return "ok"

    monkeypatch.setattr(ng, "DeepSeekClient", StubClient)
    result = ng.call_llm("prompt", api_key="my-key")

    assert result == "ok"
    assert constructed["api_url"] == "https://api.deepseek.com/v1/chat/completions"
    assert constructed["api_key"] == "my-key"
    assert constructed["model"] == "deepseek-v4-pro"
    assert constructed["timeout"] == 600


def test_call_llm_rejects_removed_backends():
    """Stale callers passing 'openai' or 'ollama' get a clear ValueError."""
    fake = _FakeDeepSeekClient()
    for removed in ("openai", "ollama"):
        try:
            ng.call_llm("p", backend=removed, api_key="k", deepseek_client=fake)
        except ValueError as exc:
            assert "deepseek" in str(exc).lower()
            assert removed in str(exc)
        else:
            raise AssertionError(f"backend {removed!r} should have been rejected")


def test_call_llm_ollama_url_parameter_is_accepted_but_ignored():
    """The ollama_url kwarg remains in the signature for backward compat but
    must not route to Ollama — it still hits the DeepSeek client."""
    fake = _FakeDeepSeekClient()
    result = ng.call_llm(
        "p", ollama_url="http://somewhere:11434/api/generate",
        api_key="k", deepseek_client=fake,
    )
    assert result == "synthesized note body"
    assert len(fake.calls) == 1


def test_call_llm_rejects_flash_model_override():
    """All Personal text generation is pinned to DeepSeek V4 Pro."""
    fake = _FakeDeepSeekClient()
    try:
        ng.call_llm("p", model="deepseek-v4-flash", api_key="k", deepseek_client=fake)
    except ValueError as exc:
        assert "deepseek-v4-pro" in str(exc)
    else:
        raise AssertionError("deepseek-v4-flash override should be rejected")


# ── CLI surface ─────────────────────────────────────────────────────────────

def test_cli_only_offers_deepseek_backend_and_deprecated_ollama_url():
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(ng.__file__)), "--help"],
        text=True, capture_output=True, check=True,
    )
    out = result.stdout
    # Only deepseek is a valid --backend choice.
    assert "'deepseek'" in out or "deepseek" in out
    assert "openai" not in out.lower().replace("'deepseek'", "")  # no openai mention
    # --ollama-url is retained but marked deprecated/ignored.
    assert "--ollama-url" in out
    assert "deprecated" in out.lower() or "ignored" in out.lower()
    # Default model is surfaced.
    assert "deepseek-v4-pro" in out
    # Default endpoint is surfaced.
    assert "api.deepseek.com/v1/chat/completions" in out


def test_cli_rejects_openai_backend_choice():
    """argparse choices enforce the allowlist before call_llm is reached."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(Path(ng.__file__)),
         "--backend", "openai", "--course", "X", "--lecture", "1", "--facts", "x.json"],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "openai" in (result.stderr + result.stdout).lower()


def test_default_note_output_path_targets_sync_staging_tree():
    path = ng.default_note_output_path("PERSONAL-ALPHA", 3)
    assert path.endswith("courses\\personal-alpha\\derived\\notes\\Lecture03.md")
