"""Shared HTTP and LLM clients for Personal KB pipeline scripts.

The module deliberately uses only the Python standard library.  A single client
instance can be passed through an operation to reuse its configured opener and
retry policy, while tests can inject an in-memory opener.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

def _default_env_path() -> Path:
    """Resolve credentials without embedding a machine-specific filesystem path."""
    explicit_path = os.environ.get("PERSONAL_KB_ENV_FILE")
    if explicit_path:
        return Path(explicit_path)

    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home) / ".env"

    return Path.home() / ".config" / "personal-kb" / ".env"


DEFAULT_ENV_PATH = _default_env_path()


class HttpClientError(RuntimeError):
    """A clear failure returned by an HTTP API client."""


class TransientHttpError(HttpClientError):
    """An error that may succeed on retry."""


def load_api_key(key_name: str = "DEEPSEEK_API_KEY", env_path: str | Path | None = None) -> str | None:
    """Return an API key from the environment, then a simple KEY=value file.

    The parser ignores unrelated .env lines, comments, and optional surrounding
    quotes so Hermes' mixed-purpose .env file is safe to read without sourcing.
    """
    value = os.environ.get(key_name)
    if value:
        return value.strip()

    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip().removeprefix("export ").strip() == key_name:
                return candidate.strip().strip('"').strip("'") or None
    except FileNotFoundError:
        pass
    return None


class JsonHttpClient:
    """JSON POST client with a reusable opener, timeout, and exponential retry."""

    def __init__(
        self,
        *,
        opener: Callable[..., Any] | Any | None = None,
        timeout: float = 30,
        retries: int = 2,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds cannot be negative")
        self._opener = opener or urllib.request.build_opener()
        self.timeout = timeout
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> Any:
        request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            method="POST",
        )
        request_timeout = self.timeout if timeout is None else timeout
        attempts = self.retries if retries is None else retries
        if request_timeout <= 0:
            raise ValueError("timeout must be positive")
        if attempts < 0:
            raise ValueError("retries cannot be negative")

        for attempt in range(attempts + 1):
            retry_delay = self.backoff_seconds * (2**attempt)
            try:
                response = self._open(request, request_timeout)
                with response:
                    raw_body = response.read()
                try:
                    return json.loads(raw_body.decode("utf-8") if isinstance(raw_body, bytes) else raw_body)
                except (TypeError, json.JSONDecodeError) as error:
                    raise HttpClientError(f"Invalid JSON response from {url}: {error}") from error
            except TransientHttpError:
                if attempt == attempts:
                    raise
            except urllib.error.HTTPError as error:
                if error.code == 429 or 500 <= error.code < 600:
                    if attempt < attempts:
                        retry_delay = self._retry_after_delay(error, attempt)
                    else:
                        detail = error.read().decode("utf-8", errors="replace")[:500]
                        raise HttpClientError(f"HTTP {error.code} from {url}: {detail}") from error
                else:
                    detail = error.read().decode("utf-8", errors="replace")[:500]
                    raise HttpClientError(f"HTTP {error.code} from {url}: {detail}") from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                if attempt == attempts:
                    raise TransientHttpError(f"Request to {url} failed after {attempt + 1} attempts: {error}") from error
            if attempt < attempts:
                self._sleep(retry_delay)

        raise AssertionError("unreachable")

    def _retry_after_delay(self, error: urllib.error.HTTPError, attempt: int) -> float:
        """Return a bounded retry delay, honoring numeric Retry-After values."""
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after is not None:
            try:
                delay = float(retry_after)
                if delay >= 0:
                    return delay
            except (TypeError, ValueError):
                pass
        return self.backoff_seconds * (2**attempt)

    def _open(self, request: urllib.request.Request, timeout: float) -> Any:
        open_method = getattr(self._opener, "open", None)
        if open_method is not None:
            return open_method(request, timeout=timeout)
        return self._opener(request, timeout)


class DeepSeekClient:
    """Small OpenAI-compatible client for the DeepSeek chat completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str = "https://api.deepseek.com/v1/chat/completions",
        *,
        http_client: JsonHttpClient | None = None,
        timeout: float = 600,
    ) -> None:
        self.api_key = api_key or load_api_key()
        self.api_url = api_url
        self.http_client = http_client or JsonHttpClient(timeout=timeout)
        self.timeout = timeout

    def complete(self, prompt: str, *, model: str, max_tokens: int = 8192, temperature: float = 0.1) -> str:
        if not self.api_key:
            raise HttpClientError(
                "DEEPSEEK_API_KEY is not set. Set it in the environment or the configured Personal-KB env file."
            )
        response = self.http_client.post_json(
            self.api_url,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise HttpClientError("DeepSeek response did not contain choices[0].message.content") from error
        if not isinstance(content, str):
            raise HttpClientError("DeepSeek response content was not text")
        return content
