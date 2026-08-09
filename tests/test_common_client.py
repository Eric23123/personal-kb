import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_json_http_client_retries_transient_failure_with_exponential_backoff():
    from scripts.core.common_client import JsonHttpClient, TransientHttpError

    attempts = []
    delays = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(request, timeout):
        attempts.append((request.full_url, timeout))
        if len(attempts) < 3:
            raise TransientHttpError("temporarily unavailable")
        return Response({"ok": True})

    client = JsonHttpClient(opener=opener, timeout=11, retries=2, backoff_seconds=0.25, sleep=delays.append)

    assert client.post_json("https://example.test/endpoint", {"value": 1}) == {"ok": True}
    assert attempts == [
        ("https://example.test/endpoint", 11),
        ("https://example.test/endpoint", 11),
        ("https://example.test/endpoint", 11),
    ]
    assert delays == [0.25, 0.5]


def test_json_http_client_retries_rate_limit_and_honors_retry_after():
    from email.message import Message
    from urllib.error import HTTPError

    from scripts.core.common_client import JsonHttpClient

    attempts = []
    delays = []

    class Response:
        def read(self):
            return b'{"ok": true}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(request, timeout):
        attempts.append(request.full_url)
        if len(attempts) == 1:
            headers = Message()
            headers["Retry-After"] = "3"
            raise HTTPError(request.full_url, 429, "busy", headers, None)
        return Response()

    client = JsonHttpClient(opener=opener, retries=1, backoff_seconds=0.25, sleep=delays.append)

    assert client.post_json("https://example.test/endpoint", {}) == {"ok": True}
    assert attempts == ["https://example.test/endpoint", "https://example.test/endpoint"]
    assert delays == [3.0]
