"""Minimal live Hindsight REST adapter for the isolated e2e gate.

This module is deliberately limited to banks whose names are supplied by the
isolated gate. It never targets the production bank implicitly.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


class LiveHindsightClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0) -> None:
        self.base_url = (base_url or os.environ.get("PERSONAL_KB_HINDSIGHT_URL", "http://127.0.0.1:8888")).rstrip("/")
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else {}

    def post_json(self, url: str, payload: dict[str, Any], **kwargs: Any) -> Any:
        prefix = self.base_url
        path = url[len(prefix):] if url.startswith(prefix) else url
        return self._request("POST", path, payload)

    def create_bank(self, bank: str, name: str) -> Any:
        return self._request("PUT", f"/v1/default/banks/{bank}", {"name": name})

    def delete_bank(self, bank: str) -> Any:
        return self._request("DELETE", f"/v1/default/banks/{bank}")

    def retain(self, bank: str, items: list[dict[str, Any]]) -> Any:
        return self._request(
            "POST",
            f"/v1/default/banks/{bank}/memories",
            {"items": items, "async": False},
        )

    def recall(
        self,
        bank: str,
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
    ) -> Any:
        payload: dict[str, Any] = {
            "query": query,
            "max_tokens": 256,
        }
        if tags:
            payload["tags"] = tags
            payload["tags_match"] = "all_strict"
        return self._request("POST", f"/v1/default/banks/{bank}/memories/recall", payload)

    def count_bank(self, bank: str) -> int:
        data = self._request("GET", f"/v1/default/banks/{bank}/stats")
        for key in ("fact_count", "memory_count", "count", "total"):
            value = data.get(key)
            if isinstance(value, int):
                return value
        nested = data.get("stats")
        if isinstance(nested, dict):
            for key in ("fact_count", "memory_count", "count", "total"):
                value = nested.get(key)
                if isinstance(value, int):
                    return value
        return 0

    def cleanup_stale_e2e_banks(self) -> None:
        data = self._request("GET", "/v1/default/banks")
        for bank in data.get("banks", []) if isinstance(data, dict) else []:
            bank_id = str(bank.get("bank_id", ""))
            if bank_id.startswith("hermes-e2e-gate-"):
                try:
                    self.delete_bank(bank_id)
                except Exception:
                    pass
