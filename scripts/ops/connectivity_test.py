"""Repeatable connectivity test for Personal KB services.

No network calls happen at import time — the default probe only imports
``urllib`` inside the function body.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Endpoint:
    """A named service endpoint to probe without mutating remote state."""

    name: str
    url: str
    method: str = "GET"
    payload: dict[str, Any] | None = None
    expected_status: int = 200


# ---------------------------------------------------------------------------
# Default probe
# ---------------------------------------------------------------------------


def _default_probe(endpoint: Endpoint, timeout: float) -> dict[str, Any]:
    """Probe one endpoint with a no-proxy HTTP request."""
    import urllib.request

    start = time.monotonic()
    try:
        body = None
        headers = {"Accept": "application/json"}
        if endpoint.payload is not None:
            body = json.dumps(endpoint.payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            endpoint.url,
            data=body,
            headers=headers,
            method=endpoint.method.upper(),
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            status = int(resp.status)
            resp.read(256)
        elapsed = round((time.monotonic() - start) * 1000, 2)
        ok = status == endpoint.expected_status
        return {
            "ok": ok,
            "status": status,
            "detail": "healthy" if ok else f"HTTP {status}; expected {endpoint.expected_status}",
            "latency_ms": elapsed,
        }
    except Exception as exc:
        elapsed = round((time.monotonic() - start) * 1000, 2)
        return {
            "ok": False,
            "status": None,
            "detail": str(exc)[:300],
            "latency_ms": elapsed,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_connectivity_test(
    endpoints: list[Endpoint],
    *,
    rounds: int = 3,
    timeout: float = 5.0,
    probe: Callable[[Endpoint, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe every endpoint for *rounds* iterations and return a stability report.

    Parameters
    ----------
    endpoints:
        One or more services to test.
    rounds:
        How many times to probe each endpoint in sequence.
    timeout:
        Seconds per probe (passed to *probe*).
    probe:
        Callable ``(endpoint, timeout) -> dict``.  When *None*, a built-in
        no-proxy HTTP GET is used.

    Returns
    -------
    dict with:
    * ``stable`` — *True* only when every endpoint succeeded in every round
    * ``rounds`` — number of rounds executed
    * ``endpoints`` — per-endpoint stats (``successes``, ``failures``,
      ``last_error``)
    * ``observations`` — list of per-round dicts, each keyed by endpoint name
    """
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if len({endpoint.name for endpoint in endpoints}) != len(endpoints):
        raise ValueError("endpoint names must be unique")
    if probe is None:
        probe = _default_probe

    endpoint_stats: dict[str, dict[str, Any]] = {}
    for ep in endpoints:
        endpoint_stats[ep.name] = {
            "url": ep.url,
            "successes": 0,
            "failures": 0,
            "last_error": None,
            "latencies_ms": [],
        }

    observations: list[dict[str, Any]] = []

    for _ in range(rounds):
        round_obs: dict[str, Any] = {}
        for ep in endpoints:
            result = probe(ep, timeout)
            round_obs[ep.name] = result
            stats = endpoint_stats[ep.name]
            latency = result.get("latency_ms")
            if isinstance(latency, (int, float)):
                stats["latencies_ms"].append(float(latency))
            if result.get("ok"):
                stats["successes"] += 1
            else:
                stats["failures"] += 1
                stats["last_error"] = result.get("detail", "probe failed")
        observations.append(round_obs)

    for stats in endpoint_stats.values():
        latencies = stats.pop("latencies_ms")
        stats["latency_ms"] = {
            "min": round(min(latencies), 2) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        }

    stable = all(stats["failures"] == 0 for stats in endpoint_stats.values())

    return {
        "stable": stable,
        "rounds": rounds,
        "endpoints": endpoint_stats,
        "observations": observations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_LAPTOP_HOST = os.environ.get("PERSONAL_KB_LAPTOP_HOST", "127.0.0.1")
_EMBEDDING_URL = os.environ.get("PERSONAL_KB_EMBEDDING_URL", f"http://{_LAPTOP_HOST}:8001")
_OPENVIKING_URL = os.environ.get("PERSONAL_KB_OPENVIKING_URL", "http://127.0.0.1:1934")
_HINDSIGHT_URL = os.environ.get("PERSONAL_KB_HINDSIGHT_URL", "http://127.0.0.1:8888")
_DEFAULT_ENDPOINTS: list[Endpoint] = [
    Endpoint("embedding", f"{_EMBEDDING_URL.rstrip('/')}/health"),
    Endpoint("openviking", _OPENVIKING_URL),
    Endpoint("hindsight-health", f"{_HINDSIGHT_URL.rstrip('/')}/health"),
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Personal KB connectivity test — probe services and print JSON",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        action="append",
        dest="endpoints",
        metavar="NAME=URL",
        help="Add one endpoint (repeatable).  Overrides built-in defaults.",
    )
    parser.add_argument(
        "--rounds",
        "-n",
        type=int,
        default=3,
        help="Probe rounds per endpoint (default: 3)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=5.0,
        help="Seconds per probe (default: 5.0)",
    )
    parser.add_argument(
        "--defaults",
        action="store_true",
        help="Include sensible laptop / mini-PC defaults even when --endpoint is given.",
    )
    args = parser.parse_args()

    endpoints: list[Endpoint] = []

    # When the user provides explicit endpoints, they replace the defaults
    # unless --defaults is also set.
    if args.endpoints:
        for spec in args.endpoints:
            if "=" not in spec:
                parser.error(f"invalid endpoint spec {spec!r} (expected NAME=URL)")
            name, _, url = spec.partition("=")
            endpoints.append(Endpoint(name=name.strip(), url=url.strip()))
        if args.defaults:
            endpoints = _DEFAULT_ENDPOINTS + endpoints
    else:
        endpoints = list(_DEFAULT_ENDPOINTS)

    report = run_connectivity_test(
        endpoints,
        rounds=args.rounds,
        timeout=args.timeout,
    )

    print(json.dumps(report, indent=2, default=str))
    sys.exit(0 if report["stable"] else 1)
