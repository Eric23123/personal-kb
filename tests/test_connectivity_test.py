from __future__ import annotations

from scripts.ops.connectivity_test import Endpoint, run_connectivity_test


def test_connectivity_test_records_each_round_and_latency():
    calls = []

    def probe(endpoint, timeout):
        calls.append((endpoint.name, timeout))
        return {"ok": True, "status": 200, "detail": "healthy", "latency_ms": 2.5}

    report = run_connectivity_test(
        [Endpoint("ollama", "http://laptop:11434/api/tags")],
        rounds=3,
        timeout=4.0,
        probe=probe,
    )

    assert report["stable"] is True
    assert report["rounds"] == 3
    assert report["endpoints"]["ollama"]["successes"] == 3
    assert report["endpoints"]["ollama"]["failures"] == 0
    assert len(report["observations"]) == 3
    assert calls == [("ollama", 4.0)] * 3


def test_connectivity_test_fails_stability_when_any_round_fails():
    calls = 0

    def probe(endpoint, timeout):
        nonlocal calls
        calls += 1
        return {"ok": calls != 2, "status": 200 if calls != 2 else None, "detail": "ok" if calls != 2 else "timeout", "latency_ms": None}

    report = run_connectivity_test(
        [Endpoint("embedding", "http://laptop:18002")],
        rounds=3,
        timeout=1.0,
        probe=probe,
    )

    assert report["stable"] is False
    assert report["endpoints"]["embedding"]["successes"] == 2
    assert report["endpoints"]["embedding"]["failures"] == 1
    assert report["endpoints"]["embedding"]["last_error"] == "timeout"
