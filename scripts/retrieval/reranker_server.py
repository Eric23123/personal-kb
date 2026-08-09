"""Persistent reranker server — loads the cross-encoder once and serves requests.

Avoids the ~19s model load on every CLI call. Once the server is running,
reranking takes <1s per query.

Usage:
    # Start the server (loads model once, stays running)
    python scripts/retrieval/reranker_server.py --port 1940

    # Then use reranker.py with --server flag (no model load overhead)
    python scripts/retrieval/reranker.py search "kanban" --limit 10 --rerank-top 5 --server localhost:1940

Health check:
    curl http://localhost:1940/health

Rerank request:
    POST http://localhost:1940/rerank
    {"query": "kanban pull system", "candidates": [{"abstract": "...", ...}, ...], "top_k": 5}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

try:
    from ..retrieval.reranker import document_family_priority, infer_document_metadata
except ImportError:  # pragma: no cover
    from scripts.retrieval.reranker import document_family_priority, infer_document_metadata

# Suppress HF symlink warnings on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DEFAULT_MODEL = "BAAI/bge-reranker-base"
DEFAULT_PORT = 1940
DEFAULT_HOST = "127.0.0.1"

# Module-level model cache (loaded once at startup).
_model: Any = None
_model_name: str | None = None


def _get_model(model_name: str = DEFAULT_MODEL) -> Any:
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder
        print(f"Loading reranker model: {model_name} ...", flush=True)
        start = time.monotonic()
        _model = CrossEncoder(model_name)
        _model_name = model_name
        elapsed = time.monotonic() - start
        print(f"Model loaded in {elapsed:.1f}s", flush=True)
    return _model


def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Rerank candidates using the loaded cross-encoder."""
    if not candidates or len(candidates) <= top_k:
        return candidates

    model = _get_model()
    pairs = [
        (query, c.get("rerank_context") or c.get("abstract", "") or c.get("uri", ""))
        for c in candidates
    ]
    scores = model.predict(pairs)

    raw_scores = [float(score) for score in scores]
    max_raw = max(raw_scores, default=0.0)
    for candidate, score in zip(candidates, raw_scores):
        metadata = candidate.get("source_metadata") or infer_document_metadata(
            candidate, candidate.get("rerank_context", "")
        )
        candidate["source_metadata"] = metadata
        candidate["rerank_score_raw"] = score
        candidate["family_priority"] = document_family_priority(query, metadata)
        adjusted = score
        if max_raw - score <= 0.05:
            if candidate["family_priority"] > 0:
                adjusted += 0.05
            elif candidate["family_priority"] < 0:
                adjusted -= 0.02
        candidate["rerank_score"] = adjusted
    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]


def _handle_health() -> bytes:
    return json.dumps({
        "status": "healthy",
        "model": _model_name or "not_loaded",
        "port": DEFAULT_PORT,
    }).encode()


def _handle_rerank(body: bytes) -> bytes:
    try:
        data = json.loads(body)
        query = data["query"]
        candidates = data["candidates"]
        top_k = data.get("top_k", 5)
    except (json.JSONDecodeError, KeyError) as exc:
        return json.dumps({"error": f"bad request: {exc}"}).encode()

    try:
        start = time.monotonic()
        reranked = _rerank(query, candidates, top_k)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        has_scores = any("rerank_score" in c for c in reranked)
        return json.dumps({
            "results": reranked,
            "mode": "dense_plus_reranker" if has_scores else "dense_fallback",
            "latency_ms": elapsed_ms,
        }, default=str, ensure_ascii=False).encode()
    except Exception as exc:
        # Fallback to dense ranking
        fallback = sorted(candidates, key=lambda c: c.get("score", 0.0), reverse=True)[:top_k]
        return json.dumps({
            "results": fallback,
            "mode": "dense_fallback",
            "error": str(exc),
        }, default=str, ensure_ascii=False).encode()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="reranker_server",
        description="Persistent cross-encoder reranker server for Personal KB",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    # Pre-load the model at startup
    _get_model(args.model)

    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_handle_health())
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if self.path == "/rerank":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(_handle_rerank(body))
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            # Suppress default request logging; print compact instead
            pass

    server = HTTPServer((args.host, args.port), Handler)
    print(f"Reranker server listening on http://{args.host}:{args.port}", flush=True)
    print(f"  GET  /health  — health check", flush=True)
    print(f"  POST /rerank  — rerank candidates", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())