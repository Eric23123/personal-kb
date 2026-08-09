"""Personal KB source-grounded retrieval CLI — /search, /source, /trace, /hybrid.

Usage:
    python scripts/search.py search "feedback stability"
    python scripts/search.py search "kanban" --limit 5
    python scripts/search.py source "viking://resources/personal-kb/.../SKILL-abc123"
    python scripts/search.py trace "closed-loop transfer function"
    python scripts/search.py hybrid "kanban" --index-path data/lexical_index.json
    python scripts/search.py hybrid-trace "KB 1001" --index-path data/lexical_index.json
    python scripts/search.py health

Requires the OpenViking SDK (openviking_sdk) and a running OpenViking server
(default port 1934). All search/read operations are namespace-locked to
viking://resources/personal-kb.

Results are returned as JSON to stdout for programmatic use.
"""


from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from ..core.openviking_backend import PersonalOpenVikingBackend, PERSONAL_NAMESPACE
    from ..retrieval.hybrid_retrieval import DEFAULT_MAX_SOURCES, DEFAULT_RERANK_TOP, DEFAULT_TOP_K
    from ..retrieval.query_orchestrator import execute_planned_query
    from ..retrieval.source_reader import read_selected_generation_sources
except ImportError:  # pragma: no cover - exercised by direct CLI use
    from scripts.core.openviking_backend import PersonalOpenVikingBackend, PERSONAL_NAMESPACE
    from scripts.retrieval.hybrid_retrieval import DEFAULT_MAX_SOURCES, DEFAULT_RERANK_TOP, DEFAULT_TOP_K
    from scripts.retrieval.query_orchestrator import execute_planned_query
    from scripts.retrieval.source_reader import read_selected_generation_sources

DEFAULT_OPENVIKING_URL = "http://127.0.0.1:1934"
DEFAULT_LIMIT = 8
DEFAULT_LEXICAL_INDEX_PATH = "data/lexical_index.json"


def _get_backend(base_url: str, timeout: float) -> PersonalOpenVikingBackend:
    """Create a backend with the real OpenViking SDK client (lazy-loaded)."""
    return PersonalOpenVikingBackend(
        base_url=base_url,
        timeout=timeout,
        root=Path.cwd(),
    )


def cmd_search(backend: PersonalOpenVikingBackend, query: str, limit: int) -> dict[str, Any]:
    """Search the Personal namespace and return ranked results with abstracts."""
    raw = backend.search(query, limit=limit)
    results = []
    for hit in raw.get("resources", []):
        results.append({
            "uri": hit.get("uri", ""),
            "score": round(hit.get("score", 0.0), 4),
            "level": hit.get("level", 0),
            "abstract": (hit.get("abstract") or "")[:500],
        })
    return {
        "query": query,
        "namespace": PERSONAL_NAMESPACE,
        "total": raw.get("total", len(results)),
        "results": results,
    }


def cmd_source(backend: PersonalOpenVikingBackend, uri: str, read_limit: int) -> dict[str, Any]:
    """Read the full source content at a specific OpenViking URI."""
    content = backend.read(uri, limit=read_limit)
    return {
        "uri": uri,
        "content": content,
    }


def cmd_read_selected(
    backend: PersonalOpenVikingBackend,
    candidates: list[dict[str, Any]],
    selected_ids: list[str],
) -> dict[str, Any]:
    """Read L2 content for model-selected candidate IDs only."""
    return read_selected_generation_sources(backend, candidates, selected_ids)


def cmd_trace(backend: PersonalOpenVikingBackend, query: str, limit: int) -> dict[str, Any]:
    """Search + read: return the search results plus the top result's full source."""
    search_result = cmd_search(backend, query, limit)
    if search_result["results"]:
        top_uri = search_result["results"][0]["uri"]
        source = cmd_source(backend, top_uri, read_limit=10000)
        search_result["top_source"] = {
            "uri": top_uri,
            "content": (source["content"] or "")[:2000],
        }
    return search_result


def cmd_hybrid(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    index_path: str,
    limit: int,
    lexical_limit: int,
    top_k: int,
    k: int,
    dense_weight: float,
    lexical_weight: float,
    rerank_top: int = DEFAULT_RERANK_TOP,
    rerank_server: str = "",
    max_sources: int = DEFAULT_MAX_SOURCES,
) -> dict[str, Any]:
    """Search with dense + BM25 lexical fusion."""
    return backend.hybrid_search(
        query,
        index_path=index_path,
        search_limit=limit,
        lexical_limit=lexical_limit,
        top_k=top_k,
        k=k,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
        rerank_top=rerank_top,
        rerank_server_url=rerank_server,
        max_sources=max_sources,
    )


def cmd_hybrid_trace(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    index_path: str,
    limit: int,
    lexical_limit: int,
    top_k: int,
    k: int,
    dense_weight: float,
    lexical_weight: float,
    read_limit: int,
    rerank_top: int = DEFAULT_RERANK_TOP,
    rerank_server: str = "",
    max_sources: int = DEFAULT_MAX_SOURCES,
) -> dict[str, Any]:
    """Hybrid search + read the top fused result's source."""
    result = cmd_hybrid(
        backend, query,
        index_path=index_path, limit=limit, lexical_limit=lexical_limit,
        top_k=top_k, k=k, dense_weight=dense_weight, lexical_weight=lexical_weight,
        rerank_top=rerank_top, rerank_server=rerank_server, max_sources=max_sources,
    )
    if result["results"]:
        top_uri = result["results"][0]["uri"]
        source = cmd_source(backend, top_uri, read_limit)
        result["top_source"] = {
            "uri": top_uri,
            "content": (source["content"] or "")[:2000],
        }
    return result


def cmd_planned_hybrid(
    backend: PersonalOpenVikingBackend,
    query: str,
    *,
    planner: Any,
    index_path: str = DEFAULT_LEXICAL_INDEX_PATH,
    max_sources: int = DEFAULT_MAX_SOURCES,
    hindsight_recall: Any = None,
) -> dict[str, Any]:
    """Plan with an injected planner, then execute with visible trace.

    ``planner`` must be a ``QueryPlanner`` configured with an LLM callable
    by the caller (typically the personal-kb skill, which injects the active
    Hermes main-model callable). This function does NOT build its own LLM
    client — that would break OAuth providers and duplicate credential logic.
    """
    planning = planner.plan(query)
    return execute_planned_query(
        planning,
        backend,
        index_path=index_path,
        max_sources=max_sources,
        hindsight_recall=hindsight_recall,
    )


def cmd_health(backend: PersonalOpenVikingBackend) -> dict[str, Any]:
    """Check OpenViking server health and namespace accessibility."""
    try:
        # Force lazy client init — will raise if SDK missing or server unreachable.
        client = backend.client
        # SyncHTTPClient.health() returns dict or falsy; bool() it for status.
        raw_health = client.health()
        is_healthy = bool(raw_health) if not isinstance(raw_health, dict) else raw_health.get("healthy", True)
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "url": backend.base_url,
            "namespace": PERSONAL_NAMESPACE,
            "health": raw_health,
        }
    except Exception as exc:
        return {
            "status": "error",
            "url": backend.base_url,
            "error": str(exc),
        }


def main() -> int:
    # Windows may default to a GBK console; the desktop bridge and command line
    # both return UTF-8 JSON because source excerpts can contain any Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        prog="search",
        description="Personal KB source-grounded retrieval via OpenViking",
    )
    parser.add_argument(
        "command",
        choices=["search", "source", "trace", "health", "hybrid", "hybrid-trace", "read-selected"],
        help=(
            "search: query the index | source: read a URI | trace: search+read top hit | "
            "health: check server | hybrid: dense+BM25 fusion | hybrid-trace: hybrid+read top"
        ),
    )
    parser.add_argument("query", nargs="?", help="Search query or source URI")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Max results (search/trace/hybrid)")
    parser.add_argument("--read-limit", type=int, default=20000, help="Max chars to read (source/hybrid-trace)")
    parser.add_argument("--url", default=DEFAULT_OPENVIKING_URL, help="OpenViking server URL")
    parser.add_argument("--timeout", type=float, default=30, help="Request timeout in seconds")
    parser.add_argument(
        "--candidate-manifest", default="",
        help="JSON file containing a generation result manifest (read-selected)",
    )
    parser.add_argument(
        "--selected-ids", default="",
        help="Comma-separated candidate IDs selected for L2 reading (read-selected)",
    )
    parser.add_argument(
        "--index-path", default=DEFAULT_LEXICAL_INDEX_PATH,
        help="Path to the BM25 lexical index (hybrid commands)",
    )
    parser.add_argument(
        "--lexical-limit", type=int, default=20,
        help="Number of lexical candidates to retrieve (hybrid commands)",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="Default final source count (default 5)",
    )
    parser.add_argument(
        "--rrf-k", type=int, default=60,
        help="RRF smoothing constant (hybrid commands)",
    )
    parser.add_argument(
        "--dense-weight", type=float, default=1.0,
        help="Weight for dense ranks in RRF (hybrid commands)",
    )
    parser.add_argument(
        "--lexical-weight", type=float, default=1.0,
        help="Weight for lexical ranks in RRF (hybrid commands)",
    )
    parser.add_argument(
        "--rerank-top", type=int, default=DEFAULT_RERANK_TOP,
        help="Fused candidates to rerank; default 0 (RRF-only); set >0 to opt in",
    )
    parser.add_argument(
        "--rerank-server", default="",
        help="Persistent reranker server host:port",
    )
    parser.add_argument(
        "--max-sources", type=int, default=DEFAULT_MAX_SOURCES,
        help="Hard maximum final source documents (default 8)",
    )
    args = parser.parse_args()

    if args.command in ("search", "source", "trace", "hybrid", "hybrid-trace") and not args.query:
        parser.error(f"{args.command} requires a query or URI argument")
    if args.command == "read-selected" and (not args.candidate_manifest or not args.selected_ids):
        parser.error("read-selected requires --candidate-manifest and --selected-ids")

    backend = _get_backend(args.url, args.timeout)

    if args.command == "health":
        result = cmd_health(backend)
    elif args.command == "search":
        result = cmd_search(backend, args.query, args.limit)
    elif args.command == "source":
        result = cmd_source(backend, args.query, args.read_limit)
    elif args.command == "read-selected":
        manifest_payload = json.loads(Path(args.candidate_manifest).read_text(encoding="utf-8"))
        candidates = manifest_payload.get("results", manifest_payload)
        if not isinstance(candidates, list):
            parser.error("candidate manifest must be a list or an object with a results list")
        selected_ids = [item.strip() for item in args.selected_ids.split(",") if item.strip()]
        result = cmd_read_selected(backend, candidates, selected_ids)
    elif args.command == "trace":
        result = cmd_trace(backend, args.query, args.limit)
    elif args.command == "hybrid":
        result = cmd_hybrid(
            backend, args.query,
            index_path=args.index_path,
            limit=args.limit,
            lexical_limit=args.lexical_limit,
            top_k=args.top_k,
            k=args.rrf_k,
            dense_weight=args.dense_weight,
            lexical_weight=args.lexical_weight,
            rerank_top=args.rerank_top,
            rerank_server=args.rerank_server,
            max_sources=args.max_sources,
        )
    elif args.command == "hybrid-trace":
        result = cmd_hybrid_trace(
            backend, args.query,
            index_path=args.index_path,
            limit=args.limit,
            lexical_limit=args.lexical_limit,
            top_k=args.top_k,
            k=args.rrf_k,
            dense_weight=args.dense_weight,
            lexical_weight=args.lexical_weight,
            read_limit=args.read_limit,
            rerank_top=args.rerank_top,
            rerank_server=args.rerank_server,
            max_sources=args.max_sources,
        )
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
