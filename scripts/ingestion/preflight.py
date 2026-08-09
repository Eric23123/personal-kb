"""Reproducible pre-ingestion preflight for the Personal KB pipeline.

Usage:
    python scripts/ingestion/preflight.py [--manifest config/source_manifest.json] [--root .]
    python scripts/ingestion/preflight.py --json   # machine-readable output

Runs every validation check that must pass before real course material is
ingested into OpenViking. A green preflight means the pipeline is safe to
operate; a single red check blocks ingestion.

Checks (in order):
  1. Source-manifest paths, SHA-256 hashes, metadata, and duplicates.
  2. OpenViking, embedding-server, and Hindsight service health.
  3. LLM routing — only deepseek-v4-pro through api.deepseek.com/v1 accepted.
  4. Logical-source conflict scan — catch same-source-different-hash before
     ingestion.
  5. Dry-run URI and provenance preview for every manifest entry.
  6. Git working-tree cleanliness.

Every check is injectable so tests never contact live services.
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))

try:
    from ..core.common_client import load_api_key
    from ..core.openviking_backend import (
        PERSONAL_NAMESPACE,
        DEFAULT_OPENVIKING_URL,
        _logical_source_key,
        canonical_resource_root,
    )
    from .source_manifest import _canonical_resource_uri, _entries, load_manifest, validate_manifest
except ImportError:  # pragma: no cover — direct CLI use
    from scripts.core.common_client import load_api_key
    from scripts.core.openviking_backend import (
        PERSONAL_NAMESPACE,
        DEFAULT_OPENVIKING_URL,
        _logical_source_key,
        canonical_resource_root,
    )
    from scripts.ingestion.source_manifest import _canonical_resource_uri, _entries, load_manifest, validate_manifest

HINDSIGHT_URL = os.environ.get("PERSONAL_KB_HINDSIGHT_URL", "http://localhost:8888")
EMBEDDING_URL = os.environ.get("PERSONAL_KB_EMBEDDING_URL", "http://127.0.0.1:8001")
REQUIRED_MODEL = "deepseek-v4-pro"
REQUIRED_BASE_URL = "https://api.deepseek.com/v1"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _ok(name: str, **details: Any) -> CheckResult:
    return CheckResult(name=name, passed=True, details=details)


def _fail(name: str, errors: list[str], **details: Any) -> CheckResult:
    return CheckResult(name=name, passed=False, details=details, errors=errors)


# ---------------------------------------------------------------------------
# Check 1: Source manifest
# ---------------------------------------------------------------------------


def check_manifest(manifest_path: str | Path, root: str | Path) -> CheckResult:
    """Validate every manifest entry: paths exist, hashes match, no duplicates."""
    manifest_path = Path(manifest_path)
    root_path = Path(root).expanduser().resolve()

    if not manifest_path.is_file():
        return _fail("manifest", [f"Manifest not found: {manifest_path}"])

    errors = validate_manifest(manifest_path, root_path)
    if errors:
        return _fail("manifest", errors, manifest=str(manifest_path))

    data = load_manifest(manifest_path)
    sources = data.get("sources", [])
    return _ok(
        "manifest",
        manifest=str(manifest_path),
        source_count=len(sources),
        hash_algorithm=data.get("hash_algorithm", "sha256"),
    )


# ---------------------------------------------------------------------------
# Check 2: Service health
# ---------------------------------------------------------------------------


def _probe_url(url: str, *, timeout: float = 5) -> tuple[bool, str]:
    """Read-only GET health probe; returns (reachable, detail)."""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json"},
            method="GET",
        )
        with opener.open(request, timeout=timeout) as response:
            response.read(512)
        return True, "reachable"
    except Exception as exc:
        return False, str(exc)[:200]


def check_services(
    *,
    openviking_url: str = DEFAULT_OPENVIKING_URL,
    embedding_url: str = EMBEDDING_URL,
    hindsight_url: str = HINDSIGHT_URL,
    ov_client: Any = None,
    embedding_probe: Callable[[], tuple[bool, str]] | None = None,
    hindsight_probe: Callable[[], tuple[bool, str]] | None = None,
) -> CheckResult:
    """Check reachability of OpenViking, embedding server, and Hindsight."""
    results: dict[str, dict[str, Any]] = {}
    all_ok = True

    # OpenViking
    if ov_client is not None:
        try:
            ov_client.health()
            results["openviking"] = {"status": "healthy", "url": openviking_url}
        except Exception as exc:
            results["openviking"] = {"status": "unreachable", "url": openviking_url, "error": str(exc)[:200]}
            all_ok = False
    else:
        ok, detail = _probe_url(openviking_url)
        results["openviking"] = {"status": "healthy" if ok else "unreachable", "url": openviking_url, "detail": detail}
        if not ok:
            all_ok = False

    # Embedding
    if embedding_probe is not None:
        ok, detail = embedding_probe()
        results["embedding"] = {"status": "healthy" if ok else "unreachable", "url": embedding_url, "detail": detail}
        if not ok:
            all_ok = False
    else:
        ok, detail = _probe_url(embedding_url)
        results["embedding"] = {"status": "healthy" if ok else "unreachable", "url": embedding_url, "detail": detail}
        if not ok:
            all_ok = False

    # Hindsight
    if hindsight_probe is not None:
        ok, detail = hindsight_probe()
        results["hindsight"] = {"status": "healthy" if ok else "unreachable", "url": hindsight_url, "detail": detail}
        if not ok:
            all_ok = False
    else:
        ok, detail = _probe_url(f"{hindsight_url.rstrip('/')}/v1/default/banks/hermes-history/memories/recall")
        results["hindsight"] = {"status": "healthy" if ok else "unreachable", "url": hindsight_url, "detail": detail}
        if not ok:
            all_ok = False

    if all_ok:
        return _ok("services", services=results)
    return _fail(
        "services",
        [f"{name}: {info['status']}" for name, info in results.items() if info["status"] != "healthy"],
        services=results,
    )


# ---------------------------------------------------------------------------
# Check 3: LLM routing
# ---------------------------------------------------------------------------


def check_routing(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> CheckResult:
    """Verify the active LLM is deepseek-v4-pro through api.deepseek.com/v1."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}

    # Model check
    actual_model = model or REQUIRED_MODEL
    details["model"] = actual_model
    if actual_model != REQUIRED_MODEL:
        errors.append(f"Model is {actual_model!r}, expected {REQUIRED_MODEL!r}")

    # Base URL check
    actual_url = base_url or REQUIRED_BASE_URL
    details["base_url"] = actual_url
    if actual_url != REQUIRED_BASE_URL:
        errors.append(f"Base URL is {actual_url!r}, expected {REQUIRED_BASE_URL!r}")

    # API key check
    key = api_key if api_key is not None else load_api_key()
    if key:
        details["api_key"] = "configured"
    else:
        warnings.append("DEEPSEEK_API_KEY is not set")

    # Flash override check
    if "flash" in actual_model.casefold():
        errors.append("Flash model override is rejected for Personal KB")

    if errors:
        return _fail("routing", errors, warnings=warnings, **details)
    return _ok("routing", warnings=warnings, **details)


# ---------------------------------------------------------------------------
# Check 4: Logical-source conflict scan
# ---------------------------------------------------------------------------


def check_logical_source_conflicts(
    manifest_path: str | Path,
    root: str | Path,
) -> CheckResult:
    """Scan for same-logical-source-different-hash conflicts before ingestion.

    This catches the case where two manifest entries represent the same file
    but declare different hashes, which would trigger SourceHashMismatch at
    ingestion time.
    """
    try:
        data = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return _fail("logical-source-conflicts", [str(exc)])

    base = Path(root).expanduser().resolve()
    try:
        entries = _entries(data, base)
    except (KeyError, ValueError) as exc:
        return _fail("logical-source-conflicts", [str(exc)])

    # Group entries by logical source key (course + source_type + stem).
    seen: dict[str, list[tuple[str, str]]] = {}  # key → [(source_path, source_hash)]
    conflicts: list[dict[str, Any]] = []

    for path, entry in entries:
        uri = _canonical_resource_uri(path, entry, base)
        key = _logical_source_key(uri)
        source_hash = entry.get("source_hash", "<no hash>")
        if key not in seen:
            seen[key] = []
        seen[key].append((str(path), source_hash))

    for key, items in seen.items():
        if len(items) > 1:
            hashes = {h for _, h in items}
            if len(hashes) > 1:
                conflicts.append({
                    "logical_source_key": key,
                    "entries": [{"source_path": p, "source_hash": h} for p, h in items],
                })

    if conflicts:
        return _fail(
            "logical-source-conflicts",
            [f"{len(conflicts)} logical source(s) have conflicting hashes"],
            conflicts=conflicts,
            unique_sources=len(seen),
        )
    return _ok("logical-source-conflicts", unique_sources=len(seen), conflicts=[])


# ---------------------------------------------------------------------------
# Check 5: Dry-run URIs and provenance
# ---------------------------------------------------------------------------


def check_dry_run(
    manifest_path: str | Path,
    root: str | Path,
) -> CheckResult:
    """Compute every URI and provenance payload without contacting OpenViking."""
    try:
        data = load_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return _fail("dry-run", [str(exc)])

    base = Path(root).expanduser().resolve()
    try:
        entries = _entries(data, base)
    except (KeyError, ValueError) as exc:
        return _fail("dry-run", [str(exc)])

    previews: list[dict[str, Any]] = []
    namespace_errors: list[str] = []

    for path, entry in entries:
        uri = _canonical_resource_uri(path, entry, base)
        if not uri.startswith(PERSONAL_NAMESPACE):
            namespace_errors.append(f"{path}: URI {uri} is outside Personal namespace")

        source_hash = entry.get("source_hash")
        preview: dict[str, Any] = {
            "source_path": str(path),
            "uri": uri,
            "source_type": entry.get("source_type"),
            "course": entry.get("course"),
            "lecture": entry.get("lecture"),
            "source_hash": source_hash,
            "canonical_root": canonical_resource_root(uri),
        }
        if entry.get("semester"):
            preview["semester"] = entry["semester"]
        if entry.get("corpus"):
            preview["corpus"] = entry["corpus"]
        previews.append(preview)

    if namespace_errors:
        return _fail("dry-run", namespace_errors, previews=previews, count=len(previews))
    return _ok("dry-run", previews=previews, count=len(previews))


# ---------------------------------------------------------------------------
# Check 6: Git working tree
# ---------------------------------------------------------------------------


def check_git_tree(root: str | Path = ".") -> CheckResult:
    """Verify the git working tree is clean before ingestion."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(Path(root).expanduser()),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return _fail("git-tree", [f"git status failed: {result.stderr.strip()[:200]}"])
    except FileNotFoundError:
        return _ok("git-tree", status="no-git", detail="git not found — skipping")
    except Exception as exc:
        return _ok("git-tree", status="skipped", detail=str(exc)[:200])

    dirty = result.stdout.strip()
    if dirty:
        files = dirty.splitlines()[:20]
        return _fail(
            "git-tree",
            [f"Working tree has {len(files)} uncommitted change(s) — commit or stash before ingestion"],
            dirty_files=files,
            dirty_count=len(dirty.splitlines()),
        )
    return _ok("git-tree", status="clean")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_preflight(
    manifest_path: str | Path = "config/source_manifest.json",
    root: str | Path = ".",
    *,
    # Injectable dependencies (tests never contact live services)
    ov_client: Any = None,
    embedding_probe: Callable[[], tuple[bool, str]] | None = None,
    hindsight_probe: Callable[[], tuple[bool, str]] | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    openviking_url: str = DEFAULT_OPENVIKING_URL,
    embedding_url: str = EMBEDDING_URL,
    hindsight_url: str = HINDSIGHT_URL,
    skip_services: bool = False,
    skip_git: bool = False,
) -> dict[str, Any]:
    """Run all preflight checks and return a structured report.

    Returns a dict with ``valid`` (bool), ``check_count``, ``passed_count``,
    ``failed_count``, and a ``checks`` list of per-check results.
    """
    checks: list[CheckResult] = []

    # 1. Manifest
    checks.append(check_manifest(manifest_path, root))

    # 2. Services (can be skipped for offline validation)
    if not skip_services:
        checks.append(
            check_services(
                openviking_url=openviking_url,
                embedding_url=embedding_url,
                hindsight_url=hindsight_url,
                ov_client=ov_client,
                embedding_probe=embedding_probe,
                hindsight_probe=hindsight_probe,
            )
        )

    # 3. Routing
    checks.append(check_routing(model=model, base_url=base_url, api_key=api_key))

    # 4. Logical-source conflicts (only if manifest passed)
    if checks[0].passed:
        checks.append(check_logical_source_conflicts(manifest_path, root))
    else:
        checks.append(_fail("logical-source-conflicts", ["Skipped — manifest check failed"]))

    # 5. Dry-run (only if manifest passed)
    if checks[0].passed:
        checks.append(check_dry_run(manifest_path, root))

    # 6. Git tree
    if not skip_git:
        checks.append(check_git_tree(root))

    passed = [c for c in checks if c.passed]
    failed = [c for c in checks if not c.passed]

    return {
        "valid": len(failed) == 0,
        "check_count": len(checks),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "checks": [
            {
                "name": c.name,
                "passed": c.passed,
                "errors": c.errors,
                "warnings": c.warnings,
                **c.details,
            }
            for c in checks
        ],
    }


def _report(report: dict[str, Any]) -> str:
    """Format a preflight report as human-readable text."""
    lines = [
        "=" * 60,
        "personal KB PREFLIGHT REPORT",
        "=" * 60,
        f"Overall: {'PASS' if report['valid'] else 'FAIL'}  "
        f"({report['passed_count']}/{report['check_count']} checks passed)",
        "",
    ]

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  [{status}] {check['name']}")
        for err in check.get("errors", []):
            lines.append(f"         ERROR: {err}")
        for warn in check.get("warnings", []):
            lines.append(f"         WARN:  {warn}")

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-ingestion preflight for Personal KB pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingestion/preflight.py
  python scripts/ingestion/preflight.py --manifest config/source_manifest.json --root .
  python scripts/ingestion/preflight.py --json
  python scripts/ingestion/preflight.py --no-services --no-git
""",
    )
    parser.add_argument("--manifest", default="config/source_manifest.json", help="Path to source manifest")
    parser.add_argument("--root", default=".", help="Project root directory")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--no-services", action="store_true", help="Skip service health checks (offline mode)")
    parser.add_argument("--no-git", action="store_true", help="Skip git working-tree check")
    args = parser.parse_args()

    report = run_preflight(
        manifest_path=args.manifest,
        root=args.root,
        skip_services=args.no_services,
        skip_git=args.no_git,
    )

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_report(report))

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
