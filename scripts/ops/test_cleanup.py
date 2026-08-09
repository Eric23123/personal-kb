"""Systematic test resource cleanup for Personal KB.

Removes test resources from OpenViking and Hindsight based on manifest
entries, URI patterns, or explicit URI lists. Designed for safe,
auditable cleanup with a dry-run mode.

Usage:
    # Dry-run: show what would be removed without touching anything.
    python scripts/ops/test_cleanup.py --manifest config/source_manifest.json --dry-run

    # Remove specific test URIs.
    python scripts/ops/test_cleanup.py --uris viking://.../test1 viking://.../test2

    # Remove all resources matching a pattern.
    python scripts/ops/test_cleanup.py --pattern "E2E-TEST"

    # Remove all resources with corpus=test in the manifest.
    python scripts/ops/test_cleanup.py --test-only --manifest config/source_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from scripts.core.openviking_backend import PERSONAL_NAMESPACE, canonical_resource_root


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


class CleanupClient(Protocol):
    """Subset of OpenViking client needed for cleanup."""

    def rm(self, uri: str, **kwargs: Any) -> Any: ...

    def ls(self, uri: str, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class CleanupReport:
    total_found: int = 0
    removed: int = 0
    failed: int = 0
    dry_run: bool = False
    details: list[dict[str, Any]] = field(default_factory=list)

    def add_removed(self, uri: str) -> None:
        self.removed += 1
        self.details.append({"uri": uri, "action": "removed"})

    def add_skipped(self, uri: str, reason: str) -> None:
        self.details.append({"uri": uri, "action": "skipped", "reason": reason})

    def add_failed(self, uri: str, error: str) -> None:
        self.failed += 1
        self.details.append({"uri": uri, "action": "failed", "error": error})

    def add_dry_run(self, uri: str) -> None:
        self.details.append({"uri": uri, "action": "would_remove"})


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def _collect_from_manifest(
    manifest_path: str | Path,
    *,
    test_only: bool = False,
) -> list[str]:
    """Collect URIs from a manifest.

    When ``test_only=True``, only entries with ``corpus=test`` are returned.
    """
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sources = data.get("sources", [])
    uris: list[str] = []
    for entry in sources:
        if test_only and entry.get("corpus") != "test":
            continue
        # Compute the URI using resource_uri so it matches what was ingested.
        from scripts.ingestion.source_manifest import _canonical_resource_uri
        from pathlib import Path as P

        root = P(".").resolve()
        path = P(str(entry["source_path"]))
        if not path.is_absolute():
            # Assume relative to project root.
            path = _PROJECT_ROOT / path
        try:
            uri = _canonical_resource_uri(path, entry, _PROJECT_ROOT)
            uris.append(canonical_resource_root(uri))
        except Exception:
            # Fall back to a pattern match based on course/hash.
            pass
    return sorted(set(uris))


def _collect_from_namespace(
    client: CleanupClient,
    pattern: str,
    namespace: str = PERSONAL_NAMESPACE,
) -> list[str]:
    """Collect URIs containing a substring pattern."""
    try:
        tree = client.ls(namespace, recursive=True, node_limit=5000)
    except Exception:
        return []

    uris: list[str] = []
    for item in (tree or []):
        uri = str(item.get("uri", ""))
        if item.get("isDir") or not uri:
            continue
        if pattern in uri:
            uris.append(uri)
    return sorted(uris)


def run_cleanup(
    client: CleanupClient,
    *,
    uris: list[str] | None = None,
    pattern: str | None = None,
    manifest_path: str | Path | None = None,
    test_only: bool = False,
    dry_run: bool = True,
) -> CleanupReport:
    """Remove test resources from OpenViking.

    Resources are found by explicit URIs, pattern matching, or manifest
    entries.  When ``dry_run=True``, nothing is actually removed.
    """
    report = CleanupReport(dry_run=dry_run)

    # Collect targets.
    targets: set[str] = set()

    if uris:
        targets.update(uris)

    if pattern:
        found = _collect_from_namespace(client, pattern)
        targets.update(found)

    if manifest_path:
        man_uris = _collect_from_manifest(manifest_path, test_only=test_only)
        # For manifest URIs, we also collect child entries by listing.
        for mu in man_uris:
            targets.add(mu)
            try:
                children = client.ls(mu, recursive=True, node_limit=500)
                for item in (children or []):
                    cu = str(item.get("uri", ""))
                    if cu and not item.get("isDir"):
                        targets.add(cu)
            except Exception:
                pass

    report.total_found = len(targets)

    if dry_run:
        for u in sorted(targets):
            report.add_dry_run(u)
        return report

    # Do the actual removal.
    for uri in sorted(targets):
        try:
            client.rm(uri, recursive=True, wait=True)
            report.add_removed(uri)
        except Exception as exc:
            report.add_failed(uri, str(exc)[:200])

    return report


# ---------------------------------------------------------------------------
# Hindsight cleanup (injectable)
# ---------------------------------------------------------------------------


def cleanup_hindsight_bank(
    hs_client: Any,
    bank: str,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Remove an isolated Hindsight test bank.

    Returns a dict with ``bank``, ``removed`` (bool), and ``error`` (str or None).
    """
    result: dict[str, Any] = {"bank": bank, "removed": False, "error": None}

    if dry_run:
        result["dry_run"] = True
        return result

    try:
        hs_client.delete_bank(bank)
        result["removed"] = True
    except Exception as exc:
        result["error"] = str(exc)[:500]

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Test resource cleanup for Personal KB")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Preview only (default)")
    parser.add_argument("--execute", dest="dry_run", action="store_false", help="Actually remove resources")
    parser.add_argument("--uris", nargs="*", help="Specific URIs to remove")
    parser.add_argument("--pattern", help="Substring pattern to match")
    parser.add_argument("--manifest", help="Manifest to derive URIs from")
    parser.add_argument("--test-only", action="store_true", help="Only corpus=test entries")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if not any([args.uris, args.pattern, args.manifest]):
        parser.error("At least one of --uris, --pattern, or --manifest is required")

    from openviking_sdk import SyncHTTPClient

    client = SyncHTTPClient(url="http://127.0.0.1:1934", timeout=300)
    client.initialize()

    report = run_cleanup(
        client,
        uris=args.uris,
        pattern=args.pattern,
        manifest_path=args.manifest,
        test_only=args.test_only,
        dry_run=args.dry_run,
    )

    if args.json:
        out = {
            "total_found": report.total_found,
            "removed": report.removed,
            "failed": report.failed,
            "dry_run": report.dry_run,
            "details": report.details,
        }
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        mode = "DRY RUN" if report.dry_run else "LIVE"
        print(f"Personal KB Cleanup — {mode}")
        print(f"  Found:   {report.total_found}")
        print(f"  Removed: {report.removed}")
        print(f"  Failed:  {report.failed}")
        for d in report.details:
            action = d["action"].upper()
            uri = d["uri"]
            if "error" in d:
                print(f"  [{action}] {uri} — ERROR: {d['error'][:100]}")
            elif "reason" in d:
                print(f"  [{action}] {uri} — {d['reason']}")
            else:
                print(f"  [{action}] {uri}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
