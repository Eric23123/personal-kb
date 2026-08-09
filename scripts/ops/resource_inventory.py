"""Resource inventory for Personal KB — snapshot OpenViking before/after ingestion.

Usage:
    python scripts/ops/resource_inventory.py snapshot --output inventory_before.json
    python scripts/ops/resource_inventory.py diff before.json after.json

All OpenViking interactions are injectable so tests never contact live services.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from scripts.core.openviking_backend import PERSONAL_NAMESPACE, canonical_resource_root


# ---------------------------------------------------------------------------
# Client protocol
# ---------------------------------------------------------------------------


class InventoryClient(Protocol):
    """Subset of OpenViking client needed for inventory snapshots."""

    def ls(self, uri: str, **kwargs: Any) -> Any: ...

    def read(self, uri: str, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ResourceEntry:
    uri: str
    canonical_root: str
    size_hint: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inventory:
    namespace: str
    timestamp: str
    resource_count: int
    resources: list[ResourceEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def snapshot(
    client: InventoryClient | None = None,
    *,
    namespace: str = PERSONAL_NAMESPACE,
) -> Inventory:
    """Take a point-in-time inventory of all resources in the namespace.

    ``client`` is injectable — pass None for tests that populate
    ``_fake_files`` directly on the backend's client attribute.
    """
    from datetime import datetime, timezone

    if client is None:
        from openviking_sdk import SyncHTTPClient

        client = SyncHTTPClient(url="http://127.0.0.1:1934", timeout=300)
        client.initialize()

    resources: list[ResourceEntry] = []
    try:
        tree = client.ls(namespace, recursive=True, node_limit=5000)
    except Exception as exc:
        # Return an error inventory rather than crashing.
        return Inventory(
            namespace=namespace,
            timestamp=datetime.now(timezone.utc).isoformat(),
            resource_count=-1,
            metadata={"error": str(exc)[:500]},
        )

    for item in (tree or []):
        uri = str(item.get("uri", ""))
        if not uri or item.get("isDir"):
            continue
        try:
            root = canonical_resource_root(uri)
        except Exception:
            root = uri

        resources.append(ResourceEntry(
            uri=uri,
            canonical_root=root,
            size_hint=item.get("size", 0) if isinstance(item, dict) else 0,
        ))

    return Inventory(
        namespace=namespace,
        timestamp=datetime.now(timezone.utc).isoformat(),
        resource_count=len(resources),
        resources=resources,
    )


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class InventoryDiff:
    before_count: int
    after_count: int
    added: list[str]  # URIs that appear only in 'after'
    removed: list[str]  # URIs that appear only in 'before'
    unchanged: int

    @property
    def net_change(self) -> int:
        return self.after_count - self.before_count


def diff(before: Inventory, after: Inventory) -> InventoryDiff:
    """Compute the difference between two inventory snapshots."""
    before_uris = {r.uri for r in before.resources}
    after_uris = {r.uri for r in after.resources}

    added = sorted(after_uris - before_uris)
    removed = sorted(before_uris - after_uris)
    unchanged = len(before_uris & after_uris)

    return InventoryDiff(
        before_count=before.resource_count,
        after_count=after.resource_count,
        added=added,
        removed=removed,
        unchanged=unchanged,
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def save_inventory(inventory: Inventory, path: str | Path) -> Path:
    """Save an inventory to a JSON file."""
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "namespace": inventory.namespace,
        "timestamp": inventory.timestamp,
        "resource_count": inventory.resource_count,
        "resources": [
            {
                "uri": r.uri,
                "canonical_root": r.canonical_root,
                "size_hint": r.size_hint,
            }
            for r in inventory.resources
        ],
        "metadata": inventory.metadata,
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_inventory(path: str | Path) -> Inventory:
    """Load an inventory from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return Inventory(
        namespace=raw["namespace"],
        timestamp=raw["timestamp"],
        resource_count=raw["resource_count"],
        resources=[
            ResourceEntry(
                uri=r["uri"],
                canonical_root=r.get("canonical_root", r["uri"]),
                size_hint=r.get("size_hint", 0),
            )
            for r in raw.get("resources", [])
        ],
        metadata=raw.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Resource inventory for Personal KB")
    sub = parser.add_subparsers(dest="command")

    snap = sub.add_parser("snapshot", help="Take a point-in-time inventory")
    snap.add_argument("--output", "-o", required=True, help="Output JSON path")
    snap.add_argument("--namespace", default=PERSONAL_NAMESPACE, help="Namespace URI")
    snap.add_argument("--url", default="http://127.0.0.1:1934", help="OpenViking URL")

    diff_parser = sub.add_parser("diff", help="Diff two inventories")
    diff_parser.add_argument("before", help="Path to before.json")
    diff_parser.add_argument("after", help="Path to after.json")

    args = parser.parse_args()

    if args.command == "snapshot":
        inv = snapshot(namespace=args.namespace)
        path = save_inventory(inv, args.output)
        print(f"Inventory saved: {path}")
        print(f"  Resources: {inv.resource_count}")
        return 0 if inv.resource_count >= 0 else 1

    if args.command == "diff":
        before = load_inventory(args.before)
        after = load_inventory(args.after)
        d = diff(before, after)
        print(f"Before: {d.before_count}  After: {d.after_count}  Net: {d.net_change:+d}")
        print(f"  Added:   {len(d.added)}")
        print(f"  Removed: {len(d.removed)}")
        print(f"  Unchanged: {d.unchanged}")
        if d.added:
            print("\nAdded URIs:")
            for u in d.added[:20]:
                print(f"  + {u}")
            if len(d.added) > 20:
                print(f"  ... and {len(d.added) - 20} more")
        if d.removed:
            print("\nRemoved URIs:")
            for u in d.removed[:20]:
                print(f"  - {u}")
            if len(d.removed) > 20:
                print(f"  ... and {len(d.removed) - 20} more")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
