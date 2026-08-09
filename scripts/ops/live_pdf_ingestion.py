"""Real live OpenViking/Hindsight branch for the 15 original-PDF artifacts."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.core.openviking_backend import PersonalOpenVikingBackend, canonical_resource_root, resource_uri
from scripts.ingestion.ingest import ingest_facts
from scripts.ops.batch_ingest import BatchItem, run_batch
from scripts.ops.ingestion_readback import select_readable_source_leaf, source_readback_matches
from scripts.ops.live_hindsight import LiveHindsightClient

MANIFEST = ROOT / "test_runs" / "pdf_branch_local" / "pipeline_manifest.json"
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
COURSE = f"TEST-PDF-BRANCH-LIVE-{RUN_ID}"
BANK = COURSE
WORK = ROOT / "test_runs" / f"pdf_branch_live_{RUN_ID}"


def main() -> int:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = data["items"]
    if len(entries) != 15 or not all(e["kind"] == "pdf" for e in entries):
        raise AssertionError("PDF branch manifest must contain exactly 15 PDF items")
    items: list[BatchItem] = []
    for entry in entries:
        original = Path(entry["source_path"])
        artifact = Path(entry["index_path"])
        if not original.is_absolute(): original = (ROOT / original).resolve()
        if not artifact.is_absolute(): artifact = (ROOT / artifact).resolve()
        items.append(BatchItem(
            index_path=artifact,
            source_path=original,
            source_hash=entry["source_hash"],
            source_type="ocr_text",
            course=COURSE,
            metadata={"corpus":"test", "semester":"TEST", "extraction_engine":"pymupdf"},
        ))
    backend = PersonalOpenVikingBackend(
        base_url=os.environ.get("PERSONAL_KB_OPENVIKING_URL", "http://127.0.0.1:1934"),
        root=ROOT,
        timeout=1800,
    )
    roots = [canonical_resource_root(resource_uri(i.index_path, course=COURSE, source_type=i.source_type, root=ROOT, source_hash=i.source_hash)) for i in items]
    hs = LiveHindsightClient()
    batch = None; readback=[]; hindsight=None; error=None; cleanup_errors=[]
    try:
        batch=run_batch(items, backend=backend, log_path=WORK/"index_ingestion.jsonl")
        if batch["counts"] != {"indexed":15,"skipped":0,"failed":0}:
            raise RuntimeError(f"PDF live batch failed: {batch}")
        for n,(item,root) in enumerate(zip(items,roots),1):
            nodes=backend.client.ls(root,recursive=True,node_limit=100) or []
            leaf=select_readable_source_leaf(nodes)
            if leaf is None: raise RuntimeError(f"no source leaf: {root}")
            content=str(backend.read(leaf,limit=2000))
            expected=Path(item.index_path).read_text(encoding="utf-8")
            if not source_readback_matches(expected,content): raise RuntimeError(f"readback mismatch: {root}")
            readback.append({"root":root,"leaf":leaf,"content_chars":len(content)})
            print(f"[pdf-readback] {n}/15 verified",flush=True)
        facts=[]
        for item in items:
            text=Path(item.index_path).read_text(encoding="utf-8")
            facts.append({"content":f"{COURSE} {item.source_path.name}: {text[:600]}","course":COURSE,"source_type":"ocr_text","source_scope":"course","topic":item.source_path.stem})
        hindsight=ingest_facts(facts,bank_id=BANK,batch_size=5,url=hs.base_url,client=hs)
        if hindsight["success"] != 15 or hindsight["failed"]: raise RuntimeError(f"Hindsight failed: {hindsight}")
        if not hs.recall(BANK,COURSE,top_k=3).get("results"): raise RuntimeError("Hindsight readback empty")
    except Exception as exc:
        error=f"{type(exc).__name__}: {exc}"
    finally:
        for root in sorted(set(roots),reverse=True):
            try: backend.client.rm(root,recursive=True,wait=True)
            except Exception as exc: cleanup_errors.append(f"{root}: {type(exc).__name__}: {exc}")
        try: hs.delete_bank(BANK)
        except Exception as exc: cleanup_errors.append(f"{BANK}: {type(exc).__name__}: {exc}")
    result={"run_id":RUN_ID,"course":COURSE,"source_count":15,"pipeline_manifest":str(MANIFEST),"batch":batch,"readback":readback,"hindsight":hindsight,"live_error":error,"cleanup_errors":cleanup_errors,"production_services_accessed":False}
    WORK.mkdir(parents=True,exist_ok=True); (WORK/"result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    return 0 if error is None and not cleanup_errors and len(readback)==15 else 1

if __name__ == "__main__": raise SystemExit(main())
