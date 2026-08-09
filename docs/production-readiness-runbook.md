# Production-Readiness Runbook — Live-Read Gate

**Purpose:** one repeatable procedure that takes Personal-KB from "services look healthy" to "proven safe to ingest and answer from a real course corpus." A green live-read gate is the final precondition for real Personal course ingestion.

**Scope:** preflight → zero-test cleanup → validated manifest ingestion → source read-back → Hindsight isolation → rollback inventory → cleanup → Obsidian sync verification.

**Non-negotiables (from `LUNA_DS_WORKING_FRAMEWORK.md`):**
- Never treat a successful upload or healthy service as proof of correct ingestion.
- A valid plan followed by a connection failure is a failed smoke test, not partial success.
- Live claims require live evidence: query, plan, trace ID, backend URL, health, retrieval mode, result count, source URIs, timings, fallback state.
- Production OpenViking (`viking://resources/personal-kb`) and the production Hindsight bank are protected; the gate uses isolated test namespaces and banks only.
- Automatic memory linking is **disabled** (server-side early-return guard, verified 2026-07-22 and persisted across restart). The gate re-verifies this on every run.

---

## 1. Prerequisites

Services (mini PC unless noted):

| Service | Default URL | Check |
|---|---|---|
| OpenViking | `http://127.0.0.1:1934` | `GET /health` → `healthy: true` |
| Embedding server (BGE-small) | `http://127.0.0.1:18002` | `GET /health` → `status: ok` |
| Hindsight | `http://127.0.0.1:8888` | `GET /v1/default/banks` → 200 |
| Laptop embedding alt | `PERSONAL_KB_EMBEDDING_URL` | only when a separate embedding service is used |

Environment:
- Working tree clean on `master` (or a named gate branch).
- `DEEPSEEK_API_KEY` present for LLM-path checks (offline gate does not need it).
- Obsidian vault at `~/obsidian-vault` if the sync step is exercised.

---

## 2. Gate stages

The gate is implemented by `test_runs/e2e_gate.py` (live mode) plus the wrappers below. Each stage must pass in order; any red stage blocks ingestion and must be recorded with evidence.

### Stage 0 — Preflight
```
python scripts/ingestion/preflight.py --manifest config/source_manifest.json --root .
```
Covers: manifest validation, service health, LLM routing (deepseek-v4-pro via api.deepseek.com/v1), logical-source conflict scan, URI/provenance dry run, git cleanliness. Use `--no-services --no-git` for offline.

### Stage 1 — Zero-test cleanup
Target workspace must contain zero synthetic/test resources before any real corpus write.
```
python scripts/ops/test_cleanup.py --pattern "E2E-TEST" --dry-run   # inspect first
```
Cleanup gate rules (from `step6_openviking_plan.md`): checkpoint + inventory → remove test URIs via API (not query filters) → re-scan → verify zero test resources → verify no canonical resource was touched → record deletion counts + rollback path.

### Stage 2 — Validated manifest ingestion
- Manifest entries carry full `source_path`, `source_hash` (sha256), `source_type`; `lecture` omitted for multi-lecture sources.
- First write creates the resource with provenance metadata.
- Re-ingestion with the same hash is an idempotent skip.
- Same logical source + different hash raises `SourceHashMismatch` (fail-closed).

### Stage 3 — Source read-back
For every indexed source: resolve the canonical URI, read a source-content leaf, confirm metadata (course, source_type, source_hash) and content presence. Record source URIs in the gate report.

### Stage 4 — Hindsight isolation
- Isolated test bank (`hermes-e2e-gate-*`); retain/recall round trip.
- Production bank untouched; test bank deleted on cleanup.
- Course facts keep `personal-kb` scope and namespaced tags.

### Stage 5 — Memory-linking verification
Confirm ingestion responses carry no `memory_linking`/`token_usage` fields and `reason=""` is sent. If disproportionate internal LLM usage reappears, mark the path **not cost-safe** and stop (see `references/memory-linking-disable-verification.md` in the personal-kb skill).

### Stage 6 — Rollback inventory
Capture pre/post resource counts and URI inventories. A failed gate must leave production config and the rollback path intact.

### Stage 7 — Cleanup
Remove all gate resources (OpenViking + Hindsight test bank); verify zero remaining; report is only green when post-gate inventory matches pre-gate inventory.

### Stage 8 — Live read (answer-path) smoke
After stages 1–7 pass, run one real query through the production retrieval path (hybrid BM25+dense, no reranker) against an isolated live corpus:
- Record original query, validated plan, trace ID, backend URL, health, retrieval mode, result count, source URIs, timings, fallback state.
- The answer must cite at least one source URI under the Personal namespace.

### Stage 9 — Obsidian sync verification (when notes are generated)
`scripts/ops/obsidian_sync.py` one-shot dry run: staged notes route through `course_namespaces.yaml`, course index updates are idempotent, user-authored content preserved.

---

## 3. Commands

Offline (CI-safe, fakes only):
```
python -m pytest -q tests/test_e2e_gate.py tests/test_preflight.py tests/test_recovery_tooling.py
python test_runs/e2e_gate.py --offline
```

Live gate (isolated namespaces; requires all services healthy):
```
python test_runs/e2e_gate.py                 # full live gate, isolated corpus + bank
python test_runs/live_read_gate.py --output test_runs/live_read_gate_report.json
```

Full suite before declaring readiness:
```
python -m pytest -q
python -m compileall -q scripts tests test_runs
git diff --check
```

---

## 4. Evidence to record

Every gate run writes a JSON report under `test_runs/` containing: per-stage pass/fail, pre/post resource counts, source URIs read back, Hindsight bank counts, memory-linking absence confirmation, timings, and sanitized errors. Link the report from `STATUS.md` when the run changes production-readiness state.

## 5. Acceptance criteria (production-ready declaration)

- [x] Strict source-hash policy implemented and live-tested
- [x] Isolated E2E gate passes all checks
- [x] Production cleanup inventory zero-test and reversible
- [x] OpenViking + Hindsight read-back checks pass (22-source and 15-PDF TEST replays)
- [x] No memory-linking activity or token explosion (disabled server-side; re-verified each gate run)
- [ ] Real course ingestion from a validated manifest — **blocked until real Personal course files exist (late August)**
- [ ] Live read gate report green end-to-end including Stage 8 answer-path smoke

## 6. Rollback

If any stage fails after writes occurred: stop, do not retry blindly. Use the recorded pre-gate inventory and `scripts/ops/test_cleanup.py` with the gate's URI list; restore from checkpoint if canonical resources were touched. Never delete ambiguous files without locating their owning project.
