# Personal-KB Project Report

**Report date:** 2026-08-04  
**Scope:** repository review, offline verification, and improvement assessment.  
**Production changes:** none.

## Executive Summary

Personal-KB is a thoughtfully bounded, source-grounded study-material pipeline. Its strongest engineering decisions are the fail-closed source-hash manifest, OpenViking/Hindsight separation, namespace validation, provenance requirements, isolated ingestion gate, and explicit production rollback rules. The intended production retrieval policy is hybrid BM25 plus dense RRF without reranking; that is appropriate until representative course data supports a measured promotion.

The repository is **not currently ready to claim a green offline gate**. The current environment produced four test failures and a source-manifest hash mismatch. Those are localized and reproducible, but must be resolved before any real-course ingestion. The verified E2E fake-client gate and Python compilation both pass.

## What The Project Does

1. Accepts local text, PDF, audio, image, and slide inputs.
2. Produces hash-bound extraction artifacts through PyMuPDF, Whisper/MOSS, OCR, and diagram-processing paths.
3. Sends canonical source material to OpenViking for source-grounded retrieval and keeps personal/temporal learning state in Hindsight.
4. Uses validated planning, namespace-safe queries, provenance-bearing source reads, and local lexical plus dense RRF fusion.
5. Generates structured DeepSeek-backed study notes, then synchronizes managed Markdown into course-routed Obsidian locations.
6. Protects real ingestion with manifests, inventories, idempotency, read-back, test-resource cleanup, and rollback evidence.

## Architecture Assessment

| Area | Current design | Assessment |
|---|---|---|
| Canonical knowledge | OpenViking under `viking://resources/personal-kb` | Correctly source-grounded; URI and provenance preservation are explicit invariants. |
| Personal learning memory | Hindsight | Correct separation from canonical course facts and retrieval. |
| Retrieval | BM25 + dense RRF; reranker opt-in | Sensible conservative default; avoid promotion without held-out course data. |
| Ingestion safety | SHA-256 manifest, idempotency, inventory, cleanup | Strong design, but its checked-in fixture is currently inconsistent. |
| Media pipeline | PyMuPDF, Whisper/MOSS, GLM-OCR, Qwen3-VL | Appropriate staged model roles; optional GPU dependencies are isolated. |
| Note output | DeepSeek synthesis, correction pass, Obsidian sync | Good workflow boundary; command-line Unicode handling needs repair on Windows. |
| CI | GitHub Actions, Python 3.11, pytest plus compileall | Baseline is sound; current tests do not pass locally. |

## Verification Performed

| Command | Result | Evidence |
|---|---|---|
| `python -m pytest -q` | FAIL: 327 passed, 4 failed | Two CLI help tests and two injected-client inventory tests failed. |
| `python -m compileall -q scripts tests test_runs` | PASS | Exit code 0. |
| `python test_runs/e2e_gate.py` | PASS: 8/8 | First write, idempotency, hash mismatch, provenance, source read-back, Hindsight isolation, rollback inventory, and cleanup. This is the fake-client/default gate, not a live-service claim. |
| `python scripts/ingestion/source_manifest.py validate --manifest config/source_manifest.json --root .` | FAIL | `data/Test_whisper.txt` SHA-256 does not match the checked-in fixture manifest. |

The workspace has no `.git` directory. Consequently, `git status`, recent-commit inspection, and `git diff --check` cannot be performed in this copy. No conclusion about branch cleanliness or historical provenance can be made from this workspace.

## Highest-Priority Findings

### P0: Fixture manifest is invalid

`data/Test_whisper.txt` hashes to `D924F9105299C7266A161C1A4C2197CF4222EC15C0F060A019B475A07466B0EC`, while `config/source_manifest.json` records `f132644a77efd61b5c19d8c72b167e0c13951dd22eae0511bee9bf737643bd92`.

This is correctly rejected by the fail-closed policy. Do not simply replace the manifest hash: first establish whether the fixture text or the manifest is the canonical expected artifact, then update the non-canonical side and add a regression assertion for the public fixture manifest.

### P0: Full offline suite is red

1. `tests/test_note_generator_optimization.py::test_cli_exposes_bounded_worker_and_source_output_controls`
2. `tests/test_note_generator_routing.py::test_cli_only_offers_deepseek_backend_and_deprecated_ollama_url`

Both invoke `scripts/notes/note_generator.py --help`. Argparse tries to print a warning symbol (`U+26A0 U+FE0F`) and fails with `UnicodeEncodeError` under the current Windows GBK console. Make CLI help portable: use ASCII help copy for flags, or explicitly configure a UTF-8 stdout path before argument parsing. Retain Unicode only where the output contract requires it, and test the chosen policy on Windows.

3. `tests/test_recovery_tooling.py::TestResourceInventory::test_snapshot_with_fake_client`
4. `tests/test_recovery_tooling.py::TestResourceInventory::test_snapshot_skips_dirs`

`scripts/ops/resource_inventory.py:snapshot()` unconditionally imports `openviking_sdk`, even when callers supply a fake `InventoryClient`. Import and construct `SyncHTTPClient` only inside the `client is None` branch. This preserves the stated injectable/offline contract and lets `requirements-dev.txt` remain sufficient for offline tests.

## Documentation And Readiness Risks

- Verification counts are stale or inconsistent: the Step 6 plan says 276 tests, `STATUS.md` records 313, 331, and 334 in different checkpoints, while this audit observed 327 passing and 4 failing. Use a timestamped command output as the single source for the next status update.
- `docs/production-readiness-runbook.md` and the `e2e_gate.py` usage text document `python test_runs/e2e_gate.py --offline`, but the parser accepts no `--offline` argument. The default invocation is the fake/offline path; document that exact command or add the flag with coverage.
- Real course ingestion and an end-to-end live answer-path smoke remain intentionally blocked pending real course material and a recorded isolated live-read report. This is an appropriate production boundary, not an implementation omission.
- The repository snapshot lacks Git metadata. Restore it before release gating so the required clean-tree and diff checks are meaningful.

## Skills Installed

The following Codex skills were installed into the user-level Codex skills directory and will be available in the next session:

| Skill | Why It Fits Personal-KB |
|---|---|
| `pdf` | PDF extraction, inspection, rendering, and artifact QA for the media pipeline. |
| `speech` | Audio-oriented workflows relevant to lecture capture and media preparation. |
| `transcribe` | Transcript production and verification for Whisper/MOSS paths. |
| `security-best-practices` | Review API-key handling, local file ingestion, service endpoints, and provenance boundaries. |

## Recommended Next Milestone

1. Repair the public fixture hash mismatch after determining the canonical fixture content.
2. Repair the two focused code defects above and add/retain their regression tests.
3. Run `python -m pytest -q`, `python -m compileall -q scripts tests test_runs`, manifest validation, and the default E2E gate; capture outputs in a dated evidence artifact.
4. Reconcile `STATUS.md`, the Step 6 plan, and the production runbook with those artifacts, including the actual E2E command.
5. Only after the offline gate is green, collect a user-authorized isolated live-read report. Do not modify the production OpenViking namespace or Hindsight bank as part of that verification.

## Sources Reviewed

- `LUNA_DS_WORKING_FRAMEWORK.md`
- `README.md`
- `STATUS.md`
- `docs/plans/step6_openviking_plan.md`
- `docs/plans/step4_note_generator_plan.md`
- `docs/production-readiness-runbook.md`
- `.github/workflows/tests.yml`
- `scripts/notes/note_generator.py`
- `scripts/ops/resource_inventory.py`
