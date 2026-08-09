# Step 2: OvisOCR2 vs GLM-OCR Document Extraction A/B Plan

> **For Hermes:** Execute this plan as an isolated media benchmark first. Do not change the production OCR default or ingest benchmark artifacts into OpenViking/Hindsight until the acceptance gate passes.

**Goal:** Determine whether OvisOCR2 should replace GLM-OCR as Personal-KB's primary scanned-page parser while preserving exact source provenance, structured Markdown output, and a safe fallback path.

**Architecture:** Keep digital PDFs on the pymupdf path. Render scanned pages once at the same DPI, send identical page images to two isolated OCR branches, normalize outputs into the existing Personal media-artifact contract, and score raw transcription/structure before any summarization or indexing. The first comparison is OvisOCR2 versus the current GLM-OCR incumbent; Luna is not a primary OCR branch.

**Tech Stack:** Python 3.12 environment on the RTX 4080 laptop for the OvisOCR2 runtime, official OvisOCR2 BF16/F16 or Q6-GGUF weights, vLLM 0.22.1 where supported or llama.cpp with a matching multimodal projector, existing GLM-OCR path, pymupdf at 150 DPI, deterministic JSONL/JSON result artifacts, and the existing media-artifact validation tests.

---

## Decision already recorded: Luna's role in OCR

GPT-5.6 Luna supports image input and is a strong general reasoning/vision model, but it is not the right bulk OCR engine for Personal-KB.

Use Luna only as an optional escalation or adjudicator for a small difficult-page sample when image input is available on the selected OpenAI route. Do not use it as the default page extractor because:

- local OvisOCR2/GLM-OCR are purpose-trained for exact transcription, reading order, formulas, and tables;
- Luna may normalize, paraphrase, omit, or hallucinate text when asked to be helpful rather than literal;
- Luna has no demonstrated Personal-specific OCR benchmark in this project;
- remote image upload adds latency, privacy exposure, and quota/cost consumption;
- the same money/quota is more valuable for final semantic verification, difficult diagram interpretation, or architecture/review work.

The intended Personal pipeline is:

```text
Digital PDF       -> pymupdf
Scanned page      -> OvisOCR2 candidate / GLM-OCR incumbent
Low-confidence page -> Qwen3-VL or optional Luna escalation
Diagram semantics -> Qwen3-VL
Text summarization -> DeepSeek V4 Pro
Architecture/review -> GPT-5.6 Luna
```

Luna may be used to compare two OCR outputs against a page image, but that result is supplemental evidence. It must not be the sole ground truth or the only promotion criterion.

---

## Current context and constraints

- Current scanned-page incumbent: `glm-ocr` through `scripts.media.transcribe.ocr_image()` and `ocr_pdf()`.
- Current fallback: `qwen3-vl:8b`.
- Current scanned-PDF render default: 150 DPI.
- Existing scanned-PDF corpus list: `test_runs/pdf_benchmark/run_scanned_ocr.py`, currently seven control-systems PDFs.
- Existing artifact contract: `scripts/ingestion/media_artifact.py`.
- Existing laptop orchestration boundary: `scripts/ops/laptop_pipeline.py`.
- Production OpenViking and Hindsight must remain untouched during the A/B run.
- Existing working-tree changes are pre-existing (`STATUS.md` modified and `docs/production-readiness-runbook.md` untracked); do not overwrite, reset, or commit them as part of this work.
- The OvisOCR2 paper reports 96.58 on OmniDocBench v1.6 and 75.06 Avg3 on PureDocBench, but those are author-reported/public-benchmark results, not Personal-corpus evidence.
- OvisOCR2 is page-level. PDF rendering, page labels, cross-page assembly, and cross-page table handling remain pipeline responsibilities.

---

## Files and artifact boundaries

### Likely implementation files

- Modify: `scripts/media/transcribe.py`
  - Add an injectable OvisOCR2 image/PDF adapter without breaking the existing Ollama transport.
  - Keep runtime-specific imports lazy.
  - Preserve deterministic page ordering and explicit engine labels.
- Modify: `scripts/ops/laptop_pipeline.py`
  - Preserve the explicit local-processing/indexing boundary.
  - Ensure the selected OCR engine is carried into the result and artifact metadata.
- Modify: `scripts/ingestion/media_artifact.py`
  - Add `ovisocr2` as a valid extraction engine only if the adapter is implemented.
  - Keep source hash, timestamp, and downstream status mandatory.
- Modify: `config/pipeline_laptop.yaml`
  - Document the incumbent, candidate, fallback, runtime endpoint, and opt-in selection; do not silently change the production default during the benchmark.
- Modify: `tests/test_media_fixture_validation.py`
  - Add offline artifact coverage for `ovisocr2` without downloading or calling a model.
- Modify: `tests/test_laptop_pipeline.py`
  - Add injected OCR-engine propagation and deterministic-output coverage if the pipeline contract changes.

### Isolated benchmark files

- Create: `test_runs/ocr_ovisocr2_ab/run_ocr_ab.py`
- Create: `test_runs/ocr_ovisocr2_ab/corpus_manifest.json`
- Create: `test_runs/ocr_ovisocr2_ab/results.jsonl`
- Create: `test_runs/ocr_ovisocr2_ab/report.md`
- Create: `test_runs/ocr_ovisocr2_ab/outputs/glm-ocr/`
- Create: `test_runs/ocr_ovisocr2_ab/outputs/ovisocr2/`

These outputs remain benchmark artifacts. They must not be added to the Personal source manifest or indexed into OpenViking/Hindsight.

### Documentation files

- Modify after the benchmark is complete: `README.md` to record the measured decision and link this plan.
- Modify after the benchmark is complete: `STATUS.md` to record actual evidence, not the plan itself.

---

## Task 1: Freeze and validate the common corpus

**Objective:** Create one immutable page manifest so both OCR branches receive byte-identical page images.

1. Reuse the seven scanned-PDF names from `test_runs/pdf_benchmark/run_scanned_ocr.py`, but parameterize the source root rather than relying on its hard-coded laptop path.
2. Render each scanned page exactly once at 150 DPI into the isolated benchmark directory.
3. Record for every page: source PDF path, source SHA-256, page number, rendered-image SHA-256, width, height, DPI, and document category.
4. Include a balanced subset if available: clean print, two-column pages, formulas, dense tables, handwriting, diagrams, and degraded/fax-like pages.
5. Fail before inference if any source or rendered page is missing, empty, duplicated, or changes after manifest creation.

**Acceptance:** both branches report the same page count and identical rendered-image hashes.

---

## Task 2: Establish the frozen GLM-OCR baseline

**Objective:** Record a fresh baseline with the current production candidate and no hidden fallback substitution.

1. Run GLM-OCR on every page in the frozen manifest.
2. Record the actual engine used per page; a failure must be visible rather than silently recorded as GLM-OCR success.
3. Save raw output, normalized Markdown, page latency, total latency, output characters/tokens, and error/retry information.
4. Run a deterministic structural check for empty output, `[OCR FAILED]`, repeated tails, truncation markers, malformed HTML tables, and malformed LaTeX delimiters.
5. Keep the old `qwen3-vl:8b` fallback out of the first primary A/B score. A separate fallback experiment can be added later.

**Acceptance:** the baseline has one result row per manifest page and no unreported engine substitutions.

---

## Task 3: Bring up OvisOCR2 in an isolated runtime

**Objective:** Verify the candidate runtime independently before adding it to the Personal pipeline.

1. Prefer official `ATH-MaaS/OvisOCR2` BF16/F16 or a Q6 GGUF for the first quality run; do not begin with an aggressive Q4 quantization.
2. If using vLLM, isolate a Python 3.12 environment and pin `vllm==0.22.1`; use the model-card preprocessing range and deterministic decoding.
3. If vLLM is blocked by the Windows/WSL2 runtime, use the community GGUF plus the matching `mmproj` file through a verified llama.cpp multimodal CLI.
4. Use one fixed prompt that requests literal transcription, natural reading order, Markdown output, HTML tables, LaTeX formulas, and bounding-box image tags. Do not ask for summarization or translation.
5. Verify one page each for text, formula, table, multi-column layout, and visual-region output.
6. Record the exact model revision, quantization, runtime version, CUDA/driver environment, prompt, image preprocessing limits, and decoding parameters.

**Acceptance:** OvisOCR2 produces non-empty structured output for the smoke pages and the runtime can process a unique image without using a multimodal cache hit.

---

## Task 4: Add the OvisOCR2 adapter without changing the default

**Objective:** Make OvisOCR2 selectable and testable through the existing media boundary while leaving GLM-OCR as the incumbent.

1. Add a lazy, injectable OvisOCR2 transport or local-runtime adapter to `scripts/media/transcribe.py`.
2. Keep `ocr_image()` and `ocr_pdf()` usable with fake transports in offline tests.
3. Preserve page headers and engine metadata, for example `OCR: ovisocr2`.
4. Extend `EXTRACTION_ENGINES` only after the adapter can produce a validated artifact.
5. Keep `pipeline_laptop.yaml` explicit: `glm-ocr` remains the current default until the promotion gate passes; `ovisocr2` is an opt-in candidate.
6. Do not make the Ovis runtime import at module import time; a machine without the optional dependency must still pass ordinary offline imports/tests.

**Acceptance:** existing GLM-OCR/Ollama behavior remains compatible, offline tests do not contact a model, and the candidate can be selected explicitly.

---

## Task 5: Run the paired page-level A/B

**Objective:** Compare both parsers on exactly the same images and settings.

For each page, record:

- engine and exact revision
- cold/warm status
- wall-clock latency and time to first output where available
- output token/character count
- success, error, retry, and repetition-loop flags
- text, formula, table, reading-order, and visual-region checks
- peak GPU memory and process memory when measurable

Do not compare only total PDF time. Report per-page p50/p95 and pages/minute, because a single pathological page can make a pipeline operationally unpleasant even when the average looks fine.

Run at least:

- one cold single-page run
- one warm sequential run
- one bounded batch/concurrency run appropriate to the laptop
- one no-cache run with unique images

**Acceptance:** the report contains complete paired rows, no missing pages, no unreported fallback calls, and separate cold/warm/batch measurements.

---

## Task 6: Score extraction quality

**Objective:** Measure what Personal actually needs rather than copying a leaderboard number.

Use a manually checked held-out subset of at least 20 pages, expanded to 50–100 pages if time permits. Include pages with tables, formulas, handwriting, diagrams, columns, and degraded scans.

Score:

1. **Text completeness/fidelity — 35%**
   - normalized character/word edit distance against a human-corrected reference
2. **Table reconstruction — 25%**
   - valid HTML parse, row/column preservation, merged-cell handling, and cell-content accuracy
3. **Formula fidelity — 15%**
   - delimiter validity plus manual/math-render comparison
4. **Reading order — 15%**
   - human sequence check for columns, headers, footers, captions, and side notes
5. **Operational reliability — 10%**
   - non-empty output, no repetition loop, no truncation, retry rate, and latency

Use a blind human review for the difficult subset. Luna may provide a secondary page-image comparison, but it must not replace human-corrected references or deterministic structural checks.

**Acceptance:** promote OvisOCR2 only if it wins the weighted score or delivers a clearly justified latency/operational gain without a material regression in tables, formulas, reading order, or worst-case pages. A higher average with catastrophic table loss is a failed result; OCR has a talent for looking impressive right before it deletes the one row that mattered.

---

## Task 7: Decide the production routing policy

### If OvisOCR2 wins

- Make OvisOCR2 the explicit primary scanned-page parser.
- Keep GLM-OCR as the first fallback for runtime/model failures or pages where Ovis confidence checks fail.
- Keep Qwen3-VL as a semantic/diagram fallback, not the default literal OCR engine.
- Add repetition/truncation detection and one bounded retry.
- Preserve `extraction_engine="ovisocr2"` in every artifact.

### If GLM-OCR wins or results are mixed

- Keep GLM-OCR as primary.
- Use OvisOCR2 selectively for complex tables/multi-column pages if the page classifier can identify them reliably.
- Do not route by benchmark score alone; use per-page failure categories and measured latency.

### Luna policy in either case

- Do not make Luna the bulk OCR engine.
- Use it only for a small, explicit escalation set: ambiguous handwriting, difficult visual-region interpretation, or blind secondary review.
- Record Luna as a separate `ocr-adjudication` or `vision-review` stage rather than mislabeling its output as the primary OCR engine.

---

## Task 8: Integrate only after the evidence gate

**Objective:** Promote the chosen engine without mixing OCR quality with retrieval or summarization changes.

1. Run offline media-artifact and laptop-pipeline tests.
2. Run `python -m py_compile` on changed modules.
3. Run `git diff --check`.
4. Re-run the isolated A/B report after any adapter change.
5. Update `README.md`, `STATUS.md`, and this plan with the actual model revision, benchmark counts, score breakdown, latency, and decision.
6. If the chosen engine is subsequently indexed into OpenViking/Hindsight, run the required offline e2e gate first and ask before the live gate; do not ingest the A/B outputs themselves.
7. Preserve the current production configuration and rollback path until the promoted branch passes a clean post-change smoke test.

Suggested verification commands:

```text
python -m pytest -q tests/test_media_fixture_validation.py tests/test_laptop_pipeline.py
python -m py_compile scripts/media/transcribe.py scripts/ops/laptop_pipeline.py scripts/ingestion/media_artifact.py
python test_runs/ocr_ovisocr2_ab/run_ocr_ab.py --validate-only
python test_runs/e2e_gate.py
python -m pytest -q
git diff --check
```

---

## Open questions

- Which OvisOCR2 runtime is stable on the RTX 4080 laptop: isolated vLLM, llama.cpp GGUF, or another supported backend?
- Does the selected GGUF preserve enough vision quality at Q6 compared with BF16/F16?
- How does OvisOCR2 perform on Personal's Chinese/English mix and handwritten annotations?
- Does it preserve diagrams as useful cropped visual regions, or should Qwen3-VL continue to process every diagram separately?
- What is the real failure rate on the seven scanned control-systems PDFs rather than on public benchmarks?

## Promotion rule

Until Task 6 is complete, Personal-KB keeps the current GLM-OCR default and treats OvisOCR2 as an isolated candidate. Luna remains the architecture/review authority and an optional hard-page adjudicator, not the OCR engine.
