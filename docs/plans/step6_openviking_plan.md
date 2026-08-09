# Step 6: OpenViking Retrieval Backend — Detailed Plan

**Status:** Hardening milestone complete; all 7 pre-semester priorities (P1-P7) implemented. Production OpenViking uses bge-small embeddings with DeepSeek V4 Pro as VLM. Hybrid BM25+dense RRF is production default; reranker installed but deferred. 276 tests pass. Project reorganized into module subdirectories (core/ingestion/media/retrieval/notes/study/ops). **Last checkpoint:** 2026-07-21 — P1-P7 done, project reorganized, 5 doable items identified for pre-school window (laptop pipeline, expanded benchmark, batch ingestion, connectivity test, embedding A/B test).
**Created:** 2026-07-16
**Last checkpoint:** 2026-07-20 11:14:23 -0500 — pre-semester reliability plan recorded: strict source-change rejection, reproducible preflight, complete isolated gate, media fixtures, recovery tooling, student workflow smoke tests, and course onboarding. Retrieval optimization is explicitly deferred until representative course material exists.
**Depends on:** Step 1 (Whisper/MOSS), Step 2 (pymupdf/GLM-OCR/Qwen3-VL), Step 3 (Hindsight metadata and ingestion), Step 4 (lecture note generation)

---

## Goal

Make OpenViking the source-grounded retrieval backend for `personal-kb`, while retaining Hindsight for personal, temporal, entity, graph, and learning-history memory. Run the media-to-retrieval pipeline on the RTX 4080 laptop, where Whisper and OCR already run, and promote a stronger English embedding/retrieval stack only after a grounded A/B evaluation.

## Current Status — 2026-07-20

- Personal-safe OpenViking adapter implemented with namespace checks, provenance metadata, dry-run support, injectable clients, and offline tests.
- Audio and raw book binaries are excluded from future ingestion. Previously indexed test media and unrelated reference material were removed.
- Production OpenViking remains the small-model embedding index on mini-PC port `1934`; its active VLM is `deepseek-v4-pro` through `https://api.deepseek.com/v1`.
- BGE-large A/B remains paused because the CUDA Python runtime failed Windows DLL/mount loading and the CPU fallback was too slow.
- EmbeddingGemma A/B setup and full isolated evaluation are complete but it was not promoted because the remote tunnel/prefix-proxy path still needs interactive-latency optimization and a held-out benchmark.
- Curated ingestion policy v1 is enforced for the production Personal namespace through an explicit allowlist of 17 resources; production was rebuilt and verified with zero queue errors/requeues and zero missing vectors.
- Step 6.4 reranking and Step 6.5 BM25 + dense RRF hybrid retrieval are complete. The lexical index has been rebuilt with JSON-derived course/lecture/source metadata while preserving OpenViking URI alignment.
- **Step 6.6 core implementation is complete.** The planner requires exactly one compact `lexical` and one natural-language `semantic` query; the orchestrator routes them to BM25 and dense retrieval respectively, performs one shared RRF/reranker/read pipeline, preserves provenance, and applies one global cutoff.
- **Realistic source-read retest complete.** Ten planner-format Personal queries produced one hybrid execution record each. Hybrid without reranking achieved 70% read hit@5 and 70% top-1 support; the new context/metadata reranker achieved 60% and 60%; zero final source-read errors. The Kp/Routh miss is a verified corpus-coverage gap: neither branch's readable source contains `kp` or `routh`.
- **Current next gate:** bound or disable OpenViking automatic memory linking during source ingestion, then run a cost-safe fake live smoke test before any real-course write. Keep RRF without reranking as the rollback-safe production default; defer larger formula-heavy/multi-source reranker work until representative course material is available.

### Fake live-ingestion smoke test — complete, production-gated

- Test partition: `TEST-LIVE-2W`; one synthetic textbook source and one Hindsight fact.
- OpenViking: first run ingested 1, second run skipped the canonical root, and the readable leaf matched the fixture.
- Hindsight: 1 fact retained and expected observations returned on recall.
- Cleanup and isolation: fake resource/tree, test bank, and exact internal reason archive removed; no test URIs remained under the Personal namespace and production Hindsight had zero test-tag matches.
- Finding: OpenViking memory linking reported 975,780 internal LLM tokens for the tiny fixture. Do not promote this live path to real textbook ingestion until memory linking is bounded, isolated, or disabled.

### Step 6.6 terminology

- **Query plan:** the validated intended strategy: intent, backend(s), rewritten queries, explicit filters, retrieval mode, limits, and trace ID.
- **Query trace:** the execution record: actual backends, query statuses, timings, result counts, source URIs, fallback reasons, and sanitized errors. It must never contain secrets.

## Corpus import policy and pre-ingestion cleanup gate

OpenViking must receive a curated, source-grounded corpus—not every text file in the repository and not only LLM-generated notes.

### Import policy

- Import canonical verified study text: cleaned transcripts, PDF/OCR text, slide and diagram descriptions, textbook chapters, homework/exams, formulas, glossaries, course catalogs, and final human-edited notes.
- Import final LLM-generated lecture notes, summaries, and study facts only when they contain source references, are grounded in canonical files, and add useful structure rather than duplicating transient summaries.
- Keep canonical material under a source layer and generated material under a derived layer; preserve `source_uri`, `source_hash`, generation model, and provenance metadata.
- Exclude raw audio, images, unprocessed PDFs, code, scripts, logs, caches, benchmark outputs, intermediate LLM artifacts, duplicate copies, and OpenViking-generated `.abstract.md`/`.overview.md` files.
- Keep project/infrastructure documentation in a separate optional namespace instead of the default study-retrieval corpus.
- Route personal learning history, mastery, review history, preferences, and temporal/entity relationships to Hindsight rather than bulk-importing them into OpenViking.

### Mandatory cleanup before actual course ingestion

Before ingesting the real course allowlist into any target OpenViking workspace, clean all synthetic/test resources from that workspace:

1. Create a rollback checkpoint and export an inventory of existing URIs.
2. Identify every resource whose URI, source path, or metadata matches test/smoke fixtures, including `TEST`, `test_runs`, `Test_`, `test_`, smoke-test namespaces, and synthetic benchmark artifacts.
3. Remove those test resources through the OpenViking SDK/API; do not merely hide them with a query filter.
4. Re-scan the tree and inventory and verify that zero test resources remain.
5. Verify that no canonical course resource was removed accidentally.
6. Only after the zero-test-resource check passes, ingest the canonical source and approved derived manifests.
7. Record the cleanup manifest, deletion count, remaining resource count, queue status, and rollback path.

This cleanup gate is required because the EmbeddingGemma comparison surfaced duplicated `TEST` transcripts and synthetic resources, making retrieval-quality results harder to interpret. Production OpenViking must remain untouched during experiments; when this gate is applied to production, use a checkpoint and an explicit user-approved migration window.

## Architecture

```text
Laptop: audio/PDF/image inputs
        |
        +--> Whisper/MOSS transcription
        +--> pymupdf digital extraction
        +--> GLM-OCR scanned-page extraction
        +--> Qwen3-VL/vision diagram extraction
        |
        +--> OpenViking resources
        |      dense embedding + lexical/hybrid retrieval
        |      reranking + source-grounded read()
        |
        +--> Hindsight memories
               personal facts + temporal events + entity/graph relationships

Hermes personal-kb skill
        |
        +--> OpenViking for document/source questions
        +--> Hindsight for personal-memory/temporal/graph questions
        +--> Combined context for study, review, and quiz generation
```

OpenViking is the document retrieval system, not a replacement for Hindsight. The router should select the backend based on query intent and may query both for mixed questions.

---

## 6.1 Implement OpenViking as the Personal KB backend

**Objective:** Make `personal-kb` use OpenViking for source-grounded retrieval and reading.

**Files:**
- Modify: `openviking/config/ov_bge_en.conf`
- Modify: `openviking/scripts/hermes_openviking_mcp.py`
- Modify: `openviking/scripts/ingest_personal.py`
- Modify: `openviking/scripts/ingest_missing_personal.py`
- Modify: `Personal_KB/scripts/common_client.py`
- Modify: `Personal_KB/scripts/ingest.py`
- Create: `Personal_KB/scripts/openviking_backend.py`
- Create: `Personal_KB/tests/test_openviking_backend.py`

**Requirements:**

- OpenViking remains bound to the laptop service endpoint.
- `personal-kb` can search OpenViking by course, lecture, source type, and topic.
- Search results preserve the `viking://` URI and source file path.
- The backend can retrieve the actual source context after search.
- Re-ingestion is idempotent and does not duplicate resources.
- Hindsight retains only memory-oriented facts and generated learning-history artifacts unless a future task explicitly requires dual retention.

**Verification:**

```text
python -m pytest Personal_KB/tests/test_openviking_backend.py -v
```

Expected: backend health, search, source read, namespace filtering, and idempotent-ingestion tests pass.

---

## 6.2 Run the full pipeline on the laptop

**Objective:** Make the laptop the media-processing and indexing host because it already runs Whisper and OCR models.

**Files:**
- Create: `Personal_KB/config/pipeline_laptop.yaml`
- Create: `Personal_KB/scripts/laptop_pipeline.py`
- Modify: `Personal_KB/transcribe.py`
- Modify: `Personal_KB/scripts/diagrams.py`
- Modify: `Personal_KB/scripts/ingest.py`
- Modify: `openviking/config/ov_bge_en.conf`

**Pipeline stages:**

1. Detect input type.
2. Run Whisper/MOSS for audio.
3. Run pymupdf for digital PDFs.
4. Run GLM-OCR for scanned pages.
5. Run Qwen3-VL or the configured fallback for diagrams.
6. Generate structured extraction artifacts.
7. Upload/index the source and derived text into OpenViking.
8. Retain only appropriate personal/temporal/graph memories into Hindsight.
9. Emit a manifest containing source hash, extraction engine, timestamps, and index status.

**Verification:**

- Process one audio file, one digital PDF, one scanned PDF, and one diagram.
- Verify every artifact has an OpenViking URI.
- Verify failures are resumable and do not reprocess unchanged source hashes.
- Verify the pipeline can be launched from the mini PC while execution occurs on the laptop.

---

## 6.3 Upgrade the primary English embedding model

**Objective:** Test `bge-large-en-v1.5` and stronger English candidates because the laptop is already required for OCR/Whisper and can host the heavier retrieval model.

**Candidate order:**

1. `bge-large-en-v1.5` — primary candidate
2. `bge-base-en-v1.5` — quality/latency fallback
3. `bge-m3` or another stronger English/multilingual model — only if the corpus becomes multilingual or hybrid retrieval is needed
4. Keep `bge-small-en-v1.5` as the measured baseline and emergency fallback

**Files:**
- Modify: `openviking/config/ov_bge_en.conf`
- Create: `openviking/config/ov_bge_large_en.conf`
- Create: `openviking/config/ov_bge_base_en.conf`
- Create: separate workspaces under `openviking/data_bge_large_en/` and `openviking/data_bge_base_en/`
- Modify: `openviking/scripts/retrieval_benchmark.py`

**Rules:**

- Never overwrite the current production index during an A/B test.
- Each embedding dimension requires a fresh OpenViking vector workspace and full re-embedding.
- Compare the same corpus, query set, extraction artifacts, and retrieval settings.
- Promote the larger model only if quality gains justify latency, VRAM/RAM, and re-indexing cost.

**Metrics:**

- Recall@1 and Recall@5
- MRR
- First correct source rank
- Actual top-1 context concept coverage
- Source traceability
- Query latency
- Indexing throughput
- Memory/VRAM use
- Queue errors and retry count

**Acceptance gate:**

The promoted model must improve the expanded difficult-query benchmark, not merely produce a larger embedding vector. The existing 10-query benchmark is already saturated at 100% with `bge-small-en-v1.5`, so the new evaluation must add ambiguous, formula-heavy, multi-source, and multi-hop questions.

---

## 6.4 Add a reranking layer — COMPLETE (2026-07-17)

**Objective:** Retrieve a broad candidate set cheaply, then apply a stronger cross-encoder to improve ordering.

**Architecture:**

```text
Dense/hybrid retrieval top 20
        |
        v
Cross-encoder reranker
        |
        v
Top 3-5 source contexts
        |
        v
OpenViking read()
```

**Candidates:**

- `bge-reranker-base` for the default local option
- `bge-reranker-large` if quality gains justify latency and memory
- Supported remote reranker only if local deployment is impractical

**Files:**
- Create: `Personal_KB/scripts/reranker.py`
- Modify: `Personal_KB/scripts/openviking_backend.py`
- Modify: `openviking/scripts/hermes_openviking_mcp.py`
- Create: `Personal_KB/tests/test_reranker.py`

**Verification:**

- Compare dense-only versus dense-plus-reranker on the same top-20 candidate lists.
- Verify the reranker cannot return a source outside the allowed Personal namespace.
- Verify timeout/failure falls back to dense ranking.
- Measure top-1 source accuracy and added latency.

---

## 6.5 Add lexical/hybrid retrieval — COMPLETE (2026-07-17)

**Objective:** Recover exact terms that dense embeddings may underweight, including equations, symbols, acronyms, and identifiers.

**Candidate design:**

```text
Dense BGE candidates
        +
BM25/sparse lexical candidates
        |
        v
Rank fusion / deduplication
        |
        v
Cross-encoder reranking
```

**Lexical fields:**

- Source filename
- Course and lecture metadata
- Headings
- Formula text and LaTeX
- Variable names
- Acronyms
- Full extracted text

**Files:**
- Create: `Personal_KB/scripts/lexical_index.py`
- Create: `Personal_KB/scripts/hybrid_retrieval.py`
- Modify: `Personal_KB/scripts/openviking_backend.py`
- Create: `Personal_KB/tests/test_hybrid_retrieval.py`

**Verification:**

- Test exact queries such as `DG/(1+DGH)`, `N = Z - P`, `kanban`, `Jidoka`, and `KB 1001`.
- Compare dense-only, lexical-only, and fused retrieval.
- Verify duplicate results are merged by canonical source URI.
- Tune fusion weights only against the held-out benchmark set.

---

## 6.6 Add an LLM-assisted query layer

**Objective:** Improve retrieval for vague, ambiguous, or multi-part student questions before backend search.

**Query-layer functions:**

1. Classify query intent:
   - document/source lookup
   - personal memory
   - temporal event
   - entity/relationship
   - formula/exact term
   - multi-source synthesis
2. Rewrite vague student language into technical retrieval terms.
3. Generate multiple queries when one query is insufficient.
4. Extract filters such as course, lecture, source type, semester, and date.
5. Decide whether to query OpenViking, Hindsight, or both.
6. Preserve the original query for answer phrasing.

**Example:**

```text
User: How do I know if the feedback system is stable?

Generated retrieval queries:
- closed-loop stability criterion
- Nyquist criterion
- Routh-Hurwitz stability
- closed-loop pole locations
- gain margin and phase margin

Backend: OpenViking resources
```

**Files:**
- Create: `Personal_KB/scripts/query_planner.py`
- Modify: `Personal_KB/scripts/openviking_backend.py`
- Modify: `Personal_KB/scripts/common_client.py`
- Create: `Personal_KB/tests/test_query_planner.py`

**Requirements:**

- Query planning must have a timeout and deterministic fallback.
- If the LLM fails, use the original query directly.
- The planner must not invent course filters or source constraints.
- Every generated query must retain a link to the original user query for debugging.
- Query traces should record rewritten queries, selected backends, and final source URIs without storing secrets.

---

## 6.7 Integrate with the `personal-kb` Hermes skill

**Objective:** Expose the dual-backend retrieval architecture through stable study commands.

**Commands:**

- `/personal-kb search` — query planner + OpenViking/Hindsight routing
- `/personal-kb source` — exact OpenViking source lookup and read
- `/personal-kb explain` — retrieve source context then explain
- `/personal-kb quiz` — source-grounded active recall
- `/personal-kb review` — combine OpenViking resources with Hindsight weak-area memory
- `/personal-kb trace` — show query plan, backend, scores, and source URIs

**Verification:**

- Every answer includes source traceability where a document source exists.
- Personal-memory questions do not require OpenViking resources.
- Source questions do not depend on Hindsight having a retained fact.
- Mixed questions can use both systems without duplicating context.

---

## 6.8 Evaluation and production promotion

**Objective:** Select the production configuration using evidence rather than model size.

**Benchmark categories:**

- Direct factual lookup
- Ambiguous terminology
- Formula and symbol lookup
- Exact identifiers and acronyms
- Multi-source synthesis
- Cross-lecture relationships
- Temporal/personal-memory routing
- Vague student questions
- OCR/diagram-derived content
- Long-context source reading

**Required comparison:**

- Hindsight-only
- OpenViking dense with `bge-small-en`
- OpenViking dense with `bge-base-en`/`bge-large-en`
- OpenViking dense + reranker
- OpenViking hybrid + reranker
- Full LLM query-layer pipeline

**Promotion criteria:**

- Better held-out source accuracy and context usefulness
- No unacceptable regression in latency
- Stable laptop resource usage
- Clean failure fallback
- Reproducible index rebuild
- Traceable source URIs
- No duplicate or cross-course contamination

---

## 6.9 Rollout order

1. Implement OpenViking as the Personal source backend.
2. Move the complete OCR/Whisper-to-index pipeline to the laptop.
3. Establish the existing `bge-small-en` baseline.
4. Build isolated `bge-base-en` and `bge-large-en` candidate indexes.
5. Add reranking.
6. Add lexical/hybrid retrieval.
7. Add LLM query planning and backend routing.
8. Run the expanded benchmark.
9. Promote the best configuration and retain the previous index as rollback until production validation is complete.

---

## Open Questions

- Whether the laptop can run the selected larger embedding model concurrently with Whisper and OCR without unacceptable latency.
- Whether local reranking should use `bge-reranker-base`, `bge-reranker-large`, or a supported remote endpoint.
- Whether OpenViking's native sparse/hybrid backend is sufficient or a separate BM25 index is needed.
- Which LLM should power query planning: the existing laptop model, the configured OpenViking VLM, or a remote low-latency model.
- How much context should be returned for `/quiz` versus `/review`.
