# Expanded Retrieval Benchmark Results

**Run date:** 2026-07-22
**Mode:** offline/local only; `production_services_accessed: false`

## Synthetic regression set

- Corpus: 18 synthetic text sources (`data/test_sources/s01..s18.txt`)
- Queries: 18 hardened paraphrase/formula queries
- Model: `BAAI/bge-small-en-v1.5`

| Strategy | Recall@1 | Recall@5 | MRR |
|---|---:|---:|---:|
| BM25 lexical | 1.000 | 1.000 | 1.000 |
| BGE-small semantic | 1.000 | 1.000 | 1.000 |
| Hybrid RRF | 1.000 | 1.000 | 1.000 |

This is a regression gate, not evidence to promote a larger embedding model: the synthetic domains remain cleanly separated.

## Local-only real-material evaluation

The private working copy also contains an adversarial control-systems
benchmark over local notes, textbook extracts, lectures, homework, and exams.
Those source materials, manifests, runner, and tests are intentionally omitted
from the public snapshot because they are not part of the redistributable
project. The results are therefore not a public reproduction target.

## Reproduction

From the project root:

    TORCHINDUCTOR_CACHE_DIR=./.torch-cache python test_runs/synthetic_retrieval_benchmark.py --output test_runs/synthetic_benchmark_results.json
    python -m pytest -q tests/test_synthetic_retrieval_benchmark.py

The public runner is offline and does not ingest benchmark material into
OpenViking or Hindsight. The local-only evaluation remains available in the
private working copy.
