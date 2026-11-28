# Research evaluation

The evaluation package currently provides deterministic scoring utilities for saved reports and retrieved source rankings. It does **not** claim that a report is factually correct simply because it has valid citation IDs or a high model-verifier score.

## Available metrics

- Retrieval: Precision@K, Recall@K, MRR@K and nDCG@K, using human-labelled relevant source URLs.
- Citation integrity: citation precision and invalid citation count.
- Claim-verification bookkeeping: verification coverage, supported-claim rate, contradicted-claim count and unresolved-claim count.
- Reviewer output: records the reviewer score and publication-readiness flag without treating the LLM score as calibrated truth.

## Score a saved report

```bash
research-eval path/to/report.json \
  --relevant-source-url https://example.org/primary-paper \
  --relevant-source-url https://example.org/official-record
```

The URL labels must come from a human-reviewed relevance annotation. Do not use generated citation IDs as ground truth labels.

## Build the benchmark before setting quality gates

No report-quality baseline is enabled yet. Before enforcing thresholds in CI, curate a dataset with:

1. Questions and an explicit scope/timeframe.
2. Frozen search snapshots or a licensed static corpus, including provenance and retrieval date.
3. Human-labelled relevant sources and atomic expected claims.
4. Adjudicated supported/contradicted/insufficient evidence labels with supporting passages.
5. Rubrics for completeness, reasoning quality, uncertainty handling and temporal correctness.
6. At least two annotators for a meaningful subset, with adjudication and agreement reporting.

Keep a small deterministic CI suite for schema, citation and scoring regressions. Run model-based evaluations separately (nightly or on prompt/model changes) and compare them with reviewed baselines. Do not automatically replace a baseline after a regression.
