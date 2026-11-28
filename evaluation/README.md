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


## Human-adjudicated ResearchBench tooling

The benchmark tools separate gold dataset labels from model outputs and human reviews. Start with the empty `evaluation/datasets/benchmark_template.json` (status `draft`); it is not a benchmark dataset and must not be used for performance claims.

- Dataset schema: `evaluation/schemas.py`.
- Snapshot and annotation validation: `evaluation/dataset.py`.
- Cohen's kappa and quadratic-weighted kappa: `evaluation/agreement.py`.
- Graded retrieval, claim classification, claim coverage/alignment, evidence-source matching, exact quote matching, and report-rubric scoring: `evaluation/scoring.py`.
- Annotation instructions: `evaluation/ANNOTATION_GUIDE.md`.
- Scoring rubric: `evaluation/rubric.json`.

### Commands

```bash
research-bench validate evaluation/datasets/benchmark_template.json
research-bench validate evaluation/datasets/benchmark_v1.json --require-ready --verify-snapshots --repo-root .
research-bench agreement evaluation/datasets/benchmark_v1.json --case-id CASE_ID
research-bench score --dataset evaluation/datasets/benchmark_v1.json --case-id CASE_ID \
  --report path/to/report.json --run-annotation path/to/run_annotation.json
```

A ready case requires two independent, complete source/claim annotations and a complete final adjudication. The scorer requires two independent reviews of the generated report, one-to-one mapping of each gold and generated claim, and scores for all seven rubric dimensions. It also checks that the run annotation identifies the same benchmark ID and version being scored.

### Licensing and frozen evidence

Redistributable and excerpt-permitted sources use repository snapshots with SHA-256. Restricted sources use a `controlled_snapshot_ref`; the repository validator reports them as not locally verified and they must be checked in their controlled store. Metadata-only sources do not have stored text and cannot be used as stored gold evidence spans. Do not copy restricted full text into this repository.

The tooling is ready for dataset curation; the actual corpus, adjudicated labels, and reviewed baselines still need to be created by human annotators. No model-generated labels are treated as gold data.
