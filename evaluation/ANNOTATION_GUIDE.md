# Human annotation and adjudication guide

## Non-negotiable rules

1. Do not treat the model's claim-verifier status, confidence, or report-review score as ground truth.
2. Annotators work independently and must not see one another's labels before submitting.
3. Every judgment needs a rationale. Supported and contradicted claim judgments must cite an exact span in a frozen source snapshot.
4. Freeze evidence before annotation. Live web results must not silently change the evidence available to an already-versioned case.
5. Record each source license and permitted redistribution. Do not commit restricted full-text content to this repository. For restricted sources, preserve only permitted metadata/excerpts and use controlled storage for full snapshots.
6. Keep benchmark labels, model outputs and human run annotations in separate artifacts.

## Source snapshot procedure

For each source, record its stable ID, exact URL, title, publisher, source type, publication date if known, retrieval time, capture method, license identifier and redistribution permission. Where storage is allowed, preserve a UTF-8 frozen text snapshot and its SHA-256 digest. Do not fabricate a digest for content that was not captured.

Use the redistribution field as follows:
- `redistributable`: full snapshot may be stored under the stated license;
- `excerpt_only`: only a limited excerpt may be stored;
- `metadata_only`: retain bibliographic metadata, but not source text;
- `restricted`: source text must remain in controlled storage and its `controlled_snapshot_ref` must identify that storage location;
- `unknown`: licensing has not yet been established; a case cannot be promoted as ready.

## Source relevance scale

| Grade | Meaning |
|---:|---|
| 0 | Irrelevant or misleading for this question |
| 1 | Tangential background only |
| 2 | Relevant context or partial support |
| 3 | Directly answers or strongly substantiates an important part |

Judge usefulness for the specific question and scope, not general publisher prestige.

## Atomic claim labels

Preserve meaningful qualifiers, populations, time ranges, units and conditions when splitting the question into independently checkable propositions.

- `supported`: frozen evidence directly supports the proposition at its stated scope.
- `contradicted`: frozen evidence directly conflicts with the proposition.
- `insufficient_evidence`: the frozen source set cannot establish or refute the proposition.
- `uncertain`: sources are ambiguous, materially inconsistent, or cannot be confidently interpreted.

For supported and contradicted labels, cite the shortest exact span demonstrating the judgment. For insufficient evidence, explain what is missing instead of inventing a quote. Restricted-source spans are reviewed against controlled evidence and are not locally hash-verified by the repository validator. Metadata-only sources cannot be used as stored gold evidence spans.

## Independent annotation and adjudication

For every case intended for a ready release:
1. At least two annotators independently grade every source and label every expected claim.
2. Calculate quadratic-weighted Cohen's kappa for 0–3 source relevance and nominal Cohen's kappa for claim status.
3. Report the number of cases/items. Kappa can be unstable or undefined for small or one-class samples; do not force a value.
4. An adjudicator resolves disagreements against the frozen evidence and records the rationale.
5. The adjudicated labels—not a majority vote without evidence—become the benchmark reference.

## Generated report review

Use `templates/run_annotation_template.json` for each system output. Record the benchmark/case version, run ID, Git commit, model and prompt versions. Human annotators map generated claims to benchmark claims using:
- `equivalent`: same proposition and material scope;
- `partial`: overlap, but a material qualifier or scope is missing/changed;
- `missing`: an expected gold claim is absent from the report;
- `extra`: a material generated claim has no gold counterpart.

Score all seven report rubric dimensions from 0 to 4. Each score requires a rationale, and factual/citation judgments should identify the sentence or evidence being assessed.

Two annotators review independently. An adjudicator records the final alignments and scores. Claim-status accuracy is computed only for equivalent alignments; report claim coverage and alignment precision must be shown alongside it. Partial matches must not silently count as fully correct.

## Promotion to a benchmark release

Set `status` to `ready` only after every case has:
- a frozen snapshot with provenance and explicit license/redistribution handling;
- at least two independent, complete annotations;
- complete adjudication of every source and expected claim;
- exact evidence spans that validate against frozen text where stored.

Run:

```bash
research-bench validate evaluation/datasets/benchmark_v1.json --require-ready --verify-snapshots
research-bench agreement evaluation/datasets/benchmark_v1.json --case-id CASE_ID
```

A draft dataset is useful for annotation operations but must not be used for public performance claims or enforced quality thresholds.


## Rights-aware snapshot and annotation CLI

The candidate seed pack is intentionally unfrozen. Before exporting forms, register a rights decision and freeze every source in the case. Snapshot registration is manual: the command never downloads or copies web pages on its own.

### 1. Register source rights and permitted text

For a source whose full text may be redistributed under the verified license:

~~~bash
research-bench register-source evaluation/datasets/benchmark_candidates_v0.2.json \
  --case-id CASE_ID --source-id SOURCE_ID \
  --redistribution redistributable --license-id LICENSE_IDENTIFIER \
  --input-file /path/to/verified-source-text.txt --confirm-rights \
  --output /tmp/benchmark-next.json --repo-root .
~~~

For an excerpt-only permission, supply a file containing only the permitted excerpt and use --redistribution excerpt_only. The tool hashes exactly the UTF-8 bytes supplied; it cannot decide whether your license interpretation is correct.

For a source that cannot be stored in the repository:

~~~bash
research-bench register-source /tmp/benchmark-next.json \
  --case-id CASE_ID --source-id SOURCE_ID \
  --redistribution restricted --license-id RIGHTS_BASIS \
  --controlled-ref vault://YOUR-CONTROLLED-STORE/SNAPSHOT-ID \
  --output /tmp/benchmark-next.json --repo-root . --overwrite-dataset
~~~

For metadata-only references, use --redistribution metadata_only and provide a documented rights basis in --license-id. Metadata-only sources cannot be used for stored gold evidence quotations.

The --confirm-rights switch is required before a local full-text or excerpt file can be copied. Never use it unless the supplied file is legally permitted to be stored in the repository. Restricted source text must never be passed with --input-file.

### 2. Freeze the evidence for a case

After every source in that case has an explicit rights decision and an appropriate local snapshot or controlled reference:

~~~bash
research-bench freeze-case /tmp/benchmark-next.json \
  --case-id CASE_ID --output /tmp/benchmark-frozen.json --repo-root .
~~~

This verifies local SHA-256 hashes and UTF-8 text before setting the case's frozen timestamp. A frozen case cannot be changed by the registration command. Create a new dataset version if evidence must change.

### 3. Export independent forms

~~~bash
research-bench export-annotations /tmp/benchmark-frozen.json \
  --output-dir .research-bench/forms \
  --annotator-ids annotator_01 annotator_02 \
  --case-id CASE_ID
~~~

Each annotator receives a separate file. Forms contain no other annotator's judgments and have blank labels; they are not complete annotations until the annotator fills in the date, every source grade, every claim status, rationales and required evidence spans.

### 4. Import completed annotations

~~~bash
research-bench import-annotations /tmp/benchmark-frozen.json \
  --annotation .research-bench/forms/CASE_ID/annotator_01.json \
  --annotation .research-bench/forms/CASE_ID/annotator_02.json \
  --output /tmp/benchmark-annotated.json --repo-root .
~~~

Import rejects duplicate annotators, missing or extra source/claim labels, supported/contradicted claims without evidence spans, mismatched benchmark versions, invalid quotations and hash mismatches. It writes a new dataset file instead of overwriting the input.

### 5. Export and import adjudication

~~~bash
research-bench export-adjudication /tmp/benchmark-annotated.json \
  --case-id CASE_ID --adjudicator-id adjudicator_01 \
  --output .research-bench/forms/CASE_ID/adjudication.json

research-bench import-adjudication /tmp/benchmark-annotated.json \
  --form .research-bench/forms/CASE_ID/adjudication.json \
  --output /tmp/benchmark-adjudicated.json --repo-root .
~~~

Fill the adjudication form with a final source grade and claim status, rationale and timestamp before importing. The adjudicator must be different from the independent annotators. The form includes both independent submissions for comparison and leaves final labels blank.

Keep .research-bench/ local. Do not commit annotator working forms or controlled-store references that reveal sensitive internal locations. Only publish benchmark files after the rights, snapshot, annotation and adjudication review is complete.
