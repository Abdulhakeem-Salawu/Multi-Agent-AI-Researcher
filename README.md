# Multi-Agent AI Researcher

An evidence-grounded research assistant built with **LangGraph**, Tavily and an OpenAI-compatible LLM interface. The orchestration separates search, claim verification and report-level review so that retrieval relevance, claim evidence and overall report quality are not conflated.

## Current graph

```text
Question
   |
   v
Intake & validation -> Research planner
                            |
              +-------------+-------------+
              v             v             v
          Researcher 1  Researcher 2  Researcher N
              +-------------+-------------+
                            v
                 Evidence aggregation/ranking
                            |
                            v
                    Evidence-grounded writer
                            |
                            v
                    Citation / lexical audit
                            |
                            v
                   Atomic claim extraction
                            |
              +-------------+-------------+
              v             v             v
          Verifier 1     Verifier 2    Verifier N
              +-------------+-------------+
                            v
                    Verification aggregation
                            |
                unresolved claim + budget?
                       /          \
                     yes           no
                      |             |
                Gap research    Report reviewer
                      |             |
                 Search again       |
                      |             |
               Rewrite & reverify  |
                       \            /
                        Final report
```

Search tasks and claim verifiers are dispatched independently with LangGraph `Send`. State reducers merge parallel search outputs, errors and per-claim verification results. Search results are sorted deterministically before ranking, and a verifier's evidence quotes are checked against the exact retrieved source text before they are retained.

## Quick start

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
cp .env.example .env
```

Set `TAVILY_API_KEY` and configure an OpenAI-compatible model endpoint in `.env`. Ollama is supported through its OpenAI-compatible API.

Run a research question:

```bash
research-agent "How do hybrid retrieval methods improve evidence selection?"
```

Run the API:

```bash
uvicorn research_agent.api:app --reload
```

Then POST JSON to `/research`:

```json
{"question": "How do hybrid retrieval methods improve evidence selection?"}
```

The API also exposes `GET /health` and `GET /runs/{run_id}`.

## Verification and review semantics

- **Lexical audit:** checks citation IDs and token overlap only. A lexical pass is not proof of entailment.
- **Claim verification:** the verifier returns one of `supported`, `contradicted`, `insufficient_evidence` or `uncertain`, with source IDs and exact quoted spans.
- **Span validation:** quote spans that cannot be matched to the retrieved source text are discarded. A `supported` or `contradicted` verdict without a valid corresponding span is downgraded to `uncertain`.
- **Knowledge-gap loop:** claims marked unresolved and requiring more research can trigger targeted follow-up queries. `MAX_RESEARCH_ROUNDS` bounds the number of follow-up rounds (0–3).
- **Report review:** a distinct reviewer scores completeness, analytical depth and citation quality. It cannot rewrite or override claim-verification results. Any unresolved claim prevents a positive publication-readiness decision.
- **No verifier model:** claims remain `uncertain`; the system does not present lexical overlap as verification. Report review is marked `not_run`.

The confidence values and review scores are model assessments, not calibrated probabilities or guarantees. They must be evaluated against human-labelled examples before they are used as quality thresholds.

## Observability

Install the optional telemetry stack:

```bash
pip install -e ".[observability]"
```

The application instruments graph nodes, web-search requests, LLM calls and end-to-end runs. Prometheus metrics are available at GET /metrics when the optional Prometheus client is installed. A scrape configuration is provided in deployment/prometheus.yml; import deployment/grafana-dashboard.json into Grafana as a starter dashboard.

For Langfuse, configure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL. Prompt management is opt-in through LANGFUSE_PROMPT_MANAGEMENT_ENABLED=true. Create the six named text prompts in Langfuse and promote the intended version to the configured label. Missing/unavailable prompts fall back to the checked-in system instructions.

The Langfuse Python SDK v5 requires Python 3.11+. The rest of the project retains its Python 3.10 floor; Python 3.10 can use the OpenTelemetry and Prometheus integrations without the Langfuse SDK. Configure OTLP export with OTEL_EXPORTER_OTLP_ENDPOINT.

### GitHub Actions credentials

For GitHub-hosted CI, add these under **Settings → Secrets and variables → Actions → Repository secrets**:

- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_BASE_URL
- LANGFUSE_TRACE_CONTENT

The URL and content-capture setting may alternatively be repository variables; the workflow accepts either. The dedicated **Langfuse connectivity** workflow maps these values into a one-off smoke test. It checks authentication, sends only synthetic metadata/output, and flushes the observation. It does not send a research question or model prompt.

Run it from **Actions → Langfuse connectivity → Run workflow**, selecting the feature branch. The same workflow also runs automatically when its own file or the smoke-test script changes on the feature branch. Do not expose these credentials to workflows triggered by untrusted fork pull requests.

**Privacy:** trace content is excluded by default. Prompt text, research questions, provider queries, retrieved passages and model inputs/outputs are not sent to telemetry unless LANGFUSE_TRACE_CONTENT=true is explicitly enabled. Review your deployment's data-handling requirements before enabling content capture.

## Offline evaluation

Install the development tools and score a saved report:

```bash
pip install -e '.[dev]'
research-eval path/to/report.json \
  --relevant-source-url https://example.org/primary-source
```

Retrieval relevance labels must be human-reviewed. The current metrics measure retrieval ranking, citation bookkeeping and verification coverage; they do not independently establish factual truth. See evaluation/README.md before creating CI quality thresholds.

## Human-adjudicated ResearchBench

The benchmark tooling is implemented, but the benchmark corpus is intentionally empty. Start with `evaluation/datasets/benchmark_template.json`; it is marked `draft` and contains no gold claims or performance labels.

Each case is designed to hold a frozen evidence snapshot, provenance and licensing metadata, atomic expected claims, two independent source/claim annotations, and a final adjudication. The seven-dimension report rubric and detailed instructions are in `evaluation/rubric.json` and `evaluation/ANNOTATION_GUIDE.md`. A run annotation template records the benchmark version, Git commit, model and prompt versions.

Validate the schema template:

```bash
research-bench validate evaluation/datasets/benchmark_template.json
```

Validate a populated release and verify locally stored snapshots:

```bash
research-bench validate evaluation/datasets/benchmark_v1.json --require-ready --verify-snapshots --repo-root .
research-bench agreement evaluation/datasets/benchmark_v1.json --case-id CASE_ID
```

Score a report only after two independent human reviews and adjudication:

```bash
research-bench score \
  --dataset evaluation/datasets/benchmark_v1.json \
  --case-id CASE_ID \
  --report path/to/report.json \
  --run-annotation path/to/run_annotation.json
```

The scorer reports graded retrieval, claim alignment/coverage, four-way claim-status classification, evidence-source precision/recall, exact quote matches and human report-rubric scores. It does not treat the model's own verifier or reviewer output as ground truth.

Sources marked `restricted` must use a `controlled_snapshot_ref` and must not be committed as full text. `metadata_only` sources cannot be cited as stored evidence spans. Locally stored redistributable or excerpt-permitted snapshots are SHA-256 checked.

### Curate and annotate a candidate case

The 24-case seed pack is intentionally unfrozen. For each case, register every source's rights status first. For restricted source material, keep the full document in an approved controlled store and record only its reference. For local snapshots, pass only text whose redistribution or excerpt permission has been verified.

Example using the LangGraph candidate case and a controlled source store:

~~~bash
research-bench register-source evaluation/datasets/benchmark_candidates_v0.2.json \
  --case-id LG-SEND-001 --source-id langgraph_graph_api \
  --redistribution restricted --license-id RIGHTS_BASIS_RECORDED \
  --controlled-ref vault://research-bench/LG-SEND-001/langgraph-graph-api-v1 \
  --output /tmp/research-bench-rights.json --repo-root .

research-bench freeze-case /tmp/research-bench-rights.json \
  --case-id LG-SEND-001 --output /tmp/research-bench-frozen.json --repo-root .

research-bench export-annotations /tmp/research-bench-frozen.json \
  --output-dir .research-bench/forms \
  --annotator-ids annotator_01 annotator_02 \
  --case-id LG-SEND-001
~~~

Give each annotator only their own file under .research-bench/forms/LG-SEND-001/. They must access controlled source content using the approved vault reference. After both forms are completed independently:

~~~bash
research-bench import-annotations /tmp/research-bench-frozen.json \
  --annotation .research-bench/forms/LG-SEND-001/annotator_01.json \
  --annotation .research-bench/forms/LG-SEND-001/annotator_02.json \
  --output /tmp/research-bench-annotated.json --repo-root .

research-bench export-adjudication /tmp/research-bench-annotated.json \
  --case-id LG-SEND-001 --adjudicator-id adjudicator_01 \
  --output .research-bench/forms/LG-SEND-001/adjudication.json

research-bench import-adjudication /tmp/research-bench-annotated.json \
  --form .research-bench/forms/LG-SEND-001/adjudication.json \
  --output /tmp/research-bench-adjudicated.json --repo-root .
~~~

Change RIGHTS_BASIS_RECORDED and the controlled-store reference to the actual documented permission and location. Do not use these placeholders literally. Repeat source registration for every source in a case before freezing it. See evaluation/ANNOTATION_GUIDE.md for redistributable, excerpt-only, metadata-only and restricted-source handling.

## Design decisions

- **LangGraph owns orchestration.** Search, retrieval, model access and verification remain ordinary Python services behind small interfaces.
- **Parallelism and loops are bounded.** The planner caps tasks per round; `MAX_RESEARCH_ROUNDS` caps follow-up cycles. Provider calls have timeouts and bounded retries.
- **Evidence is explicit.** Sources retain URL, title, publication date when available, retrieval timestamp, provider score and excerpts. Verification spans identify the exact quoted text used by the verifier.
- **Failures are visible.** Per-query failures, invalid evidence spans and planner/verifier/reviewer failures are recorded. The system does not fabricate successful searches or verified claims.
- **Completed runs are persisted.** The report store uses SQLite. LangGraph checkpoint persistence is injectable; the default graph does not pretend that an in-memory checkpoint is durable across restarts.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TAVILY_API_KEY` | Tavily search credential | required |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:11434/v1` |
| `LLM_API_KEY` | Optional model API credential | empty |
| `LLM_MODEL` | Model identifier | `gemma3:1b` |
| `REQUEST_TIMEOUT_SECONDS` | Provider request timeout | `45` |
| `MAX_RESEARCH_QUERIES` | Maximum parallel queries per round | `4` |
| `MAX_RESULTS_PER_QUERY` | Search results per task | `8` |
| `MAX_EVIDENCE_SOURCES` | Sources passed to synthesis and verification | `8` |
| `MAX_RESEARCH_ROUNDS` | Maximum follow-up research rounds | `1` |
| `RETRIEVAL_MODE` | `hybrid` or `bm25` ranking | `hybrid` |
| `RESEARCH_DB_PATH` | SQLite completed-run database | `data/research_runs.sqlite3` |

## Development roadmap

1. **Foundation:** LangGraph fan-out/fan-in, structured contracts, deterministic source ordering, report persistence and CI.
2. **Claim verification:** atomic extraction, parallel verification, quote-span validation and a bounded research-gap loop. Implemented in this milestone; still requires calibration against human-labelled claims.
3. **Observability:** Langfuse traces/prompts/evaluations, OpenTelemetry instrumentation and Prometheus/Grafana operational metrics.
4. **Evaluation:** human-adjudicated ResearchBench schemas, snapshot validation, inter-annotator agreement, report scoring and the annotation guide are implemented. Populate, annotate and adjudicate a real corpus before setting CI quality thresholds.
5. **Runtime:** durable external checkpoint storage, API hardening and Modal deployment.

This remains an evolving research system, not a claim of production readiness. Model-produced verification and review outputs are fallible and must be measured against expert annotations.


## External research benchmark tracks

The evaluation package includes optional adapters for DeepResearch Bench II, FRAMES and BrowseComp-Plus. These remain separate from the internal human-adjudicated ResearchBench because their source labels and scoring semantics differ. See [evaluation/EXTERNAL_DATASETS.md](evaluation/EXTERNAL_DATASETS.md) for licence scope, local download instructions, reproducibility requirements and track-specific scoring.


### External benchmark experiments

Pinned dataset downloads, local-only BrowseComp-Plus preparation and FTS5 indexing, and task-level LangGraph experiment runs are documented in [evaluation/EXTERNAL_DATASETS.md](evaluation/EXTERNAL_DATASETS.md). Experiment outputs include SHA-256 and run manifests; no upstream corpora are checked into this repository.


## Modal cloud benchmark execution

The repository includes a manually dispatched GitHub Actions workflow that runs external benchmark preparation and experiments inside Modal. Your `TAVILY_KEY` secret is recognized. Configure `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` GitHub secrets (separate from your local `modal setup`), plus an HTTPS OpenAI-compatible agent endpoint and judge/extractor endpoint as described in [evaluation/EXTERNAL_DATASETS.md](evaluation/EXTERNAL_DATASETS.md). Your existing `TAVILY_KEY` GitHub secret is used without renaming.

Large dataset and BrowseComp-Plus corpus downloads are made directly by Modal workers and saved to a persistent Modal Volume. They are not downloaded by your local computer or copied into GitHub. Retrieve only the small `summary.json` outputs unless you deliberately intend to transfer the corpus.
