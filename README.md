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

**Privacy:** trace content is excluded by default. Prompt text, research questions, provider queries, retrieved passages and model inputs/outputs are not sent to telemetry unless LANGFUSE_TRACE_CONTENT=true is explicitly enabled. Review your deployment's data-handling requirements before enabling content capture.

## Offline evaluation

Install the development tools and score a saved report:

```bash
pip install -e '.[dev]'
research-eval path/to/report.json \
  --relevant-source-url https://example.org/primary-source
```

Retrieval relevance labels must be human-reviewed. The current metrics measure retrieval ranking, citation bookkeeping and verification coverage; they do not independently establish factual truth. See evaluation/README.md before creating CI quality thresholds.

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
4. **Evaluation:** expand retrieval and report-quality benchmarks with human-adjudicated examples and CI regression gates.
5. **Runtime:** durable external checkpoint storage, API hardening and Modal deployment.

This remains an evolving research system, not a claim of production readiness. Model-produced verification and review outputs are fallible and must be measured against expert annotations.
