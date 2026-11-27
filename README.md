# Multi-Agent AI Researcher

An evidence-grounded research assistant built with **LangGraph**, Tavily and an OpenAI-compatible LLM interface. This repository is the active development home for the research-agent project; the initial implementation focuses on explicit orchestration, evidence traceability and deterministic tests.

## Current graph

```text
Question
   |
   v
Intake & validation
   |
   v
Research planner
   |
   +---- Send(task 1) ---> Researcher 1 ---+
   +---- Send(task 2) ---> Researcher 2 ---+--> Evidence aggregation & ranking
   +---- Send(task N) ---> Researcher N ---+             |
                                                        v
                                               Evidence-grounded writer
                                                        |
                                                        v
                                                Citation / claim audit
                                                        |
                                                        v
                                                  Research report
```

Each planned query becomes a separate LangGraph `Send` task. Search results and task errors are merged through explicit state reducers. Results are ordered deterministically before ranking so worker completion order does not silently change source IDs.

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

## Design decisions

- **LangGraph owns orchestration.** Search, retrieval, model access, and claim screening remain ordinary Python services behind small interfaces.
- **Parallelism is bounded.** The planner produces at most `MAX_RESEARCH_QUERIES` tasks. Each task has a stable index, and each provider call has a timeout and bounded retries.
- **Evidence is explicit.** A report source retains URL, title, publication date when available, retrieval timestamp, provider score and an excerpt.
- **Citation screening is not fact verification.** The first milestone audits citation IDs and lexical overlap. A passing lexical score is only a candidate signal; it does not establish entailment or truth.
- **Failures are visible.** Per-query search errors and planning/synthesis fallbacks are recorded in report limitations. The system does not fabricate successful search results.
- **Completed runs are persisted.** The report store uses SQLite. LangGraph checkpoint persistence is injectable; the default graph does not pretend that an in-memory checkpoint is durable across restarts.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `TAVILY_API_KEY` | Tavily search credential | required |
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `http://localhost:11434/v1` |
| `LLM_API_KEY` | Optional model API credential | empty |
| `LLM_MODEL` | Model identifier | `gemma3:1b` |
| `REQUEST_TIMEOUT_SECONDS` | Provider request timeout | `45` |
| `MAX_RESEARCH_QUERIES` | Maximum parallel research tasks | `4` |
| `MAX_RESULTS_PER_QUERY` | Search results per task | `8` |
| `MAX_EVIDENCE_SOURCES` | Sources passed to synthesis | `8` |
| `RETRIEVAL_MODE` | `hybrid` or `bm25` ranking | `hybrid` |
| `RESEARCH_DB_PATH` | SQLite completed-run database | `data/research_runs.sqlite3` |

## Development roadmap

1. **Foundation (this milestone):** LangGraph fan-out/fan-in, structured contracts, deterministic source ordering, report persistence and tests.
2. **Verification:** atomic claim extraction, parallel claim-verifier tasks, evidence-span alignment, contradiction handling and calibrated abstention.
3. **Reflection:** bounded knowledge-gap research loops with explicit budget and stopping rules.
4. **Observability:** Langfuse traces/prompts/evaluations, OpenTelemetry instrumentation and Prometheus/Grafana operational metrics.
5. **Evaluation:** expand the research benchmark with human-adjudicated report-level rubrics and CI regression gates.
6. **Runtime:** durable checkpointer backend, API hardening and Modal deployment.

The current implementation is a foundation, not a claim of production readiness. Evaluation thresholds and verification confidence must be calibrated on labelled examples before they are used as quality guarantees.
