# Observability deployment notes

## Install the optional integrations

```bash
pip install -e ".[observability]"
```

Langfuse Python SDK v5 is installed only on Python 3.11+. Python 3.10 deployments can still use Prometheus and OpenTelemetry; their Langfuse prompt-management calls fall back to checked-in prompts until the runtime is upgraded to Python 3.11+.

## Langfuse

Create a Langfuse project and add its public and secret keys to the environment. Set LANGFUSE_BASE_URL to an absolute URL including the scheme, for example https://cloud.langfuse.com (or your Langfuse region/self-hosted URL). The application automatically prefixes https:// when only a hostname is supplied. For GitHub Actions, store LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY under Settings → Secrets and variables → Actions → Repository secrets. LANGFUSE_BASE_URL and LANGFUSE_TRACE_CONTENT can also be stored as repository variables. Create text prompts using the names below and promote the version you intend to run to the production label (or set LANGFUSE_PROMPT_LABEL to a label you maintain).

| Prompt name | Purpose |
|---|---|
| research-planner | Initial search-query generation |
| research-writer | Evidence-grounded report synthesis |
| atomic-claim-extractor | Atomic factual-claim extraction |
| claim-verifier | Claim-to-evidence verification |
| knowledge-gap-planner | Follow-up query planning |
| report-reviewer | Whole-report quality review |

Each prompt should contain the corresponding system instructions currently present in research_agent/graph.py. Start by copying those instructions exactly; prompt changes should then be versioned and evaluated before promotion.

Prompt management is disabled by default. Enable it with LANGFUSE_PROMPT_MANAGEMENT_ENABLED=true. If a prompt cannot be fetched, the graph falls back to the checked-in system prompt and records the prompt source in observation metadata.

By default, trace metadata contains prompt names/versions, node names, model name and outcome/summary counts. It does not include research questions, prompt text, LLM inputs/outputs, web-search queries or retrieved passages. Set LANGFUSE_TRACE_CONTENT=true only after reviewing your data-retention and privacy requirements.

## OpenTelemetry

Set OTEL_EXPORTER_OTLP_ENDPOINT to an OTLP/HTTP collector endpoint, such as http://otel-collector:4318. The service creates spans for each graph node and LLM/search call. Use OTEL_SERVICE_NAME to name this service in your backend.

## Prometheus and Grafana

The API exposes Prometheus metrics on /metrics when prometheus-client is installed. Start Prometheus with prometheus.yml in this folder, changing the target to the address of the API service on your deployment network.

The metrics include:
- run counts and run duration by entrypoint/outcome;
- node counts and node durations;
- web-search calls and latency;
- LLM calls and latency by model/outcome;
- claim-verification outcomes.

Import a dashboard into Grafana and build panels from those series. Keep labels low-cardinality: do not add user IDs, run IDs, full queries, URLs or claim text as metric labels.

## Evaluation

Run deterministic offline scoring against a saved report:

```bash
research-eval path/to/report.json \\
  --relevant-source-url https://example.org/primary-source
```

Relevant-source URLs must be human-labelled. A high citation-integrity or reviewer score is not proof that the report is factually correct; use adjudicated claim/evidence labels for that evaluation.


## GitHub Actions connectivity check

The `.github/workflows/langfuse-smoke.yml` workflow uses the repository secret names above. Run **Actions → Langfuse connectivity → Run workflow** on `feature/langgraph-foundation`, or let it run when its workflow/script files are pushed to that branch.

It calls the SDK's `auth_check()`, sends one synthetic observation, and flushes it. It does not include a real research question or source content. The workflow is not attached to pull-request events, so credentials are not passed to untrusted fork PR jobs.
