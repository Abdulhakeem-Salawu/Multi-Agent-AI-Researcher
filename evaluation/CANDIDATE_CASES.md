# Candidate research cases — curation seed pack

**Status: draft only. Do not use for quality claims, leaderboard scores, or CI acceptance thresholds.**

This file introduces 24 source-linked research questions and candidate atomic propositions for the human-adjudicated ResearchBench. It is an annotation work queue, not ground truth.

## Scope

The cases cover:
- LangGraph dynamic fan-out, parallel super-steps, and source-bounded questions.
- Python asyncio.gather, TaskGroup, exception propagation, and cancellation.
- SQLite WAL and write isolation.
- OpenTelemetry context propagation and trace correlation.
- Prometheus counters, gauges, histograms, and summaries.
- HTTP caching freshness, validation, and heuristic freshness.
- OAuth 2.0 PKCE.
- W3C PROV-O entity/activity/agent concepts and relationships.
- NIST AI RMF functions and Playbook interpretation.
- OWASP prompt injection, including direct and indirect forms.

## Important data-quality status

Every case currently has:
- One or more official source URLs.
- Candidate questions and draft atomic propositions.
- snapshot.frozen = false.
- redistribution = "unknown" and no license identifier.
- No local source text, SHA-256 digest, source relevance labels, claim-status labels, or adjudication.

The propositions are **not gold labels**. They were drafted to help reviewers structure the questions and must be checked against a frozen version of each source. One proposition may be intentionally false or unsupported to exercise contradicted/insufficient-evidence judgments; annotators must decide from the source, not from the case author's wording.

## Promotion workflow

For each case:

1. **Review scope and claim atomicity.** Preserve qualifiers, versions, units, exception modes, and time boundaries. Split propositions that contain multiple independent facts.
2. **Freeze sources.** Capture the exact URL, title, publisher, publication date if available, retrieval time, retrieval method, and immutable content digest.
3. **Clear rights.** Record the license identifier and permitted redistribution. Store a local text snapshot only when permitted. Otherwise use an approved controlled snapshot reference or metadata-only record. Never commit restricted full text.
4. **Add evidence spans.** For every supported or contradicted proposition, annotate the shortest exact passage that substantiates the judgment. Preserve quote offsets when a local snapshot is stored.
5. **Collect independent labels.** Two annotators independently grade every source and assign every claim a four-way status. They must not see each other's judgments.
6. **Adjudicate.** A separate adjudicator resolves disagreements against the frozen evidence and records the rationale.
7. **Validate.** Run the strict release validator and quote/hash checks only after the annotation package is complete.

~~~bash
research-bench validate evaluation/datasets/benchmark_candidates_v0.2.json
research-bench validate evaluation/datasets/benchmark_candidates_v0.2.json --require-ready --verify-snapshots --repo-root .
~~~

The first command checks schema/draft structure. The second is expected to fail until all source snapshots, license declarations, two annotations, complete adjudications, and evidence spans are supplied.

## Source list

| Case family | Primary source |
|---|---|
| LangGraph | [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| Python concurrency | [Coroutines and tasks](https://docs.python.org/3/library/asyncio-task.html) |
| SQLite WAL | [Write-Ahead Logging](https://www.sqlite.org/wal.html) |
| SQLite isolation | [Isolation in SQLite](https://www.sqlite.org/isolation.html) |
| OpenTelemetry | [Context propagation](https://opentelemetry.io/docs/concepts/context-propagation/) |
| Prometheus | [Metric types](https://prometheus.io/docs/concepts/metric_types/) |
| HTTP caching | [RFC 9111](https://www.rfc-editor.org/rfc/rfc9111.html) |
| OAuth PKCE | [RFC 7636](https://www.rfc-editor.org/rfc/rfc7636.html) |
| Provenance | [W3C PROV-O](https://www.w3.org/TR/prov-o/) |
| AI risk management | [NIST AI RMF Playbook](https://airc.nist.gov/airmf-resources/playbook/) |
| LLM security | [OWASP Prompt Injection glossary](https://genai.owasp.org/glossary/prompt-injection/) |

The source URLs are research leads checked on 2026-11-28. Their content and licensing must be reviewed again when the source snapshots are frozen. The benchmark's as_of date should be populated for any case whose answer depends on a changing standard, implementation, policy, or release.
