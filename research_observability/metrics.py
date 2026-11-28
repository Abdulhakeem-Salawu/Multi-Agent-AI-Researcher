from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
except ImportError:  # Metrics remain optional for lightweight local use.
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = Counter = Histogram = generate_latest = None  # type: ignore[assignment]

PROMETHEUS_AVAILABLE = Counter is not None
REGISTRY = CollectorRegistry(auto_describe=True) if PROMETHEUS_AVAILABLE else None

if PROMETHEUS_AVAILABLE:
    RUNS = Counter(
        "research_runs_total",
        "Completed or failed research runs.",
        ("entrypoint", "outcome"),
        registry=REGISTRY,
    )
    RUN_DURATION = Histogram(
        "research_run_duration_seconds",
        "End-to-end research run duration.",
        ("entrypoint",),
        registry=REGISTRY,
    )
    NODE_RUNS = Counter(
        "research_node_executions_total",
        "LangGraph node executions.",
        ("node", "outcome"),
        registry=REGISTRY,
    )
    NODE_DURATION = Histogram(
        "research_node_duration_seconds",
        "LangGraph node execution duration.",
        ("node",),
        registry=REGISTRY,
    )
    SEARCH_CALLS = Counter(
        "research_search_calls_total",
        "Web-search provider calls.",
        ("outcome",),
        registry=REGISTRY,
    )
    SEARCH_DURATION = Histogram(
        "research_search_duration_seconds",
        "Web-search provider call duration.",
        registry=REGISTRY,
    )
    LLM_CALLS = Counter(
        "research_llm_calls_total",
        "LLM completion calls.",
        ("model", "outcome"),
        registry=REGISTRY,
    )
    LLM_DURATION = Histogram(
        "research_llm_duration_seconds",
        "LLM completion duration.",
        ("model",),
        registry=REGISTRY,
    )
    VERIFICATIONS = Counter(
        "research_claim_verifications_total",
        "Final claim-verification outcomes.",
        ("status",),
        registry=REGISTRY,
    )


def record_node(node: str, outcome: str, duration_seconds: float) -> None:
    if PROMETHEUS_AVAILABLE:
        NODE_RUNS.labels(node=node, outcome=outcome).inc()
        NODE_DURATION.labels(node=node).observe(max(0.0, duration_seconds))


def record_search(outcome: str, duration_seconds: float) -> None:
    if PROMETHEUS_AVAILABLE:
        SEARCH_CALLS.labels(outcome=outcome).inc()
        SEARCH_DURATION.observe(max(0.0, duration_seconds))


def record_llm(model: str, outcome: str, duration_seconds: float) -> None:
    if PROMETHEUS_AVAILABLE:
        LLM_CALLS.labels(model=model or "unknown", outcome=outcome).inc()
        LLM_DURATION.labels(model=model or "unknown").observe(max(0.0, duration_seconds))


def record_verification(status: str) -> None:
    if PROMETHEUS_AVAILABLE:
        VERIFICATIONS.labels(status=status).inc()


@contextmanager
def track_research_run(entrypoint: str) -> Iterator[None]:
    started = perf_counter()
    outcome = "error"
    try:
        yield
        outcome = "success"
    finally:
        if PROMETHEUS_AVAILABLE:
            RUNS.labels(entrypoint=entrypoint, outcome=outcome).inc()
            RUN_DURATION.labels(entrypoint=entrypoint).observe(perf_counter() - started)


def prometheus_payload() -> tuple[bytes, str] | None:
    if not PROMETHEUS_AVAILABLE:
        return None
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
