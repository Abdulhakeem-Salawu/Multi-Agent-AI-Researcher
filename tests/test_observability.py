from __future__ import annotations

from contextlib import contextmanager

from research_observability import instrumentation, tracing
from research_core.schemas import SearchHit
from research_observability.instrumentation import InstrumentedChatModel, InstrumentedSearchProvider
from research_observability.prompts import (
    ResolvedPrompt,
    current_prompt_metadata,
    prompt_context,
    resolve_text_prompt,
)
from research_observability.tracing import instrument_node, observation, trace_content_enabled


class FakeLLM:
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        return f"{system_prompt}|{user_prompt}|{temperature}"


class FakeSearch:
    def search(self, query: str, max_results: int = 8):
        return [{"query": query, "max_results": max_results}]


def test_instrumented_llm_preserves_delegate_contract(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    model = InstrumentedChatModel(FakeLLM(), "test-model")

    result = model.complete("system", "user", temperature=0.25)

    assert result == "system|user|0.25"


def test_instrumented_search_preserves_provider_result():
    provider = InstrumentedSearchProvider(FakeSearch())

    result = provider.search("test query", max_results=3)

    assert result == [{"query": "test query", "max_results": 3}]


def test_local_prompt_fallback_and_prompt_context(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PROMPT_MANAGEMENT_ENABLED", "false")
    prompt = resolve_text_prompt("research-planner", "local system instructions")

    assert prompt == ResolvedPrompt(
        name="research-planner",
        text="local system instructions",
        source="local",
    )
    with prompt_context(prompt):
        metadata = current_prompt_metadata()
    assert metadata["prompt.name"] == "research-planner"
    assert metadata["prompt.source"] == "local"
    assert current_prompt_metadata() == {}


def test_observation_and_instrumented_node_are_safe_without_credentials(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    node = instrument_node("test_node", lambda state: {"answer": state["question"]})
    with observation("test-observation"):
        result = node({"question": "hello research"})

    assert result == {"answer": "hello research"}



def test_flush_telemetry_is_safe_without_external_services(monkeypatch):
    from research_observability.tracing import flush_telemetry

    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    flush_telemetry()



class RecordingSpan:
    def __init__(self):
        self.updates = []
        self.attributes = {}

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _record_observations(monkeypatch, module):
    records = []

    @contextmanager
    def record(name, **kwargs):
        span = RecordingSpan()
        records.append({"name": name, "kwargs": kwargs, "span": span})
        yield span

    monkeypatch.setattr(module, "observation", record)
    return records


def test_trace_content_is_enabled_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_TRACE_CONTENT", raising=False)
    assert trace_content_enabled() is True

    monkeypatch.setenv("LANGFUSE_TRACE_CONTENT", "")
    assert trace_content_enabled() is True

    monkeypatch.setenv("LANGFUSE_TRACE_CONTENT", "false")
    assert trace_content_enabled() is False

    monkeypatch.setenv("LANGFUSE_TRACE_CONTENT", "0")
    assert trace_content_enabled() is False


def test_instrumented_llm_records_full_prompt_and_completion_by_default(monkeypatch):
    records = _record_observations(monkeypatch, instrumentation)
    monkeypatch.setattr(instrumentation, "trace_content_enabled", lambda: True)

    result = InstrumentedChatModel(FakeLLM(), "test-model").complete(
        "complete system instructions",
        "full research question and evidence",
        temperature=0.25,
    )

    record = records[0]
    assert record["name"] == "llm.completion"
    assert record["kwargs"]["input"] == {
        "system_prompt": "complete system instructions",
        "user_prompt": "full research question and evidence",
    }
    assert record["span"].updates[-1]["output"] == result


def test_instrumented_search_records_query_and_full_returned_source_content(monkeypatch):
    records = _record_observations(monkeypatch, instrumentation)
    monkeypatch.setattr(instrumentation, "trace_content_enabled", lambda: True)
    hit = SearchHit(
        title="Primary evidence",
        url="https://example.org/research",
        content="Search result snippet",
        raw_content="Full retrieved source passage with the substantive evidence.",
        score=0.95,
        published_date="2026-11-28",
    )

    class Provider:
        def search(self, query: str, max_results: int = 8):
            return [hit]

    result = InstrumentedSearchProvider(Provider(), "tavily").search(
        "the full search query",
        max_results=5,
    )

    assert result == [hit]
    record = records[0]
    assert record["kwargs"]["input"] == {
        "query": "the full search query",
        "max_results": 5,
    }
    output = record["span"].updates[-1]["output"]
    assert output["result_count"] == 1
    assert output["results"][0]["title"] == "Primary evidence"
    assert output["results"][0]["content"] == "Search result snippet"
    assert output["results"][0]["raw_content"] == (
        "Full retrieved source passage with the substantive evidence."
    )


def test_instrumented_node_records_full_state_and_output_by_default(monkeypatch):
    records = _record_observations(monkeypatch, tracing)
    monkeypatch.setattr(tracing, "trace_content_enabled", lambda: True)
    state = {"question": "the complete user research question", "sources": [{"text": "full evidence"}]}
    node = instrument_node("test_node", lambda _: {"answer": "complete node output"})

    result = node(state)

    record = records[0]
    assert record["kwargs"]["input"] == state
    assert record["span"].updates[-1]["output"] == result
