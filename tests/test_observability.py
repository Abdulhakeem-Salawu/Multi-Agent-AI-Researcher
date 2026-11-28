from __future__ import annotations

from research_observability.instrumentation import InstrumentedChatModel, InstrumentedSearchProvider
from research_observability.prompts import (
    ResolvedPrompt,
    current_prompt_metadata,
    prompt_context,
    resolve_text_prompt,
)
from research_observability.tracing import instrument_node, observation


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
