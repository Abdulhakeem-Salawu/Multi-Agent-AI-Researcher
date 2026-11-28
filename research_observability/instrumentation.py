from __future__ import annotations

from time import perf_counter
from typing import Any, Protocol

from research_core.schemas import SearchHit

from .metrics import record_llm, record_search
from .prompts import current_prompt_metadata
from .tracing import observation, trace_content_enabled


class ChatModel(Protocol):
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str: ...


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 8) -> list[SearchHit]: ...


class InstrumentedChatModel:
    def __init__(self, delegate: ChatModel, model_name: str = "unknown") -> None:
        self.delegate = delegate
        self.model_name = model_name or "unknown"

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        started = perf_counter()
        outcome = "error"
        input_payload = (
            {"system_prompt": system_prompt, "user_prompt": user_prompt}
            if trace_content_enabled()
            else None
        )
        with observation(
            "llm.completion",
            as_type="generation",
            model=self.model_name,
            model_parameters={"temperature": temperature},
            metadata={"llm.model": self.model_name, **current_prompt_metadata()},
            input=input_payload,
        ) as span:
            try:
                result = self.delegate.complete(system_prompt, user_prompt, temperature)
                outcome = "success"
                if trace_content_enabled() and hasattr(span, "update"):
                    span.update(output=result)
                return result
            except Exception as exc:
                if hasattr(span, "update"):
                    span.update(
                        level="ERROR",
                        status_message=(
                            f"{type(exc).__name__}: {exc}"
                            if trace_content_enabled()
                            else type(exc).__name__
                        ),
                    )
                raise
            finally:
                record_llm(self.model_name, outcome, perf_counter() - started)


def _search_result_for_trace(result: Any) -> Any:
    """Convert a provider hit into a JSON-compatible, content-complete trace value."""
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", exclude_none=True)
    if isinstance(result, dict):
        return dict(result)
    if isinstance(result, (list, tuple)):
        return [_search_result_for_trace(item) for item in result]
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if hasattr(result, "__dict__"):
        return dict(vars(result))
    return str(result)


class InstrumentedSearchProvider:
    def __init__(self, delegate: SearchProvider, provider_name: str = "tavily") -> None:
        self.delegate = delegate
        self.provider_name = provider_name or "custom"

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        started = perf_counter()
        outcome = "error"
        input_payload = {"query": query, "max_results": max_results} if trace_content_enabled() else None
        with observation(
            "search.web",
            metadata={"search.provider": self.provider_name, "search.max_results": max_results},
            input=input_payload,
        ) as span:
            try:
                results = self.delegate.search(query, max_results)
                outcome = "success"
                if trace_content_enabled() and hasattr(span, "update"):
                    span.update(
                        output={
                            "result_count": len(results),
                            "results": [_search_result_for_trace(item) for item in results],
                        }
                    )
                return results
            except Exception as exc:
                if hasattr(span, "update"):
                    span.update(
                        level="ERROR",
                        status_message=(
                            f"{type(exc).__name__}: {exc}"
                            if trace_content_enabled()
                            else type(exc).__name__
                        ),
                    )
                raise
            finally:
                record_search(outcome, perf_counter() - started)
