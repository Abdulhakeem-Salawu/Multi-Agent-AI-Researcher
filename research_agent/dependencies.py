from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from research_core.llm import ChatModel, OpenAICompatibleChatModel
from research_core.search import SearchProvider, TavilySearchProvider
from research_core.storage import ResearchRunStore


@dataclass(frozen=True)
class ResearchSettings:
    tavily_api_key: str | None = None
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str | None = None
    llm_model: str = "gemma3:1b"
    request_timeout_seconds: float = 45.0
    max_queries: int = 4
    max_results_per_query: int = 8
    max_evidence_sources: int = 8
    retrieval_mode: str = "hybrid"
    database_path: str = "data/research_runs.sqlite3"

    @classmethod
    def from_environment(cls) -> ResearchSettings:
        load_dotenv()
        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_model=os.getenv("LLM_MODEL", "gemma3:1b"),
            request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45")),
            max_queries=max(1, min(8, int(os.getenv("MAX_RESEARCH_QUERIES", "4")))),
            max_results_per_query=max(1, min(20, int(os.getenv("MAX_RESULTS_PER_QUERY", "8")))),
            max_evidence_sources=max(1, min(20, int(os.getenv("MAX_EVIDENCE_SOURCES", "8")))),
            retrieval_mode=os.getenv("RETRIEVAL_MODE", "hybrid"),
            database_path=os.getenv("RESEARCH_DB_PATH", "data/research_runs.sqlite3"),
        )


@dataclass
class ResearchDependencies:
    search_provider: SearchProvider
    llm: ChatModel | None = None
    max_queries: int = 4
    max_results_per_query: int = 8
    max_evidence_sources: int = 8
    retrieval_mode: str = "hybrid"
    store: ResearchRunStore | None = None


def build_default_dependencies() -> ResearchDependencies:
    settings = ResearchSettings.from_environment()
    if not settings.tavily_api_key:
        raise RuntimeError("Set TAVILY_API_KEY in .env to enable live web research.")

    return ResearchDependencies(
        search_provider=TavilySearchProvider(
            settings.tavily_api_key,
            timeout=settings.request_timeout_seconds,
        ),
        llm=OpenAICompatibleChatModel(
            settings.llm_base_url,
            settings.llm_model,
            api_key=settings.llm_api_key,
            timeout=settings.request_timeout_seconds,
        ),
        max_queries=settings.max_queries,
        max_results_per_query=settings.max_results_per_query,
        max_evidence_sources=settings.max_evidence_sources,
        retrieval_mode=settings.retrieval_mode,
        store=ResearchRunStore(settings.database_path),
    )
