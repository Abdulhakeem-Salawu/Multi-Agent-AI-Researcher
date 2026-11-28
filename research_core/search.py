from __future__ import annotations

import time
from typing import Protocol

import httpx

from .schemas import SearchHit


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int = 8) -> list[SearchHit]: ...


class TavilySearchProvider:
    """Tavily adapter with bounded retries for transient transport/server failures."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 30.0,
        max_attempts: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required.")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.api_key = api_key
        self.timeout = timeout
        self.max_attempts = max_attempts

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        payload = None
        for attempt in range(self.max_attempts):
            try:
                response = httpx.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "search_depth": "advanced",
                        "max_results": max_results,
                        "include_raw_content": True,
                        "include_answer": False,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                retryable = status in {408, 429} or status >= 500
                if not retryable or attempt + 1 >= self.max_attempts:
                    raise
                time.sleep(0.5 * (2**attempt))
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt + 1 >= self.max_attempts:
                    raise
                time.sleep(0.5 * (2**attempt))

        if payload is None:
            raise RuntimeError("Tavily did not return a search payload.")

        hits: list[SearchHit] = []
        for item in payload.get("results", []):
            url = item.get("url")
            if not url:
                continue
            hits.append(
                SearchHit(
                    title=item.get("title") or url,
                    url=url,
                    content=item.get("content") or "",
                    raw_content=item.get("raw_content"),
                    score=min(1.0, max(0.0, float(item.get("score") or 0.0))),
                    published_date=item.get("published_date"),
                )
            )
        return hits


class StaticSearchProvider:
    """A deterministic provider for unit tests and offline graph runs."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self.hits = hits
        self.calls: list[str] = []

    def search(self, query: str, max_results: int = 8) -> list[SearchHit]:
        self.calls.append(query)
        return self.hits[:max_results]
