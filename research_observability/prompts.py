from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    text: str
    source: str
    version: int | None = None


_CURRENT_PROMPT: ContextVar[dict[str, str | int] | None] = ContextVar(
    "research_current_prompt",
    default=None,
)


def current_prompt_metadata() -> dict[str, str | int]:
    current = _CURRENT_PROMPT.get()
    return dict(current or {})


@contextmanager
def prompt_context(prompt: ResolvedPrompt) -> Iterator[None]:
    token = _CURRENT_PROMPT.set(
        {
            "prompt.name": prompt.name,
            "prompt.source": prompt.source,
            **({"prompt.version": prompt.version} if prompt.version is not None else {}),
        }
    )
    try:
        yield
    finally:
        _CURRENT_PROMPT.reset(token)


def resolve_text_prompt(
    name: str,
    default_text: str,
    *,
    label: str | None = None,
) -> ResolvedPrompt:
    """Resolve a text prompt from Langfuse, with a checked-in fallback."""
    enabled = os.getenv("LANGFUSE_PROMPT_MANAGEMENT_ENABLED", "false").strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return ResolvedPrompt(name=name, text=default_text, source="local")

    from .tracing import get_langfuse_client

    client = get_langfuse_client()
    if client is None:
        return ResolvedPrompt(name=name, text=default_text, source="local")

    try:
        prompt = client.get_prompt(
            name,
            label=label or os.getenv("LANGFUSE_PROMPT_LABEL", "production"),
            type="text",
        )
        compiled = prompt.compile()
        if not isinstance(compiled, str) or not compiled.strip():
            raise ValueError("Langfuse returned an empty or non-text prompt.")
        version = getattr(prompt, "version", None)
        return ResolvedPrompt(
            name=name,
            text=compiled,
            source="langfuse",
            version=version if isinstance(version, int) else None,
        )
    except Exception as exc:  # noqa: BLE001 - prompt management failure uses local versioned source.
        logger.warning("Could not resolve Langfuse prompt %s; using local fallback: %s", name, exc)
        return ResolvedPrompt(name=name, text=default_text, source="local")
