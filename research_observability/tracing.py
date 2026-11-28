from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from time import perf_counter
from typing import Any, TypeVar
from urllib.parse import urlsplit

from .metrics import record_node
from .prompts import current_prompt_metadata

logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])
_OTLP_LOCK = Lock()
_OTLP_PROVIDER_IDS: set[int] = set()


def trace_content_enabled() -> bool:
    return os.getenv("LANGFUSE_TRACE_CONTENT", "false").strip().lower() in {"1", "true", "yes"}


def normalize_langfuse_base_url() -> str | None:
    """Normalize a configured Langfuse host to an absolute HTTP(S) base URL."""
    configured = os.getenv("LANGFUSE_BASE_URL")
    if configured is None or not configured.strip():
        return None

    base_url = configured.strip()
    if base_url.lower().startswith("export "):
        base_url = base_url[7:].strip()
    if base_url.upper().startswith("LANGFUSE_BASE_URL"):
        key, separator, value = base_url.partition("=")
        if separator and key.strip().upper() == "LANGFUSE_BASE_URL":
            base_url = value.strip()
        elif ":" in base_url:
            key, separator, value = base_url.partition(":")
            if separator and key.strip().upper() == "LANGFUSE_BASE_URL":
                base_url = value.strip()
    base_url = base_url.strip().strip("\"'")
    if "://" not in base_url:
        base_url = f"https://{base_url}"

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
        raise ValueError(
            "LANGFUSE_BASE_URL must be an absolute HTTP(S) URL, for example "
            "https://cloud.langfuse.com. The configured value was not displayed."
        )

    base_url = base_url.rstrip("/")
    os.environ["LANGFUSE_BASE_URL"] = base_url
    return base_url


def _configure_otlp_exporter() -> None:
    """Attach an OTLP exporter when explicitly configured by deployment environment."""
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") and not os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        logger.warning("OTLP endpoint is configured but the OTLP exporter is not installed: %s", exc)
        return

    with _OTLP_LOCK:
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            provider = TracerProvider(
                resource=Resource.create(
                    {"service.name": os.getenv("OTEL_SERVICE_NAME", "multi-agent-ai-researcher")}
                )
            )
            trace.set_tracer_provider(provider)
        provider_id = id(provider)
        if provider_id not in _OTLP_PROVIDER_IDS:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
            _OTLP_PROVIDER_IDS.add(provider_id)


def get_langfuse_client():
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        normalize_langfuse_base_url()
        from langfuse import get_client

        client = get_client()
        _configure_otlp_exporter()
        return client
    except Exception as exc:  # noqa: BLE001 - telemetry configuration must not break research execution.
        logger.warning("Langfuse client unavailable; using OpenTelemetry/no-op tracing: %s", exc)
        return None


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        del kwargs

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value


def _set_otel_attributes(span: Any, metadata: dict[str, Any]) -> None:
    for key, value in metadata.items():
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        elif value is not None:
            span.set_attribute(key, str(value))


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    metadata: dict[str, Any] | None = None,
    input: Any = None,
    model: str | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Create a Langfuse observation when configured, otherwise an OTel span or no-op."""
    safe_metadata = dict(metadata or {})
    safe_metadata.update(current_prompt_metadata())

    langfuse = get_langfuse_client()
    if langfuse is not None:
        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
            "metadata": safe_metadata,
        }
        if trace_content_enabled() and input is not None:
            kwargs["input"] = input
        if as_type == "generation" and model:
            kwargs["model"] = model
        if as_type == "generation" and model_parameters:
            kwargs["model_parameters"] = model_parameters

        try:
            manager = langfuse.start_as_current_observation(**kwargs)
            span = manager.__enter__()
        except Exception as exc:  # noqa: BLE001 - SDK errors must not prevent application work.
            logger.warning("Could not start Langfuse observation %s: %s", name, exc)
        else:
            try:
                yield span
            except BaseException as exc:
                try:
                    span.update(
                        level="ERROR",
                        status_message=(
                            f"{type(exc).__name__}: {exc}"
                            if trace_content_enabled()
                            else type(exc).__name__
                        ),
                    )
                except Exception as telemetry_exc:  # noqa: BLE001 - preserve the original application exception.
                    logger.debug("Could not annotate failed Langfuse observation: %s", telemetry_exc)
                manager.__exit__(type(exc), exc, exc.__traceback__)
                raise
            else:
                manager.__exit__(None, None, None)
            return

    _configure_otlp_exporter()
    try:
        from opentelemetry import trace
    except ImportError:
        yield _NoopObservation()
        return

    tracer = trace.get_tracer("multi_agent_ai_researcher")
    with tracer.start_as_current_span(name) as span:
        _set_otel_attributes(span, safe_metadata)
        try:
            yield span
        except BaseException as exc:
            try:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
            except (ImportError, AttributeError):
                pass
            raise


def instrument_node(name: str, function: F) -> F:
    @wraps(function)
    def wrapped(state: Any) -> Any:
        started = perf_counter()
        outcome = "error"
        with observation(name, metadata={"research.node": name}) as span:
            try:
                result = function(state)
                outcome = "success"
                if trace_content_enabled() and hasattr(span, "update"):
                    span.update(output=result)
                elif hasattr(span, "set_attribute"):
                    span.set_attribute("research.outcome", outcome)
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
                record_node(name, outcome, perf_counter() - started)

    return wrapped  # type: ignore[return-value]



def flush_telemetry(timeout_millis: int = 5000) -> None:
    """Flush buffered Langfuse and OpenTelemetry telemetry during process shutdown."""
    client = get_langfuse_client()
    if client is not None:
        try:
            client.flush()
        except Exception as exc:  # noqa: BLE001 - telemetry shutdown must not fail the application.
            logger.warning("Could not flush Langfuse telemetry: %s", exc)

    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=timeout_millis)
    except Exception as exc:  # noqa: BLE001 - telemetry shutdown must not fail the application.
        logger.debug("Could not flush OpenTelemetry spans: %s", exc)
