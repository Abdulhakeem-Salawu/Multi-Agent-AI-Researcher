from __future__ import annotations

import os
import sys
from typing import Any

REQUIRED_ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def run_smoke_test(client: Any | None = None) -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing GitHub Actions environment variables: " + ", ".join(missing)
        )

    from research_observability.tracing import normalize_langfuse_base_url, trace_content_enabled

    normalize_langfuse_base_url()

    if client is None:
        try:
            from langfuse import get_client
        except ImportError as exc:
            raise RuntimeError(
                'Langfuse SDK is not installed. Install with: pip install -e ".[observability]"'
            ) from exc
        client = get_client()

    if not client.auth_check():
        raise RuntimeError("Langfuse rejected the configured API credentials.")

    content_capture = trace_content_enabled()
    metadata = {
        "source": "github-actions",
        "branch": os.getenv("GITHUB_REF_NAME", "local"),
        "content_capture_enabled": content_capture,
    }

    with client.start_as_current_observation(
        as_type="span",
        name="github-actions.langfuse-connection-check",
        metadata=metadata,
    ) as span:
        # Deliberately synthetic output; this test never sends user research content.
        span.update(
            output={
                "status": "connected",
                "authentication": "passed",
                "content_capture_enabled": content_capture,
            }
        )

    client.flush()
    print("Langfuse authentication succeeded and the smoke-test observation was flushed.")


def main() -> int:
    try:
        run_smoke_test()
    except Exception as exc:  # noqa: BLE001 - CLI converts all connectivity failures into a nonzero exit.
        print(f"Langfuse smoke test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
