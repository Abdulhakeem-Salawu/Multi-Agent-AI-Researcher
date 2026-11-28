from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ExternalBenchmarkItem:
    """Thin wrapper that preserves dataset-specific upstream label semantics."""

    dataset_id: str
    track: str
    task_id: str
    question: str
    annotation_type: str
    reference_answer: str | None = None
    annotations: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    license_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
