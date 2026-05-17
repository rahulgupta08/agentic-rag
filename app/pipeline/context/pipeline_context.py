from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass(slots=True)
class PipelineContext:
    """
    Shared ingestion execution context.

    This context travels across all ingestion stages
    and eventually across distributed workers.

    Responsibilities:
    - correlation tracking
    - ingestion metadata
    - retry tracking
    - execution observability
    - checkpointing compatibility
    """

    correlation_id: str

    document_id: str

    ingestion_source: str

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    current_stage: Optional[str] = None

    retry_count: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    execution_metrics: Dict[str, Any] = field(
        default_factory=dict
    )

    def set_stage(self, stage_name: str) -> None:
        """
        Updates current executing stage.
        """
        self.current_stage = stage_name

    def increment_retry(self) -> None:
        """
        Retry tracking helper.
        """
        self.retry_count += 1

    def add_metric(self, key: str, value: Any) -> None:
        """
        Attach runtime metrics.
        """
        self.execution_metrics[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Attach execution metadata.
        """
        self.metadata[key] = value