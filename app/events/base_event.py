from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """
    Base immutable domain event contract.

    This class becomes the foundational event schema
    for:
    - internal async orchestration
    - RabbitMQ publishing
    - retries
    - DLQs
    - observability
    - tracing
    - future Kafka migration
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    correlation_id: str = Field(
        ...,
        description="Tracks an ingestion workflow across stages"
    )

    event_type: str = Field(
        ...,
        description="Logical event name"
    )

    event_version: str = Field(
        default="1.0"
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    payload: Dict[str, Any] = Field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

    class Config:
        frozen = True
        extra = "forbid"

    def to_dict(self) -> Dict[str, Any]:
        """
        Safe serialization helper.

        This method ensures:
        - RabbitMQ compatibility
        - future Kafka compatibility
        - structured logging compatibility
        """
        return self.model_dump(mode="json")