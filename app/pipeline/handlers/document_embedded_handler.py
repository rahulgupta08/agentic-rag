from __future__ import annotations

import logging

from app.events.base_event import BaseEvent
from app.events.event_bus import EventBus
from app.events.ingestion_events import (
    DocumentPersistedEvent,
)
from app.pipeline.context.pipeline_context import (
    PipelineContext,
)
from app.pipeline.contracts.pipeline_stage import (
    PipelineStage,
)
from app.pipeline.handlers.base_handler import (
    BaseEventHandler,
)


logger = logging.getLogger(__name__)


class DocumentEmbeddedHandler(BaseEventHandler):
    """
    Handles document.embedded events.

    Responsibilities:
    - invoke persistence stage
    - publish document.persisted event
    """

    def __init__(
        self,
        event_bus: EventBus,
        persistence_stage: PipelineStage
    ) -> None:

        super().__init__(event_bus)

        self._persistence_stage = persistence_stage

    async def handle(
        self,
        event: BaseEvent
    ) -> None:

        payload = event.payload

        context = PipelineContext(
            correlation_id=event.correlation_id,
            document_id=payload["document_id"],
            ingestion_source="pipeline_event_handler"
        )

        logger.info(
            "Starting persistence stage",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id
            }
        )

        await self._persistence_stage.execute(
            data=payload["embedded_chunks"],
            context=context
        )

        next_event = DocumentPersistedEvent(
            correlation_id=context.correlation_id,
            payload={
                "document_id": context.document_id,
                "status": "completed"
            }
        )

        logger.info(
            "Publishing document.persisted event",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id
            }
        )

        await self._event_bus.publish(
            next_event
        )