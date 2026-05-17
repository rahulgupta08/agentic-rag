from __future__ import annotations

import logging

from app.events.base_event import BaseEvent
from app.events.event_bus import EventBus
from app.events.ingestion_events import (
    DocumentEmbeddedEvent,
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


class DocumentChunkedHandler(BaseEventHandler):
    """
    Handles document.chunked events.

    Responsibilities:
    - invoke embedding stage
    - publish document.embedded event
    """

    def __init__(
        self,
        event_bus: EventBus,
        embedding_stage: PipelineStage
    ) -> None:

        super().__init__(event_bus)

        self._embedding_stage = embedding_stage

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
            "Starting embedding stage",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id
            }
        )

        embedded_chunks = await (
            self._embedding_stage.execute(
                data=payload["chunks"],
                context=context
            )
        )

        next_event = DocumentEmbeddedEvent(
            correlation_id=context.correlation_id,
            payload={
                "document_id": context.document_id,
                "embedded_chunks": embedded_chunks
            }
        )

        logger.info(
            "Publishing document.embedded event",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id
            }
        )

        await self._event_bus.publish(
            next_event
        )