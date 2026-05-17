from __future__ import annotations

import logging
from uuid import uuid4

from app.events.base_event import BaseEvent
from app.events.event_bus import EventBus
from app.events.ingestion_events import (
    DocumentChunkedEvent,
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


class DocumentLoadedHandler(BaseEventHandler):
    """
    Handles document.loaded events.

    Responsibilities:
    - invoke chunking stage
    - publish document.chunked event
    - isolate orchestration from business logic
    """

    def __init__(
        self,
        event_bus: EventBus,
        chunking_stage: PipelineStage
    ) -> None:
        super().__init__(event_bus)

        self._chunking_stage = chunking_stage

    async def handle(
        self,
        event: BaseEvent
    ) -> None:
        """
        Execute chunking workflow.
        """

        payload = event.payload

        context = PipelineContext(
            correlation_id=event.correlation_id,
            document_id=payload["document_id"],
            ingestion_source="pipeline_event_handler"
        )

        logger.info(
            "Starting chunking stage",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id
            }
        )

        chunking_result = await self._chunking_stage.execute(
            data=payload,
            context=context
        )

        next_event = DocumentChunkedEvent(
            correlation_id=context.correlation_id,
            payload={
                "document_id": context.document_id,
                "chunks": chunking_result
            }
        )

        logger.info(
            "Publishing document.chunked event",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id
            }
        )

        await self._event_bus.publish(next_event)