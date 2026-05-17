from __future__ import annotations

import logging
from uuid import uuid4

from app.events.ingestion_events import (
    DocumentLoadedEvent,
)
from app.events.in_memory_event_bus import (
    InMemoryEventBus,
)
from app.pipeline.handlers.document_chunked_handler import (
    DocumentChunkedHandler,
)
from app.pipeline.handlers.document_embedded_handler import (
    DocumentEmbeddedHandler,
)
from app.pipeline.handlers.document_loaded_handler import (
    DocumentLoadedHandler,
)
from app.pipeline.stages.chunking_stage import (
    ChunkingStage,
)
from app.pipeline.stages.embedding_stage import (
    EmbeddingStage,
)
from app.pipeline.stages.persistence_stage import (
    PersistenceStage,
)


logger = logging.getLogger(__name__)


class AsyncIngestionOrchestrator:
    """
    Production-grade async ingestion orchestrator.

    Responsibilities:
    - event bus bootstrapping
    - handler registration
    - stage orchestration
    - ingestion lifecycle coordination

    Important:
    This class contains orchestration only.
    No business logic belongs here.
    """

    def __init__(
        self,
        chunking_stage: ChunkingStage,
        embedding_stage: EmbeddingStage,
        persistence_stage: PersistenceStage
    ) -> None:

        self._event_bus = InMemoryEventBus()

        self._chunking_stage = chunking_stage
        self._embedding_stage = embedding_stage
        self._persistence_stage = persistence_stage

        self._register_handlers()

    def _register_handlers(self) -> None:
        """
        Register pipeline event handlers.
        """

        self._event_bus.subscribe(
            "document.loaded",
            DocumentLoadedHandler(
                event_bus=self._event_bus,
                chunking_stage=self._chunking_stage
            )
        )

        self._event_bus.subscribe(
            "document.chunked",
            DocumentChunkedHandler(
                event_bus=self._event_bus,
                embedding_stage=self._embedding_stage
            )
        )

        self._event_bus.subscribe(
            "document.embedded",
            DocumentEmbeddedHandler(
                event_bus=self._event_bus,
                persistence_stage=self._persistence_stage
            )
        )

    async def ingest_document(
        self,
        *,
        document: dict,
        document_id: str,
        collection: str,
        document_metadata: dict
    ) -> None:
        """
        Entry point into async event-driven ingestion.
        """

        correlation_id = str(uuid4())

        logger.info(
            "Starting ingestion workflow",
            extra={
                "document_id": document_id,
                "correlation_id": correlation_id
            }
        )

        initial_event = DocumentLoadedEvent(
            correlation_id=correlation_id,
            payload={
                "document": document,
                "document_id": document_id,
                "collection": collection,
                "document_metadata": document_metadata
            }
        )

        await self._event_bus.publish(
            initial_event
        )

        logger.info(
            "Ingestion workflow submitted",
            extra={
                "document_id": document_id,
                "correlation_id": correlation_id
            }
        )