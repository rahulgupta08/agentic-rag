from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List

from app.ingestion.chunker_new import DocumentChunker
from app.pipeline.context.pipeline_context import (
    PipelineContext,
)
from app.pipeline.contracts.pipeline_stage import (
    PipelineStage,
)
from app.schemas.document_chunk import (
    DocumentChunkSchema,
)


logger = logging.getLogger(__name__)


class ChunkingStage(
    PipelineStage[
        Dict[str, Any],
        List[DocumentChunkSchema]
    ]
):
    """
    Production-grade chunking pipeline stage.

    This stage wraps the existing DocumentChunker
    implementation and adapts it into the new
    async event-driven pipeline architecture.

    Responsibilities:
    - chunk generation
    - execution isolation
    - observability hooks
    - async stage compatibility
    - future worker portability

    Important:
    The underlying business logic remains unchanged.
    """

    stage_name = "document_chunking"

    def __init__(
        self,
        chunker: DocumentChunker | None = None
    ) -> None:

        super().__init__()

        self._chunker = chunker or DocumentChunker()

    async def process(
        self,
        data: Dict[str, Any],
        context: PipelineContext
    ) -> List[DocumentChunkSchema]:
        """
        Execute document chunking.

        Expected input payload:
        {
            "document": dict,
            "document_id": str,
            "collection": str,
            "document_metadata": dict
        }
        """

        logger.info(
            "Starting chunking stage execution",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id,
                "stage": self.stage_name
            }
        )

        started_at = time.perf_counter()

        try:

            document = data["document"]
            document_id = data["document_id"]
            collection = data["collection"]
            document_metadata = data["document_metadata"]

            """
            Offload synchronous CPU-heavy chunking
            to thread executor.

            This prevents blocking the asyncio loop.

            Extremely important for future concurrent
            ingestion execution.
            """

            chunks = await asyncio.to_thread(
                self._chunker.chunk,
                document,
                document_id,
                collection,
                document_metadata
            )

            duration_seconds = round(
                time.perf_counter() - started_at,
                4
            )

            context.add_metric(
                "chunking_duration_seconds",
                duration_seconds
            )

            context.add_metric(
                "chunks_created",
                len(chunks)
            )

            logger.info(
                "Chunking stage completed",
                extra={
                    "document_id": context.document_id,
                    "correlation_id": context.correlation_id,
                    "chunks_created": len(chunks),
                    "duration_seconds": duration_seconds
                }
            )

            return chunks

        except Exception:
            logger.exception(
                "Chunking stage failed",
                extra={
                    "document_id": context.document_id,
                    "correlation_id": context.correlation_id,
                    "stage": self.stage_name
                }
            )
            raise