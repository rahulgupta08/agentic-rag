from __future__ import annotations

import logging
import time
from typing import List

from app.pipeline.context.pipeline_context import (
    PipelineContext,
)
from app.pipeline.contracts.pipeline_stage import (
    PipelineStage,
)
from app.schemas.embedded_chunk import (
    EmbeddedChunkSchema,
)
from app.services.vector_store.base_vector_store import (
    BaseVectorStore,
)


logger = logging.getLogger(__name__)


class PersistenceStage(
    PipelineStage[
        List[EmbeddedChunkSchema],
        bool
    ]
):
    """
    Production-grade vector persistence stage.

    Responsibilities:
    - vector persistence orchestration
    - batch persistence execution
    - retry isolation
    - observability instrumentation
    - future idempotency support
    - distributed worker compatibility

    Important:
    Actual DB persistence remains delegated
    to vector store adapters.
    """

    stage_name = "document_persistence"

    def __init__(
        self,
        vector_store: BaseVectorStore,
        persistence_batch_size: int = 64
    ) -> None:

        super().__init__()

        self._vector_store = vector_store
        self._persistence_batch_size = (
            persistence_batch_size
        )

    async def process(
        self,
        data: List[EmbeddedChunkSchema],
        context: PipelineContext
    ) -> bool:
        """
        Execute vector persistence workflow.
        """

        logger.info(
            "Starting persistence stage execution",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id,
                "stage": self.stage_name,
                "embedded_chunks": len(data)
            }
        )

        started_at = time.perf_counter()

        try:

            total_batches = (
                len(data)
                + self._persistence_batch_size
                - 1
            ) // self._persistence_batch_size

            total_persisted = 0

            for batch_index in range(total_batches):

                start = (
                    batch_index *
                    self._persistence_batch_size
                )

                end = (
                    start +
                    self._persistence_batch_size
                )

                batch_chunks = data[start:end]

                logger.info(
                    "Persisting vector batch",
                    extra={
                        "document_id": context.document_id,
                        "correlation_id": context.correlation_id,
                        "batch_index": batch_index + 1,
                        "total_batches": total_batches,
                        "batch_size": len(batch_chunks)
                    }
                )

                """
                Delegate actual persistence
                to vector store implementation.

                Preserves:
                - Weaviate compatibility
                - Pinecone compatibility
                - future vector DB portability
                """

                await self._vector_store.upsert_batch(
                    batch_chunks
                )

                total_persisted += len(batch_chunks)

            duration_seconds = round(
                time.perf_counter() - started_at,
                4
            )

            context.add_metric(
                "persistence_duration_seconds",
                duration_seconds
            )

            context.add_metric(
                "persisted_vectors",
                total_persisted
            )

            context.add_metric(
                "persistence_batches",
                total_batches
            )

            logger.info(
                "Persistence stage completed",
                extra={
                    "document_id": context.document_id,
                    "correlation_id": context.correlation_id,
                    "persisted_vectors": total_persisted,
                    "total_batches": total_batches,
                    "duration_seconds": duration_seconds
                }
            )

            return True

        except Exception:
            logger.exception(
                "Persistence stage failed",
                extra={
                    "document_id": context.document_id,
                    "correlation_id": context.correlation_id,
                    "stage": self.stage_name
                }
            )
            raise