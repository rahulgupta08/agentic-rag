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
from app.schemas.document_chunk import (
    DocumentChunkSchema,
)
from app.schemas.embedded_chunk import (
    EmbeddedChunkSchema,
)
from app.services.embedding.base_embedding_provider import (
    BaseEmbeddingProvider,
)


logger = logging.getLogger(__name__)


class EmbeddingStage(
    PipelineStage[
        List[DocumentChunkSchema],
        List[EmbeddedChunkSchema]
    ]
):
    """
    Production-grade embedding pipeline stage.

    Responsibilities:
    - batch embedding orchestration
    - async embedding execution
    - execution observability
    - retry isolation
    - future GPU portability
    - future distributed worker compatibility

    Important:
    This stage intentionally delegates actual
    embedding generation to the existing
    provider abstraction layer.
    """

    stage_name = "document_embedding"

    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        embedding_batch_size: int = 32
    ) -> None:

        super().__init__()

        self._embedding_provider = embedding_provider
        self._embedding_batch_size = embedding_batch_size

    async def process(
        self,
        data: List[DocumentChunkSchema],
        context: PipelineContext
    ) -> List[EmbeddedChunkSchema]:
        """
        Execute embedding generation workflow.
        """

        logger.info(
            "Starting embedding stage execution",
            extra={
                "document_id": context.document_id,
                "correlation_id": context.correlation_id,
                "stage": self.stage_name,
                "total_chunks": len(data)
            }
        )

        started_at = time.perf_counter()

        try:

            embedded_chunks: List[
                EmbeddedChunkSchema
            ] = []

            total_batches = (
                len(data) + self._embedding_batch_size - 1
            ) // self._embedding_batch_size

            for batch_index in range(total_batches):

                start = (
                    batch_index *
                    self._embedding_batch_size
                )

                end = (
                    start +
                    self._embedding_batch_size
                )

                batch_chunks = data[start:end]

                logger.info(
                    "Processing embedding batch",
                    extra={
                        "document_id": context.document_id,
                        "correlation_id": context.correlation_id,
                        "batch_index": batch_index + 1,
                        "total_batches": total_batches,
                        "batch_size": len(batch_chunks)
                    }
                )

                """
                Use existing async provider implementation.

                This preserves:
                - provider abstraction
                - OpenAI compatibility
                - future local model compatibility
                - future Triton compatibility
                """

                batch_result = await (
                    self._embedding_provider.embed_batch(
                        batch_chunks
                    )
                )

                embedded_chunks.extend(batch_result)

            duration_seconds = round(
                time.perf_counter() - started_at,
                4
            )

            context.add_metric(
                "embedding_duration_seconds",
                duration_seconds
            )

            context.add_metric(
                "embedded_chunks",
                len(embedded_chunks)
            )

            context.add_metric(
                "embedding_batches",
                total_batches
            )

            logger.info(
                "Embedding stage completed",
                extra={
                    "document_id": context.document_id,
                    "correlation_id": context.correlation_id,
                    "embedded_chunks": len(
                        embedded_chunks
                    ),
                    "total_batches": total_batches,
                    "duration_seconds": duration_seconds
                }
            )

            return embedded_chunks

        except Exception:
            logger.exception(
                "Embedding stage failed",
                extra={
                    "document_id": context.document_id,
                    "correlation_id": context.correlation_id,
                    "stage": self.stage_name
                }
            )
            raise