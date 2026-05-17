from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from app.events.base_event import BaseEvent
from app.pipeline.context.pipeline_context import PipelineContext


InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class PipelineStage(
    ABC,
    Generic[InputType, OutputType]
):
    """
    Abstract async ingestion stage contract.

    This abstraction is the core architectural bridge
    between:
    - current monolithic ingestion execution
    - future distributed event-driven workers

    Every ingestion stage must:
    - consume typed input
    - process asynchronously
    - produce typed output
    - remain orchestration-agnostic

    This ensures:
    - clean retries
    - async scalability
    - worker portability
    - testability
    - observability
    """

    stage_name: str

    def __init__(self) -> None:
        if not getattr(self, "stage_name", None):
            raise ValueError(
                f"{self.__class__.__name__} must define stage_name"
            )

    async def execute(
        self,
        data: InputType,
        context: PipelineContext
    ) -> OutputType:
        """
        Unified stage execution lifecycle.

        This method should NEVER be overridden.

        It guarantees:
        - consistent stage tracking
        - observability
        - future retries
        - future checkpointing
        """

        context.set_stage(self.stage_name)

        await self.before_process(data, context)

        result = await self.process(data, context)

        await self.after_process(result, context)

        return result

    async def before_process(
        self,
        data: InputType,
        context: PipelineContext
    ) -> None:
        """
        Optional execution hook.

        Override only if required.
        """
        return None

    @abstractmethod
    async def process(
        self,
        data: InputType,
        context: PipelineContext
    ) -> OutputType:
        """
        Core stage business logic.
        """
        raise NotImplementedError

    async def after_process(
        self,
        result: OutputType,
        context: PipelineContext
    ) -> None:
        """
        Optional execution hook.

        Override only if required.
        """
        return None

    async def build_success_event(
        self,
        context: PipelineContext,
        payload: dict
    ) -> BaseEvent:
        """
        Optional event construction helper.

        Future RabbitMQ/Kafka publishing
        will use this.
        """

        return BaseEvent(
            correlation_id=context.correlation_id,
            event_type=f"{self.stage_name}.completed",
            payload=payload
        )

    async def build_failure_event(
        self,
        context: PipelineContext,
        error: Exception
    ) -> BaseEvent:
        """
        Standardized failure event builder.
        """

        return BaseEvent(
            correlation_id=context.correlation_id,
            event_type=f"{self.stage_name}.failed",
            payload={
                "document_id": context.document_id,
                "error": str(error),
                "stage": self.stage_name
            }
        )