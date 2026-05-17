from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

from app.events.base_event import BaseEvent


EventHandler = Callable[[BaseEvent], Awaitable[None]]


class EventBus(ABC):
    """
    Abstract async event bus contract.

    This abstraction ensures the ingestion pipeline
    remains infrastructure-agnostic.

    Implementations:
    - InMemoryEventBus
    - RabbitMQEventBus
    - KafkaEventBus

    Business logic must NEVER depend directly
    on RabbitMQ/Kafka SDKs.
    """

    @abstractmethod
    async def publish(
        self,
        event: BaseEvent
    ) -> None:
        """
        Publish domain event.
        """
        raise NotImplementedError

    @abstractmethod
    def subscribe(
        self,
        event_type: str,
        handler: EventHandler
    ) -> None:
        """
        Register async event handler.
        """
        raise NotImplementedError