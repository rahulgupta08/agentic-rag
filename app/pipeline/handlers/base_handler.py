from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from app.events.base_event import BaseEvent
from app.events.event_bus import EventBus


logger = logging.getLogger(__name__)


class BaseEventHandler(ABC):
    """
    Base async event handler.

    Responsibilities:
    - event orchestration
    - error boundary
    - observability
    - next-event publishing

    Handlers should NEVER contain business logic.
    """

    def __init__(
        self,
        event_bus: EventBus
    ) -> None:
        self._event_bus = event_bus

    async def __call__(
        self,
        event: BaseEvent
    ) -> None:
        """
        Unified event execution wrapper.
        """

        logger.info(
            "Executing event handler",
            extra={
                "event_type": event.event_type,
                "event_id": event.event_id,
                "handler": self.__class__.__name__
            }
        )

        try:
            await self.handle(event)

        except Exception:
            logger.exception(
                "Event handler failed",
                extra={
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "handler": self.__class__.__name__
                }
            )
            raise

    @abstractmethod
    async def handle(
        self,
        event: BaseEvent
    ) -> None:
        """
        Implement orchestration logic.
        """
        raise NotImplementedError