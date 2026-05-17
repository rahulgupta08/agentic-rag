from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import DefaultDict, List

from app.events.base_event import BaseEvent
from app.events.event_bus import EventBus, EventHandler


logger = logging.getLogger(__name__)


class InMemoryEventBus(EventBus):
    """
    Async in-memory event bus.

    This implementation provides:
    - internal event-driven orchestration
    - async concurrent event dispatching
    - local development simplicity
    - RabbitMQ-compatible architecture

    This is intentionally designed to mirror
    future distributed event bus semantics.
    """

    def __init__(self) -> None:
        self._subscribers: DefaultDict[
            str,
            List[EventHandler]
        ] = defaultdict(list)

    def subscribe(
        self,
        event_type: str,
        handler: EventHandler
    ) -> None:
        """
        Register async event subscriber.
        """

        logger.info(
            "Registering event handler",
            extra={
                "event_type": event_type,
                "handler": handler.__name__
            }
        )

        self._subscribers[event_type].append(handler)

    async def publish(
        self,
        event: BaseEvent
    ) -> None:
        """
        Publish event to all subscribers.

        Handlers execute concurrently.

        Important:
        Failures in one handler should NOT
        block others.
        """

        handlers = self._subscribers.get(
            event.event_type,
            []
        )

        if not handlers:
            logger.warning(
                "No subscribers found for event",
                extra={
                    "event_type": event.event_type,
                    "event_id": event.event_id
                }
            )
            return

        logger.info(
            "Publishing event",
            extra={
                "event_type": event.event_type,
                "event_id": event.event_id,
                "subscriber_count": len(handlers)
            }
        )

        tasks = [
            self._safe_execute(
                handler,
                event
            )
            for handler in handlers
        ]

        await asyncio.gather(*tasks)

    async def _safe_execute(
        self,
        handler: EventHandler,
        event: BaseEvent
    ) -> None:
        """
        Safely execute event handler.

        This isolates handler failures
        and prepares the architecture for:
        - retries
        - DLQs
        - distributed workers
        """

        try:
            await handler(event)

        except Exception as exc:
            logger.exception(
                "Event handler execution failed",
                extra={
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "handler": handler.__name__,
                    "error": str(exc)
                }
            )