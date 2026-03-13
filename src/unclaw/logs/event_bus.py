"""Small synchronous in-process event bus."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

type EventHandler = Callable[[object], None]


@dataclass(slots=True)
class EventBus:
    """Publish runtime events to in-process subscribers."""

    _subscribers: list[EventHandler] = field(default_factory=list)

    def subscribe(self, handler: EventHandler) -> None:
        """Register one synchronous event handler."""
        if handler not in self._subscribers:
            self._subscribers.append(handler)

    def publish(self, event: object) -> None:
        """Publish one event to all current subscribers."""
        for handler in tuple(self._subscribers):
            handler(event)

