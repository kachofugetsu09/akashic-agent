from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent.lifecycle.types import AfterStepCtx
from bus.event_bus import EventBus, EventSubscription


class TurnLifecycle:
    """Register the runtime progress observer on the shared event bus."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def on_after_step(
        self,
        handler: Callable[[AfterStepCtx], Awaitable[None] | None],
    ) -> EventSubscription:
        return self._bus.on(AfterStepCtx, handler)
