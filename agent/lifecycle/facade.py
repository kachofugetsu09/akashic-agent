from __future__ import annotations

from collections.abc import Awaitable, Callable

from bus.event_bus import EventBus
from agent.lifecycle.types import (
    AfterReasoningCtx,
    AfterStepCtx,
    AfterTurnCtx,
    BeforeReasoningCtx,
    BeforeStepCtx,
    BeforeTurnCtx,
)


class TurnLifecycle:
    # 1. 持有 EventBus 引用，所有注册委托给 bus.on。
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    # ── Gate handlers: (ctx) -> ctx | None，支持 async 和 sync ──

    def on_before_turn(
        self,
        handler: Callable[
            [BeforeTurnCtx],
            Awaitable[BeforeTurnCtx | None] | BeforeTurnCtx | None,
        ],
    ) -> None:
        self._bus.on(BeforeTurnCtx, handler)

    def on_before_reasoning(
        self,
        handler: Callable[
            [BeforeReasoningCtx],
            Awaitable[BeforeReasoningCtx | None] | BeforeReasoningCtx | None,
        ],
    ) -> None:
        self._bus.on(BeforeReasoningCtx, handler)

    def on_before_step(
        self,
        handler: Callable[
            [BeforeStepCtx],
            Awaitable[BeforeStepCtx | None] | BeforeStepCtx | None,
        ],
    ) -> None:
        self._bus.on(BeforeStepCtx, handler)

    def on_after_reasoning(
        self,
        handler: Callable[
            [AfterReasoningCtx],
            Awaitable[AfterReasoningCtx | None] | AfterReasoningCtx | None,
        ],
    ) -> None:
        self._bus.on(AfterReasoningCtx, handler)

    # ── Tap handlers: (ctx) -> None，支持 async 和 sync ──

    def on_after_step(
        self,
        handler: Callable[[AfterStepCtx], Awaitable[None] | None],
    ) -> None:
        self._bus.on(AfterStepCtx, handler)

    def on_after_turn(
        self,
        handler: Callable[[AfterTurnCtx], Awaitable[None] | None],
    ) -> None:
        self._bus.on(AfterTurnCtx, handler)
