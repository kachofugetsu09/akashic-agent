from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TypeAlias, TypeVar, cast

logger = logging.getLogger(__name__)

E = TypeVar("E")
Handler: TypeAlias = Callable[[E], Awaitable[E | None] | E | None]


class EventBus:
    """Typed lifecycle hooks: observe + ordered intercept pipeline."""

    def __init__(self) -> None:
        self._handlers: dict[type[object], list[Handler[object]]] = {}

    def on(
        self,
        event_type: type[E],
        handler: Handler[E],
    ) -> None:
        # 1. 按注册顺序保存 handler，语义由 emit / observe 调用点决定。
        handlers = self._handlers.setdefault(cast(type[object], event_type), [])
        handlers.append(cast(Handler[object], handler))

    async def emit(
        self,
        event: E,
    ) -> E:
        # 1. 依次执行干预链，handler 返回新事件时替换当前事件。
        for raw_handler in self._handlers.get(cast(type[object], type(event)), []):
            handler = cast(Handler[E], raw_handler)
            result = handler(event)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                event = cast(E, result)
        return event

    async def observe(
        self,
        event: object,
    ) -> None:
        # 1. 依次执行观察者，单个观察者失败不打断主流程。
        for handler in self._handlers.get(type(event), []):
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("observer error for %s", type(event).__name__)
