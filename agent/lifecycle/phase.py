from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bus.event_bus import EventBus

logger = logging.getLogger(__name__)

I = TypeVar("I")
C = TypeVar("C")
O = TypeVar("O")


class GatePhase(ABC, Generic[I, C, O]):
    """Chain = sequential emit. Each handler can mutate ctx."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, input: I) -> O:
        try:
            # 1. 先执行内部 setup，生成本轮上下文。
            ctx = await self._setup(input)

            # 2. 再通过 EventBus.emit 依次执行 handler 链，各 handler 可以修改 ctx。
            ctx = await self._bus.emit(ctx)

            # 3. 最后执行内部 finalize，返回最终输出。
            return await self._finalize(ctx, input)
        except Exception:
            logger.exception("Phase %s failed", self.__class__.__name__)
            raise

    @abstractmethod
    async def _setup(self, input: I) -> C:
        ...

    @abstractmethod
    async def _finalize(self, ctx: C, input: I) -> O:
        ...


class TapPhase(ABC, Generic[I, C, O]):
    """Chain = parallel fanout. Handlers are read-only observers."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, input: I) -> O:
        try:
            # 1. 先执行内部 setup，生成本轮上下文。
            ctx = await self._setup(input)

            # 2. 再通过 EventBus.fanout 并发通知所有 observer，handler 不可修改 ctx。
            await self._bus.fanout(ctx)

            # 3. 最后执行内部 finalize，返回最终输出。
            return await self._finalize(ctx, input)
        except Exception:
            logger.exception("Phase %s failed", self.__class__.__name__)
            raise

    @abstractmethod
    async def _setup(self, input: I) -> C:
        ...

    @abstractmethod
    async def _finalize(self, ctx: C, input: I) -> O:
        ...
