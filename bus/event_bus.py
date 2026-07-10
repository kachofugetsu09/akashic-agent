from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease, RuntimeSnapshotStore

logger = logging.getLogger(__name__)

E = TypeVar("E")
Handler: TypeAlias = Callable[[E], Awaitable[E | None] | E | None]


class _AnyEvent:
    pass


@dataclass(frozen=True)
class _QueuedEvent:
    event: object
    snapshot_lease: RuntimeSnapshotLease | None


class EventBus:
    """Typed lifecycle hooks: observe + ordered intercept pipeline."""

    def __init__(self) -> None:
        self._handlers: dict[type[object], list[Handler[object]]] = {}
        self._observe_queue: asyncio.Queue[_QueuedEvent] | None = None
        self._observe_task: asyncio.Task[None] | None = None
        self._closed = False
        self._runtime_snapshot_store: RuntimeSnapshotStore | None = None
        self._pending_enqueue_tasks: set[asyncio.Task[None]] = set()

    def bind_runtime_snapshot_store(self, store: RuntimeSnapshotStore) -> None:
        self._runtime_snapshot_store = store

    def on(
        self,
        event_type: type[E],
        handler: Handler[E],
    ) -> EventSubscription:
        # 1. 按注册顺序保存 handler，语义由 emit / observe 调用点决定。
        handlers = self._handlers.setdefault(cast(type[object], event_type), [])
        raw_handler = cast(Handler[object], handler)
        handlers.append(raw_handler)
        return EventSubscription(self, cast(type[object], event_type), raw_handler)

    def on_any(self, handler: Handler[object]) -> EventSubscription:
        handlers = self._handlers.setdefault(_AnyEvent, [])
        handlers.append(handler)
        return EventSubscription(self, _AnyEvent, handler)

    def handler_count(self) -> int:
        return sum(len(handlers) for handlers in self._handlers.values())

    def off(
        self,
        event_type: type[object],
        handler: Handler[object],
    ) -> None:
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            del self._handlers[event_type]

    async def emit(
        self,
        event: E,
    ) -> E:
        lease = await self._runtime_lease()
        if lease is not None:
            from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

            async with lease:
                token = bind_runtime_snapshot(lease)
                try:
                    return await self.emit(event)
                finally:
                    reset_runtime_snapshot(token)
        # 1. 依次执行干预链，handler 返回新事件时替换当前事件。
        for raw_handler in self._handlers_for(cast(type[object], type(event))):
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
        lease = await self._runtime_lease()
        if lease is not None:
            from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

            async with lease:
                token = bind_runtime_snapshot(lease)
                try:
                    await self.observe(event)
                    return
                finally:
                    reset_runtime_snapshot(token)
        # 1. 依次执行观察者，单个观察者失败不打断主流程。
        for handler in self._handlers_for(type(event)):
            _ = await self._run_observer(event, handler)

    async def fanout(
        self,
        event: object,
    ) -> None:
        lease = await self._runtime_lease()
        if lease is not None:
            from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

            async with lease:
                token = bind_runtime_snapshot(lease)
                try:
                    await self.fanout(event)
                    return
                finally:
                    reset_runtime_snapshot(token)
        # 1. 并发执行观察者；每个观察者自己记录异常，fanout 只汇总失败数量。
        handlers = self._handlers_for(type(event))
        if not handlers:
            return
        from agent.plugins.snapshot import get_current_runtime_lease

        source_lease = get_current_runtime_lease()
        results = await asyncio.gather(
            *(
                self._run_observer(event, handler, source_lease)
                for handler in handlers
            )
        )
        failed_count = results.count(False)
        if failed_count:
            logger.warning(
                "fanout completed with observer errors: event=%s failed=%d total=%d",
                type(event).__name__,
                failed_count,
                len(handlers),
            )

    def enqueue(
        self,
        event: object,
    ) -> None:
        # 1. 后台队列只负责把事件交给 fanout，避免主回复等待后处理。
        if self._closed:
            logger.warning("event enqueue ignored after close: %s", type(event).__name__)
            return
        queue = self._ensure_observe_queue()
        from agent.plugins.snapshot import lease_current_runtime_snapshot

        snapshot_lease = lease_current_runtime_snapshot()
        if snapshot_lease is None and self._runtime_snapshot_store is not None:
            snapshot = self._runtime_snapshot_store.current
            if snapshot is not None and not snapshot.accepting_leases:
                task = asyncio.create_task(
                    self._enqueue_after_admission(event, queue),
                    name="event_enqueue_admission",
                )
                self._pending_enqueue_tasks.add(task)
                task.add_done_callback(self._pending_enqueue_tasks.discard)
                return
            snapshot_lease = self._runtime_snapshot_store.lease()
        queue.put_nowait(
            _QueuedEvent(
                event=event,
                snapshot_lease=snapshot_lease,
            )
        )

    async def _enqueue_after_admission(
        self,
        event: object,
        queue: asyncio.Queue[_QueuedEvent],
    ) -> None:
        assert self._runtime_snapshot_store is not None
        lease = await self._runtime_snapshot_store.acquire()
        queue.put_nowait(_QueuedEvent(event=event, snapshot_lease=lease))

    async def _runtime_lease(self) -> RuntimeSnapshotLease | None:
        if self._runtime_snapshot_store is None:
            return None
        from agent.plugins.snapshot import get_current_runtime_snapshot

        if get_current_runtime_snapshot() is not None:
            return None
        if self._runtime_snapshot_store.current is None:
            return None
        return await self._runtime_snapshot_store.acquire()

    async def drain(
        self,
    ) -> None:
        queue = self._observe_queue
        if queue is None:
            return
        self._ensure_observe_task()
        await queue.join()

    async def aclose(
        self,
    ) -> None:
        self._closed = True
        for task in self._pending_enqueue_tasks:
            _ = task.cancel()
        if self._pending_enqueue_tasks:
            _ = await asyncio.gather(
                *self._pending_enqueue_tasks,
                return_exceptions=True,
            )
        self._pending_enqueue_tasks.clear()
        await self.drain()
        task = self._observe_task
        if task is None:
            return
        _ = task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_observer(
        self,
        event: object,
        handler: Handler[object],
        source_lease: RuntimeSnapshotLease | None = None,
    ) -> bool:
        from agent.plugins.snapshot import get_current_runtime_lease

        source_lease = source_lease or get_current_runtime_lease()
        snapshot_lease = source_lease.fork() if source_lease is not None else None
        handler_task = asyncio.create_task(
            self._invoke_observer(handler, event, snapshot_lease),
            name=f"event_observer:{_handler_name(handler)}",
        )
        try:
            await asyncio.shield(handler_task)
            return True
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                _ = handler_task.cancel()
                _ = await asyncio.gather(handler_task, return_exceptions=True)
                raise
            logger.warning(
                "observer cancelled for %s handler=%s",
                type(event).__name__,
                _handler_name(handler),
            )
            return False
        except Exception:
            logger.exception(
                "observer error for %s handler=%s",
                type(event).__name__,
                _handler_name(handler),
            )
            return False
        finally:
            if snapshot_lease is not None and snapshot_lease.active:
                await snapshot_lease.release()

    async def _invoke_observer(
        self,
        handler: Handler[object],
        event: object,
        snapshot_lease: RuntimeSnapshotLease | None,
    ) -> None:
        if snapshot_lease is not None:
            from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

            async with snapshot_lease:
                token = bind_runtime_snapshot(snapshot_lease)
                try:
                    await self._invoke_observer(handler, event, None)
                finally:
                    reset_runtime_snapshot(token)
            return
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    def _ensure_observe_queue(
        self,
    ) -> asyncio.Queue[_QueuedEvent]:
        if self._observe_queue is None:
            self._observe_queue = asyncio.Queue()
        self._ensure_observe_task()
        return self._observe_queue

    def _ensure_observe_task(
        self,
    ) -> None:
        if self._closed:
            return
        if self._observe_task is not None and not self._observe_task.done():
            return
        task = asyncio.create_task(
            self._run_observe_queue(),
            name="event_bus_observe_queue",
        )
        self._observe_task = task
        task.add_done_callback(self._on_observe_task_done)

    async def _run_observe_queue(
        self,
    ) -> None:
        while True:
            queue = self._observe_queue
            if queue is None:
                return
            envelope = await queue.get()
            try:
                await self._fanout_queued(envelope)
            finally:
                queue.task_done()

    async def _fanout_queued(self, envelope: _QueuedEvent) -> None:
        lease = envelope.snapshot_lease
        if lease is None:
            await self.fanout(envelope.event)
            return
        from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

        async with lease:
            token = bind_runtime_snapshot(lease)
            try:
                await self.fanout(envelope.event)
            finally:
                reset_runtime_snapshot(token)

    def _handlers_for(self, event_type: type[object]) -> list[Handler[object]]:
        handlers = list(self._handlers.get(event_type, []))
        handlers.extend(self._handlers.get(_AnyEvent, []))
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        if snapshot is not None:
            handlers.extend(snapshot.event_handlers.get(event_type, ()))
        return handlers

    def _on_observe_task_done(
        self,
        task: asyncio.Task[None],
    ) -> None:
        if self._observe_task is task:
            self._observe_task = None
        if self._closed or task.cancelled():
            return
        try:
            exc = task.exception()
        except Exception as e:
            logger.warning("event dispatcher inspect failed: %s", e)
            exc = None
        if exc is not None:
            logger.error(
                "event dispatcher stopped unexpectedly",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if self._observe_queue is not None:
            self._ensure_observe_task()


class EventSubscription:
    def __init__(
        self,
        bus: EventBus,
        event_type: type[object],
        handler: Handler[object],
    ) -> None:
        self._bus = bus
        self._event_type = event_type
        self._handler = handler
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def close(self) -> None:
        if not self._active:
            return
        self._active = False
        self._bus.off(self._event_type, self._handler)


def _handler_name(handler: Handler[object]) -> str:
    return str(
        getattr(
            handler,
            "__qualname__",
            getattr(handler, "__name__", repr(handler)),
        )
    )
