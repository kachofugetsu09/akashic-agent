from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TypeAlias, TypeVar, cast

logger = logging.getLogger(__name__)

E = TypeVar("E")
Handler: TypeAlias = Callable[[E], Awaitable[E | None] | E | None]


class EventBus:
    """提供 observe 隔离、fanout 并行和 ordered intercept 生命周期语义。"""

    def __init__(self) -> None:
        self._handlers: dict[type[object], list[Handler[object]]] = {}
        self._observe_queue: asyncio.Queue[object] | None = None
        self._observe_task: asyncio.Task[None] | None = None
        self._closed = False
        self._dispatcher_failures: list[BaseException] = []
        self._dispatcher_failure_tasks: set[asyncio.Task[None]] = set()
        self._dispatcher_stopping = False

    def on(
        self,
        event_type: type[E],
        handler: Handler[E],
    ) -> EventSubscription:
        # 1. 按注册顺序保存 handler，语义由 emit / observe 调用点决定。
        raw_event_type = _event_type(event_type)
        handlers = self._handlers.setdefault(raw_event_type, [])
        raw_handler = cast(Handler[object], handler)
        handlers.append(raw_handler)
        return EventSubscription(self, raw_event_type, raw_handler)

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
        # 1. 依次执行观察者，单个观察者失败不打断主流程。
        for handler in self._handlers_for(type(event)):
            _ = await self._run_observer(event, handler)

    async def fanout(
        self,
        event: object,
    ) -> None:
        # 1. 并发执行观察者；每个观察者自己记录异常，fanout 只汇总失败数量。
        handlers = self._handlers_for(type(event))
        if handlers:
            results = await asyncio.gather(
                *(
                    self._run_observer(event, handler)
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
        queue.put_nowait(event)

    async def drain(
        self,
    ) -> None:
        """等待已入队事件完成，并报告 dispatcher 的内部失败。"""

        queue = self._observe_queue
        if queue is None:
            return
        self._ensure_observe_task()
        await queue.join()
        self._raise_dispatcher_failures()

    async def aclose(
        self,
    ) -> None:
        """关闭接纳、排空队列并关闭 dispatcher，同时保留清理错误。"""

        errors: list[BaseException] = []

        # 1. 关闭 admission，后续事件不再进入已关闭总线。
        self._closed = True

        # 2. 先排空已有事件，确保队列 owner 等待每个 callback 完成。
        try:
            await self.drain()
        except BaseException as error:
            errors.append(error)

        # 3. 排空后停止 dispatcher；取消本身正常，其他退出错误必须保留。
        self._dispatcher_stopping = True
        task = self._observe_task
        if task is not None:
            _ = task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                errors.append(error)
            finally:
                if self._observe_task is task:
                    self._observe_task = None

        for error in self._dispatcher_failures:
            if not any(existing is error for existing in errors):
                errors.append(error)
        self._dispatcher_failures.clear()
        _raise_event_bus_errors("EventBus 关闭失败", errors)

    async def _run_observer(
        self,
        event: object,
        handler: Handler[object],
    ) -> bool:
        """在隔离 task 中运行单个 observer，并区分 observer 与调用方取消。"""

        from agent.plugin_composition.channels import (
            get_current_channel_turn_binding,
        )

        channel_binding = get_current_channel_turn_binding()
        caller_task = asyncio.current_task()
        caller_cancelling = (
            caller_task.cancelling() if caller_task is not None else 0
        )
        handler_task = asyncio.create_task(
            self._invoke_observer(
                handler,
                event,
                channel_binding,
            ),
            name=f"event_observer:{_handler_name(handler)}",
        )
        try:
            await asyncio.shield(handler_task)
            return True
        except asyncio.CancelledError:
            caller_cancelled = caller_task is not None and (
                caller_task.cancelling() > caller_cancelling
                or not handler_task.cancelled()
            )
            if caller_cancelled:
                _ = handler_task.cancel()
                try:
                    await handler_task
                except asyncio.CancelledError:
                    pass
                raise
            logger.warning(
                "observer handler cancelled for %s handler=%s",
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

    async def _invoke_observer(
        self,
        handler: Handler[object],
        event: object,
        channel_binding: object | None,
    ) -> None:
        channel_token = None
        if channel_binding is not None:
            from agent.plugin_composition.channels import (
                bind_channel_turn_binding,
            )

            channel_token = bind_channel_turn_binding(channel_binding)
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        finally:
            if channel_token is not None:
                from agent.plugin_composition.channels import (
                    reset_channel_turn_binding,
                )

                reset_channel_turn_binding(channel_token)

    def _ensure_observe_queue(
        self,
    ) -> asyncio.Queue[object]:
        if self._observe_queue is None:
            self._observe_queue = asyncio.Queue()
        self._ensure_observe_task()
        return self._observe_queue

    def _ensure_observe_task(
        self,
    ) -> None:
        if self._dispatcher_stopping:
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
        try:
            while True:
                queue = self._observe_queue
                if queue is None:
                    return
                event = await queue.get()
                try:
                    await self.fanout(event)
                finally:
                    queue.task_done()
        except asyncio.CancelledError:
            if not self._dispatcher_stopping:
                self._record_dispatcher_failure(
                    RuntimeError("EventBus dispatcher 被意外取消"),
                    task=asyncio.current_task(),
                )
            raise
        except Exception as error:
            self._record_dispatcher_failure(error, task=asyncio.current_task())
            raise

    def _handlers_for(self, event_type: type[object]) -> list[Handler[object]]:
        return list(self._handlers.get(event_type, []))

    def _on_observe_task_done(
        self,
        task: asyncio.Task[None],
    ) -> None:
        if self._observe_task is task:
            self._observe_task = None
        if task.cancelled():
            if not self._dispatcher_stopping:
                if task not in self._dispatcher_failure_tasks:
                    self._record_dispatcher_failure(
                        RuntimeError("EventBus dispatcher 被意外取消"),
                        task=task,
                    )
                else:
                    self._dispatcher_failure_tasks.discard(task)
                logger.error("event dispatcher cancelled unexpectedly")
                if self._observe_queue is not None:
                    self._ensure_observe_task()
            return

        exc = task.exception()
        if exc is not None:
            if task not in self._dispatcher_failure_tasks:
                self._record_dispatcher_failure(exc, task=task)
            self._dispatcher_failure_tasks.discard(task)
            logger.error(
                "event dispatcher stopped unexpectedly",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        if self._observe_queue is not None:
            self._ensure_observe_task()

    def _raise_dispatcher_failures(self) -> None:
        errors = list(self._dispatcher_failures)
        self._dispatcher_failures = []
        _raise_event_bus_errors("EventBus dispatcher 失败", errors)

    def _record_dispatcher_failure(
        self,
        error: BaseException,
        *,
        task: asyncio.Task[None] | None = None,
    ) -> None:
        if task is not None:
            self._dispatcher_failure_tasks.add(task)
        if any(existing is error for existing in self._dispatcher_failures):
            return
        self._dispatcher_failures.append(error)


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


def _event_type(value: object) -> type[object]:
    return cast(type[object], value)


def _raise_event_bus_errors(message: str, errors: list[BaseException]) -> None:
    """保留单个错误的原始类型，多个错误用异常组完整返回。"""

    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    raise BaseExceptionGroup(message, errors)
