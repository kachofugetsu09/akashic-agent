from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

from agent.plugin_composition import CompositionRoot, FiberState, ServiceKey
from agent.plugin_contracts.reply import (
    REPLY_STATUS as REPLY_STATUS,
)


@dataclass(frozen=True, slots=True)
class _StatusEvent:
    value: object
    boundary: bool


_ROOT_CLOSED = object()


class _StatusChannel:
    """Keep one current frame while preserving provider lifecycle boundaries."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[_StatusEvent] = asyncio.Queue(maxsize=1)
        self._boundary_pending = False
        self._boundary_consumed = asyncio.Event()
        self._boundary_consumed.set()
        self._terminal_error: BaseException | None = None
        self._closed = False

    def _replace(self, event: _StatusEvent) -> None:
        """Replace only a non-terminal frame and wake a waiting consumer."""
        while self._events.full():
            _ = self._events.get_nowait()
        self._boundary_pending = event.boundary
        if event.boundary:
            self._boundary_consumed.clear()
        else:
            self._boundary_consumed.set()
        self._events.put_nowait(event)

    async def publish(self, value: object) -> None:
        """Coalesce temporary frames without delaying provider cleanup."""
        while self._boundary_pending and not self._closed and self._terminal_error is None:
            await self._boundary_consumed.wait()
        if self._closed or self._terminal_error is not None:
            return
        if self._events.full():
            _ = self._events.get_nowait()
        self._events.put_nowait(_StatusEvent(value, boundary=False))

    def publish_boundary(self, value: object) -> None:
        """Replace temporary data unless a terminal error already won."""
        if self._closed or self._terminal_error is not None:
            return
        self._replace(_StatusEvent(value, boundary=True))

    def publish_error(self, error: BaseException) -> None:
        """Keep the first reader error visible until the consumer observes it."""
        if self._closed or self._terminal_error is not None:
            return
        self._terminal_error = error
        self._replace(_StatusEvent(error, boundary=True))
        self._boundary_pending = False
        self._boundary_consumed.set()

    def close(self) -> None:
        """Wake this subscription without replacing an already queued error."""
        if self._closed:
            return
        self._closed = True
        if self._terminal_error is None:
            self._replace(_StatusEvent(_ROOT_CLOSED, boundary=True))
        else:
            self._boundary_pending = False
            self._boundary_consumed.set()

    async def receive(self) -> object:
        event = await self._events.get()
        if event.boundary:
            self._boundary_pending = False
            self._boundary_consumed.set()
        return event.value


class RuntimeReplyStatus:
    """Subscribe to the live Root without retaining a provider call."""

    def __init__(self, root: CompositionRoot):
        if not isinstance(root, CompositionRoot):
            raise TypeError("RuntimeReplyStatus 需要 CompositionRoot")
        self._root = root

    async def follow(self, session_id: str) -> AsyncGenerator[dict[str, object], None]:
        """Follow one optional provider Fiber until the Root or caller closes."""
        root = self._root
        channel = _StatusChannel()
        subscriber = None
        root_effect = None
        subscription_id = uuid4().hex

        def unavailable() -> dict[str, object]:
            return {
                "version": 2,
                "session_id": session_id,
                "snapshot_id": None,
                "available": False,
                "items": [],
            }

        async def apply(context) -> None:
            """Start the Fiber-owned pump for one frozen provider activation."""
            try:
                reader = context.require(REPLY_STATUS)
                provider = context._fiber.dependency_store[  # pyright: ignore[reportPrivateUsage]
                    cast(ServiceKey[Any], REPLY_STATUS)
                ]
                snapshot_id = f"{root.generation_id}:{provider.revision}"

                async def pump() -> None:
                    boundary_sent = False
                    try:
                        follower = reader.follow(session_id)
                        async with aclosing(follower):
                            async for items in follower:
                                await channel.publish({
                                    "version": 2,
                                    "session_id": session_id,
                                    "snapshot_id": snapshot_id,
                                    "available": True,
                                    "items": list(items),
                                })
                        channel.publish_boundary(unavailable())
                        boundary_sent = True
                        await asyncio.Event().wait()
                    except asyncio.CancelledError as error:
                        current = asyncio.current_task()
                        if current is None or not current.cancelling():
                            channel.publish_error(error)
                            boundary_sent = True
                        raise
                    except BaseException as error:
                        channel.publish_error(error)
                        boundary_sent = True
                        raise
                    finally:
                        if not boundary_sent:
                            channel.publish_boundary(unavailable())

                await context.spawn(
                    pump(),
                    name=f"reply-status-pump:{subscription_id}",
                )
            except BaseException as error:
                if not isinstance(error, asyncio.CancelledError):
                    channel.publish_error(error)
                raise

        try:
            root_effect = await root.context.effect(
                lambda: channel.close,
                label=f"reply-status-root-close:{subscription_id}",
            )
            subscriber = await root.context.inject(
                (cast(ServiceKey[Any], REPLY_STATUS),),
                apply,
                name=f"reply-status:{subscription_id}",
            )
            if subscriber.state == FiberState.PENDING:
                yield unavailable()

            while True:
                value = await channel.receive()
                if value is _ROOT_CLOSED:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield cast(dict[str, object], value)
        finally:
            try:
                if subscriber is not None:
                    await subscriber.dispose()
            finally:
                if root_effect is not None:
                    await root_effect.aclose()
