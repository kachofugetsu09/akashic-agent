from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Generic, Protocol, TypeVar, cast

from agent.plugin_composition.model import CompositionError, ServiceKey
from agent.plugins.snapshot import (
    RuntimeSnapshotLease,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    get_lifecycle_runtime_snapshot,
    reset_runtime_snapshot,
)

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")


class ServiceCall(Protocol, Generic[T_co]):
    """Call one fixed service through one exact snapshot lease."""

    async def call(self, action: Callable[[T_co], Awaitable[R]]) -> R: ...


class _StableSource:
    __slots__ = ("_store",)

    def __init__(self, store: RuntimeSnapshotStore) -> None:
        self._store = store

    async def acquire(self) -> RuntimeSnapshotLease:
        return await self._store.acquire()


class _BoundCall(Generic[T]):
    __slots__ = ("_key", "_source")

    def __init__(self, key: ServiceKey[T], source: _StableSource) -> None:
        self._key = key
        self._source = source

    async def call(self, action: Callable[[T], Awaitable[R]]) -> R:
        if get_lifecycle_runtime_snapshot() is not None:
            raise CompositionError(
                "TASK_BOUND",
                "Service call 只能从未绑定的 host task 调用",
            )
        lease = await self._source.acquire()
        token = bind_runtime_snapshot(lease)
        call_error: BaseException | None = None
        result: object | None = None
        try:
            root = lease.snapshot.composition_root
            if root is None:
                raise CompositionError(
                    "ROOT_MISSING",
                    "Service call 的 snapshot 缺少 Root",
                )
            service = root.context.require(self._key)
            result = await action(service)
        except BaseException as error:
            call_error = error
        finally:
            reset_runtime_snapshot(token)

        release_error: BaseException | None = None
        try:
            await _release_critical(lease)
        except BaseException as error:
            release_error = error

        if call_error is not None and release_error is not None:
            raise BaseExceptionGroup(
                "Service call 与 lease release 同时失败",
                [call_error, release_error],
            )
        if call_error is not None:
            raise call_error
        if release_error is not None:
            raise release_error
        return cast(R, result)


async def _release_critical(lease: RuntimeSnapshotLease) -> None:
    task = asyncio.create_task(lease.release(), name="service-lease-release")
    cancelled = False
    while not task.done():
        try:
            _ = await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    if cancelled:
        raise asyncio.CancelledError


def _bind_service(
    store: RuntimeSnapshotStore,
    key: ServiceKey[T],
) -> ServiceCall[T]:
    """Build the Core-owned stable call for one ServiceKey."""

    return _BoundCall(key, _StableSource(store))


__all__ = ["ServiceCall"]
