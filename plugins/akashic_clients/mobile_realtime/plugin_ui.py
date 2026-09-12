from __future__ import annotations

import asyncio
import logging
from time import monotonic
from dataclasses import dataclass

from ..services import MobileUiProvider, MobileUiQueryOverloaded

logger = logging.getLogger(__name__)

_MAX_DEVICE_QUERIES = 4
_MAX_BACKGROUND_DEVICE_QUERIES = 2
_MAX_PLUGIN_QUERIES = 2
_MAX_DEVICE_OUTSTANDING = 32


@dataclass(frozen=True, slots=True)
class PluginUiQuery:
    request_id: str
    owner_id: str
    plugin_id: str
    plugin_revision: str
    method: str
    payload: dict[str, object]
    slot: str
    session_id: str | None
    turn_id: str | None


@dataclass(slots=True)
class _TrackedQuery:
    owner_id: str
    task: asyncio.Task[dict[str, object]]


@dataclass(slots=True)
class _DeviceGates:
    total: asyncio.Semaphore
    background: asyncio.Semaphore


class PluginUiQueryScheduler:
    """隔离并发执行每台设备的临时只读插件查询。"""

    def __init__(self, provider: MobileUiProvider) -> None:
        self._provider = provider
        self._lock = asyncio.Lock()
        self._devices: dict[str, _DeviceGates] = {}
        self._plugins: dict[tuple[str, str], asyncio.Semaphore] = {}
        self._queries: dict[tuple[str, str], _TrackedQuery] = {}

    async def execute(
        self,
        device_id: str,
        query: PluginUiQuery,
    ) -> dict[str, object]:
        """登记、调度并清理一次查询。"""

        queued_at = monotonic()
        _trace_query("received", query, queued_at)

        # 1. 在设备边界限制全部运行中和排队中的请求
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("plugin UI query 缺少 asyncio task")
        key = (device_id, query.request_id)
        async with self._lock:
            if key in self._queries:
                raise MobileUiQueryOverloaded("plugin UI request_id 重复")
            outstanding = sum(
                1 for tracked_device, _ in self._queries if tracked_device == device_id
            )
            if outstanding >= _MAX_DEVICE_OUTSTANDING:
                raise MobileUiQueryOverloaded("plugin UI 查询队列已满")
            self._queries[key] = _TrackedQuery(
                owner_id=query.owner_id,
                task=task,
            )
            device_gates = self._devices.setdefault(
                device_id,
                _DeviceGates(
                    total=asyncio.Semaphore(_MAX_DEVICE_QUERIES),
                    background=asyncio.Semaphore(_MAX_BACKGROUND_DEVICE_QUERIES),
                ),
            )
            plugin_gate = self._plugins.setdefault(
                (device_id, query.plugin_id),
                asyncio.Semaphore(_MAX_PLUGIN_QUERIES),
            )

        # 2. dashboard 和 drawer 不占后台 gate，始终保留两个交互槽
        completed = False
        try:
            async with plugin_gate:
                if query.slot in {"dashboard.main", "drawer.panel"}:
                    async with device_gates.total:
                        result = await self._run(query, queued_at)
                else:
                    async with device_gates.background:
                        async with device_gates.total:
                            result = await self._run(query, queued_at)
            completed = True
            return result
        finally:
            _trace_query("completed" if completed else "interrupted", query, queued_at)
            async with self._lock:
                _ = self._queries.pop(key, None)

    async def cancel_owner(self, device_id: str, owner_id: str) -> int:
        """取消一个已卸载 WebView owner 的所有请求。"""

        async with self._lock:
            tasks = [
                tracked.task
                for (tracked_device, _), tracked in self._queries.items()
                if tracked_device == device_id and tracked.owner_id == owner_id
            ]
        for task in tasks:
            _ = task.cancel()
        return len(tasks)

    async def cancel_device(self, device_id: str) -> None:
        """断线时取消设备全部临时查询并释放调度状态。"""

        async with self._lock:
            tasks = [
                tracked.task
                for (tracked_device, _), tracked in self._queries.items()
                if tracked_device == device_id
            ]
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            if any(key[0] == device_id for key in self._queries):
                return
            _ = self._devices.pop(device_id, None)
            for key in tuple(self._plugins):
                if key[0] == device_id:
                    del self._plugins[key]

    async def _run(self, query: PluginUiQuery, queued_at: float) -> dict[str, object]:
        """记录真实开始执行的时间，再调用已验证的插件查询。"""
        _trace_query("started", query, queued_at)
        return await self._provider.query(
            query.plugin_id,
            query.plugin_revision,
            query.method,
            query.payload,
            session_id=query.session_id,
            turn_id=query.turn_id,
        )


def _trace_query(phase: str, query: PluginUiQuery, queued_at: float) -> None:
    """用 wire owner 关联网页与服务端的等待，不记录查询正文或授权。"""
    event = f"mobile.plugin_query.{phase}"
    logger.info(event, extra={
        "event": event, "request_id": query.request_id, "owner_id": query.owner_id,
        "plugin_id": query.plugin_id, "method": query.method,
        "elapsed_ms": round((monotonic() - queued_at) * 1000, 3),
    })
