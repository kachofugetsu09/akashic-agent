"""连接额度的读取缓存：每个连接一分钟内只问一次服务商，失败时保留上一次读数。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from plugins.models.contract import QuotaReading

_TTL_SECONDS = 60.0
# 手动刷新也不能连点打满服务商；间隔内直接给缓存。
_FRESH_FLOOR_SECONDS = 10.0
_READ_TIMEOUT_SECONDS = 20.0

_LOG = logging.getLogger(__name__)

ReadQuota = Callable[[str], Awaitable[QuotaReading]]


@dataclass(frozen=True, slots=True)
class ConnectionQuota:
    connection_id: str
    reading: QuotaReading | None
    # 最近一次成功读数的时间（ISO-8601）；从未读到时为空。
    read_at: str = ""
    # 本次读取失败的原因；有旧读数时界面照常显示旧值并标出时间。
    error: str = ""

    def payload(self) -> dict[str, object]:
        reading = self.reading
        return {
            "connectionId": self.connection_id,
            "plan": reading.plan if reading else "",
            "account": reading.account if reading else "",
            "balance": reading.balance if reading else "",
            "note": reading.note if reading else "",
            "windows": [
                {"label": window.label, "usedPercent": window.used_percent,
                 "resetsAt": window.resets_at, "windowSeconds": window.window_seconds}
                for window in (reading.windows if reading else ())
            ],
            "readAt": self.read_at,
            "error": self.error,
        }


@dataclass(slots=True)
class _Entry:
    checked: float
    value: ConnectionQuota


class QuotaCache:
    def __init__(self, read: ReadQuota) -> None:
        self._read = read
        self._entries: dict[str, _Entry] = {}
        self._pending: dict[str, asyncio.Task[ConnectionQuota]] = {}

    async def read(self, connection_ids: Sequence[str], *, fresh: bool) -> tuple[ConnectionQuota, ...]:
        return tuple(await asyncio.gather(*(self._one(item, fresh) for item in connection_ids)))

    def forget(self, keep: Sequence[str]) -> None:
        for connection_id in set(self._entries) - set(keep):
            del self._entries[connection_id]

    # 同一连接的并发请求合并成一次读取；调用方取消不打断正在进行的读取。
    async def _one(self, connection_id: str, fresh: bool) -> ConnectionQuota:
        entry = self._entries.get(connection_id)
        age = time.monotonic() - entry.checked if entry else None
        limit = _FRESH_FLOOR_SECONDS if fresh else _TTL_SECONDS
        if entry is not None and age is not None and age < limit:
            return entry.value
        task = self._pending.get(connection_id)
        if task is None:
            task = asyncio.create_task(self._refresh(connection_id))
            self._pending[connection_id] = task
            task.add_done_callback(lambda _: self._pending.pop(connection_id, None))
        return await asyncio.shield(task)

    async def _refresh(self, connection_id: str) -> ConnectionQuota:
        previous = self._entries.get(connection_id)
        try:
            async with asyncio.timeout(_READ_TIMEOUT_SECONDS):
                reading = await self._read(connection_id)
            value = ConnectionQuota(connection_id, reading, datetime.now(timezone.utc).isoformat())
        except asyncio.CancelledError:
            raise
        # 额度只是展示：任何驱动失败都转成这一张卡片的错误，不影响其它连接与对话。
        except Exception as error:
            message = "读取超时" if isinstance(error, TimeoutError) else str(error) or type(error).__name__
            _LOG.info("quota read failed for %s: %s", connection_id, message)
            value = ConnectionQuota(
                connection_id,
                previous.value.reading if previous else None,
                previous.value.read_at if previous else "",
                message,
            )
        self._entries[connection_id] = _Entry(time.monotonic(), value)
        return value

