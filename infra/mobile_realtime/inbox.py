from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from infra.mobile_realtime.storage import (
    AckAdvance,
    DeviceCursor,
    DurableInboxEvent,
    MobileRealtimeStorage,
)


class InboxResetRequired(RuntimeError):
    """表示 durable inbox 已超出恢复窗口，客户端必须重建投影。"""


@dataclass(frozen=True, slots=True)
class DurableReplay:
    events: tuple[DurableInboxEvent, ...]
    cursor: DeviceCursor


class DurableInboxManager:
    """管理每设备 P0 入箱、累计 ACK 和显式恢复窗口。"""

    def __init__(
        self,
        storage: MobileRealtimeStorage,
        *,
        retention: timedelta = timedelta(days=7),
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if retention <= timedelta(0):
            raise ValueError("retention 必须大于零")
        self._storage = storage
        self._retention = retention
        self._clock = clock

    def enqueue(
        self,
        *,
        device_id: str,
        event_id: str,
        envelope_json: str,
    ) -> DurableInboxEvent:
        return self._storage.append_durable_event(
            device_id=device_id,
            event_id=event_id,
            envelope_json=envelope_json,
            created_at=self._now(),
        )

    def enqueue_many(
        self,
        *,
        device_ids: tuple[str, ...],
        event_id: str,
        envelope_json: str,
    ) -> tuple[DurableInboxEvent, ...]:
        """用同一提交时间原子写入一个事件的全部设备副本。"""

        return self._storage.append_durable_events(
            device_ids=device_ids,
            event_id=event_id,
            envelope_json=envelope_json,
            created_at=self._now(),
        )

    def rebase_with_event(
        self,
        *,
        device_id: str,
        through_event_seq: int,
        event_id: str,
        envelope_json: str,
    ) -> DurableInboxEvent:
        """原子重定位回退游标，并持久化下一条重建事件。"""

        return self._storage.rebase_cursor_with_durable_event(
            device_id,
            through_event_seq=through_event_seq,
            event_id=event_id,
            envelope_json=envelope_json,
            created_at=self._now(),
        )

    def replay(
        self,
        device_id: str,
        *,
        after_event_seq: int,
        limit: int,
    ) -> DurableReplay:
        """返回仍在恢复窗口内的 P0 事件，过期时显式要求 reset。"""

        # 1. 先判断 durable 历史是否已经越过保留窗口
        cutoff = self._now() - self._retention
        if self._storage.has_unacked_event_before(device_id, cutoff=cutoff):
            raise InboxResetRequired(
                f"设备 durable inbox 已超过恢复窗口: {device_id}"
            )

        # 2. 返回严格按 event_seq 排序的恢复批次与 cursor 快照
        events = self._storage.read_durable_events(
            device_id,
            after_event_seq=after_event_seq,
            limit=limit,
        )
        return DurableReplay(events=events, cursor=self._storage.read_cursor(device_id))

    def mark_sent(self, device_id: str, *, through_event_seq: int) -> DeviceCursor:
        return self._storage.mark_events_sent(
            device_id,
            through_event_seq=through_event_seq,
        )

    def acknowledge(self, device_id: str, *, through_event_seq: int) -> AckAdvance:
        return self._storage.acknowledge_durable_events(
            device_id,
            through_event_seq=through_event_seq,
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("inbox clock 必须返回带时区的 datetime")
        return now
