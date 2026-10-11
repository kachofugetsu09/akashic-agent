"""eventmail 发布的只读查询与结算合同。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


class ContentWakeServicesV2(Protocol):
    async def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    async def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    async def expire(
        self,
        item_refs: Sequence[Mapping[str, object]],
        now: datetime,
    ) -> Mapping[str, object]: ...

    async def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    async def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    async def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    async def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]: ...

    async def mail_watermark(self) -> int: ...

    async def alert_deadline(self, now: datetime) -> datetime | None: ...

    async def alert_status(
        self, source_id: str, event_id: str, *, mail_id: str | None = None
    ) -> str | None: ...

    async def change_alert(
        self,
        item_ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        action: str,
        now: datetime,
        *,
        not_before: datetime | None = None,
    ) -> bool: ...

    async def peek_alert(self, now: datetime) -> Mapping[str, object] | None: ...

    async def select_alert(
        self,
        accepted_turn: Mapping[str, object],
        now: datetime,
        *,
        item_ref: Mapping[str, object] | None = None,
    ) -> Mapping[str, object] | None: ...

    async def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    async def selected_alerts(self) -> tuple[Mapping[str, object], ...]: ...

    async def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool: ...

    async def defer_alert(
        self, source_id: str, event_id: str, not_before: datetime
    ) -> None: ...

    async def close_alert(self, source_id: str, event_id: str, status: str) -> None: ...

    async def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]: ...


    def alert_start(self, item_ref: Mapping[str, object], expires_at: datetime | None,
                    now: Callable[[], datetime]) -> AbstractAsyncContextManager[str | None]:
        """保护原版本检查到 Delivery 首次 started 提交，退出后才发送。"""
        ...


class EventMailDeliveryServicesV2(Protocol):
    async def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    async def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    async def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...


EVENTMAIL_WAKE_V2 = ServiceKey[ContentWakeServicesV2]("eventmail.wake.v2")


EVENTMAIL_DELIVERY_V2 = ServiceKey[EventMailDeliveryServicesV2]("eventmail.delivery.v2")

# Fleet 的 Calendar/Feed/Fitbit/Steam 从这些来源端口提交；不是零消费者能力。
class BoundContentSource(Protocol):
    def close(self) -> None: ...

    async def submit(
        self, batch_id: str, items: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]: ...

    async def read_submission(self, batch_id: str) -> Mapping[str, object] | None:
        """Read a checkpointed receipt during offline handoff verification."""
        ...

    async def read_revision(self, item_id: str, revision: str) -> Mapping[str, object] | None:
        """Read a checkpointed revision during offline handoff verification."""
        ...

    async def unsettled(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    async def ack(self, settlement_ref: str) -> Mapping[str, object]: ...


class ContentSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContentSource: ...


class BoundAlertSource(Protocol):
    def close(self) -> None: ...

    async def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]: ...

    async def status(self, *, event_id: str) -> str | None: ...


class AlertSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundAlertSource: ...


class BoundContextSource(Protocol):
    def close(self) -> None: ...

    async def report(
        self,
        *,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]: ...


class ContextSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContextSource: ...


# Fleet Feed/Steam/Calendar/Fitbit 通过 bind(source_id) 使用三种来源端口；主仓目录不扫描外部源码。
EVENTMAIL_CONTENT_SOURCE = ServiceKey[ContentSourceServices]("eventmail.content_source.v2")
EVENTMAIL_ALERT_SOURCE = ServiceKey[AlertSourceServices]("eventmail.alert_source.v2")
EVENTMAIL_CONTEXT_SOURCE = ServiceKey[ContextSourceServices]("eventmail.context_source.v2")
