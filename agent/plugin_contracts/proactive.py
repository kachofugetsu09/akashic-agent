"""Wake 读取和结算普通来源的合同；来源继续拥有持久状态。"""

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


class DriftWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def select(
        self,
        ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def transition(self, token: str, action: str) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...


class DeliveryServices(Protocol):
    def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...


class EventMailDeliveryServicesV2(Protocol):
    async def pending(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    async def lookup(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    async def settle(
        self, selection_token: str, settlement_ref: str
    ) -> Mapping[str, object]: ...


class SemanticInterest(Protocol):
    def decision(self) -> bool | None: ...
    def status(self) -> str | None: ...
    async def score(
        self, texts: Sequence[str], *, cutoff: str
    ) -> tuple[float, ...]: ...


EVENTMAIL_WAKE_V2 = ServiceKey[ContentWakeServicesV2]("eventmail.wake.v2")
EVENTMAIL_DELIVERY_V2 = ServiceKey[EventMailDeliveryServicesV2]("eventmail.delivery.v2")
DRIFT_WAKE = ServiceKey[DriftWakeServices]("drift.wake.v1")
DRIFT_DELIVERY = ServiceKey[DeliveryServices]("drift.delivery.v1")
SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")


class DriftWakeServicesV2(Protocol):
    """Drift 查询与完整事务；取消等待已开始的存储工作退出。"""

    async def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    async def select(
        self, ref: Mapping[str, object], accepted_turn: Mapping[str, object], now: datetime,
    ) -> Mapping[str, object]: ...

    async def transition(self, token: str, action: str) -> Mapping[str, object]: ...

    async def selection(
        self, accepted_turn: Mapping[str, object],
    ) -> Mapping[str, object] | None: ...


class DriftDeliveryServicesV2(Protocol):
    """读取原 Drift 领取并等待真实送达结算，不替代 Delivery 回执。"""

    async def lookup(
        self, accepted_turn: Mapping[str, object],
    ) -> Mapping[str, object] | None: ...

    async def settle(
        self, selection_token: str, settlement_ref: str,
    ) -> Mapping[str, object]: ...


DRIFT_WAKE_V2 = ServiceKey[DriftWakeServicesV2]("drift.wake.v2")
DRIFT_DELIVERY_V2 = ServiceKey[DriftDeliveryServicesV2]("drift.delivery.v2")
