"""Wake 读取和结算普通来源的合同；来源继续拥有持久状态。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


class ContentWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...

    def selected(self, limit: int = 100) -> tuple[Mapping[str, object], ...]: ...

    def expire(
        self,
        item_refs: Sequence[Mapping[str, object]],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def selection(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def select(
        self,
        item_ref: Mapping[str, object],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def select_batch(
        self,
        item_refs: Sequence[Mapping[str, object]],
        snapshot_seq: int,
        accepted_turn: Mapping[str, object],
        now: datetime,
    ) -> Mapping[str, object]: ...

    def transition(
        self,
        selection_token: str,
        action: str,
        *,
        not_before: datetime | None = None,
        selected_refs: Sequence[Mapping[str, object]] | None = None,
    ) -> Mapping[str, object]: ...

    def mail_watermark(self) -> int: ...

    def alert_deadline(self, now: datetime) -> datetime | None: ...

    def alert_status(
        self, source_id: str, event_id: str, *, mail_id: str | None = None
    ) -> str | None: ...

    def change_alert(
        self,
        item_ref: Mapping[str, object],
        accepted_turn: Mapping[str, object],
        action: str,
        now: datetime,
        *,
        not_before: datetime | None = None,
    ) -> bool: ...

    def peek_alert(self, now: datetime) -> Mapping[str, object] | None: ...

    def select_alert(
        self,
        accepted_turn: Mapping[str, object],
        now: datetime,
        *,
        item_ref: Mapping[str, object] | None = None,
    ) -> Mapping[str, object] | None: ...

    def selected_alert(
        self, accepted_turn: Mapping[str, object]
    ) -> Mapping[str, object] | None: ...

    def selected_alerts(self) -> tuple[Mapping[str, object], ...]: ...

    def expire_alert(self, source_id: str, event_id: str, now: datetime) -> bool: ...

    def defer_alert(
        self, source_id: str, event_id: str, not_before: datetime
    ) -> None: ...

    def close_alert(self, source_id: str, event_id: str, status: str) -> None: ...

    def active_context(self, now: datetime) -> tuple[Mapping[str, object], ...]: ...


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


class SemanticInterest(Protocol):
    async def score(
        self, texts: Sequence[str], *, cutoff: str
    ) -> tuple[float, ...]: ...


EVENTMAIL_WAKE = ServiceKey[ContentWakeServices]("eventmail.wake.v1")
EVENTMAIL_DELIVERY = ServiceKey[DeliveryServices]("eventmail.delivery.v1")
DRIFT_WAKE = ServiceKey[DriftWakeServices]("drift.wake.v1")
DRIFT_DELIVERY = ServiceKey[DeliveryServices]("drift.delivery.v1")
SEMANTIC_INTEREST = ServiceKey[SemanticInterest]("akasha.semantic-interest.v1")
