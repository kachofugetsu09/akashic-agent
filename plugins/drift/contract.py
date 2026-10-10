"""drift 发布的只读查询与结算合同。"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


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
