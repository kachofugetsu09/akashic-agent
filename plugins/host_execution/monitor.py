"""执行插件拥有 Bridge 连通状态；它不承诺旧 manager 仍存活。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Protocol
from .bridge.factory import HostBridgeRpcError
from core.common.diagnostic_log import log_event
from agent.plugin_composition.context import HealthHandle

_MONITOR_INTERVAL_S = 2.0
logger = logging.getLogger(__name__)

class BridgeProbe(Protocol):
    async def probe(self) -> Mapping[str, object]: ...
    async def close_transport(self) -> None: ...


@dataclass
class HostBridgeStatus:
    """由执行插件监控任务更新的短命状态；只描述连接，不承诺旧 manager 仍存活。"""

    state: Literal["disabled", "checking", "healthy", "degraded"] = "disabled"
    failures: int = 0
    code: str | None = None
    checked_at: str | None = None
    # Bridge 没配置出口时为 None；RPC 健康而出口不可用时浏览器出不了网。
    egress: Literal["ready", "unavailable"] | None = None

    def snapshot(self) -> dict[str, object]:
        return asdict(self)


async def _monitor(
    manager: BridgeProbe,
    *,
    status: HostBridgeStatus,
    health: HealthHandle | None = None,
) -> None:
    try:
        while True:
            try:
                identity = await manager.probe()
            except HostBridgeRpcError as exc:
                status.checked_at = datetime.now(UTC).isoformat()
                status.state = "degraded"
                status.failures += 1
                if health is not None:
                    health.degrade(exc.code.name)
                status.code = exc.code.name
                log_event(logger, logging.WARNING, "host_bridge.degraded",
                          reason=exc.code.name, counts=f"failures:{status.failures}")
                if not exc.transient:
                    raise
            else:
                if status.failures:
                    log_event(logger, logging.INFO, "host_bridge.recovered",
                              counts=f"failures:{status.failures}")
                status.checked_at = datetime.now(UTC).isoformat()
                status.state = "healthy"
                status.failures = 0
                status.code = None
                if health is not None:
                    health.recover()
                _observe_egress(status, identity)
            await asyncio.sleep(min(_MONITOR_INTERVAL_S * (2 ** min(status.failures, 3)), 10))
    finally:
        await manager.close_transport()


def _observe_egress(status: HostBridgeStatus, identity: Mapping[str, object]) -> None:
    capabilities = identity.get("capabilities")
    names = set(capabilities) if isinstance(capabilities, list) else set()
    egress: Literal["ready", "unavailable"] | None = (
        "ready" if "host-egress" in names else "unavailable" if "host-egress-unavailable" in names else None)
    if egress == status.egress:
        return
    if egress == "unavailable":
        log_event(logger, logging.WARNING, "host_egress.unavailable", outcome="failed")
    elif egress == "ready" and status.egress == "unavailable":
        log_event(logger, logging.INFO, "host_egress.recovered", outcome="ready")
    status.egress = egress
