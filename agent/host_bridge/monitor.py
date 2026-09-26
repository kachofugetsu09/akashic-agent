from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Coroutine
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from core.common.diagnostic_log import log_event

from agent.host_bridge.client import HostBridgeRpcError, HostBridgeShellProcessManager

_MONITOR_INTERVAL_S = 2.0
logger = logging.getLogger(__name__)


@dataclass
class HostBridgeStatus:
    """由 App 监控任务更新的短命状态；只描述连接，不承诺旧 manager 仍存活。"""

    state: Literal["disabled", "checking", "healthy", "degraded"] = "disabled"
    failures: int = 0
    code: str | None = None
    checked_at: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


async def claim_host_bridge_boot() -> dict[str, Any] | None:
    """Claim host execution ownership before Core initializes mutable runtime state."""

    identity = _configured_bridge_identity()
    if identity is None:
        return None
    manager = HostBridgeShellProcessManager(*identity)
    try:
        return await manager.claim_boot()
    finally:
        await manager.close_transport()


def build_host_bridge_monitor(status: HostBridgeStatus) -> Coroutine[Any, Any, None] | None:
    """Build the required Core liveness monitor for host-bridge mode."""

    identity = _configured_bridge_identity()
    if identity is None:
        return None
    status.state = "checking"
    return _monitor(*identity, status=status)


def _configured_bridge_identity() -> tuple[Path, str, str, str, str] | None:
    """Load the complete bridge identity or fail at the environment boundary."""

    mode = os.environ.get("AKASHIC_EXECUTION_MODE", "local")
    if mode == "local":
        return None
    if mode != "host-bridge":
        raise RuntimeError("AKASHIC_EXECUTION_MODE 只能是 local 或 host-bridge")
    socket_text = os.environ.get("AKASHIC_HOST_BRIDGE_SOCKET", "")
    token = os.environ.get("AKASHIC_HOST_BRIDGE_TOKEN", "")
    boot_id = os.environ.get("AKASHIC_BOOT_ID", "")
    release_commit = os.environ.get("AKASHIC_RUNTIME_COMMIT", "")
    toolchain_digest = os.environ.get("AKASHIC_HOST_TOOLCHAIN_DIGEST", "")
    if not all((socket_text, token, boot_id, release_commit, toolchain_digest)):
        raise RuntimeError(
            "host-bridge monitor 缺少 socket/token/boot/release identity"
        )
    socket_path = Path(socket_text)
    if not socket_path.is_absolute():
        raise RuntimeError("AKASHIC_HOST_BRIDGE_SOCKET 必须是绝对路径")
    return socket_path, boot_id, token, release_commit, toolchain_digest


async def _monitor(
    socket_path: Path,
    boot_id: str,
    token: str,
    release_commit: str,
    toolchain_digest: str,
    *,
    status: HostBridgeStatus,
) -> None:
    manager = HostBridgeShellProcessManager(
        socket_path,
        boot_id,
        token,
        release_commit,
        toolchain_digest,
    )
    try:
        while True:
            try:
                await manager.probe()
            except HostBridgeRpcError as exc:
                status.checked_at = datetime.now(UTC).isoformat()
                status.state = "degraded"
                status.failures += 1
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
            await asyncio.sleep(min(_MONITOR_INTERVAL_S * (2 ** min(status.failures, 3)), 10))
    finally:
        await manager.close_transport()
