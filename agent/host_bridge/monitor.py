from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from agent.host_bridge.client import HostBridgeShellProcessManager

_MONITOR_INTERVAL_S = 2.0


def build_host_bridge_monitor() -> Coroutine[Any, Any, None] | None:
    """Build the required Core liveness monitor for host-bridge mode."""

    mode = os.environ.get("AKASHIC_EXECUTION_MODE", "local")
    if mode == "local":
        return None
    if mode != "host-bridge":
        raise RuntimeError("AKASHIC_EXECUTION_MODE 只能是 local 或 host-bridge")
    socket_text = os.environ.get("AKASHIC_HOST_BRIDGE_SOCKET", "")
    token = os.environ.get("AKASHIC_HOST_BRIDGE_TOKEN", "")
    boot_id = os.environ.get("AKASHIC_BOOT_ID", "")
    if not socket_text or not token or not boot_id:
        raise RuntimeError("host-bridge monitor 缺少 socket/token/boot identity")
    return _monitor(Path(socket_text), boot_id, token)


async def _monitor(socket_path: Path, boot_id: str, token: str) -> None:
    manager = HostBridgeShellProcessManager(socket_path, boot_id, token)
    try:
        while True:
            await manager.probe()
            await asyncio.sleep(_MONITOR_INTERVAL_S)
    finally:
        await manager.close_transport()
