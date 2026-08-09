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
    release_commit = os.environ.get("AKASHIC_RUNTIME_COMMIT", "")
    toolchain_digest = os.environ.get("AKASHIC_HOST_TOOLCHAIN_DIGEST", "")
    if not all((socket_text, token, boot_id, release_commit, toolchain_digest)):
        raise RuntimeError(
            "host-bridge monitor 缺少 socket/token/boot/release identity"
        )
    return _monitor(Path(socket_text), boot_id, token, release_commit, toolchain_digest)


async def _monitor(
    socket_path: Path,
    boot_id: str,
    token: str,
    release_commit: str,
    toolchain_digest: str,
) -> None:
    manager = HostBridgeShellProcessManager(
        socket_path,
        boot_id,
        token,
        expected_release_commit=release_commit,
        expected_toolchain_digest=toolchain_digest,
    )
    try:
        while True:
            await manager.probe()
            await asyncio.sleep(_MONITOR_INTERVAL_S)
    finally:
        await manager.close_transport()
