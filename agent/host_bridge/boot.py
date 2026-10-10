"""在持久 owner 构造前取得当前 boot 的宿主执行认领回执。"""
from __future__ import annotations

from typing import Any
from agent.host_bridge.factory import build_file_bridge


async def claim_host_bridge_boot() -> dict[str, Any] | None:
    """只在显式 Bridge 模式认领；关闭探测连接不撤销外部认领。"""
    manager = build_file_bridge()
    if manager is None:
        return None
    try:
        return await manager.claim_boot()
    finally:
        await manager.close_transport()
