"""宿主执行插件拥有实际 Controller 客户端与启动清理。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from agent.plugin_composition import Context
from agent.plugin_composition.execution import EXECUTION
from agent.plugin_composition.host import HOST_INFO
from core.common.file_io import run_file_io
from infra.persistence.json_store import load_json, atomic_save_json
from plugins.host_execution.contract import WORKLOAD_CONTROLLER
from .controller_access import ControllerAccess, cleanup_workloads_for_boot
from .controller_client import UnixWorkloadController

api_version = 3
name = "host_execution"
version = "1.0.0"
desc = "拥有宿主进程与容器控制，代码 owner 与 generation 授权仍属内核"
inject = (EXECUTION, HOST_INFO,)
entrypoints = {"workload-controller": "controller.main"}


async def apply(ctx: Context) -> None:
    """先核对旧候选清理回执，再发布实际 Context 的控制授权。"""
    socket = os.environ.get("AKASHIC_WORKLOAD_SOCKET", "").strip()
    controller = None if not socket else UnixWorkloadController(Path(socket))
    workspace_id = hashlib.sha256(str(ctx.runtime.workspace.resolve()).encode()).hexdigest()[:16]
    if controller is not None:
        # 同一进程的 provider 换代不重复清理；只有完整回执后才记本 boot 已完成。
        boot_id = ctx.require(HOST_INFO).boot_id
        marker = ctx.data_root / "last-cleaned-boot.json"
        missing = object()
        previous = await run_file_io(lambda: load_json(marker, missing))
        if previous is not missing and (not isinstance(previous, str) or not previous or previous.strip() != previous):
            raise ValueError("宿主执行启动清理标记损坏")
        if previous != boot_id:
            await cleanup_workloads_for_boot(controller, workspace_id)
            await run_file_io(lambda: atomic_save_json(marker, boot_id))
    await ctx.provide(WORKLOAD_CONTROLLER, ControllerAccess(ctx, controller, workspace_id))
