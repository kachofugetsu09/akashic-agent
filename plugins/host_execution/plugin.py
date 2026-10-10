"""宿主执行插件拥有实际 Controller 客户端与启动清理。"""
from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from agent.plugin_composition import Context
from agent.plugin_composition.execution import EXECUTION
from agent.plugin_composition.host import HOST_INFO
from core.common.file_io import run_file_io
from infra.persistence.json_store import load_json, atomic_save_json
from plugins.host_execution.contract import HOST_STATUS, WORKLOAD_CONTROLLER, PROCESSES, FILES
from agent.host_bridge.factory import HostBridgeRpcError, build_file_bridge
from plugins.ui.contract import UI
from .processes import PluginProcesses
from .files import Files
from .monitor import HostBridgeStatus, _monitor
from . import dashboard
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
    await ctx.provide(FILES, Files())
    processes = PluginProcesses(formal=not ctx.require(HOST_INFO).validation)
    await ctx.effect(lambda: processes.close, label="host.processes")
    await ctx.provide(PROCESSES, processes)
    await start_monitor(ctx)
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


async def start_monitor(ctx: Context) -> None:
    """先登记监控的关闭责任，再启动实际探测；关闭等待原任务退出。"""
    # 1. 状态只属于本 generation；候选不连接正式 Bridge。
    status = HostBridgeStatus()
    await ctx.provide(HOST_STATUS, status)
    manager = None if ctx.require(HOST_INFO).validation else build_file_bridge()
    if manager is not None:
        status.state = "checking"
        health = await ctx.health("bridge", required=False)
        health.degrade("checking")

        async def monitor() -> None:
            try:
                await _monitor(manager, status=status, health=health)
            except HostBridgeRpcError as error:
                status.state = "degraded"
                health.degrade(str(error) or type(error).__name__)
                ctx.report_incident("bridge.monitor.failed", str(error) or type(error).__name__)
                # 已关闭探测连接；显式 degraded/Incident 保留拒绝，换代可重新取得连接。

        # 2. Effect 登记完成后，监控任务才有机会等待真实 RPC。
        def setup():
            task = asyncio.create_task(monitor(), name="host-bridge-monitor")
            async def close() -> None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            return close
        await ctx.effect(setup, label="bridge.monitor")

    # 3. UI 缺席只影响这一可选路由，不阻止执行服务。
    async def register_status(child: Context) -> None:
        await child.require(UI).register(child, dashboard=lambda: dashboard)

    await ctx.inject((UI, HOST_STATUS), register_status, name="host-status-ui")
