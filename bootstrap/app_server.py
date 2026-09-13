from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from agent.config_models import Config
from agent.control.service import ControlService
from agent.plugin_composition.rpc import RpcMethod, rpc_method_key
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugins.snapshot import (
    follow_reply_status,
    lease_runtime_snapshot,
    project_message_rows,
)
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.tools import CoreRuntime, build_core_runtime
from bootstrap.workspace_lock import WorkspaceInstanceLock
from core.net.http import SharedHttpResources
from infra.control.stdio import StdioAppServer
from session.log import MessageCatalog, MessagePage
from session.message import Message


def build_control_service(
    core: CoreRuntime, *, workspace_token: str | None = None,
    boot_id: str | None = None, ready: Callable[[], bool] | None = None,
) -> ControlService:
    """控制端只取得输入、读取和管理端口；协议层不持有插件宿主或任意 SQL。"""
    manager = core.plugin_manager

    async def accept(session_id: str, message_id: str, incoming: ChannelInboundMessage) -> Message:
        # 当前 Root 的同一租约覆盖输入接纳；Conversation 在 append 前核对 gate。
        async with lease_runtime_snapshot(manager.snapshot_store) as snapshot:
            root = snapshot.composition_root
            assert root is not None
            return await root.context.require(CHANNEL_INPUT)(session_id, message_id, incoming)

    @asynccontextmanager
    async def resolve_method(name: str) -> AsyncIterator[RpcMethod | None]:
        """宿主只解析扩展入口；旧请求保留旧代参数与处理函数。"""
        async with lease_runtime_snapshot(manager.snapshot_store) as snapshot:
            root = snapshot.composition_root
            assert root is not None
            yield root.context.get(rpc_method_key(name))

    async def install(source: str, marketplace: str, ref: str, sparse: list[str],
                      update_id: str) -> dict[str, object]:
        _ = await manager.install_candidate(source=source, marketplace=marketplace,
            ref_name=ref, sparse_paths=sparse, update_id=update_id)
        return asdict(manager.read_update(update_id))

    async def promote(update_id: str) -> dict[str, object]:
        manager.start_update_publication(update_id)
        return asdict(manager.read_update(update_id))

    async def discard(update_id: str) -> dict[str, object]:
        await manager.discard_update(update_id, reason="administrator discarded candidate")
        return asdict(manager.read_update(update_id))

    async def drain(plugin_id: str) -> str:
        await manager.reconcile_disabled_and_drain(plugin_id)
        return f"插件已停用并排空: {plugin_id}"

    async def uninstall(plugin_id: str) -> dict[str, object]:
        """安装 owner 停用并排空代码，原 plugin-data 与恢复归档保留。"""
        from agent.plugins.install import finalize_uninstall_plugin, set_installed_plugin_enabled

        # 1. 明确停用原安装记录，再等待所有实际资源归还。
        _ = set_installed_plugin_enabled(plugin_id, enabled=False,
                                        plugins_home=manager.installed_plugins_home)
        await manager.reconcile_disabled_and_drain(plugin_id)
        # 2. 原安装 owner 完成代码移除，不获得 workspace 数据减少权。
        cache_path, data_path = finalize_uninstall_plugin(plugin_id,
            workspace=core.workspace, plugins_home=manager.installed_plugins_home)
        return {"plugin_id": plugin_id, "cache_path": str(cache_path), "data_path": str(data_path)}

    async def message_display(page: MessagePage, *, display_only: bool) -> list[dict[str, object]]:
        return await project_message_rows(
            manager.snapshot_store,
            page,
            display_only=display_only,
        )

    return ControlService(
        MessageCatalog(core.message_log), core.workspace, accept=accept,
        attachments=core.channel_attachment_store.resolve_refs,
        reply_status=lambda session_id: follow_reply_status(
            manager.snapshot_store,
            session_id,
        ),
        message_display=message_display,
        plugin_install=install, plugin_status=manager.candidate_status,
        plugin_update=lambda identity: asdict(manager.read_update(identity)),
        plugin_promote=promote, plugin_discard=discard, plugin_drain=drain,
        plugin_uninstall=uninstall, workspace_token=workspace_token,
        boot_id=boot_id, ready=ready,
        control_frames=core.control_frames,
        resolve_method=resolve_method,
    )


async def run_stdio_app_server(config: Config, workspace: Path) -> None:
    """运行同一套消息插件；EOF 关闭本进程，消息与领域回执仍可恢复。"""
    http = SharedHttpResources()
    lock = WorkspaceInstanceLock(workspace)
    lock.acquire()
    core: CoreRuntime | None = None
    service: ControlService | None = None
    try:
        # 1. 与 Gateway 共用正式构造和控制端口，不创建另一种执行模型。
        core = build_core_runtime(config, workspace, http, clear_stale_session_admissions=True)
        await core.start()
        service = build_control_service(core)
        # 2. 读循环与插件服务并行，follow 不阻塞下一条控制请求。
        async with asyncio.TaskGroup() as group:
            runtime = group.create_task(core.plugin_manager.run_runtime_services())
            try:
                await StdioAppServer(service, max_message_bytes=config.app_server.max_message_bytes).run()
            finally:
                _ = runtime.cancel()
    finally:
        await run_cleanup_steps(
            ("control_service.shutdown", service.shutdown if service else _noop),
            ("message_bus.close", core.bus.aclose if core else _noop),
            ("core.stop", core.stop if core else _noop),
            ("http_resources.aclose", http.aclose),
            ("workspace_lock.release", lambda: _release_lock(lock)),
        )


async def _noop() -> None:
    return None


async def _release_lock(lock: WorkspaceInstanceLock) -> None:
    lock.release()
