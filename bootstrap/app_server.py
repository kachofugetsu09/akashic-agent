from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from agent.config_models import Config
from agent.control.errors import RuntimeClosedError
from agent.control.service import ControlService
from agent.plugin_composition.rpc import RpcMethod, rpc_method_key
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.message_view import project_message_rows
from bootstrap.cleanup import run_cleanup_steps
from bootstrap.reply_status import RuntimeReplyStatus
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
        root = manager.live_root
        if root is None:
            raise RuntimeClosedError("正式 live Root 不可用")
        provider_context, accept_input = root._service_provider(CHANNEL_INPUT)
        async with provider_context.runtime_scope():
            return await accept_input(session_id, message_id, incoming)

    @asynccontextmanager
    async def resolve_method(name: str) -> AsyncIterator[RpcMethod | None]:
        """宿主只解析扩展入口；旧请求保留旧代参数与处理函数。"""
        root = manager.live_root
        if root is None:
            raise RuntimeClosedError("正式 live Root 不可用")
        key = rpc_method_key(name)
        if root.service_value(key) is None:
            yield None
            return
        provider_context, method = root._service_provider(key)
        async with provider_context.runtime_scope():
            yield method

    async def install(source: str, marketplace: str, ref: str, sparse: list[str],
                      update_id: str) -> dict[str, object]:
        _ = await manager.install(source=source, marketplace=marketplace,
            ref_name=ref, sparse_paths=sparse, update_id=update_id)
        return asdict(manager.read_update(update_id))

    async def drain(plugin_id: str) -> str:
        await manager.reconcile_disabled_and_drain(plugin_id)
        return f"插件已停用并排空: {plugin_id}"

    async def uninstall(plugin_id: str) -> dict[str, object]:
        """委托 Manager owner；调用方只收到 selection CAS 后的 accepted。"""
        return await manager.uninstall(plugin_id)

    async def message_display(page: MessagePage, *, display_only: bool) -> list[dict[str, object]]:
        root = manager.live_root
        if root is None:
            raise RuntimeClosedError("正式 live Root 不可用")
        return await project_message_rows(root, page, display_only=display_only)

    def reply_status(session_id: str) -> AsyncGenerator[dict[str, object], None]:
        root = manager.live_root
        if root is None:
            raise RuntimeClosedError("正式 live Root 不可用")
        return RuntimeReplyStatus(root).follow(session_id)

    return ControlService(
        MessageCatalog(core.message_log), core.workspace, accept=accept,
        attachments=core.channel_attachment_store.resolve_refs,
        reply_status=reply_status,
        message_display=message_display,
        plugin_install=install, plugin_status=manager.plugin_status,
        plugin_update=lambda identity: asdict(manager.read_update(identity)),
        plugin_drain=drain,
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
        # 2. EOF/错误由真实 stdio 入口结束宿主，Core.stop 负责 Fiber 结算。
        await StdioAppServer(
            service, max_message_bytes=config.app_server.max_message_bytes
        ).run()
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
