"""Gateway 用声明端口组装控制协议，不取得 Root 或 Manager。"""
from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import cast
from pydantic import BaseModel

from agent.plugin_composition import Context
from plugins.channels.contract import CHANNEL_INPUT_V2, ChannelInboundMessage
from plugins.ledger.contract import CHANNEL_ATTACHMENT_READ
from .contract import CONTROL_FRAMES
from agent.plugin_composition.host import HOST_INFO
from plugins.ledger.contract import MESSAGE_CATALOG
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from plugins.ledger.contract import CallRef
from plugins.ui.contract import MESSAGE_DISPLAY, message_rows
from agent.plugin_composition.tasks import RESTART_GATE
from plugins.ui.contract import MessagePage
from plugins.ledger.contract import Message
from .contract import RpcMethod
from .errors import RuntimeClosedError
from .protocol.errors import JsonRpcError, SERVER_OVERLOADED
from .runtime_stop import CancelStopParams, StopParams, prepare_stop
from .service import ControlService
from .status_feed import RuntimeReplyStatus


def build_control_service(ctx: Context, *, workspace_token: str | None) -> ControlService:
    """参数校验属于协议层，持久接纳和管理提交仍由各自 owner 完成。"""
    updates = ctx.require(PLUGIN_UPDATES)
    host = ctx.require(HOST_INFO)

    async def accept(session_id: str, message_id: str, incoming: ChannelInboundMessage) -> Message:
        return await ctx.require(CHANNEL_INPUT_V2)(session_id, message_id, incoming)

    @asynccontextmanager
    async def resolve_method(name: str) -> AsyncIterator[RpcMethod | None]:
        with ctx.borrow(RpcMethod.key(name)) as method:
            yield method

    @ctx.entrypoint
    async def install(source: str, marketplace: str, ref: str, sparse: list[str],
                      update_id: str) -> dict[str, object]:
        return asdict(await updates.install(ctx, update_id, source=source,
            marketplace=marketplace, ref=ref, sparse=tuple(sparse)))

    @ctx.entrypoint
    async def drain(plugin_id: str) -> str:
        if plugin_id == ctx.runtime.plugin_id:
            raise JsonRpcError(SERVER_OVERLOADED, "不能从 Gateway 当前请求同步排空自身；请提交卸载操作")
        await updates.drain(ctx, plugin_id)
        return f"插件已停用并排空: {plugin_id}"

    @ctx.entrypoint
    async def uninstall(plugin_id: str) -> dict[str, object]:
        return await updates.uninstall(ctx, plugin_id)

    def update(identity: str) -> dict[str, object]:
        status = updates.read(ctx, identity)
        if status is None:
            raise KeyError(identity)
        return asdict(status)

    async def message_display(page: MessagePage, *, display_only: bool) -> list[dict[str, object]]:
        with ctx.borrow(MESSAGE_DISPLAY) as display:
            if display is None:
                return message_rows(page, display_only=display_only)
            return await display(page, display_only=display_only)

    def reply_status(session_id: str) -> AsyncGenerator[dict[str, object], None]:
        return RuntimeReplyStatus(ctx).follow(session_id)

    async def stop(params: BaseModel) -> object:
        try:
            return await prepare_stop(ctx, cast(StopParams, params))
        except (RuntimeError, ValueError, TimeoutError, ConnectionError) as error:
            raise JsonRpcError(SERVER_OVERLOADED, str(error) or "停止准备超时") from error

    async def cancel_stop(params: BaseModel) -> object:
        request = cast(CancelStopParams, params)
        gate = ctx.require(RESTART_GATE)
        if gate.boot_id != request.boot_id:
            raise JsonRpcError(SERVER_OVERLOADED, "取消请求不属于当前 boot")
        claim = ctx.require(CONTROL_FRAMES).claim_for(request.session_id,
            CallRef(request.call_message_id, request.call_part_index))
        if claim is not None:
            claim.abort()
        gate.abort(request.request_id)
        return {"accepting": gate.accepting}

    return ControlService(
        ctx.require(MESSAGE_CATALOG), ctx.runtime.workspace, accept=accept,
        attachments=ctx.require(CHANNEL_ATTACHMENT_READ).resolve_refs,
        reply_status=reply_status, message_display=message_display,
        plugin_install=install, plugin_status=lambda: updates.status(ctx),
        plugin_update=update, plugin_drain=drain, plugin_uninstall=uninstall,
        workspace_token=workspace_token, boot_id=host.boot_id, ready=host.ready,
        control_frames=ctx.require(CONTROL_FRAMES), resolve_method=resolve_method,
        request_scope=ctx.runtime_scope,
        methods={"runtime/prepare-stop": RpcMethod(StopParams, stop),
                 "runtime/cancel-stop": RpcMethod(CancelStopParams, cancel_stop)},
    )
