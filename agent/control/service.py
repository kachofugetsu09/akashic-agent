from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import aclosing
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from types import MappingProxyType
from uuid import uuid4

from agent.control.protocol.errors import JsonRpcError, UNAUTHORIZED
from agent.control.protocol.models import InitializeParams, MessageSendParams
from agent.control.protocol.method import RpcMethod
from agent.plugin_composition.channels import ChannelInboundMessage
from infra.channels.message_view import follow_messages, message_rows, session_row
from session.artifacts import AttachmentRef
from session.log import MessageCatalog
from session.message import Message

logger = logging.getLogger(__name__)
Accept = Callable[[str, str, ChannelInboundMessage], Awaitable[Message]]
ReplyStatus = Callable[[str], AsyncGenerator[dict[str, object], None]]
PluginInstall = Callable[[str, str, str, list[str], str], Awaitable[dict[str, object]]]
PluginAction = Callable[[str], Awaitable[dict[str, object]]]


class ControlService:
    """控制端只接纳输入和读取日志，不创建 Turn、回复任务或父级发布屏障。"""

    def __init__(
        self, messages: MessageCatalog, workspace: Path, *, accept: Accept,
        reply_status: ReplyStatus,
        attachments: Callable[[tuple[str, ...]], tuple[AttachmentRef, ...]],
        plugin_install: PluginInstall | None = None,
        plugin_status: Callable[[], dict[str, object]] | None = None,
        plugin_update: Callable[[str], dict[str, object]] | None = None,
        plugin_promote: PluginAction | None = None,
        plugin_discard: PluginAction | None = None,
        plugin_drain: Callable[[str], Awaitable[str]] | None = None,
        plugin_uninstall: PluginAction | None = None,
        workspace_token: str | None = None, boot_id: str | None = None,
        ready: Callable[[], bool] | None = None,
        methods: Mapping[str, RpcMethod] | None = None,
    ) -> None:
        self.messages = messages
        self.workspace = workspace.resolve()
        self._accept = accept
        self._reply_status = reply_status
        self._attachments = attachments
        self._plugin_install = plugin_install
        self._plugin_status = plugin_status
        self._plugin_update = plugin_update
        self._plugin_promote = plugin_promote
        self._plugin_discard = plugin_discard
        self._plugin_drain = plugin_drain
        self._plugin_uninstall = plugin_uninstall
        self._workspace_token = workspace_token
        self._boot_id = boot_id
        self._ready = ready
        self._operations: set[asyncio.Task[object]] = set()
        self._closed = False
        self.methods = MappingProxyType(dict(methods or {}))

    def initialize(self, params: InitializeParams) -> dict[str, object]:
        if self._workspace_token is not None and not secrets.compare_digest(
            params.workspaceToken or "", self._workspace_token,
        ):
            raise JsonRpcError(UNAUTHORIZED, "Invalid workspace token")
        return {
            "protocolVersion": "2.0", "serverInfo": {"name": "akashic-agent", "version": "0.2.0"},
            "workspace": str(self.workspace),
            "capabilities": {"messageLog": True, "replyStatus": True},
        }

    def status(self) -> dict[str, object]:
        return {"ready": self._ready() if self._ready is not None else not self._closed,
                "bootId": self._boot_id, "workspace": str(self.workspace), "protocolVersion": "2.0"}

    def create_session(self) -> dict[str, object]:
        # 分配身份不创建空 Session；首个 Input 的 writer 才提交会话。
        return {"version": 2, "session_id": "akashic:" + uuid4().hex}

    def list_sessions(self, cursor: list[str] | None, limit: int) -> dict[str, object]:
        page = self.messages.sessions(visibility="listed", limit=limit,
            after=None if cursor is None else (cursor[0], cursor[1]))
        return {"version": 2, "items": [session_row(entry) for entry in page.items],
                "total": page.total, "next_cursor": page.next_cursor}

    def read_messages(self, session_id: str, after_seq: int, through_seq: int | None,
                      limit: int) -> dict[str, object]:
        page = self.messages.reader(session_id).read_page(
            after_seq=after_seq, through_seq=through_seq, limit=limit,
        )
        return {"version": 2, "session_id": session_id, "items": message_rows(page),
                "after_seq": after_seq, "through_seq": page.through_seq,
                "next_after_seq": page.messages[-1].seq if page.messages else after_seq,
                "has_more": page.has_more}

    async def send_message(self, request: MessageSendParams) -> dict[str, object]:
        """在传输边界构造原始输入；重投使用相同消息 ID，由唯一 writer 核对正文。"""
        # 1. 控制端是 Akashic 客户端，不接受来源、作者或内部会话属性覆盖。
        if self._closed:
            raise RuntimeError("控制服务已关闭")
        if not request.session_id.startswith("akashic:") or not request.session_id[8:]:
            raise ValueError("控制端输入需要 Akashic Session")
        if not request.text.strip() and not request.attachment_ids and request.retry_of is None:
            raise ValueError("text 与 attachment_ids 不能同时为空")
        if request.retry_of is not None and (
            request.text or request.attachment_ids or request.reply_to_message_id is not None
            or request.model_id is not None or request.reasoning_effort is not None
        ):
            raise ValueError("重试只引用原 Input，不能同时修改正文或模型")
        if request.reasoning_effort is not None and request.model_id is None:
            raise ValueError("reasoning_effort 必须与 model_id 一起提交")
        metadata: dict[str, str] = {}
        if request.retry_of is not None:
            metadata["retry_of_client_message_id"] = request.retry_of
        if request.reply_to_message_id is not None:
            metadata["reply_to_message_id"] = request.reply_to_message_id
        if request.model_id is not None:
            metadata["model_runtime_id"] = request.model_id
            metadata["model_reasoning_effort"] = request.reasoning_effort or ""
        refs = self._attachments(tuple(request.attachment_ids)) if request.attachment_ids else ()
        incoming = ChannelInboundMessage(
            channel="akashic", sender="control", chat_id=request.session_id[8:],
            content=request.text, timestamp=datetime.now(UTC), metadata=metadata, attachments=refs,
        )
        # 2. ACK 只证明 Input/Control 已提交，不等待模型、观察者或客户端投影。
        message = await self._accept(request.session_id, request.message_id, incoming)
        return {"version": 2, "session_id": message.session_id,
                "message_id": message.message_id, "seq": message.seq}

    async def follow(self, session_id: str, after_seq: int) -> AsyncGenerator[dict[str, object], None]:
        """合并两个只读订阅；取消只关闭读取任务，不取消已经接纳的工作。"""
        queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(1)

        async def messages() -> None:
            async with aclosing(follow_messages(self.messages.reader(session_id), after_seq=after_seq)) as feed:
                async for page in feed:
                    await queue.put({"type": "messages.appended", **page})

        async def replies() -> None:
            async with aclosing(self._reply_status(session_id)) as feed:
                async for status in feed:
                    await queue.put({"type": "reply.status", **status})

        async def run() -> None:
            async with asyncio.TaskGroup() as group:
                _ = group.create_task(messages())
                _ = group.create_task(replies())

        runner = asyncio.create_task(run(), name="control-read:" + session_id)
        pending: asyncio.Task[dict[str, object]] | None = None
        try:
            while True:
                pending = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait((runner, pending), return_when=asyncio.FIRST_COMPLETED)
                if runner in done:
                    runner.result()
                    raise RuntimeError("消息与回复订阅已关闭")
                yield pending.result()
                pending = None
        finally:
            if pending is not None:
                _ = pending.cancel()
                _ = await asyncio.gather(pending, return_exceptions=True)
            _ = runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass

    def plugin_status(self) -> dict[str, object]:
        if self._plugin_status is None:
            raise RuntimeError("控制服务没有插件管理能力")
        return self._plugin_status()

    def plugin_update(self, update_id: str) -> dict[str, object]:
        if self._plugin_update is None:
            raise RuntimeError("控制服务没有插件更新读取能力")
        return self._plugin_update(update_id)

    async def install_plugin(self, source: str, marketplace: str, ref: str,
                             sparse: list[str], update_id: str) -> dict[str, object]:
        if self._plugin_install is None:
            raise RuntimeError("控制服务没有插件安装能力")
        return cast(dict[str, object], await self._operate(
            self._plugin_install(source, marketplace, ref, sparse, update_id),
        ))

    async def promote_plugin(self, update_id: str) -> dict[str, object]:
        if self._plugin_promote is None:
            raise RuntimeError("控制服务没有插件发布能力")
        return cast(dict[str, object], await self._operate(self._plugin_promote(update_id)))

    async def discard_plugin(self, update_id: str) -> dict[str, object]:
        if self._plugin_discard is None:
            raise RuntimeError("控制服务没有插件回退能力")
        return cast(dict[str, object], await self._operate(self._plugin_discard(update_id)))

    async def disable_and_drain_plugin(self, plugin_id: str) -> dict[str, object]:
        if self._plugin_drain is None:
            raise RuntimeError("控制服务没有插件停用能力")
        message = await self._operate(self._plugin_drain(plugin_id))
        return {"plugin_id": plugin_id, "drained": True, "message": message}

    async def uninstall_plugin(self, plugin_id: str) -> dict[str, object]:
        if self._plugin_uninstall is None:
            raise RuntimeError("控制服务没有插件卸载能力")
        return cast(dict[str, object], await self._operate(self._plugin_uninstall(plugin_id)))

    async def _operate(self, operation: Awaitable[object]) -> object:
        """控制连接只等待管理操作；断线不取消已交给应用 owner 的操作。"""
        async def run() -> object:
            return await operation

        task = asyncio.create_task(run(), name="control-plugin-operation")
        self._operations.add(task)

        def finished(task: asyncio.Task[object]) -> None:
            self._operations.remove(task)
            if not task.cancelled() and (error := task.exception()) is not None:
                logger.error("插件管理操作失败", exc_info=(type(error), error, error.__traceback__))

        task.add_done_callback(finished)
        return await asyncio.shield(task)

    async def shutdown(self) -> None:
        self._closed = True
        tasks = tuple(self._operations)
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)
