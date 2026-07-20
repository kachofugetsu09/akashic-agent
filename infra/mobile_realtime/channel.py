from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import TYPE_CHECKING, cast
from uuid import UUID

from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from agent.plugins.mobile_ui import (
    MobileUiPluginUnavailable,
    MobileUiQueryOverloaded,
    MobileUiQueryTimeout,
    MobileUiRpcExecutionError,
    MobileUiRpcInvalidRequest,
    MobileUiStaleRevision,
)
from infra.channels.contract import ChannelContext
from infra.channels.reply_context import build_reply_inbound_text
from infra.mobile_realtime.attachments import (
    AttachmentChunk,
    AttachmentRequestError,
    AttachmentTransferService,
    MAX_ATTACHMENT_CHUNK_BYTES,
    attachment_descriptor,
)
from infra.mobile_realtime.protocol import (
    AttachmentBeginCommand,
    AttachmentDownloadCommand,
    AttachmentFinishCommand,
    ClientCommand,
    GenericCommand,
    MessageReplyReference,
    MessageSendCommand,
    MAX_JSON_FRAME_BYTES,
)
from infra.mobile_realtime.plugin_ui import PluginUiQuery, PluginUiQueryScheduler
from infra.mobile_realtime.remote_media import (
    RemoteMediaError,
    RemoteMediaSnapshot,
    snapshot_remote_media,
)
from infra.mobile_realtime.storage import AttachmentStateError, CommandReceipt

if TYPE_CHECKING:
    from agent.plugins.mobile_ui import MobileUiProvider
    from infra.mobile_realtime.gateway import MobileGatewayRuntime


logger = logging.getLogger(__name__)


class MobileCommandError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CommandReply:
    type: str
    payload: dict[str, object]
    session_id: str | None = None
    turn_id: str | None = None
    binary: AttachmentChunk | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedReply:
    message_id: str
    role: str
    content: str
    preview: str


@dataclass(slots=True)
class _DeltaBatch:
    segments: list[tuple[str, str, str | None, int | None]]
    byte_count: int
    timer: asyncio.Task[None]


@dataclass(slots=True)
class _ProcessTurnState:
    next_ordinal: int
    thinking_block: tuple[str, int] | None
    tool_blocks: dict[str, tuple[str, int, float]]


_DELTA_FLUSH_SECONDS = 0.05
_DELTA_FLUSH_BYTES = 4 * 1024
_MAX_DELTA_BATCHES = 256
_BOT_COMMAND_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_PLUGIN_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_PLUGIN_ID_PATTERN = re.compile(rf"^{_PLUGIN_SEGMENT}(?:@{_PLUGIN_SEGMENT})?$")
_MOBILE_TOOL_ARGUMENT_MAX_DEPTH = 5
_MOBILE_TOOL_ARGUMENT_MAX_ITEMS = 256
_MOBILE_TOOL_ARGUMENT_MAX_CONTAINER_ITEMS = 64
_MOBILE_TOOL_ARGUMENT_MAX_STRING_CHARS = 2_000
_MOBILE_TOOL_ARGUMENT_MAX_BYTES = 8 * 1024
_MOBILE_HISTORY_TOOL_ARGUMENT_MAX_BYTES = 8 * 1024
_MOBILE_HISTORY_PAYLOAD_MAX_BYTES = 240 * 1024
_MOBILE_TOOL_ARGUMENT_REDACTED = "[已隐藏]"
_MOBILE_TOOL_ARGUMENT_TRUNCATED = "[已截断]"
_MOBILE_TOOL_SECRET_KEYS = frozenset(
    {
        "secret",
        "auth",
        "token",
        "password",
        "passwd",
        "authorization",
        "cookie",
        "apikey",
        "privatekey",
        "secretaccesskey",
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "credential",
        "credentials",
    }
)
_MOBILE_TOOL_SECRET_TEXT_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![a-z0-9_])(?:[a-z][a-z0-9]*[-_])*(?:authorization|"
    r"proxy[-_ ]?authorization|secret[-_ ]?access[-_ ]?key|api[-_ ]?key|"
    r"access[-_ ]?token|refresh[-_ ]?token|client[-_ ]?secret|token|secret|"
    r"password|passwd|cookie)\s*[:=]"
    r"|--?(?:api[-_]?key|access[-_]?token|refresh[-_]?token|client[-_]?secret|"
    r"password|passwd|cookie|token)\s+(?:[^\s]|$)"
    r"|\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
)


class MobileRealtimeChannel:
    """把移动协议接入现有消息、生命周期和主动推送总线。"""

    name = "mobile"

    def __init__(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime
        self._ctx: ChannelContext | None = None
        self._processing_commands: set[tuple[str, str]] = set()
        self._active_turn_ids: dict[str, str] = {}
        self._process_turns: dict[tuple[str, str], _ProcessTurnState] = {}
        self._delta_batches: dict[tuple[str, str], _DeltaBatch] = {}
        self._delta_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._delta_failure: BaseException | None = None
        self._attachments: AttachmentTransferService | None = None
        self._mobile_ui_provider: MobileUiProvider | None = None
        self._mobile_ui_scheduler: PluginUiQueryScheduler | None = None
        self._mobile_ui_catalog_identity = ""
        self._mobile_ui_hot_connections: dict[str, int] = {}

    def bind_mobile_ui_provider(self, provider: MobileUiProvider) -> None:
        """绑定读取当前插件快照的移动 UI 提供器。"""

        if self._mobile_ui_provider is not None:
            raise RuntimeError("Mobile UI provider 已绑定")
        self._mobile_ui_provider = provider
        self._mobile_ui_scheduler = PluginUiQueryScheduler(provider)
        self._mobile_ui_catalog_identity = _mobile_ui_catalog_identity(provider.catalog())

    async def refresh_mobile_ui_catalog(self) -> None:
        """目录内容变化时通知所有手机重新拉取插件 UI。"""

        provider = self._mobile_ui_provider
        if provider is None:
            return
        catalog = provider.catalog()
        identity = _mobile_ui_catalog_identity(catalog)
        if identity == self._mobile_ui_catalog_identity:
            return
        _ = await asyncio.gather(
            *(
                self._runtime.publish_connection_control(
                    control_type="plugin.ui.changed",
                    payload={"catalog_revision": identity},
                    device_id=device_id,
                    connection_epoch=connection_epoch,
                )
                for device_id, connection_epoch in tuple(
                    self._mobile_ui_hot_connections.items()
                )
            )
        )
        self._mobile_ui_catalog_identity = identity

    async def start(self, ctx: ChannelContext) -> None:
        """注册移动渠道的出站、流事件和主动推送入口。"""

        if self._ctx is not None:
            raise RuntimeError("MobileRealtimeChannel 已启动")
        self._ctx = ctx
        self._attachments = AttachmentTransferService(
            self._runtime.storage,
            ctx.attachment_store,
            max_attachment_bytes=self._runtime.config.max_attachment_mb * 1024 * 1024,
        )
        _ = ctx.bus.subscribe_outbound(self.name, self._on_response)
        _ = ctx.event_bus.on(TurnStarted, self._on_turn_started)
        _ = ctx.event_bus.on(StreamDeltaReady, self._on_stream_delta)
        _ = ctx.event_bus.on(ToolCallStarted, self._on_tool_call_started)
        _ = ctx.event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
        _ = ctx.push_tool.register_channel(
            self.name,
            text=self.send,
            stream_text=self.send_stream,
            text_with_metadata=self.send_with_metadata,
        )

    async def stop(self) -> None:
        for batch in self._delta_batches.values():
            _ = batch.timer.cancel()
        if self._delta_batches:
            _ = await asyncio.gather(
                *(batch.timer for batch in self._delta_batches.values()),
                return_exceptions=True,
            )
        self._delta_batches.clear()
        self._delta_locks.clear()
        self._delta_failure = None
        self._attachments = None
        self._ctx = None
        self._active_turn_ids.clear()
        self._process_turns.clear()
        self._processing_commands.clear()

    async def handle_command(
        self,
        *,
        device_id: str,
        frame: ClientCommand,
    ) -> CommandReply:
        """幂等执行业务命令，并持久化可跨重连复用的回复。"""

        # 1. 先持久化命令占用，避免重连重复触发 Agent turn
        self._raise_delta_failure()
        receipt, created = self._runtime.storage.reserve_command(
            device_id=device_id,
            command_id=frame.id,
            command_type=frame.type,
            request_hash=_command_hash(frame),
            created_at=_utc_now(),
        )
        if not created:
            replay = self._recover_message_send_receipt(
                device_id=device_id,
                frame=frame,
                receipt=receipt,
            )
            if (
                isinstance(frame, AttachmentDownloadCommand)
                and replay.type == "attachment.download.ok"
            ):
                return self._download_attachment(frame, replay)
            return replay

        # 2. 当前实例只在命令实际执行期间拥有 processing 收据
        command_key = (device_id, frame.id)
        self._processing_commands.add(command_key)
        try:
            reply = await self._execute_command(device_id=device_id, frame=frame)
        except MobileCommandError as error:
            reply = CommandReply(
                type=f"{frame.type}.error",
                payload={"code": error.code, "message": str(error)},
                session_id=frame.session_id,
                turn_id=frame.turn_id,
            )

        # 3. 只有收据完成后才释放当前进程对未决副作用的所有权
        _validate_reply_frame_size(frame, reply)
        completed = self._runtime.storage.complete_command(
            device_id=device_id,
            command_id=frame.id,
            reply_type=reply.type,
            reply_payload_json=json.dumps(
                reply.payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            session_id=reply.session_id,
            turn_id=reply.turn_id,
            completed_at=_utc_now(),
        )
        self._processing_commands.discard(command_key)
        stored = _reply_from_receipt(completed)
        return CommandReply(
            type=stored.type,
            payload=stored.payload,
            session_id=stored.session_id,
            turn_id=stored.turn_id,
            binary=reply.binary,
        )

    async def handle_plugin_ui_command(
        self,
        *,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """执行不写 command receipt 的 Mobile Plugin UI v2 临时请求。"""

        try:
            if frame.type == "plugin.ui.catalog":
                return self._plugin_ui_catalog(device_id, frame)
            if frame.type == "plugin.ui.asset.get":
                return self._plugin_ui_asset(frame)
            if frame.type == "plugin.ui.query":
                return await self._plugin_ui_query(device_id, frame)
            if frame.type == "plugin.ui.cancel":
                return await self._cancel_plugin_ui(device_id, frame)
        except MobileUiPluginUnavailable as error:
            raise MobileCommandError("plugin_unavailable", str(error)) from error
        except MobileUiStaleRevision as error:
            raise MobileCommandError("stale_revision", str(error)) from error
        except MobileUiRpcInvalidRequest as error:
            raise MobileCommandError("plugin_invalid_request", str(error)) from error
        except MobileUiRpcExecutionError as error:
            raise MobileCommandError("plugin_failed", str(error)) from error
        except MobileUiQueryTimeout as error:
            raise MobileCommandError("plugin_timeout", str(error)) from error
        except MobileUiQueryOverloaded as error:
            raise MobileCommandError("plugin_overloaded", str(error)) from error
        raise MobileCommandError("unsupported_command", f"尚不支持命令: {frame.type}")

    async def cancel_plugin_ui_device(self, device_id: str) -> None:
        """断线时取消设备的全部临时插件查询。"""

        scheduler = self._mobile_ui_scheduler
        if scheduler is not None:
            await scheduler.cancel_device(device_id)
        _ = self._mobile_ui_hot_connections.pop(device_id, None)

    def _recover_message_send_receipt(
        self,
        *,
        device_id: str,
        frame: ClientCommand,
        receipt: CommandReceipt,
    ) -> CommandReply:
        """从已持久化消息修复中断的 message.send 收据。"""

        # 1. 已完成或非消息命令继续复用稳定回复
        if receipt.status == "completed":
            self._processing_commands.discard((device_id, frame.id))
            return _reply_from_receipt(receipt)
        if not isinstance(frame, MessageSendCommand):
            return _reply_from_receipt(receipt)

        # 2. 只有同一 client_message_id 已落库时，才能确认副作用已完成
        session_id = self._normalize_session_id(frame.session_id)
        message = self._require_ctx().session_manager.control_store.get_message_by_client_id(
            session_id,
            frame.payload.client_message_id,
        )
        if message is None:
            if (device_id, frame.id) in self._processing_commands:
                return _reply_from_receipt(receipt)
            return self._complete_interrupted_message_send(
                device_id=device_id,
                frame=frame,
                session_id=session_id,
            )
        if message["role"] != "user":
            raise RuntimeError(
                f"client_message_id 绑定了非用户消息: {frame.payload.client_message_id}"
            )

        # 3. 原子补写成功收据，后续重放继续得到相同 ACK
        completed = self._runtime.storage.complete_command(
            device_id=device_id,
            command_id=frame.id,
            reply_type="message.send.ok",
            reply_payload_json=json.dumps(
                {
                    "accepted": True,
                    "client_message_id": frame.payload.client_message_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            session_id=session_id,
            turn_id=None,
            completed_at=_utc_now(),
        )
        self._processing_commands.discard((device_id, frame.id))
        return _reply_from_receipt(completed)

    def _complete_interrupted_message_send(
        self,
        *,
        device_id: str,
        frame: MessageSendCommand,
        session_id: str,
    ) -> CommandReply:
        """把服务重启前未落库的发送收据收束为可安全重试。"""

        completed = self._runtime.storage.complete_command(
            device_id=device_id,
            command_id=frame.id,
            reply_type="message.send.error",
            reply_payload_json=json.dumps(
                {
                    "code": "command_interrupted",
                    "message": "上次发送在服务重启时中断，可以安全重试",
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            session_id=session_id,
            turn_id=None,
            completed_at=_utc_now(),
        )
        return _reply_from_receipt(completed)

    async def send(self, chat_id: str, message: str) -> None:
        await self._send_proactive(chat_id, message, delivery_id=None)

    async def send_with_metadata(
        self,
        chat_id: str,
        message: str,
        metadata: dict[str, object],
    ) -> None:
        """发送携带核心主动投递身份的消息。"""

        delivery_id = metadata.get("delivery_id")
        if (
            not isinstance(delivery_id, str)
            or not delivery_id
            or len(delivery_id) > 128
        ):
            raise ValueError("mobile proactive delivery_id 无效")
        await self._send_proactive(chat_id, message, delivery_id=delivery_id)

    async def _send_proactive(
        self,
        chat_id: str,
        message: str,
        *,
        delivery_id: str | None,
    ) -> None:
        """把主动文字发布为移动端持久事件。"""

        self._raise_delta_failure()
        session_id = self._session_id(chat_id)
        payload: dict[str, object] = {
            "content": message,
            "attachments": [],
            "metadata": {"source": "message_push"},
        }
        if delivery_id is not None:
            payload["delivery_id"] = delivery_id
        await self._runtime.publish_event(
            event_type="message.proactive",
            session_id=session_id,
            payload=payload,
        )

    async def send_stream(self, chat_id: str, message: str) -> None:
        await self.send(chat_id, message)

    async def _execute_command(
        self,
        *,
        device_id: str,
        frame: ClientCommand,
    ) -> CommandReply:
        if frame.type == "session.list":
            return await self._list_sessions(device_id, frame)
        if frame.type == "session.create":
            raise MobileCommandError(
                "unsupported_command",
                "当前版本由手机本地生成 mobile session_id",
            )
        if frame.type == "session.open":
            return await self._open_session(device_id, frame)
        if frame.type == "history.get":
            return await self._get_history(device_id, frame)
        if frame.type == "command.list":
            return self._list_commands(frame)
        if frame.type == "message.send":
            return await self._send_message(device_id, frame)
        if frame.type == "turn.stop":
            return await self._stop_turn(device_id, frame)
        if frame.type == "attachment.begin":
            return await self._begin_attachment(device_id, frame)
        if frame.type == "attachment.finish":
            return await self._finish_attachment(device_id, frame)
        if frame.type == "attachment.download":
            return self._download_attachment(frame)
        raise MobileCommandError("unsupported_command", f"尚不支持命令: {frame.type}")

    def _list_commands(self, frame: GenericCommand) -> CommandReply:
        """返回当前已启用插件的快捷命令目录。"""

        # 1. 命令目录由 ChannelContext 统一提供
        _expect_keys(frame.payload, set())
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw_command, raw_description in self._require_ctx().mobile_bot_commands:
            command = raw_command.strip().removeprefix("/")
            description = raw_description.strip()
            if not _BOT_COMMAND_PATTERN.fullmatch(command):
                raise RuntimeError(f"插件命令名无效: {raw_command!r}")
            if not description or len(description) > 256:
                raise RuntimeError(f"插件命令描述无效: {command}")
            if command == "stop" or command in seen:
                continue
            seen.add(command)
            items.append({"command": command, "description": description})

        # 2. 保持插件注册顺序，便于管理高频命令的位置
        return CommandReply(type="command.list.ok", payload={"items": items})

    def _plugin_ui_catalog(self, device_id: str, frame: GenericCommand) -> CommandReply:
        """返回不含源码的 Mobile Plugin UI v2 catalog。"""

        _expect_keys(frame.payload, {"subscribe", "if_revision"})
        subscribe = frame.payload.get("subscribe", False)
        if not isinstance(subscribe, bool):
            raise MobileCommandError("invalid_payload", "subscribe 必须是布尔值")
        if subscribe:
            self._mobile_ui_hot_connections[device_id] = frame.connection_epoch
        else:
            _ = self._mobile_ui_hot_connections.pop(device_id, None)
        if_revision = frame.payload.get("if_revision")
        if if_revision is not None and (
            not isinstance(if_revision, str)
            or not re.fullmatch(r"[0-9a-f]{64}", if_revision)
        ):
            raise MobileCommandError("invalid_revision", "if_revision 无效")
        provider = self._mobile_ui_provider
        catalog: dict[str, object]
        if provider is None:
            catalog = {
                "catalog_revision": hashlib.sha256(b"[]").hexdigest(),
                "items": [],
            }
        else:
            catalog = provider.catalog()
        if catalog["catalog_revision"] == if_revision:
            return CommandReply(
                type="plugin.ui.catalog.not_modified",
                payload={"catalog_revision": if_revision},
            )
        return CommandReply(type="plugin.ui.catalog.ok", payload=catalog)

    def _plugin_ui_asset(self, frame: GenericCommand) -> CommandReply:
        """按 revision 和摘要返回一个未缓存资源。"""

        _expect_keys(
            frame.payload,
            {"plugin_id", "plugin_revision", "kind", "sha256"},
        )
        plugin_id = frame.payload["plugin_id"]
        plugin_revision = frame.payload["plugin_revision"]
        kind = frame.payload["kind"]
        sha256 = frame.payload["sha256"]
        if not isinstance(plugin_id, str) or not _PLUGIN_ID_PATTERN.fullmatch(plugin_id):
            raise MobileCommandError("invalid_plugin", "plugin_id 无效")
        if not isinstance(plugin_revision, str) or not 1 <= len(plugin_revision) <= 128:
            raise MobileCommandError("invalid_revision", "plugin_revision 无效")
        if kind not in {"module", "stylesheet"}:
            raise MobileCommandError("invalid_asset", "kind 无效")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise MobileCommandError("invalid_asset", "sha256 无效")
        asset = self._require_mobile_ui_provider().asset(
            plugin_id,
            plugin_revision,
            cast(str, kind),
            sha256,
        )
        return CommandReply(type="plugin.ui.asset.get.ok", payload=asset)

    async def _plugin_ui_query(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """校验并调度一个有 owner 的只读插件查询。"""

        _expect_keys(
            frame.payload,
            {"owner_id", "plugin_id", "plugin_revision", "method", "payload", "slot"},
        )
        owner_id = frame.payload["owner_id"]
        plugin_id = frame.payload["plugin_id"]
        plugin_revision = frame.payload["plugin_revision"]
        method = frame.payload["method"]
        payload = frame.payload["payload"]
        slot = frame.payload["slot"]
        if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 128:
            raise MobileCommandError("invalid_owner", "owner_id 无效")
        if not isinstance(plugin_id, str) or not _PLUGIN_ID_PATTERN.fullmatch(plugin_id):
            raise MobileCommandError("invalid_plugin", "plugin_id 无效")
        if not isinstance(plugin_revision, str) or not 1 <= len(plugin_revision) <= 128:
            raise MobileCommandError("invalid_revision", "plugin_revision 无效")
        if not isinstance(method, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", method):
            raise MobileCommandError("invalid_method", "插件方法无效")
        if not isinstance(payload, dict):
            raise MobileCommandError("invalid_payload", "插件参数必须是对象")
        if _mobile_tool_argument_encoded_size(cast(dict[str, object], payload)) > 64 * 1024:
            raise MobileCommandError("invalid_payload", "插件参数超过 64 KiB")
        if slot not in {
            "dashboard.main",
            "turn.before_reasoning",
            "turn.before_tool",
            "turn.after_answer",
            "drawer.panel",
        }:
            raise MobileCommandError("invalid_slot", "插件 slot 无效")
        session_id = (
            None
            if frame.session_id is None
            else self._require_mobile_session(frame.session_id)
        )
        scheduler = self._require_mobile_ui_scheduler()
        result = await scheduler.execute(
            device_id,
            PluginUiQuery(
                request_id=frame.id,
                owner_id=owner_id,
                plugin_id=plugin_id,
                plugin_revision=plugin_revision,
                method=method,
                payload=cast(dict[str, object], payload),
                slot=cast(str, slot),
                session_id=session_id,
                turn_id=frame.turn_id,
            ),
        )
        return CommandReply(type="plugin.ui.query.ok", payload={"result": result})

    async def _cancel_plugin_ui(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """取消一个已卸载 owner 的全部查询。"""

        _expect_keys(frame.payload, {"owner_id"})
        owner_id = frame.payload["owner_id"]
        if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 128:
            raise MobileCommandError("invalid_owner", "owner_id 无效")
        cancelled = await self._require_mobile_ui_scheduler().cancel_owner(
            device_id,
            owner_id,
        )
        return CommandReply(
            type="plugin.ui.cancel.ok",
            payload={"cancelled": cancelled},
        )

    def _require_mobile_ui_provider(self) -> MobileUiProvider:
        provider = self._mobile_ui_provider
        if provider is None:
            raise MobileCommandError("plugin_unavailable", "服务端没有启用移动 UI 插件")
        return provider

    def _require_mobile_ui_scheduler(self) -> PluginUiQueryScheduler:
        scheduler = self._mobile_ui_scheduler
        if scheduler is None:
            raise MobileCommandError("plugin_unavailable", "服务端没有启用移动 UI 插件")
        return scheduler

    async def handle_attachment_chunk(
        self,
        *,
        device_id: str,
        chunk: AttachmentChunk,
    ) -> None:
        """落盘一个二进制分片，并按稀疏确认发布进度。"""

        try:
            record, should_report = await asyncio.to_thread(
                self._require_attachments().append_chunk,
                device_id=device_id,
                chunk=chunk,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_chunk_rejected", str(error)) from error
        if should_report:
            await self._runtime.publish_event(
                event_type="attachment.progress",
                device_id=device_id,
                session_id=record.session_id,
                payload={
                    "attachment_id": record.attachment_id,
                    "transferred_bytes": record.transferred_bytes,
                    "size_bytes": record.size_bytes,
                },
            )

    async def _begin_attachment(
        self,
        device_id: str,
        frame: AttachmentBeginCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        try:
            record = await asyncio.to_thread(
                self._require_attachments().begin_upload,
                device_id=device_id,
                attachment_id=frame.payload.attachment_id,
                session_id=session_id,
                filename=frame.payload.filename,
                content_type=frame.payload.content_type,
                size_bytes=frame.payload.size_bytes,
                sha256=frame.payload.sha256,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_begin_rejected", str(error)) from error
        return CommandReply(
            type="attachment.begin.ok",
            session_id=session_id,
            payload={
                **attachment_descriptor(record),
                "next_offset": record.transferred_bytes,
                "chunk_size": MAX_ATTACHMENT_CHUNK_BYTES,
                "state": record.state,
            },
        )

    async def _finish_attachment(
        self,
        device_id: str,
        frame: AttachmentFinishCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        try:
            record = await asyncio.to_thread(
                self._require_attachments().finish_upload,
                device_id=device_id,
                session_id=session_id,
                attachment_id=frame.payload.attachment_id,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_finish_rejected", str(error)) from error
        await self._runtime.publish_event(
            event_type="attachment.ready",
            device_id=device_id,
            session_id=session_id,
            payload=attachment_descriptor(record),
        )
        return CommandReply(
            type="attachment.finish.ok",
            session_id=session_id,
            payload={**attachment_descriptor(record), "state": "ready"},
        )

    def _download_attachment(
        self,
        frame: AttachmentDownloadCommand,
        stored: CommandReply | None = None,
    ) -> CommandReply:
        """读取一个出站附件分片，并让二进制帧先于确认回复发送。"""

        session_id = self._normalize_session_id(frame.session_id)
        try:
            outbound = self._require_attachments().read_outbound_chunk(
                session_id=session_id,
                attachment_id=frame.payload.attachment_id,
                offset=frame.payload.offset,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_download_rejected", str(error)) from error
        next_offset = outbound.offset + len(outbound.data)
        payload: dict[str, object] = {
            **outbound.descriptor,
            "offset": outbound.offset,
            "next_offset": next_offset,
            "complete": outbound.eof,
        }
        if stored is not None and stored.payload != payload:
            raise RuntimeError("已完成的附件下载回复与当前文件状态不一致")
        return CommandReply(
            type="attachment.download.ok",
            session_id=session_id,
            payload=payload,
            binary=AttachmentChunk(
                attachment_id=frame.payload.attachment_id,
                offset=outbound.offset,
                data=outbound.data,
            ),
        )

    async def _list_sessions(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """发布全部移动会话索引，供手机按需分页拉取缺失历史。"""

        # 1. 所有已认证手机共享 mobile 渠道的完整会话空间
        _expect_keys(frame.payload, set())
        ctx = self._require_ctx()
        session_rows = {item["key"]: item for item in ctx.session_manager.list_sessions()}
        session_ids = tuple(
            session_id
            for session_id in session_rows
            if session_id.startswith(f"{self.name}:")
        )

        # 2. 补充抽屉标题和历史消息总数
        items: list[dict[str, object]] = []
        for session_id in session_ids:
            session = session_rows.get(session_id)
            if session is None:
                raise RuntimeError(f"已绑定移动会话在 session store 中不存在: {session_id}")
            messages, total = ctx.session_manager.control_store.list_messages_for_dashboard(
                session_key=session_id,
                page=1,
                page_size=1,
                sort_by="seq",
                sort_order="asc",
            )
            first_content = str(messages[0]["content"]).strip() if messages else ""
            items.append(
                {
                    "session_id": session_id,
                    "title": first_content.splitlines()[0][:32] or "新对话",
                    "updated_at": str(session["updated_at"]),
                    "message_count": total,
                }
            )

        # 3. 索引也走 durable event，断线后仍会重放
        await self._runtime.publish_event(
            event_type="session.list",
            device_id=device_id,
            payload={"items": cast(list[object], items)},
        )
        return CommandReply(type="session.list.ok", payload={"total": len(items)})

    async def _open_session(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        _expect_keys(frame.payload, set())
        session_id = self._require_mobile_session(frame.session_id)
        await self._runtime.publish_event(
            event_type="session.updated",
            session_id=session_id,
            payload={"session_id": session_id, "state": "opened"},
        )
        return CommandReply(
            type="session.open.ok",
            session_id=session_id,
            payload={"session_id": session_id},
        )

    async def _get_history(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        session_id = self._require_mobile_session(frame.session_id)
        pagination = _pagination_payload(frame.payload)
        (
            items,
            total,
        ) = self._require_ctx().session_manager.control_store.list_messages_for_dashboard(
            session_key=session_id,
            page=pagination["page"],
            page_size=pagination["page_size"],
            sort_by="seq",
            sort_order="asc",
        )
        mobile_items = [await self._mobile_history_item(item) for item in items]
        page_payload: dict[str, object] = {
            "items": cast(list[object], mobile_items),
            "total": total,
            **pagination,
        }
        _fit_mobile_history_payload(page_payload)
        await self._runtime.publish_event(
            event_type="history.page",
            session_id=session_id,
            device_id=device_id,
            payload=page_payload,
        )
        return CommandReply(
            type="history.get.ok",
            session_id=session_id,
            payload={"total": total, **pagination},
        )

    async def _send_message(
        self,
        device_id: str,
        frame: MessageSendCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        if frame.id != frame.payload.client_message_id:
            raise MobileCommandError(
                "client_message_id_mismatch",
                "message.send 的命令 ID 必须与 client_message_id 一致",
            )
        if not frame.payload.text.strip() and not frame.payload.media_refs:
            raise MobileCommandError("empty_message", "文字和附件不能同时为空")
        ctx = self._require_ctx()
        claimed_session = self._runtime.storage.has_session_claim(session_id)
        if claimed_session and not ctx.session_manager.session_exists(session_id):
            raise MobileCommandError(
                "session_not_found",
                "会话已从电脑端删除，请在手机上新建会话后继续",
            )
        try:
            media = self._require_attachments().resolve_uploads(
                device_id=device_id,
                session_id=session_id,
                attachment_ids=list(frame.payload.media_refs),
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_not_ready", str(error)) from error
        reply = self._resolve_reply(session_id, frame.payload.reply_to)
        inbound_content = frame.payload.text
        metadata: dict[str, object] = {
            "client_request_id": frame.id,
            "client_message_id": frame.payload.client_message_id,
            "client_created_at": frame.payload.client_created_at,
            "device_id": device_id,
            "require_existing_session": True,
        }
        if reply is not None:
            metadata.update(
                {
                    "display_content": frame.payload.text,
                    "reply_to_message_id": reply.message_id,
                    "reply_role": reply.role,
                    "reply_preview": reply.preview,
                }
            )
            inbound_content = build_reply_inbound_text(
                frame.payload.text,
                reply.content,
                sender_label="你" if reply.role == "user" else "Akashic",
            )
        self._runtime.storage.claim_session(
            device_id=device_id,
            session_id=session_id,
            created_at=_utc_now(),
        )
        if not claimed_session:
            _ = ctx.session_manager.get_or_create(session_id)
        try:
            _, admission_id = ctx.session_manager.admit_existing(session_id)
        except KeyError as error:
            raise MobileCommandError(
                "session_not_found",
                "会话已从电脑端删除，请在手机上新建会话后继续",
            ) from error
        inbound = InboundMessage(
            channel=self.name,
            sender=f"device:{device_id}",
            chat_id=self._chat_id(session_id),
            content=inbound_content,
            media=media,
            metadata=metadata,
            session_admission_id=admission_id,
        )
        try:
            await ctx.bus.publish_inbound(inbound)
        except BaseException:
            ctx.session_manager.release_admission(admission_id)
            raise
        return CommandReply(
            type="message.send.ok",
            session_id=session_id,
            payload={
                "accepted": True,
                "client_message_id": frame.payload.client_message_id,
            },
        )

    def _resolve_reply(
        self,
        session_id: str,
        reference: MessageReplyReference | None,
    ) -> _ResolvedReply | None:
        """把客户端引用解析为同会话的 canonical 消息摘要。"""

        if reference is None:
            return None
        store = self._require_ctx().session_manager.control_store
        if reference.message_id is not None:
            target = store.get_message(reference.message_id)
        elif reference.client_message_id is not None:
            target = store.get_message_by_client_id(
                session_id,
                reference.client_message_id,
            )
        else:
            target = store.get_message_by_delivery_id(
                session_id,
                cast(str, reference.delivery_id),
            )
        if target is None:
            raise MobileCommandError("reply_target_missing", "被引用的消息不存在或尚未同步")
        if target["session_key"] != session_id:
            raise MobileCommandError("reply_target_session_mismatch", "不能引用其他会话的消息")
        role = str(target["role"])
        if role not in {"user", "assistant"}:
            raise RuntimeError(f"被引用消息角色无效: {target['id']} {role}")
        content = _reply_source_text(target)
        return _ResolvedReply(
            message_id=str(target["id"]),
            role=role,
            content=content,
            preview=_reply_preview(content),
        )

    async def _stop_turn(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        _expect_keys(frame.payload, set())
        session_id = self._require_mobile_session(frame.session_id)
        turn_id = frame.turn_id
        if turn_id is None:
            raise MobileCommandError("turn_id_required", "停止生成必须携带 turn_id")
        active_turn_id = self._active_turn_ids.get(session_id)
        if active_turn_id is None:
            raise MobileCommandError("turn_not_active", "当前会话没有正在生成的内容")
        if active_turn_id != turn_id:
            raise MobileCommandError("stale_turn", "目标 turn 已结束或已被新一轮替代")
        interrupt = self._require_ctx().interrupt_controller
        if interrupt is None:
            raise MobileCommandError("interrupt_unavailable", "当前未启用中断功能")
        await self._flush_deltas(session_id, turn_id)
        result = interrupt.request_interrupt(
            session_key=session_id,
            sender=f"device:{device_id}",
            command="/stop",
        )
        if result.status not in {"interrupted", "idle"}:
            raise RuntimeError(f"中断控制器返回未知状态: {result.status}")
        await self._runtime.publish_event(
            event_type="turn.interrupted",
            session_id=session_id,
            turn_id=turn_id,
            payload={"status": result.status, "message": result.message},
        )
        _ = self._active_turn_ids.pop(session_id, None)
        _ = self._process_turns.pop((session_id, turn_id), None)
        return CommandReply(
            type="turn.stop.ok",
            session_id=session_id,
            turn_id=turn_id,
            payload={"status": result.status, "message": result.message},
        )

    async def _on_turn_started(self, event: TurnStarted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or event.session_key
        self._active_turn_ids[event.session_key] = turn_id
        process_key = (event.session_key, turn_id)
        if process_key in self._process_turns:
            raise RuntimeError(
                f"mobile turn.started 重复: {event.session_key}/{turn_id}"
            )
        self._process_turns[process_key] = _ProcessTurnState(
            next_ordinal=0,
            thinking_block=None,
            tool_blocks={},
        )
        await self._runtime.publish_event(
            event_type="turn.started",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={"content": event.content},
        )

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or self._current_turn_id(event.session_key)
        if event.thinking_delta:
            block_id, ordinal = self._thinking_block(event.session_key, turn_id)
            await self._buffer_delta(
                session_id=event.session_key,
                turn_id=turn_id,
                event_type="react.thinking.delta",
                delta=event.thinking_delta,
                block_id=block_id,
                ordinal=ordinal,
            )
        if event.content_delta:
            await self._buffer_delta(
                session_id=event.session_key,
                turn_id=turn_id,
                event_type="answer.delta",
                delta=event.content_delta,
                block_id=None,
                ordinal=None,
            )

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or self._current_turn_id(event.session_key)
        await self._flush_deltas(event.session_key, turn_id)
        state = self._require_process_state(event.session_key, turn_id)
        state.thinking_block = None
        if event.call_id in state.tool_blocks:
            raise RuntimeError(f"mobile tool call_id 重复开始: {event.call_id}")
        ordinal = state.next_ordinal
        state.next_ordinal += 1
        block_id = f"tool:{event.call_id}"
        state.tool_blocks[event.call_id] = (block_id, ordinal, monotonic())
        await self._runtime.publish_event(
            event_type="react.tool.started",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={
                "call_id": event.call_id,
                "block_id": block_id,
                "ordinal": ordinal,
                "tool_name": event.tool_name,
                "arguments": _mobile_tool_arguments(event.arguments),
            },
        )

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or self._current_turn_id(event.session_key)
        await self._flush_deltas(event.session_key, turn_id)
        state = self._require_process_state(event.session_key, turn_id)
        block = state.tool_blocks.get(event.call_id)
        if block is None:
            raise RuntimeError(f"mobile tool completed 缺少 started: {event.call_id}")
        block_id, ordinal, started_at = block
        await self._runtime.publish_event(
            event_type="react.tool.completed",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={
                "call_id": event.call_id,
                "block_id": block_id,
                "ordinal": ordinal,
                "tool_name": event.tool_name,
                "status": event.status,
                "arguments": _mobile_tool_arguments(
                    event.final_arguments,
                ),
                "result_preview": event.result_preview,
                "duration_ms": max(0, round((monotonic() - started_at) * 1_000)),
            },
        )

    async def _on_response(self, message: OutboundMessage) -> None:
        self._raise_delta_failure()
        session_id = self._session_id(message.chat_id)
        turn_id = message.control_turn_id or self._current_turn_id(session_id)
        await self._flush_deltas(session_id, turn_id)
        message_id = message.session_message_id
        if message.media and message_id is None:
            raise RuntimeError("出站媒体缺少已持久化的 assistant 消息")
        metadata = dict(message.metadata)
        try:
            attachments = await self._outbound_descriptors(
                session_id,
                list(message.media),
                message_id=message_id,
            )
        except (
            AttachmentRequestError,
            AttachmentStateError,
            RemoteMediaError,
            OSError,
        ) as error:
            logger.warning(
                "mobile 远程媒体快照失败，保留最终文字: session=%s error=%s",
                session_id,
                error,
            )
            attachments = []
            metadata["media_delivery"] = {
                "status": "failed",
                "code": "media_unavailable",
                "message": "附件源暂时不可用",
            }
        final_payload: dict[str, object] = {
            "content": message.content,
            "thinking": message.thinking or "",
            "attachments": attachments,
            "metadata": metadata,
        }
        user_message_id = metadata.get("persisted_user_message_id")
        client_message_id = metadata.get("client_message_id")
        if user_message_id is not None or client_message_id is not None:
            if not isinstance(user_message_id, str) or not isinstance(client_message_id, str):
                raise RuntimeError("mobile final 缺少完整的 user/client 消息标识")
            final_payload["user_message_id"] = user_message_id
            final_payload["client_message_id"] = client_message_id
        if message_id is not None:
            final_payload["message_id"] = message_id
        await self._runtime.publish_event(
            event_type="message.final",
            session_id=session_id,
            turn_id=turn_id,
            payload=final_payload,
        )
        _ = self._active_turn_ids.pop(session_id, None)
        _ = self._process_turns.pop((session_id, turn_id), None)

    async def _mobile_history_item(
        self,
        item: Mapping[str, object],
    ) -> dict[str, object]:
        """裁剪内部字段，并把服务端媒体路径转换为稳定描述符。"""

        result = _mobile_history_item(item)
        media = item.get("media")
        if media is None:
            result["attachments"] = []
            return result
        if not isinstance(media, list) or not all(
            isinstance(path, str) for path in cast(list[object], media)
        ):
            raise ValueError(f"历史消息 media 不是字符串数组: {item['id']}")
        try:
            result["attachments"] = await self._outbound_descriptors(
                str(item["session_key"]),
                cast(list[str], media),
                message_id=str(item["id"]),
            )
        except (
            AttachmentRequestError,
            AttachmentStateError,
            RemoteMediaError,
            OSError,
        ) as error:
            logger.warning(
                "mobile 历史媒体恢复失败，保留文字历史: message=%s error=%s",
                item["id"],
                error,
            )
            result["attachments"] = []
            result["attachment_error"] = {
                "code": "media_unavailable",
                "message": "附件源暂时不可用",
            }
        return result

    async def _outbound_descriptors(
        self,
        session_id: str,
        media_paths: list[str],
        *,
        message_id: str | None = None,
    ) -> list[dict[str, object]]:
        """把本地媒体物化为不暴露路径的稳定附件描述符。"""

        if len(media_paths) > 10:
            raise ValueError("单条消息最多包含 10 个出站附件")
        if not media_paths:
            return []
        if message_id is not None:
            bound = await asyncio.to_thread(
                self._require_attachments().read_message_outbound,
                session_id=session_id,
                message_id=message_id,
            )
            if bound:
                if len(bound) != len(media_paths):
                    raise RuntimeError("历史消息附件槽位数量与 Session 不一致")
                return [attachment_descriptor(record) for record in bound]
        # 1. URL 先经过 SSRF 防护下载为受限持久快照
        snapshots: list[RemoteMediaSnapshot] = []
        paths: list[str] = []
        metadata: list[tuple[str, str] | None] = []
        try:
            for media_path in media_paths:
                if media_path.startswith(("http://", "https://")):
                    snapshot = await snapshot_remote_media(
                        media_path,
                        self._require_ctx().attachment_store,
                        max_bytes=(
                            self._runtime.config.max_attachment_mb * 1024 * 1024
                        ),
                    )
                    snapshots.append(snapshot)
                    paths.append(str(snapshot.path))
                    metadata.append((snapshot.filename, snapshot.content_type))
                else:
                    paths.append(media_path)
                    metadata.append(None)

            # 2. 文件复制和摘要移出事件循环，整批在单事务中注册
            records = await asyncio.to_thread(
                self._require_attachments().register_outbound_batch,
                session_id=session_id,
                local_media_paths=tuple(paths),
                metadata_overrides=tuple(metadata),
                message_id=message_id,
            )
            return [attachment_descriptor(record) for record in records]
        finally:
            for snapshot in snapshots:
                snapshot.path.unlink(missing_ok=True)

    async def _buffer_delta(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        delta: str,
        block_id: str | None,
        ordinal: int | None,
    ) -> None:
        """按 50ms 或 4KiB 合并连续 delta，限制 SQLite 写入频率。"""

        key = (session_id, turn_id)
        lock = self._delta_locks.setdefault(key, asyncio.Lock())
        flush_now = False
        async with lock:
            batch = self._delta_batches.get(key)
            if batch is None:
                if len(self._delta_batches) >= _MAX_DELTA_BATCHES:
                    raise RuntimeError("mobile delta batch 已达到 256 个活跃 turn 上限")
                timer = asyncio.create_task(
                    self._flush_after_delay(key),
                    name=f"mobile-delta-flush:{turn_id}",
                )
                timer.add_done_callback(self._on_delta_timer_done)
                batch = _DeltaBatch(segments=[], byte_count=0, timer=timer)
                self._delta_batches[key] = batch
            segment_identity = (event_type, block_id, ordinal)
            if (
                batch.segments
                and (
                    batch.segments[-1][0],
                    batch.segments[-1][2],
                    batch.segments[-1][3],
                )
                == segment_identity
            ):
                previous_type, previous_delta, previous_block, previous_ordinal = (
                    batch.segments[-1]
                )
                batch.segments[-1] = (
                    previous_type,
                    previous_delta + delta,
                    previous_block,
                    previous_ordinal,
                )
            else:
                batch.segments.append((event_type, delta, block_id, ordinal))
            batch.byte_count += len(delta.encode("utf-8"))
            flush_now = batch.byte_count >= _DELTA_FLUSH_BYTES
        if flush_now:
            await self._flush_deltas(session_id, turn_id)

    async def _flush_after_delay(self, key: tuple[str, str]) -> None:
        await asyncio.sleep(_DELTA_FLUSH_SECONDS)
        await self._flush_deltas(*key)

    async def _flush_deltas(self, session_id: str, turn_id: str) -> None:
        """按原始顺序发布一个 turn 当前已聚合的 delta 段。"""

        key = (session_id, turn_id)
        lock = self._delta_locks.setdefault(key, asyncio.Lock())
        async with lock:
            batch = self._delta_batches.pop(key, None)
            if batch is None:
                return
            current = asyncio.current_task()
            if batch.timer is not current:
                _ = batch.timer.cancel()
            for event_type, delta, block_id, ordinal in batch.segments:
                payload: dict[str, object] = {"delta": delta}
                if block_id is not None:
                    if ordinal is None:
                        raise AssertionError("thinking delta block 缺少 ordinal")
                    payload["block_id"] = block_id
                    payload["ordinal"] = ordinal
                await self._runtime.publish_event(
                    event_type=event_type,
                    session_id=session_id,
                    turn_id=turn_id,
                    payload=payload,
                )
        _ = self._delta_locks.pop(key, None)

    def _on_delta_timer_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        self._delta_failure = error
        ctx = self._ctx
        if ctx is not None:
            ctx.log.error(
                "mobile delta flush 失败",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _raise_delta_failure(self) -> None:
        if self._delta_failure is None:
            return
        error = self._delta_failure
        self._delta_failure = None
        raise error

    def _thinking_block(self, session_id: str, turn_id: str) -> tuple[str, int]:
        state = self._require_process_state(session_id, turn_id)
        if state.thinking_block is None:
            ordinal = state.next_ordinal
            state.next_ordinal += 1
            state.thinking_block = (f"thinking:{turn_id}:{ordinal}", ordinal)
        return state.thinking_block

    def _require_process_state(
        self,
        session_id: str,
        turn_id: str,
    ) -> _ProcessTurnState:
        state = self._process_turns.get((session_id, turn_id))
        if state is None:
            raise RuntimeError(f"mobile process turn 未开始: {session_id}/{turn_id}")
        return state

    def _require_mobile_session(self, value: str | None) -> str:
        session_id = self._normalize_session_id(value)
        if not self._require_ctx().session_manager.session_exists(session_id):
            raise MobileCommandError("session_not_found", f"会话不存在: {session_id}")
        return session_id

    def _normalize_session_id(self, value: object) -> str:
        if not isinstance(value, str) or not value.startswith(f"{self.name}:"):
            raise MobileCommandError(
                "invalid_session", "session_id 必须属于 mobile 渠道"
            )
        raw_id = value[len(self.name) + 1 :]
        try:
            parsed = UUID(raw_id)
        except ValueError as error:
            raise MobileCommandError(
                "invalid_session",
                "mobile session_id 必须包含 UUID",
            ) from error
        if raw_id not in {str(parsed), parsed.hex}:
            raise MobileCommandError(
                "invalid_session",
                "mobile session_id 必须使用规范小写 UUID",
            )
        return value

    def _session_id(self, chat_id: str) -> str:
        text = str(chat_id).strip()
        if not text:
            raise ValueError("chat_id 不能为空")
        if text.startswith(f"{self.name}:"):
            return self._normalize_session_id(text)
        return f"{self.name}:{text}"

    def _chat_id(self, session_id: str) -> str:
        return self._normalize_session_id(session_id)[len(self.name) + 1 :]

    def _current_turn_id(self, session_id: str) -> str:
        return self._active_turn_ids.get(session_id, session_id)

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("MobileRealtimeChannel 尚未启动")
        return self._ctx

    def _require_attachments(self) -> AttachmentTransferService:
        if self._attachments is None:
            raise RuntimeError("MobileRealtimeChannel 附件服务尚未启动")
        return self._attachments


def _command_hash(frame: ClientCommand) -> str:
    payload = frame.model_dump(mode="json", exclude_none=True)
    _ = payload.pop("connection_epoch")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reply_from_receipt(receipt: CommandReceipt) -> CommandReply:
    if receipt.status != "completed":
        return CommandReply(
            type=f"{receipt.command_type}.error",
            payload={
                "code": "command_outcome_unknown",
                "message": "该命令上次执行时中断，请使用原命令 ID 核对状态",
            },
        )
    if receipt.reply_type is None or receipt.reply_payload_json is None:
        raise AssertionError("completed 命令收据缺少回复")
    return CommandReply(
        type=receipt.reply_type,
        payload=_decode_reply_payload(receipt.reply_payload_json),
        session_id=receipt.session_id,
        turn_id=receipt.turn_id,
    )


def _decode_reply_payload(raw: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"命令回复包含重复字段: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"命令回复包含非标准常量: {value}")

    decoded = json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise TypeError("命令回复 payload 必须是 JSON object")
    return cast(dict[str, object], decoded)


def _pagination_payload(payload: Mapping[str, object]) -> dict[str, int]:
    _expect_keys(payload, {"page", "page_size"})
    page = payload.get("page", 1)
    page_size = payload.get("page_size", 50)
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise MobileCommandError("invalid_pagination", "page 必须是正整数")
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= 200
    ):
        raise MobileCommandError("invalid_pagination", "page_size 必须在 1..200")
    return {"page": page, "page_size": page_size}


def _expect_keys(payload: Mapping[str, object], allowed: set[str]) -> None:
    unexpected = set(payload) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise MobileCommandError("invalid_payload", f"payload 包含未知字段: {names}")


def _validate_reply_frame_size(frame: ClientCommand, reply: CommandReply) -> None:
    """在持久化前按真实 JSON 编码校验回复可投递。"""

    wire: dict[str, object] = {
        "v": 1,
        "kind": "reply",
        "type": reply.type,
        "id": frame.id,
        "connection_epoch": frame.connection_epoch,
        "payload": reply.payload,
    }
    if reply.session_id is not None:
        wire["session_id"] = reply.session_id
    if reply.turn_id is not None:
        wire["turn_id"] = reply.turn_id
    encoded = json.dumps(
        wire,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_JSON_FRAME_BYTES:
        raise RuntimeError(f"mobile reply 超过 {MAX_JSON_FRAME_BYTES} bytes: {reply.type}")


def _mobile_history_item(item: Mapping[str, object]) -> dict[str, object]:
    """裁剪服务端内部字段，只向手机同步可展示历史。"""

    mobile_extra: dict[str, object] = {}
    for field in ("reasoning_content", "turn_duration_ms", "proactive", "delivery_id"):
        value = item.get(field)
        if isinstance(value, (str, int, float, bool)):
            mobile_extra[field] = value

    result: dict[str, object] = {
        "id": str(item["id"]),
        "session_key": str(item["session_key"]),
        "seq": cast(int, item["seq"]),
        "role": str(item["role"]),
        "content": str(item["content"]),
        "tool_chain": _mobile_tool_chain(item.get("tool_chain")),
        "extra": mobile_extra,
        "ts": str(item["timestamp"]),
    }
    client_message_id = item.get("client_message_id")
    if isinstance(client_message_id, str) and client_message_id:
        result["client_message_id"] = client_message_id
    for field in ("reply_to_message_id", "reply_role", "reply_preview"):
        value = item.get(field)
        if isinstance(value, str) and value:
            result[field] = value
    return result


def _reply_preview(content: str) -> str:
    return " ".join(content.split())[:512] or "[无文字消息]"


def _reply_source_text(target: Mapping[str, object]) -> str:
    content = str(target["content"])
    if content.strip():
        return content
    media = target.get("media")
    if isinstance(media, list) and media:
        return "[附件]"
    return "[无文字消息]"


def _mobile_tool_chain(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    groups: list[dict[str, object]] = []
    for raw_group in cast(list[object], value):
        if not isinstance(raw_group, dict):
            continue
        group_record = cast(dict[str, object], raw_group)
        group: dict[str, object] = {}
        for field in ("reasoning_content", "text"):
            group_text = group_record.get(field)
            if isinstance(group_text, str) and group_text:
                group[field] = group_text
        raw_calls = group_record.get("calls")
        calls: list[dict[str, object]] = []
        if isinstance(raw_calls, list):
            for raw_call in cast(list[object], raw_calls):
                if not isinstance(raw_call, dict):
                    continue
                call_record = cast(dict[str, object], raw_call)
                name = call_record.get("name")
                if not isinstance(name, str) or not name:
                    continue
                arguments = call_record.get("final_arguments", call_record.get("arguments"))
                arguments_record = cast(dict[str, object], arguments) if isinstance(arguments, dict) else None
                call: dict[str, object] = {
                    "call_id": str(call_record.get("call_id") or ""),
                    "name": name,
                    "status": str(call_record.get("status") or "success"),
                }
                if arguments_record is not None:
                    projected_arguments = _mobile_tool_arguments(
                        arguments_record,
                        max_bytes=_MOBILE_HISTORY_TOOL_ARGUMENT_MAX_BYTES,
                    )
                    call["arguments"] = projected_arguments
                    description = projected_arguments.get("description")
                else:
                    description = None
                if isinstance(description, str) and description:
                    call["description"] = description
                result = call_record.get("result")
                if result is not None:
                    call["result_preview"] = str(result)[:2000]
                calls.append(call)
        group["calls"] = calls
        groups.append(group)
    return groups


def _fit_mobile_history_payload(payload: dict[str, object]) -> None:
    """在历史事件接近帧上限时回收工具详情预算。"""

    # 1. 正常页面直接保留全部安全参数
    if _mobile_tool_argument_encoded_size(payload) <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES:
        return

    # 2. 从页面末尾先回收完整参数，再回收参数派生的描述
    items = cast(list[dict[str, object]], payload["items"])
    for item in reversed(items):
        chain = cast(list[dict[str, object]] | None, item["tool_chain"])
        if chain is None:
            continue
        for group in reversed(chain):
            calls = cast(list[dict[str, object]], group["calls"])
            for call in reversed(calls):
                if "arguments" not in call:
                    continue
                del call["arguments"]
                if (
                    _mobile_tool_argument_encoded_size(payload)
                    <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
                ):
                    return
    for item in reversed(items):
        chain = cast(list[dict[str, object]] | None, item["tool_chain"])
        if chain is None:
            continue
        for group in reversed(chain):
            calls = cast(list[dict[str, object]], group["calls"])
            for call in reversed(calls):
                if "description" not in call:
                    continue
                del call["description"]
                if (
                    _mobile_tool_argument_encoded_size(payload)
                    <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
                ):
                    return


def _mobile_tool_arguments(
    arguments: Mapping[str, object],
    *,
    max_bytes: int = _MOBILE_TOOL_ARGUMENT_MAX_BYTES,
) -> dict[str, object]:
    """生成可安全持久化到手机端的有界工具参数投影。"""

    # 1. 先按结构、字段和值建立安全投影
    remaining = [_MOBILE_TOOL_ARGUMENT_MAX_ITEMS]
    projected = _mobile_tool_argument_value(arguments, depth=0, remaining=remaining)
    projected_record = cast(dict[str, object], projected)

    # 2. 再按真实 UTF-8 JSON 字节预算保留前导参数
    bounded: dict[str, object] = {}
    for key, value in projected_record.items():
        candidate = {**bounded, key: value}
        if _mobile_tool_argument_encoded_size(candidate) <= max_bytes:
            bounded[key] = value
            continue
        while bounded and _mobile_tool_argument_encoded_size(
            {**bounded, "…": _MOBILE_TOOL_ARGUMENT_TRUNCATED}
        ) > max_bytes:
            _ = bounded.popitem()
        bounded["…"] = _MOBILE_TOOL_ARGUMENT_TRUNCATED
        break
    return bounded


def _mobile_tool_argument_value(
    value: object,
    *,
    depth: int,
    remaining: list[int],
) -> object:
    """递归脱敏并裁剪单个 JSON 参数值。"""

    # 1. 在协议边界限制递归深度和总节点数
    if depth > _MOBILE_TOOL_ARGUMENT_MAX_DEPTH or remaining[0] <= 0:
        return _MOBILE_TOOL_ARGUMENT_TRUNCATED
    remaining[0] -= 1

    # 2. 统一隐藏字符串中的凭据，再限制可展示文本长度
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _mobile_tool_argument_contains_secret(value):
            return _MOBILE_TOOL_ARGUMENT_REDACTED
        if len(value) <= _MOBILE_TOOL_ARGUMENT_MAX_STRING_CHARS:
            return value
        return value[:_MOBILE_TOOL_ARGUMENT_MAX_STRING_CHARS] + "…"

    # 3. 对容器递归投影，在键名所有权层隐藏凭据
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(mapping.items()):
            if not isinstance(key, str):
                raise TypeError("mobile 工具参数对象键必须是字符串")
            if index >= _MOBILE_TOOL_ARGUMENT_MAX_CONTAINER_ITEMS or remaining[0] <= 0:
                result["…"] = _MOBILE_TOOL_ARGUMENT_TRUNCATED
                break
            if _mobile_tool_argument_is_secret(key):
                result[key] = _MOBILE_TOOL_ARGUMENT_REDACTED
                continue
            result[key] = _mobile_tool_argument_value(
                item,
                depth=depth + 1,
                remaining=remaining,
            )
        return result
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        result_list: list[object] = []
        redact_next = False
        for index, item in enumerate(sequence):
            if index >= _MOBILE_TOOL_ARGUMENT_MAX_CONTAINER_ITEMS or remaining[0] <= 0:
                result_list.append(_MOBILE_TOOL_ARGUMENT_TRUNCATED)
                break
            if redact_next:
                result_list.append(_MOBILE_TOOL_ARGUMENT_REDACTED)
                redact_next = False
                continue
            result_list.append(
                _mobile_tool_argument_value(
                    item,
                    depth=depth + 1,
                    remaining=remaining,
                )
            )
            redact_next = isinstance(item, str) and _mobile_tool_argument_is_secret_flag(item)
        return result_list
    raise TypeError(f"mobile 工具参数包含非 JSON 类型: {type(value).__name__}")


def _mobile_tool_argument_is_secret(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _MOBILE_TOOL_SECRET_KEYS or normalized.endswith(
        (
            "secret",
            "password",
            "passwd",
            "cookie",
            "privatekey",
            "secretaccesskey",
            "token",
            "apikey",
            "credentialfile",
            "credentialsfile",
        )
    )


def _mobile_tool_argument_contains_secret(value: str) -> bool:
    return _MOBILE_TOOL_SECRET_TEXT_PATTERN.search(value) is not None


def _mobile_tool_argument_is_secret_flag(value: str) -> bool:
    normalized = value.strip().lstrip("-").lower()
    return _mobile_tool_argument_is_secret(normalized)


def _mobile_tool_argument_encoded_size(value: Mapping[str, object]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _mobile_ui_catalog_identity(
    catalog: dict[str, object],
) -> str:
    revision = catalog.get("catalog_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision):
        raise RuntimeError("mobile UI catalog_revision 无效")
    return revision


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["CommandReply", "MobileRealtimeChannel"]
