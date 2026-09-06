"""
Telegram Channel

将 Telegram Bot 接入 Core v3 Channel ingress，支持 allowFrom 白名单。
"""

import logging
import asyncio
import html
from io import BytesIO
import json
import time
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.error import Conflict, NetworkError, TelegramError, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bus.event_bus import EventBus
from bus.events import (
    ChannelMessage,
    DeliveryReceipt,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from bus.queue import MessageBus
from agent.looping.interrupt import InterruptController
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelAdapter,
    ChannelCommitRole,
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelRuntimePorts,
    InboundIdentity,
    ProviderDeliveryRequest,
    RawInbound,
    StopReceipt,
    JsonValue,
)
from infra.channels.base import MessageDeduper
from infra.channels.contract import ChannelContext
from infra.channels.delivery import deliver_message_parts
from infra.channels.native_delivery import NativeChannelDeliveryAdapter
from infra.channels.reply_context import build_reply_inbound_text
from infra.channels.telegram_utils import (
    TelegramOutboundLimiter,
    TelegramLiveEditQueue,
    TelegramLiveTextMessage,
    TelegramStreamMessage,
    send_markdown,
    send_stream_markdown,
    send_thinking_block,
)
from session.identities import ChannelIdentities

logger = logging.getLogger(__name__)

_CHANNEL = "telegram"
_SEEN_MSG_MAXSIZE = 500  # 滑动窗口大小，防止内存无限增长
_THINKING_LIVE_TAIL = 1400
_TOOL_LIVE_TAIL = 1000
_REPLY_LIVE_TAIL = 1100
_TOOL_PREVIEW_LIMIT = 80
_LIVE_STREAM_MIN_INTERVAL_S = 2.5
_LIVE_STREAM_MIN_CHARS = 200
# 409 Conflict 观测参数：python-telegram-bot 的 network_retry_loop(max_retries=-1)
# 原生会持续退避重试 TelegramError（含 Conflict，上限 30s），callback 无需干预 polling，
# 这里只做日志节流，避免持续冲突时刷屏。
_CONFLICT_LOG_INTERVAL_SECONDS = 60


def _normalize_v3_content(value: str) -> str:
    """Keep provider text while replacing Core-forbidden control characters."""

    return "".join(
        "\u2028" if ord(char) in {10, 13} else " " if ord(char) < 32 else char
        for char in value
    )


@dataclass
class _ToolLiveLine:
    call_id: str
    tool_name: str
    intent: str
    target: str
    status: str = "running"


class _TelegramInboundRuntime:
    """Gate Telegram callbacks on one exact formal Core binding."""

    def __init__(self) -> None:
        self._ports: ChannelRuntimePorts | None = None
        self._open = False
        self._wake = asyncio.Event()
        self._tasks: set[asyncio.Task[Any]] = set()

    def attach(self, ports: ChannelRuntimePorts) -> None:
        if self._open:
            raise RuntimeError("Telegram v3 ingress 已打开")
        if ports.ingress is None:
            raise RuntimeError("Telegram v3 ingress 缺少 Core ingress")
        self._ports = ports
        self._open = False
        self._wake.clear()

    def open(self) -> None:
        if self._ports is None:
            raise RuntimeError("Telegram v3 ingress 尚未 attach")
        self._open = True
        self._wake.set()

    def close(self) -> None:
        self._open = False
        self._ports = None
        self._wake.set()

    async def run(self, operation: Coroutine[Any, Any, None]) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        try:
            await operation
        finally:
            if task is not None:
                self._tasks.discard(task)

    async def wait_quiescent(self) -> None:
        current = asyncio.current_task()
        tasks = tuple(task for task in self._tasks if task is not current)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_open(self) -> ChannelRuntimePorts:
        ports = self._ports
        if ports is None:
            raise RuntimeError("Telegram v3 ingress 尚未 attach")
        await self._wake.wait()
        if not self._open or self._ports is not ports:
            raise RuntimeError("Telegram v3 ingress admission 已关闭")
        return ports

    def require_open(self, ports: ChannelRuntimePorts) -> None:
        if not self._open or self._ports is not ports:
            raise RuntimeError("Telegram v3 ingress admission 已关闭")

    async def admit(
        self,
        raw: RawInbound,
        *,
        ports: ChannelRuntimePorts | None = None,
    ) -> bool:
        if ports is None:
            ports = await self.wait_open()
        if not self._open or self._ports is not ports or ports.ingress is None:
            return False
        return await ports.ingress.admit(raw)

    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
        ports: ChannelRuntimePorts | None = None,
    ) -> AttachmentRef:
        if ports is None:
            ports = await self.wait_open()
        if not self._open or self._ports is not ports or ports.attachment_import is None:
            raise RuntimeError("Telegram v3 attachment import admission 已关闭")
        return await ports.attachment_import.import_bytes(
            data,
            kind=kind,
            filename=filename,
            media_type=media_type,
        )


class TelegramChannel:

    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    def __init__(
        self,
        token: str,
        bus: MessageBus,
        identities: ChannelIdentities,
        allow_from: list[str] | None = None,
        command_catalog_provider: Callable[
            [], tuple[tuple[str, str], ...]
        ] | None = None,
        event_bus: EventBus | None = None,
        interrupt_controller: InterruptController | None = None,
        channel_name: str = _CHANNEL,
    ) -> None:
        self._bus = bus
        self._interrupt_controller = interrupt_controller
        self._channel = channel_name
        self.name = channel_name
        self._allow_from: set[str] = set(allow_from) if allow_from else set()
        self._message_deduper = MessageDeduper(_SEEN_MSG_MAXSIZE)
        self._identities = identities
        self._app = Application.builder().token(token).build()
        self._command_catalog_provider = command_catalog_provider
        self._app.add_handler(CommandHandler("stop", self._on_stop_command))
        self._app.add_handler(
            MessageHandler(filters.COMMAND, self._on_command)
        )
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO & ~filters.COMMAND, self._on_photo)
        )
        self._app.add_handler(
            MessageHandler(filters.Document.ALL & ~filters.COMMAND, self._on_document)
        )
        self._event_bus = event_bus
        self._events_bound = False
        self._conflict_count = 0
        self._last_conflict_log_at: float | None = None
        self._telegram_outbound_limiter = TelegramOutboundLimiter()
        self._active_streams: dict[str, TelegramStreamMessage] = {}
        self._live_edit_queue = TelegramLiveEditQueue(limiter=self._telegram_outbound_limiter)
        self._live_messages: dict[str, TelegramLiveTextMessage] = {}
        self._reply_buffers: dict[str, str] = {}
        self._thinking_buffers: dict[str, str] = {}
        self._thinking_live_next_at: dict[str, float] = {}
        self._live_last_lengths: dict[str, int] = {}
        self._tool_lines: dict[str, list[_ToolLiveLine]] = {}
        self._live_tasks: set[asyncio.Task[None]] = set()
        self._live_tasks_by_session: dict[str, set[asyncio.Task[None]]] = {}
        self._v3_inbound_runtime = _TelegramInboundRuntime()

    @property
    def bot(self):
        return self._app.bot

    def _start_live_task(
        self,
        session_key: str,
        coro: Coroutine[Any, Any, None],
    ) -> None:
        task = asyncio.create_task(coro)
        self._live_tasks.add(task)
        self._live_tasks_by_session.setdefault(session_key, set()).add(task)

        def _done(done_task: asyncio.Task[None]) -> None:
            self._live_tasks.discard(done_task)
            tasks = self._live_tasks_by_session.get(session_key)
            if tasks is not None:
                tasks.discard(done_task)
                if not tasks:
                    self._live_tasks_by_session.pop(session_key, None)
            try:
                _ = done_task.result()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning("[telegram] live 更新任务失败: %s", e)

        task.add_done_callback(_done)

    async def _cancel_live_tasks(self, session_key: str) -> None:
        tasks = [task for task in self._live_tasks_by_session.get(session_key, set()) if not task.done()]
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self, ctx: ChannelContext | None = None) -> None:
        if ctx is not None:
            self._bus = ctx.bus
            self._event_bus = ctx.event_bus
            self._interrupt_controller = ctx.interrupt_controller
        self._bind_runtime()
        await self._app.initialize()
        await self._app.start()
        await self._register_bot_commands()
        updater = self._app.updater
        if updater is None:
            raise RuntimeError("Telegram updater 未初始化")
        await updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            error_callback=self._on_polling_error,
        )
        logger.info(f"TelegramChannel 已启动  已知用户: {len(self._identities.load(self._channel))}")

    def _bind_runtime(self) -> None:
        if self._event_bus is not None and not self._events_bound:
            self._event_bus.on(TurnStarted, self._on_turn_started)
            self._event_bus.on(StreamDeltaReady, self._on_stream_delta)
            self._event_bus.on(ToolCallStarted, self._on_tool_call_started)
            self._event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
            self._events_bound = True

    async def stop(self) -> None:
        for session_key in tuple(self._live_tasks_by_session):
            await self._cancel_live_tasks(session_key)
        for session_key in tuple(self._live_messages):
            await self._delete_live_message(session_key)
        updater = self._app.updater
        if updater and updater.running:
            await updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("TelegramChannel 已停止")

    # ── 私有方法 ──────────────────────────────────────────────────

    def _is_allowed(self, user) -> bool:
        """检查用户是否在白名单中，白名单为空则允许所有人"""
        if not self._allow_from:
            return True
        return str(user.id) in self._allow_from or (
            user.username
            and user.username.lower() in {u.lower() for u in self._allow_from}
        )

    async def _register_bot_commands(self) -> None:
        catalog = (
            self._command_catalog_provider()
            if self._command_catalog_provider is not None
            else ()
        )
        commands = [
            BotCommand(command, description)
            for command, description in [
                *catalog,
                ("stop", "中断当前回复"),
            ]
        ]
        await self._app.bot.set_my_commands(commands)

    async def replace_command_catalog(
        self,
        commands: tuple[tuple[str, str], ...],
    ) -> None:
        """Publish one committed command catalog to Telegram discovery."""

        published = [
            BotCommand(command, description)
            for command, description in (*commands, ("stop", "中断当前回复"))
        ]
        await self._app.bot.set_my_commands(published)

    async def _on_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.text or not chat or not user:
            return

        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权用户  id={user.id}  username=@{user.username}"
            )
            return

        await self._v3_inbound_runtime.run(self._on_message_v3(update, context))

    def _v3_timestamp(self, message: object) -> datetime:
        value = getattr(message, "date", None)
        if not isinstance(value, datetime):
            return datetime.now(timezone.utc)
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    async def _download_v3_attachment(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        file_id: str,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
        ports: ChannelRuntimePorts,
    ) -> AttachmentRef:
        runtime = self._v3_inbound_runtime
        if runtime is None:
            raise RuntimeError("Telegram v3 ingress 尚未 attach")
        runtime.require_open(ports)
        telegram_file = await context.bot.get_file(file_id)
        download = getattr(telegram_file, "download_as_bytearray", None)
        if not callable(download):
            raise RuntimeError("Telegram provider 缺少 download_as_bytearray")
        payload = bytes(
            await cast(Awaitable[bytes | bytearray], download())
        )
        return await runtime.import_bytes(
            payload,
            kind=kind,
            filename=filename,
            media_type=media_type,
            ports=ports,
        )

    async def _admit_v3_message(
        self,
        message: object,
        *,
        sender: str,
        chat_id: str,
        content: str,
        metadata: dict[str, JsonValue],
        attachments: tuple[AttachmentRef, ...] = (),
        ports: ChannelRuntimePorts,
    ) -> None:
        runtime = self._v3_inbound_runtime
        if runtime is None:
            raise RuntimeError("Telegram v3 ingress 尚未 attach")
        raw = RawInbound(
            message_id=str(getattr(message, "message_id")),
            provider_identity=sender,
            recipient=chat_id,
            message=ChannelInboundMessage(
                channel=self._channel,
                sender=sender,
                chat_id=chat_id,
                content=_normalize_v3_content(content),
                timestamp=self._v3_timestamp(message),
                metadata=metadata,
                attachments=attachments,
            ),
        )
        accepted = await runtime.admit(raw, ports=ports)
        if not accepted:
            logger.debug(
                "[telegram] v3 ingress closed or duplicate message_id=%s",
                raw.message_id,
            )

    async def _on_message_v3(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not msg or not msg.text or not chat or not user:
            return
        ports = await self._v3_inbound_runtime.wait_open()
        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            return
        await self._safe_send_typing(context, chat.id)
        inbound_text, reply_meta = _build_inbound_text_with_reply(
            msg.text, msg.reply_to_message
        )
        attachments = await self._download_v3_reply_attachments(
            msg.reply_to_message,
            context,
            ports=ports,
        )
        await self._admit_v3_message(
            msg,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=inbound_text,
            metadata={"username": user.username or "", **reply_meta},
            attachments=attachments,
            ports=ports,
        )

    async def _on_command_v3(
        self, update: Update, _context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not msg or not chat or not user:
            return
        ports = await self._v3_inbound_runtime.wait_open()
        await self._admit_v3_message(
            msg,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=str(getattr(msg, "text", "") or ""),
            metadata={"username": user.username or ""},
            ports=ports,
        )

    async def _on_photo_v3(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not msg or not msg.photo or not chat or not user:
            return
        ports = await self._v3_inbound_runtime.wait_open()
        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            return
        await self._safe_send_typing(context, chat.id)
        photo = msg.photo[-1]
        main = await self._download_v3_attachment(
            context,
            str(photo.file_id),
            kind=AttachmentKind.IMAGE,
            filename=getattr(photo, "file_name", None),
            media_type="image/jpeg",
            ports=ports,
        )
        inbound_text, reply_meta = _build_inbound_text_with_reply(
            msg.caption or "", msg.reply_to_message
        )
        replies = await self._download_v3_reply_attachments(
            msg.reply_to_message,
            context,
            ports=ports,
        )
        await self._admit_v3_message(
            msg,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=inbound_text,
            metadata={"username": user.username or "", **reply_meta},
            attachments=(main, *replies),
            ports=ports,
        )

    async def _on_document_v3(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if not msg or not msg.document or not chat or not user:
            return
        ports = await self._v3_inbound_runtime.wait_open()
        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            return
        await self._safe_send_typing(context, chat.id)
        document = msg.document
        main = await self._download_v3_attachment(
            context,
            str(document.file_id),
            kind=AttachmentKind.FILE,
            filename=document.file_name or None,
            media_type=document.mime_type or None,
            ports=ports,
        )
        content, reply_meta = _build_inbound_text_with_reply(
            msg.caption or "", msg.reply_to_message
        )
        if document.file_name:
            content = f"[文件: {document.file_name}]\n{content}".strip()
        await self._admit_v3_message(
            msg,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=content,
            metadata={
                "username": user.username or "",
                "document_filename": document.file_name or "",
                "document_mime_type": document.mime_type or "",
                **reply_meta,
            },
            attachments=(main,),
            ports=ports,
        )

    async def _download_v3_reply_attachments(
        self,
        reply: object,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        ports: ChannelRuntimePorts,
    ) -> tuple[AttachmentRef, ...]:
        if reply is None:
            return ()
        result: list[AttachmentRef] = []
        photos = getattr(reply, "photo", None)
        if photos:
            photo = photos[-1]
            result.append(
                await self._download_v3_attachment(
                    context,
                    str(photo.file_id),
                    kind=AttachmentKind.IMAGE,
                    filename=getattr(photo, "file_name", None),
                    media_type="image/jpeg",
                    ports=ports,
                )
            )
        document = getattr(reply, "document", None)
        if document is not None:
            result.append(
                await self._download_v3_attachment(
                    context,
                    str(document.file_id),
                    kind=AttachmentKind.FILE,
                    filename=document.file_name or None,
                    media_type=document.mime_type or None,
                    ports=ports,
                )
            )
        return tuple(result)

    async def _on_stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        # /stop 复用已认证的同一入站路径，由来源追加 Control 并排空任务。
        await self._on_command(update, context)

    async def _on_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权命令  id={user.id}  username=@{user.username}"
            )
            return

        await self._v3_inbound_runtime.run(self._on_command_v3(update, context))

    async def _on_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.photo or not chat or not user:
            return

        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权用户  id={user.id}  username=@{user.username}"
            )
            return

        await self._v3_inbound_runtime.run(self._on_photo_v3(update, context))

    async def _on_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.document or not chat or not user:
            return

        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权用户  id={user.id}  username=@{user.username}"
            )
            return

        await self._v3_inbound_runtime.run(self._on_document_v3(update, context))

    def _resolve_chat_id(self, chat_id: str) -> str:
        resolved = chat_id.strip().lstrip("@").lower()
        if not resolved.lstrip("-").isdigit():
            resolved = self._identities.resolve(self._channel, resolved)
            if not resolved:
                raise ValueError(
                    f"找不到用户 {chat_id!r} 的 chat_id，该用户需先给 bot 发一条消息。"
                    f"已知用户：{list(self._identities.load(self._channel)) or '（无）'}"
                )
        return resolved

    async def send(self, chat_id: str, message: str) -> None:
        """发送文本消息（供 MessagePushTool 调用）"""
        await send_markdown(
            self._app.bot,
            self._resolve_chat_id(chat_id),
            message,
            self._telegram_outbound_limiter,
        )

    async def send_stream(self, chat_id: str, message: str) -> None:
        """发送流式文本消息（私聊优先 draft，其他场景降级普通发送）"""
        await send_stream_markdown(
            self._app.bot,
            self._resolve_chat_id(chat_id),
            message,
            self._telegram_outbound_limiter,
        )

    def create_stream_sender(self, chat_id: str):
        cid = int(self._resolve_chat_id(chat_id))
        if cid <= 0:
            return None
        key = str(cid)
        stream = TelegramStreamMessage(self._app.bot, cid, self._telegram_outbound_limiter)
        self._active_streams[key] = stream

        async def _push(delta: dict[str, str] | str) -> None:
            await stream.push_delta(delta)

        return _push

    async def _on_turn_started(self, event: TurnStarted) -> None:
        if event.channel != self._channel:
            return
        await self._cancel_live_tasks(event.session_key)
        await self._delete_live_message(event.session_key)
        _ = self._tool_lines.pop(event.session_key, None)
        _ = self._reply_buffers.pop(event.session_key, None)
        _ = self._thinking_buffers.pop(event.session_key, None)
        _ = self._thinking_live_next_at.pop(event.session_key, None)
        _ = self._live_last_lengths.pop(event.session_key, None)

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        if event.channel != self._channel:
            return
        if not event.content_delta and not event.thinking_delta:
            return
        cid = int(self._resolve_chat_id(event.chat_id))
        if cid <= 0:
            return
        if event.content_delta:
            reply = self._reply_buffers.get(event.session_key, "")
            self._reply_buffers[event.session_key] = reply + event.content_delta
        if event.thinking_delta:
            thinking = self._thinking_buffers.get(event.session_key, "")
            self._thinking_buffers[event.session_key] = thinking + event.thinking_delta
        live_len = _live_buffer_len(
            self._reply_buffers.get(event.session_key, ""),
            self._thinking_buffers.get(event.session_key, ""),
        )
        last_len = self._live_last_lengths.get(event.session_key, 0)
        now = asyncio.get_running_loop().time()
        next_at = self._thinking_live_next_at.get(event.session_key, 0.0)
        if now < next_at and live_len - last_len < _LIVE_STREAM_MIN_CHARS:
            return
        self._thinking_live_next_at[event.session_key] = now + _LIVE_STREAM_MIN_INTERVAL_S
        self._live_last_lengths[event.session_key] = live_len
        self._start_live_task(
            event.session_key,
            self._sync_live_message(event.session_key, cid),
        )

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        if event.channel != self._channel:
            return
        cid = int(self._resolve_chat_id(event.chat_id))
        if cid <= 0:
            return
        lines = self._tool_lines.setdefault(event.session_key, [])
        lines.append(
            _ToolLiveLine(
                call_id=event.call_id,
                tool_name=event.tool_name,
                intent=_format_tool_intent(event.arguments),
                target=_format_tool_target(event.arguments),
            )
        )
        self._start_live_task(
            event.session_key,
            self._sync_live_message(event.session_key, cid),
        )

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        if event.channel != self._channel:
            return
        cid = int(self._resolve_chat_id(event.chat_id))
        if cid <= 0:
            return
        lines = self._tool_lines.setdefault(event.session_key, [])
        line = next((item for item in lines if item.call_id == event.call_id), None)
        if line is None:
            line = _ToolLiveLine(
                call_id=event.call_id,
                tool_name=event.tool_name,
                intent=_format_tool_intent(event.final_arguments or event.arguments),
                target=_format_tool_target(event.final_arguments or event.arguments),
            )
            lines.append(line)
        line.status = "error" if event.status == "error" else "done"
        self._start_live_task(
            event.session_key,
            self._sync_live_message(event.session_key, cid),
        )

    async def _sync_live_message(
        self,
        session_key: str,
        chat_id: int,
        *,
        terminal: bool = False,
    ) -> None:
        text, html_text = _format_turn_live(
            self._tool_lines.get(session_key, []),
            self._reply_buffers.get(session_key, ""),
            self._thinking_buffers.get(session_key, ""),
            terminal=terminal,
        )
        if not text:
            return
        message = self._live_messages.get(session_key)
        if message is None:
            message = TelegramLiveTextMessage(
                self._app.bot,
                self._live_edit_queue,
                chat_id,
            )
            self._live_messages[session_key] = message
        await message.update(text, html_text=html_text, force=terminal)

    def _has_live_messages(self, session_key: str) -> bool:
        return session_key in self._live_messages

    async def _delete_live_message(self, session_key: str) -> None:
        message = self._live_messages.get(session_key)
        if message is not None and await message.delete():
            self._live_messages.pop(session_key, None)

    def _final_thinking_text(
        self,
        session_key: str,
        thinking: str | None,
    ) -> str:
        streamed = self._thinking_buffers.get(session_key, "").strip()
        final = (thinking or "").strip()
        if streamed and final:
            if final in streamed:
                return streamed
            if streamed in final:
                return final
            return f"{streamed}\n\n{final}"
        return streamed or final

    async def _send_final_tool_snapshot(
        self,
        session_key: str,
        chat_id: str,
    ) -> None:
        lines = self._tool_lines.get(session_key, [])
        if not lines:
            return
        tool_text = _tail_text(_format_tool_live(lines), _TOOL_LIVE_TAIL)
        if tool_text:
            await send_markdown(
                self._app.bot,
                chat_id,
                f"```\n{tool_text}\n```",
                self._telegram_outbound_limiter,
            )

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        name: str | None = None,
        caption: str | None = None,
    ) -> None:
        """发送文件，可附带说明文字"""
        cid = int(self._resolve_chat_id(chat_id))
        await self._telegram_outbound_limiter.run(
            cid,
            kind="send",
            label="send_document",
            action=lambda: self._send_document_file(cid, file_path, name, caption),
        )

    async def send_image(self, chat_id: str, image: str) -> None:
        """发送图片（本地路径或 URL）"""
        cid = int(self._resolve_chat_id(chat_id))
        if image.startswith(("http://", "https://")):
            await self._telegram_outbound_limiter.run(
                cid,
                kind="send",
                label="send_photo",
                action=lambda: self._app.bot.send_photo(chat_id=cid, photo=image),
            )
        else:
            await self._telegram_outbound_limiter.run(
                cid,
                kind="send",
                label="send_photo",
                action=lambda: self._send_photo_file(cid, image),
            )

    async def _deliver_message(self, message: ChannelMessage) -> DeliveryReceipt:
        """以 Telegram 原生调用提交完整消息并报告部分送达。"""

        async def send_text(chat_id: str, content: str) -> None:
            if bool(message.metadata.get("streamed_reply")):
                stream = self._active_streams.pop(str(chat_id), None)
                if stream is not None:
                    await stream.finalize(content)
                    return
            if message.metadata.get("_channel_commit_role") == "passive":
                await self.send(chat_id, content)
            else:
                await self.send_stream(chat_id, content)

        return await deliver_message_parts(
            message,
            send_text=send_text,
            send_file=self.send_file,
            send_image=self.send_image,
        )

    def build_v3_adapter(self, context: ChannelFactoryContext) -> ChannelAdapter:
        """Build a Core adapter over this already-started Telegram provider owner."""

        return TelegramV3ChannelAdapter(self, context)

    def _attach_v3_inbound(self, ports: ChannelRuntimePorts) -> None:
        """Attach one formal Core ingress without retaining a candidate context."""

        runtime = self._v3_inbound_runtime
        if runtime is None:
            runtime = _TelegramInboundRuntime()
            self._v3_inbound_runtime = runtime
        runtime.attach(ports)

    def _open_v3_inbound(self) -> None:
        runtime = self._v3_inbound_runtime
        if runtime is None:
            raise RuntimeError("Telegram v3 ingress 尚未 attach")
        runtime.open()

    def _close_v3_inbound(self) -> None:
        runtime = self._v3_inbound_runtime
        if runtime is not None:
            runtime.close()

    async def _drain_v3_inbound(self) -> None:
        runtime = self._v3_inbound_runtime
        if runtime is not None:
            await runtime.wait_quiescent()

    async def _send_document_file(
        self,
        chat_id: int,
        file_path: str,
        name: str | None,
        caption: str | None,
    ) -> object:
        with open(file_path, "rb") as f:
            return await self._app.bot.send_document(
                chat_id=chat_id, document=f, filename=name, caption=caption
            )

    async def _send_photo_file(self, chat_id: int, image: str) -> object:
        with open(image, "rb") as f:
            return await self._app.bot.send_photo(chat_id=chat_id, photo=f)

    async def _safe_send_typing(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: int
    ) -> None:
        """发送 typing 状态；失败时指数退避重试，不影响消息主流程。"""
        try:
            await self._telegram_outbound_limiter.run(
                chat_id,
                kind="typing",
                label="send_chat_action",
                action=lambda: context.bot.send_chat_action(
                    chat_id=chat_id, action=ChatAction.TYPING
                ),
            )
        except Exception as e:
            logger.warning(
                "[telegram] send_chat_action 失败，已跳过 typing chat_id=%s err=%s",
                chat_id,
                e,
            )

    def _on_polling_error(self, exc: TelegramError) -> None:
        """处理 Telegram polling 异常；409 Conflict 由 PTB 原生 network retry loop
        （max_retries=-1）持续退避重试（上限 30s），这里只做节流日志与状态观测，
        绝不干预 polling 生命周期。"""
        if isinstance(exc, Conflict):
            self._conflict_count += 1
            now = time.monotonic()
            if (
                self._last_conflict_log_at is None
                or now - self._last_conflict_log_at >= _CONFLICT_LOG_INTERVAL_SECONDS
            ):
                self._last_conflict_log_at = now
                logger.warning(
                    "[telegram] getUpdates 409 Conflict（累计 %d 次），"
                    "python-telegram-bot 将自动退避重试；若持续冲突，"
                    "请检查是否同一 bot token 运行了多个轮询实例。",
                    self._conflict_count,
                )
            return
        logger.warning("[telegram] polling 异常，框架将自动重试: %s", exc)


class TelegramV3ChannelAdapter(NativeChannelDeliveryAdapter):
    """Deliver Core requests through an already-started TelegramChannel."""

    def __init__(self, channel: TelegramChannel, context: ChannelFactoryContext) -> None:
        self._channel = channel
        super().__init__(
            context,
            channel_name=channel.name,
            validate_recipient=channel._resolve_chat_id,
            send_text=self._send_text,
            send_attachment=self._send_attachment,
        )

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        """Bind provider callbacks to one formal Core runtime."""

        self._channel._attach_v3_inbound(ports)

    def open_admission(self) -> None:
        """Release provider callbacks only after stable publication finalized."""

        self._channel._open_v3_inbound()

    def close_admission(self) -> None:
        """Stop new provider callbacks before Host drain begins."""

        self._channel._close_v3_inbound()

    async def stop(self) -> StopReceipt:
        self._channel._close_v3_inbound()
        await self._channel._drain_v3_inbound()
        return await super().stop()

    async def _send_text(self, request: ProviderDeliveryRequest) -> None:
        if bool(request.metadata.get("streamed_reply")):
            chat_id = self._channel._resolve_chat_id(request.recipient)
            stream = self._channel._active_streams.pop(str(chat_id), None)
            if stream is not None:
                await stream.finalize(request.body)
                return
        if request.commit_role is ChannelCommitRole.PASSIVE:
            await self._channel.send(request.recipient, request.body)
        else:
            await self._channel.send_stream(request.recipient, request.body)

    async def _send_attachment(
        self,
        request: ProviderDeliveryRequest,
        ref: AttachmentRef,
        payload: bytes,
    ) -> None:
        chat_id = int(self._channel._resolve_chat_id(request.recipient))
        if ref.kind.value == "image":
            await self._channel._telegram_outbound_limiter.run(
                chat_id,
                kind="send",
                label="send_photo(v3)",
                action=lambda: self._send_photo_bytes(chat_id, ref, payload),
            )
            return
        await self._channel._telegram_outbound_limiter.run(
            chat_id,
            kind="send",
            label="send_document(v3)",
            action=lambda: self._send_document_bytes(chat_id, ref, payload),
        )

    async def _send_document_bytes(
        self,
        chat_id: int,
        ref: AttachmentRef,
        payload: bytes,
    ) -> object:
        document = BytesIO(payload)
        document.name = ref.filename or ref.artifact_id
        return await self._channel._app.bot.send_document(
            chat_id=chat_id,
            document=document,
            filename=ref.filename,
        )

    async def _send_photo_bytes(
        self,
        chat_id: int,
        ref: AttachmentRef,
        payload: bytes,
    ) -> object:
        photo = BytesIO(payload)
        photo.name = ref.filename or ref.artifact_id
        return await self._channel._app.bot.send_photo(chat_id=chat_id, photo=photo)


def _format_turn_live(
    lines: list[_ToolLiveLine],
    reply: str,
    thinking: str,
    *,
    terminal: bool = False,
) -> tuple[str, str]:
    blocks: list[str] = []
    html_blocks: list[str] = []
    thinking_body = _tail_text(thinking.strip(), _THINKING_LIVE_TAIL)
    if thinking_body:
        thinking_text = f"思考过程\n{thinking_body}"
        blocks.append(thinking_text)
        html_blocks.append(f"<blockquote>{html.escape(thinking_text)}</blockquote>")
    if lines:
        tool_text = _tail_text(_format_tool_live(lines), _TOOL_LIVE_TAIL)
        blocks.append(tool_text)
        html_blocks.append(f"<pre>{html.escape(tool_text)}</pre>")
    reply_body = _tail_text(reply.strip(), _REPLY_LIVE_TAIL)
    if reply_body and not terminal:
        reply_text = f"临时回复\n{reply_body}"
        blocks.append(reply_text)
        html_blocks.append(f"<b>临时回复</b>\n{html.escape(reply_body)}")
    if terminal and not blocks:
        return "本轮预览完成", "<pre>本轮预览完成</pre>"
    return "\n\n".join(blocks), "\n\n".join(html_blocks)


def _format_tool_live(lines: list[_ToolLiveLine]) -> str:
    shown = lines[-12:]
    rows = ["工具调用"]
    hidden = len(lines) - len(shown)
    if hidden > 0:
        rows.append(f"... {hidden} more")
    for line in shown:
        status = "..."
        if line.status == "done":
            status = "✅"
        elif line.status == "error":
            status = "✗"
        target = f" {line.target}" if line.target else ""
        rows.append(
            f"{_tool_emoji(line.tool_name)} {_clip_inline(line.tool_name, 32)}: "
            f"{line.intent}{target} {status}"
        )
    if lines and all(line.status != "running" for line in lines):
        rows.append(f"Done · {len(lines)} tools")
    return "\n".join(rows)


def _format_tool_intent(arguments: dict[str, object]) -> str:
    value = arguments.get("description")
    if value is None or value == "":
        return ""
    return _clip_inline(_stringify_tool_value(value), _TOOL_PREVIEW_LIMIT)


def _format_tool_target(arguments: dict[str, object]) -> str:
    if not arguments:
        return ""
    primary_keys = (
        "cmd",
        "command",
        "query",
        "url",
        "path",
        "file",
        "text",
        "content",
        "prompt",
        "name",
    )
    for key in primary_keys:
        value = arguments.get(key)
        if value is not None and value != "":
            return f"\"{_clip_inline(_stringify_tool_value(value), _TOOL_PREVIEW_LIMIT)}\""
    return ""


def _stringify_tool_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _clip_inline(text: str, limit: int) -> str:
    plain = " ".join(str(text).split())
    if len(plain) <= limit:
        return plain
    if limit <= 3:
        return plain[:limit]
    return plain[: limit - 3] + "..."


def _tail_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return "..." + text[-(limit - 3):]


def _live_buffer_len(reply: str, thinking: str) -> int:
    return len(reply) + len(thinking)


def _tool_emoji(tool_name: str) -> str:
    name = tool_name.lower()
    if name.startswith("mcp"):
        return "📡"
    if "search" in name:
        return "🔍"
    if "web" in name or "url" in name:
        return "🌐"
    if "file" in name or "read" in name:
        return "📄"
    if "write" in name or "save" in name:
        return "💾"
    if "shell" in name or "exec" in name:
        return "⚙"
    return "🔧"


def _build_inbound_text_with_reply(
    user_text: str,
    reply_msg,
) -> tuple[str, dict[str, str | int]]:
    """将 Telegram 的 reply 上下文合并进入站文本，避免 agent 丢失引用信息。"""
    text = (user_text or "").strip()
    if not reply_msg:
        return text, {}

    reply_text = (reply_msg.text or reply_msg.caption or "").strip()
    if not reply_text:
        # 被回复消息无文字：若含图片则用占位符，否则只保留元信息
        if getattr(reply_msg, "photo", None):
            reply_text = "[图片]"
        else:
            return text, {"reply_to_message_id": int(reply_msg.message_id)}

    reply_sender = ""
    from_user = getattr(reply_msg, "from_user", None)
    if from_user:
        reply_sender = from_user.username or str(from_user.id)
    sender_label = f"@{reply_sender}" if reply_sender else "未知发送者"

    merged = build_reply_inbound_text(
        text,
        reply_text,
        sender_label=sender_label,
    )
    return merged, {
        "reply_to_message_id": int(reply_msg.message_id),
        "reply_to_sender": sender_label,
    }
