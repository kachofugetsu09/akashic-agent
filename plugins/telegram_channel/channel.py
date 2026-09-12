from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import deque
from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any, cast

from telegram import BotCommand, Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegramify_markdown.converter import convert_with_segments
from telegramify_markdown.entity import split_entities

from agent.plugin_composition import (
    AttachmentKind,
    AttachmentRef,
    ChannelAdapter,
    ChannelCleanupFailure,
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelReady,
    DeliveryStatus,
    InboundIdentity,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    RawInbound,
    StopReceipt,
)
from agent.plugin_composition.channels import (
    ChannelPresentationPorts,
    ChannelRuntimePorts,
    ControlResponseBodies,
)

from .reply_context import build_reply_inbound_text
from .telegram_utils import TelegramOutboundLimiter, strip_chunk

logger = logging.getLogger(__name__)

_CHANNEL = "telegram"
_SEEN_LIMIT = 500
_MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
_MAX_ATTACHMENT_COUNT = 16
_MAX_MESSAGE_UTF16 = 4090


class _MessageDeduper:
    """Keep a bounded provider message window for polling redelivery."""

    def __init__(self, limit: int) -> None:
        self._seen: set[str] = set()
        self._order: deque[str] = deque()
        self._limit = max(1, limit)

    def seen(self, value: str) -> bool:
        if value in self._seen:
            return True
        self._seen.add(value)
        self._order.append(value)
        while len(self._order) > self._limit:
            self._seen.discard(self._order.popleft())
        return False


def build_telegram_channel(context: ChannelFactoryContext) -> ChannelAdapter:
    """Build a side-effect-free Telegram adapter for the exact Core binding."""

    if not isinstance(context, ChannelFactoryContext):
        raise TypeError("Telegram channel factory 只接受 ChannelFactoryContext")
    if context.ingress is None:
        raise RuntimeError("Telegram channel 需要 Core ingress")
    if context.identity is None:
        raise RuntimeError("Telegram channel 需要 Core identity")
    if context.attachment_import is None:
        raise RuntimeError("Telegram channel 需要 Core attachment import")
    return TelegramChannelAdapter(context)


class TelegramChannelAdapter:
    """Own Telegram polling and delivery while Core owns admission and state."""

    name = _CHANNEL
    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    def __init__(self, context: ChannelFactoryContext) -> None:
        self._context = context
        self._binding_token = context.binding_token
        self._ingress = context.ingress
        self._identity = context.identity
        self._provider_factory = context.provider_client_factory
        self._credentials = context.credentials
        self._attachment_import = context.attachment_import
        self._attachment_read = context.attachment_read
        raw_allow_from = context.config.get("allow_from", ())
        if not isinstance(raw_allow_from, (list, tuple)):
            raise TypeError("Telegram allow_from 必须是数组")
        self._allow_from = frozenset(str(item) for item in raw_allow_from)
        raw_timeout = context.config.get("timeout_seconds", 30.0)
        if not isinstance(raw_timeout, (int, float)) or isinstance(raw_timeout, bool):
            raise TypeError("Telegram timeout_seconds 必须是数字")
        self._timeout = float(raw_timeout)
        self._runtime: ChannelRuntimePorts | None = None
        self._presentation: ChannelPresentationPorts | None = None
        self._provider_client: Any | None = None
        self._app: Application[Any, Any, Any, Any, Any, Any] | None = None
        self._inbound_tasks: set[asyncio.Task[Any]] = set()
        self._known_chats: dict[str, str] = {}
        self._deduper = _MessageDeduper(_SEEN_LIMIT)
        self._limiter = TelegramOutboundLimiter()
        self._admission_open = False
        self._started = False
        self._stopping = False
        self._stop_task: asyncio.Task[StopReceipt] | None = None
        self._start_failures: tuple[ChannelCleanupFailure, ...] = ()

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        """Bind the exact generation's ingress before the provider starts."""

        if self._runtime is not None:
            raise RuntimeError("Telegram runtime 不能重复绑定")
        if ports.ingress is None or ports.identity is None:
            raise RuntimeError("Telegram runtime 缺少 ingress 或 identity")
        self._runtime = ports

    def attach_presentation(self, ports: ChannelPresentationPorts) -> None:
        """Bind the exact control facade supplied for the declared capability."""

        if self._presentation is not None:
            raise RuntimeError("Telegram presentation 不能重复绑定")
        if ports.control is None:
            raise RuntimeError("Telegram channel 缺少 control port")
        self._presentation = ports

    def open_admission(self) -> None:
        """Release polling callbacks only after the committed snapshot is live."""

        if self._stopping or self._runtime is None:
            raise RuntimeError("Telegram channel 尚未准备好 admission")
        self._admission_open = True

    def close_admission(self) -> None:
        """Reject new callbacks before Core drains this binding."""

        self._admission_open = False

    async def start(self) -> ChannelReady:
        """Resolve the formal credential and start polling with admission closed."""

        if self._started or self._stopping:
            raise RuntimeError("Telegram channel 已启动或正在停止")
        token_ref = self._credentials.get("token")
        if token_ref is None:
            raise RuntimeError("Telegram channel 缺少 token credential")
        try:
            self._provider_client = await self._provider_factory.create(self._credentials)
            token = self._provider_client.credential(token_ref)
            self._app = (
                Application.builder()
                .token(token)
                .connect_timeout(self._timeout)
                .read_timeout(self._timeout)
                .write_timeout(self._timeout)
                .pool_timeout(self._timeout)
                .build()
            )
            self._app.add_handler(CommandHandler("stop", self._on_update))
            self._app.add_handler(MessageHandler(filters.COMMAND, self._on_update))
            self._app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_update)
            )
            self._app.add_handler(
                MessageHandler(filters.PHOTO & ~filters.COMMAND, self._on_update)
            )
            self._app.add_handler(
                MessageHandler(filters.Document.ALL & ~filters.COMMAND, self._on_update)
            )
            await self._app.initialize()
            await self._app.start()
            updater = self._app.updater
            if updater is None:
                raise RuntimeError("Telegram updater 未初始化")
            await updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                error_callback=self._on_polling_error,
            )
            await self._app.bot.set_my_commands(
                [BotCommand("stop", "中断当前回复")]
            )
            self._started = True
            return ChannelReady(
                binding_token=self._binding_token,
                subscriptions=("telegram.polling",),
                admission_open=False,
            )
        except BaseException as error:
            self._start_failures = (
                self._cleanup_failure("startup", error),
            )
            try:
                await self._close_provider_after_failed_start()
            except BaseException as cleanup_error:
                self._start_failures += (self._cleanup_failure("startup_cleanup", cleanup_error),)
            raise

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        """Verify Core attachments, then send ordered Telegram messages."""

        if request.binding_token != self._binding_token:
            raise RuntimeError("Telegram delivery binding token 不匹配")
        if self._app is None or not self._started:
            raise RuntimeError("Telegram channel 尚未 start")
        try:
            chat_id = int(self._resolve_recipient(request.recipient))
            if chat_id == 0:
                raise ValueError("Telegram chat_id 不能为 0")
            if not request.body.strip() and not request.attachments:
                return ProviderDeliveryReceipt(
                    request.delivery_id,
                    DeliveryStatus.REJECTED,
                    error="消息没有可发送正文或附件",
                )
            attachments = await self._read_attachments(request.attachments)
        except (TypeError, ValueError, RuntimeError) as error:
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.REJECTED,
                error=f"Telegram 发送准备失败: {error}",
            )

        provider_ids: list[str] = []
        try:
            if request.body.strip():
                provider_ids.extend(await self._send_markdown(chat_id, request.body))
            for ref, data in attachments:
                if ref.kind is AttachmentKind.IMAGE:
                    message = await self._limiter.run(
                        chat_id,
                        kind="send",
                        label="send_photo",
                        action=lambda: self._app.bot.send_photo(
                            chat_id=chat_id,
                            photo=data,
                            filename=ref.filename or "image",
                        ),
                    )
                else:
                    message = await self._limiter.run(
                        chat_id,
                        kind="send",
                        label="send_document",
                        action=lambda: self._app.bot.send_document(
                            chat_id=chat_id,
                            document=data,
                            filename=ref.filename or ref.artifact_id,
                        ),
                    )
                message_id = getattr(message, "message_id", None)
                if message_id is not None:
                    provider_ids.append(str(message_id))
        except (TelegramError, OSError, TimeoutError) as error:
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.FAILED,
                tuple(provider_ids),
                error=f"Telegram provider 回执未确认: {type(error).__name__}",
            )
        return ProviderDeliveryReceipt(
            request.delivery_id,
            DeliveryStatus.DELIVERED,
            tuple(provider_ids),
        )

    async def stop(self) -> StopReceipt:
        """关闭共用一次真实清理；取消等待者不能伪造资源已释放。"""
        self._stopping = True
        self._admission_open = False
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop(), name=self.name + "-channel-stop")
        return await asyncio.shield(self._stop_task)

    async def _stop(self) -> StopReceipt:
        self._stopping = True
        self._admission_open = False
        failures = list(self._start_failures)
        tasks = tuple(self._inbound_tasks)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures.extend(
                self._cleanup_failure("inbound", result)
                for result in results
                if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError)
            )
        app = self._app
        if app is not None:
            updater = app.updater
            if updater is not None and updater.running:
                try:
                    await updater.stop()
                except BaseException as error:
                    failures.append(self._cleanup_failure("updater", error))
            if app.running:
                try:
                    await app.stop()
                except BaseException as error:
                    failures.append(self._cleanup_failure("application", error))
            try:
                await app.shutdown()
            except BaseException as error:
                failures.append(self._cleanup_failure("shutdown", error))
            if not any(item.resource in {"updater", "application", "shutdown"} for item in failures):
                self._app = None
        provider = self._provider_client
        if provider is not None:
            try:
                await provider.aclose()
            except BaseException as error:
                failures.append(self._cleanup_failure("provider", error))
            else:
                self._provider_client = None
        self._started = False
        if failures or self._app is not None or self._provider_client is not None:
            self._stop_task = None
            return StopReceipt(self._binding_token, resources_closed=False, failures=tuple(failures))
        return StopReceipt(self._binding_token, resources_closed=True)

    async def _close_provider_after_failed_start(self) -> None:
        app = self._app
        if app is not None:
            app_failures: list[ChannelCleanupFailure] = []
            try:
                updater = app.updater
                if updater is not None and updater.running:
                    await updater.stop()
            except BaseException as error:
                app_failures.append(self._cleanup_failure("updater", error))
            try:
                if app.running:
                    await app.stop()
            except BaseException as error:
                app_failures.append(self._cleanup_failure("application", error))
            try:
                await app.shutdown()
            except BaseException as error:
                app_failures.append(self._cleanup_failure("shutdown", error))
            if not app_failures:
                self._app = None
            self._start_failures += tuple(app_failures)
        provider = self._provider_client
        if provider is not None:
            try:
                await provider.aclose()
            except BaseException as error:
                self._start_failures += (self._cleanup_failure("provider", error),)
            else:
                self._provider_client = None

    def _cleanup_failure(self, resource: str, error: BaseException) -> ChannelCleanupFailure:
        return ChannelCleanupFailure(
            stage="adapter",
            plugin_id="telegram_channel",
            generation_id=self._context.generation_id,
            binding_token=self._binding_token,
            resource=resource,
            error_type=type(error).__name__,
            message=str(error) or type(error).__name__,
            retry_action="再次调用 Telegram channel.stop()",
        )

    async def _on_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if not self._admission_open:
            return
        task = asyncio.current_task()
        if task is not None:
            self._inbound_tasks.add(task)
        try:
            await self._handle_update(update, context)
        finally:
            if task is not None:
                self._inbound_tasks.discard(task)

    async def _handle_update(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if message is None or chat is None or user is None:
            return
        if not self._is_allowed(user):
            logger.warning("[telegram] 拒绝未授权用户 id=%s", user.id)
            return
        message_id = str(getattr(message, "message_id", "") or "").strip()
        if not message_id or self._deduper.seen(f"{chat.id}:{message_id}"):
            return
        content = str(getattr(message, "text", None) or getattr(message, "caption", None) or "")
        attachments: list[AttachmentRef] = []
        if message.photo:
            photo = message.photo[-1]
            attachments.append(
                await self._download_attachment(
                    context,
                    str(photo.file_id),
                    kind=AttachmentKind.IMAGE,
                    filename="image.jpg",
                    media_type="image/jpeg",
                )
            )
            if not content:
                content = "[图片]"
        if message.document is not None:
            document = message.document
            attachments.append(
                await self._download_attachment(
                    context,
                    str(document.file_id),
                    kind=AttachmentKind.FILE,
                    filename=document.file_name or "attachment",
                    media_type=document.mime_type or "application/octet-stream",
                )
            )
            if not content:
                content = f"[文件: {document.file_name or 'attachment'}]"
        reply = getattr(message, "reply_to_message", None)
        metadata: dict[str, object] = {"username": user.username or ""}
        if reply is not None:
            photos = getattr(reply, "photo", ())
            if photos:
                attachments.append(await self._download_attachment(
                    context, str(photos[-1].file_id), kind=AttachmentKind.IMAGE,
                    filename="reply-image.jpg", media_type="image/jpeg",
                ))
            document = getattr(reply, "document", None)
            if document is not None:
                attachments.append(await self._download_attachment(
                    context, str(document.file_id), kind=AttachmentKind.FILE,
                    filename=document.file_name or "attachment",
                    media_type=document.mime_type or "application/octet-stream",
                ))
            reply_text = str(getattr(reply, "text", None) or getattr(reply, "caption", None) or "")
            if reply_text:
                content = build_reply_inbound_text(content, reply_text)
                metadata["reply_to_message_id"] = str(getattr(reply, "message_id", ""))
        if not content and not attachments:
            return
        sender = str(user.id)
        chat_id = str(chat.id)
        self._known_chats[sender] = chat_id
        self._known_chats[str(user.username or "").lower()] = chat_id
        timestamp = getattr(message, "date", None)
        if not isinstance(timestamp, datetime):
            timestamp = datetime.now(timezone.utc)
        elif timestamp.tzinfo is None or timestamp.utcoffset() is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        raw = RawInbound(
            message_id=message_id,
            provider_identity=sender,
            recipient=chat_id,
            message=ChannelInboundMessage(
                channel=_CHANNEL,
                sender=sender,
                chat_id=chat_id,
                content="".join("\u2028" if ord(char) in {10, 13} else " " if ord(char) < 32 else char for char in content),
                timestamp=timestamp,
                metadata=cast(dict[str, Any], metadata),
                attachments=tuple(attachments),
            ),
        )
        command = content.strip().split(maxsplit=1)[0].split("@", 1)[0] if content.strip() else ""
        if command == "/stop" and self._presentation is not None:
            control = self._presentation.control
            if control is None:
                raise RuntimeError("Telegram control port 未绑定")
            await control.interrupt(
                raw,
                response_bodies=ControlResponseBodies(
                    interrupted="已中断当前回复。",
                    idle="当前没有运行中的回复。",
                ),
            )
            return
        ingress = self._ingress
        if ingress is None:
            raise RuntimeError("Telegram ingress 未绑定")
        await ingress.admit(raw)

    async def _download_attachment(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        file_id: str,
        *,
        kind: AttachmentKind,
        filename: str,
        media_type: str,
    ) -> AttachmentRef:
        importer = self._attachment_import
        if importer is None:
            raise RuntimeError("Telegram attachment import 未绑定")
        provider_file = await context.bot.get_file(file_id)
        downloader = getattr(provider_file, "download_as_bytearray", None)
        if not callable(downloader):
            raise RuntimeError("Telegram provider 缺少 download_as_bytearray")
        payload = bytes(await cast(Awaitable[bytes | bytearray], downloader()))
        if len(payload) > _MAX_ATTACHMENT_BYTES:
            raise ValueError("Telegram 附件超过大小上限")
        return await importer.import_bytes(
            payload,
            kind=kind,
            filename=filename,
            media_type=media_type,
        )

    def _is_allowed(self, user: object) -> bool:
        if not self._allow_from:
            return True
        user_id = str(getattr(user, "id", ""))
        username = str(getattr(user, "username", "") or "").lower()
        return user_id in self._allow_from or username in {
            item.lower().lstrip("@") for item in self._allow_from
        }

    def _resolve_recipient(self, value: str) -> str:
        recipient = value.strip().lstrip("@")
        if recipient.lstrip("-").isdigit():
            return recipient
        mapped = self._known_chats.get(recipient.lower())
        if mapped is None and self._identity is not None:
            mapped = self._identity.resolve(recipient)
        if mapped is None or not mapped.lstrip("-").isdigit():
            raise ValueError(f"找不到 Telegram chat_id: {value!r}")
        return mapped

    async def _read_attachments(
        self,
        refs: tuple[AttachmentRef, ...],
    ) -> list[tuple[AttachmentRef, bytes]]:
        if len(refs) > _MAX_ATTACHMENT_COUNT:
            raise ValueError("Telegram 附件数量超过上限")
        reader = self._attachment_read
        if refs and reader is None:
            raise RuntimeError("Telegram outbound 缺少 attachment_read")
        result: list[tuple[AttachmentRef, bytes]] = []
        for ref in refs:
            assert reader is not None
            lease = await reader.acquire(ref)
            try:
                if lease.ref != ref:
                    raise RuntimeError("Telegram attachment lease ref 不匹配")
                data = await lease.read_bytes(max_bytes=max(ref.size_bytes, 1))
                if len(data) != ref.size_bytes:
                    raise ValueError("Telegram attachment size 不匹配")
                if hashlib.sha256(data).hexdigest() != ref.sha256:
                    raise ValueError("Telegram attachment sha256 不匹配")
                result.append((ref, data))
            finally:
                await lease.aclose()
        return result

    async def _send_markdown(self, chat_id: int, text: str) -> list[str]:
        assert self._app is not None
        rendered, entities, _ = convert_with_segments(text)
        chunks = split_entities(rendered, entities, _MAX_MESSAGE_UTF16)
        provider_ids: list[str] = []
        for chunk, chunk_entities in chunks:
            chunk, chunk_entities = strip_chunk(chunk, chunk_entities)
            if not chunk:
                continue
            message = await self._limiter.run(
                chat_id,
                kind="send",
                label="send_message",
                action=lambda chunk=chunk, chunk_entities=chunk_entities: self._app.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    entities=[entity.to_dict() for entity in chunk_entities] or None,
                ),
            )
            message_id = getattr(message, "message_id", None)
            if message_id is not None:
                provider_ids.append(str(message_id))
        return provider_ids

    def _on_polling_error(self, error: TelegramError) -> None:
        logger.warning("[telegram] polling 异常，provider 将继续重试: %s", type(error).__name__)


__all__ = ["TelegramChannelAdapter", "build_telegram_channel"]
