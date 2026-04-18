"""
Telegram Channel

将 Telegram Bot 接入 MessageBus，支持 allowFrom 白名单。
"""

import logging
import asyncio

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import Conflict, NetworkError, TelegramError, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from agent.looping.interrupt import InterruptController
from infra.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from infra.channels.telegram_utils import (
    TelegramStreamMessage,
    send_markdown,
    send_stream_markdown,
    send_thinking_block,
)
from session.manager import SessionManager

logger = logging.getLogger(__name__)

_CHANNEL = "telegram"
_SEEN_MSG_MAXSIZE = 500  # 滑动窗口大小，防止内存无限增长


class TelegramChannel:

    def __init__(
        self,
        token: str,
        bus: MessageBus,
        session_manager: SessionManager,
        allow_from: list[str] | None = None,
        interrupt_controller: InterruptController | None = None,
    ) -> None:
        self._bus = bus
        self._session_manager = session_manager
        self._interrupt_controller = interrupt_controller
        self._allow_from: set[str] = set(allow_from) if allow_from else set()
        self._message_deduper = MessageDeduper(_SEEN_MSG_MAXSIZE)
        self._attachments = AttachmentStore()
        self._identity_index = SessionIdentityIndex(
            session_manager,
            channel=_CHANNEL,
            metadata_key="username",
            normalizer=lambda value: value.lower(),
        )
        self._app = Application.builder().token(token).build()
        self._app.add_handler(CommandHandler("stop", self._on_stop_command))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO & ~filters.COMMAND, self._on_photo)
        )
        self._app.add_handler(
            MessageHandler(filters.Document.ALL & ~filters.COMMAND, self._on_document)
        )
        bus.subscribe_outbound(_CHANNEL, self._on_response)
        self.user_map = self._identity_index.mapping
        self._polling_conflict_task: asyncio.Task[None] | None = None
        self._active_streams: dict[str, TelegramStreamMessage] = {}

    @property
    def bot(self):
        return self._app.bot

    async def start(self) -> None:
        self._rebuild_user_map()
        await self._app.initialize()
        await self._app.start()
        updater = self._app.updater
        if updater is None:
            raise RuntimeError("Telegram updater 未初始化")
        await updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            error_callback=self._on_polling_error,
        )
        logger.info(f"TelegramChannel 已启动  已知用户: {len(self.user_map)}")

    async def stop(self) -> None:
        if self._polling_conflict_task and not self._polling_conflict_task.done():
            await self._polling_conflict_task
        updater = self._app.updater
        if updater and updater.running:
            await updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("TelegramChannel 已停止")

    # ── 私有方法 ──────────────────────────────────────────────────

    def _rebuild_user_map(self) -> None:
        """扫描已有 session 文件，从 metadata 重建 username → chat_id 索引。"""
        self._identity_index.rebuild()
        logger.debug(f"[telegram] user_map 重建完成: {self.user_map}")

    def _is_allowed(self, user) -> bool:
        """检查用户是否在白名单中，白名单为空则允许所有人"""
        if not self._allow_from:
            return True
        return str(user.id) in self._allow_from or (
            user.username
            and user.username.lower() in {u.lower() for u in self._allow_from}
        )

    async def _remember_username(self, chat_id: str, username: str | None) -> None:
        if username:
            await self._identity_index.remember(username, chat_id)

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

        # 去重：同一 (chat_id, message_id) 只处理一次，防止 Telegram 重投
        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            logger.warning(
                f"[telegram] 重复消息已忽略  chat_id={chat.id}  message_id={msg.message_id}"
            )
            return

        preview = msg.text[:60] + "..." if len(msg.text) > 60 else msg.text
        logger.info(
            f"[telegram] 收到消息  chat_id={chat.id}  "
            f"user=@{user.username or user.id}  内容: {preview!r}"
        )

        # 更新内存索引 + 持久化到 session.metadata
        chat_id_str = str(chat.id)
        await self._remember_username(chat_id_str, user.username)

        await self._safe_send_typing(context, chat.id)

        inbound_text, reply_meta = _build_inbound_text_with_reply(
            msg.text, msg.reply_to_message
        )
        reply_media: list[str] = []
        if msg.reply_to_message and getattr(msg.reply_to_message, "photo", None):
            try:
                tg_file = await context.bot.get_file(
                    msg.reply_to_message.photo[-1].file_id
                )
                tmp = self._attachments.create_path("reply_photo_", ".jpg")
                await tg_file.download_to_drive(tmp)
                reply_media.append(str(tmp))
                logger.info(f"[telegram] 下载被回复图片  chat_id={chat.id}  tmp={tmp}")
            except Exception as e:
                logger.warning(
                    f"[telegram] 被回复图片下载失败  chat_id={chat.id}  err={e}"
                )
        if msg.reply_to_message and getattr(msg.reply_to_message, "document", None):
            try:
                rdoc = msg.reply_to_message.document
                if rdoc is None:
                    raise ValueError("reply document 缺失")
                suffix = ""
                if rdoc.file_name and "." in rdoc.file_name:
                    suffix = "." + rdoc.file_name.rsplit(".", 1)[-1]
                tg_file = await context.bot.get_file(rdoc.file_id)
                tmp = self._attachments.create_path("reply_doc_", suffix)
                await tg_file.download_to_drive(tmp)
                reply_media.append(str(tmp))
                logger.info(
                    f"[telegram] 下载被回复文件  chat_id={chat.id}  filename={rdoc.file_name!r}  tmp={tmp}"
                )
            except Exception as e:
                logger.warning(
                    f"[telegram] 被回复文件下载失败  chat_id={chat.id}  err={e}"
                )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=str(chat.id),
                content=inbound_text,
                media=reply_media,
                metadata={
                    "username": user.username or "",
                    **reply_meta,
                },
            )
        )

    async def _on_stop_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not chat or not user:
            return
        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权 /stop  id={user.id}  username=@{user.username}"
            )
            return
        if self._interrupt_controller is None:
            await send_markdown(self._app.bot, str(chat.id), "当前未启用中断功能。")
            return

        session_key = f"{_CHANNEL}:{chat.id}"
        result = self._interrupt_controller.request_interrupt(
            session_key=session_key,
            sender=str(user.id),
            command="/stop",
        )
        await send_markdown(self._app.bot, str(chat.id), result.message)

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

        msg_key = f"{chat.id}:{msg.message_id}"
        if self._message_deduper.seen(msg_key):
            logger.warning(
                f"[telegram] 重复图片消息已忽略  chat_id={chat.id}  message_id={msg.message_id}"
            )
            return

        chat_id_str = str(chat.id)
        await self._remember_username(chat_id_str, user.username)

        await self._safe_send_typing(context, chat.id)

        # 下载最高分辨率的图片到持久化目录
        tg_file = await context.bot.get_file(msg.photo[-1].file_id)
        tmp = self._attachments.create_path("photo_", ".jpg")
        await tg_file.download_to_drive(tmp)
        logger.info(
            f"[telegram] 收到图片  chat_id={chat.id}  user=@{user.username or user.id}  path={tmp}"
        )

        caption_text = msg.caption or ""
        inbound_text, reply_meta = _build_inbound_text_with_reply(
            caption_text, msg.reply_to_message
        )
        media = [str(tmp)]
        if msg.reply_to_message and getattr(msg.reply_to_message, "photo", None):
            try:
                reply_file = await context.bot.get_file(
                    msg.reply_to_message.photo[-1].file_id
                )
                reply_tmp = self._attachments.create_path("reply_photo_", ".jpg")
                await reply_file.download_to_drive(reply_tmp)
                media.append(str(reply_tmp))
                logger.info(
                    f"[telegram] 下载被回复图片  chat_id={chat.id}  tmp={reply_tmp}"
                )
            except Exception as e:
                logger.warning(
                    f"[telegram] 被回复图片下载失败  chat_id={chat.id}  err={e}"
                )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=str(chat.id),
                content=inbound_text,
                media=media,
                metadata={
                    "username": user.username or "",
                    **reply_meta,
                },
            )
        )

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

        chat_id_str = str(chat.id)
        await self._remember_username(chat_id_str, user.username)

        await self._safe_send_typing(context, chat.id)

        doc = msg.document
        suffix = ""
        if doc.file_name and "." in doc.file_name:
            suffix = "." + doc.file_name.rsplit(".", 1)[-1]
        tg_file = await context.bot.get_file(doc.file_id)
        tmp = self._attachments.create_path("doc_", suffix)
        await tg_file.download_to_drive(tmp)
        logger.info(
            f"[telegram] 收到文件  chat_id={chat.id}  user=@{user.username or user.id}"
            f"  filename={doc.file_name!r}  tmp={tmp}"
        )

        caption_text = msg.caption or ""
        inbound_text, reply_meta = _build_inbound_text_with_reply(
            caption_text, msg.reply_to_message
        )
        if doc.file_name:
            inbound_text = f"[文件: {doc.file_name}]\n{inbound_text}".strip()
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=str(user.id),
                chat_id=str(chat.id),
                content=inbound_text,
                media=[str(tmp)],
                metadata={
                    "username": user.username or "",
                    "document_filename": doc.file_name or "",
                    "document_mime_type": doc.mime_type or "",
                    **reply_meta,
                },
            )
        )

    def _resolve_chat_id(self, chat_id: str) -> str:
        resolved = chat_id.lstrip("@").lower()
        if not resolved.lstrip("-").isdigit():
            resolved = self._identity_index.resolve(resolved)
            if not resolved:
                raise ValueError(
                    f"找不到用户 {chat_id!r} 的 chat_id，该用户需先给 bot 发一条消息。"
                    f"已知用户：{list(self.user_map.keys()) or '（无）'}"
                )
        return resolved

    async def send(self, chat_id: str, message: str) -> None:
        """发送文本消息（供 MessagePushTool 调用）"""
        await send_markdown(self._app.bot, self._resolve_chat_id(chat_id), message)

    async def send_stream(self, chat_id: str, message: str) -> None:
        """发送流式文本消息（私聊优先 draft，其他场景降级普通发送）"""
        await send_stream_markdown(
            self._app.bot,
            self._resolve_chat_id(chat_id),
            message,
        )

    def create_stream_sender(self, chat_id: str):
        cid = int(self._resolve_chat_id(chat_id))
        if cid <= 0:
            return None
        key = str(cid)
        stream = TelegramStreamMessage(self._app.bot, cid)
        self._active_streams[key] = stream

        async def _push(delta: dict[str, str] | str) -> None:
            await stream.push_delta(delta)

        return _push

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        name: str | None = None,
        caption: str | None = None,
    ) -> None:
        """发送文件，可附带说明文字"""
        cid = int(self._resolve_chat_id(chat_id))
        with open(file_path, "rb") as f:
            await self._app.bot.send_document(
                chat_id=cid, document=f, filename=name, caption=caption
            )

    async def send_image(self, chat_id: str, image: str) -> None:
        """发送图片（本地路径或 URL）"""
        cid = int(self._resolve_chat_id(chat_id))
        if image.startswith(("http://", "https://")):
            await self._app.bot.send_photo(chat_id=cid, photo=image)
        else:
            with open(image, "rb") as f:
                await self._app.bot.send_photo(chat_id=cid, photo=f)

    async def _on_response(self, msg: OutboundMessage) -> None:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"[telegram] 发送回复  chat_id={msg.chat_id}  内容: {preview!r}")
        streamed_reply = bool((msg.metadata or {}).get("streamed_reply"))
        if msg.content.strip():
            if streamed_reply:
                stream = self._active_streams.pop(str(msg.chat_id), None)
                if stream is not None:
                    await stream.finalize(msg.content)
                else:
                    await send_markdown(self._app.bot, msg.chat_id, msg.content)
            else:
                await send_stream_markdown(self._app.bot, msg.chat_id, msg.content)
        if msg.thinking:
            await send_thinking_block(self._app.bot, msg.chat_id, msg.thinking)
        for image in (msg.media or []):
            try:
                await self.send_image(str(msg.chat_id), image)
            except Exception as e:
                logger.warning(f"[telegram] meme 图片发送失败  chat_id={msg.chat_id}  path={image}  err={e}")

    async def _safe_send_typing(
        self, context: ContextTypes.DEFAULT_TYPE, chat_id: int
    ) -> None:
        """发送 typing 状态；失败时指数退避重试，不影响消息主流程。"""
        base_delay = 0.4
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await context.bot.send_chat_action(
                    chat_id=chat_id, action=ChatAction.TYPING
                )
                return
            except (TimedOut, NetworkError) as e:
                if attempt >= max_attempts:
                    logger.warning(
                        "[telegram] send_chat_action 重试耗尽，跳过 typing chat_id=%s attempts=%d err=%s",
                        chat_id,
                        attempt,
                        e,
                    )
                    return
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "[telegram] send_chat_action 失败，准备重试 chat_id=%s attempt=%d/%d backoff=%.1fs err=%s",
                    chat_id,
                    attempt,
                    max_attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
            except Exception as e:
                logger.warning(
                    "[telegram] send_chat_action 失败，已跳过 typing chat_id=%s err=%s",
                    chat_id,
                    e,
                )
                return

    def _on_polling_error(self, exc: TelegramError) -> None:
        """处理 Telegram polling 异常，避免 Conflict 场景下持续刷屏。"""
        if isinstance(exc, Conflict):
            if self._polling_conflict_task is None:
                logger.error(
                    "[telegram] 检测到 getUpdates 冲突，已暂停 Telegram 接收。"
                    "请确保同一 bot token 仅运行一个轮询实例。"
                )
                self._polling_conflict_task = asyncio.create_task(
                    self._disable_polling_on_conflict()
                )
            return
        logger.warning("[telegram] polling 异常，框架将自动重试: %s", exc)

    async def _disable_polling_on_conflict(self) -> None:
        """Conflict 时关闭 updater 轮询，保留 bot 发送能力。"""
        updater = self._app.updater
        if updater is None or not updater.running:
            return
        try:
            await updater.stop()
            logger.warning(
                "[telegram] polling 已停止；当前进程不再接收 Telegram 消息。"
            )
        except Exception as e:
            logger.warning("[telegram] 停止 polling 失败: %s", e)


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

    merged = (
        "【你正在回复一条历史消息】\n"
        f"被回复消息（来自 {sender_label}）：\n"
        f"{reply_text}\n\n"
        "【你当前新消息】\n"
        f"{text}"
    ).strip()
    return merged, {
        "reply_to_message_id": int(reply_msg.message_id),
        "reply_to_sender": sender_label,
    }
