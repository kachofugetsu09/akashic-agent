"""
Telegram Channel

将 Telegram Bot 接入 MessageBus，支持 allowFrom 白名单。
"""
import logging
import tempfile
import asyncio

from telegram import Update
from telegram.constants import ChatAction
from telegram.error import NetworkError, TimedOut
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from channels.telegram_utils import send_markdown
from session.manager import SessionManager

logger = logging.getLogger(__name__)

_CHANNEL = "telegram"


class TelegramChannel:

    def __init__(
        self,
        token: str,
        bus: MessageBus,
        session_manager: SessionManager,
        allow_from: list[str] | None = None,
    ) -> None:
        self._bus = bus
        self._session_manager = session_manager
        self._allow_from: set[str] = set(allow_from) if allow_from else set()
        self._app = Application.builder().token(token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self._app.add_handler(
            MessageHandler(filters.PHOTO & ~filters.COMMAND, self._on_photo)
        )
        bus.subscribe_outbound(_CHANNEL, self._on_response)
        # username.lower() → chat_id，启动时从 session 重建，运行时实时更新
        self.user_map: dict[str, str] = {}

    @property
    def bot(self):
        return self._app.bot

    async def start(self) -> None:
        self._rebuild_user_map()
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info(f"TelegramChannel 已启动  已知用户: {len(self.user_map)}")

    async def stop(self) -> None:
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("TelegramChannel 已停止")

    # ── 私有方法 ──────────────────────────────────────────────────

    def _rebuild_user_map(self) -> None:
        """扫描已有 session 文件，从 metadata 重建 username → chat_id 索引。"""
        self.user_map.clear()
        for entry in self._session_manager.get_channel_metadata(_CHANNEL):
            username = entry["metadata"].get("username", "").lower()
            if username:
                self.user_map[username] = entry["chat_id"]
        logger.debug(f"[telegram] user_map 重建完成: {self.user_map}")

    def _is_allowed(self, user) -> bool:
        """检查用户是否在白名单中，白名单为空则允许所有人"""
        if not self._allow_from:
            return True
        return (
            str(user.id) in self._allow_from
            or (user.username and user.username.lower() in {u.lower() for u in self._allow_from})
        )

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.text or not user:
            return

        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权用户  id={user.id}  username=@{user.username}"
            )
            return

        preview = msg.text[:60] + "..." if len(msg.text) > 60 else msg.text
        logger.info(
            f"[telegram] 收到消息  chat_id={chat.id}  "
            f"user=@{user.username or user.id}  内容: {preview!r}"
        )

        # 更新内存索引 + 持久化到 session.metadata
        if user.username:
            username = user.username.lower()
            chat_id_str = str(chat.id)
            self.user_map[username] = chat_id_str
            session = self._session_manager.get_or_create(f"{_CHANNEL}:{chat_id_str}")
            session.metadata["username"] = username
            await self._session_manager.save_async(session)

        await self._safe_send_typing(context, chat.id)

        inbound_text, reply_meta = _build_inbound_text_with_reply(msg.text, msg.reply_to_message)
        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=inbound_text,
            metadata={
                "username": user.username or "",
                **reply_meta,
            },
        ))

    async def _on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.effective_message
        chat = update.effective_chat
        user = update.effective_user

        if not msg or not msg.photo or not user:
            return

        if not self._is_allowed(user):
            logger.warning(
                f"[telegram] 拒绝未授权用户  id={user.id}  username=@{user.username}"
            )
            return

        if user.username:
            username = user.username.lower()
            chat_id_str = str(chat.id)
            self.user_map[username] = chat_id_str
            session = self._session_manager.get_or_create(f"{_CHANNEL}:{chat_id_str}")
            session.metadata["username"] = username
            await self._session_manager.save_async(session)

        await self._safe_send_typing(context, chat.id)

        # 下载最高分辨率的图片到临时文件
        tg_file = await context.bot.get_file(msg.photo[-1].file_id)
        tmp = tempfile.mktemp(suffix=".jpg", prefix="akasic_tg_")
        await tg_file.download_to_drive(tmp)
        logger.info(f"[telegram] 收到图片  chat_id={chat.id}  user=@{user.username or user.id}  tmp={tmp}")

        caption_text = msg.caption or ""
        inbound_text, reply_meta = _build_inbound_text_with_reply(caption_text, msg.reply_to_message)
        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=inbound_text,
            media=[tmp],
            metadata={
                "username": user.username or "",
                **reply_meta,
            },
        ))

    def _resolve_chat_id(self, chat_id: str) -> str:
        resolved = chat_id.lstrip("@").lower()
        if not resolved.lstrip("-").isdigit():
            resolved = self.user_map.get(resolved)
            if not resolved:
                raise ValueError(
                    f"找不到用户 {chat_id!r} 的 chat_id，该用户需先给 bot 发一条消息。"
                    f"已知用户：{list(self.user_map.keys()) or '（无）'}"
                )
        return resolved

    async def send(self, chat_id: str, message: str) -> None:
        """发送文本消息（供 MessagePushTool 调用）"""
        await send_markdown(self._app.bot, self._resolve_chat_id(chat_id), message)

    async def send_file(self, chat_id: str, file_path: str, name: str | None = None) -> None:
        """发送文件"""
        cid = int(self._resolve_chat_id(chat_id))
        with open(file_path, "rb") as f:
            await self._app.bot.send_document(chat_id=cid, document=f, filename=name)

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
        await send_markdown(self._app.bot, msg.chat_id, msg.content)

    async def _safe_send_typing(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        """发送 typing 状态；失败时指数退避重试，不影响消息主流程。"""
        base_delay = 0.4
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
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
                logger.warning("[telegram] send_chat_action 失败，已跳过 typing chat_id=%s err=%s", chat_id, e)
                return


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
        # 保守处理：非文本被回复消息只保留结构化元信息，不污染正文。
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
