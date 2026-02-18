"""
Telegram Channel

将 Telegram Bot 接入 MessageBus，支持 allowFrom 白名单。
"""
import logging

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus

logger = logging.getLogger(__name__)

_CHANNEL = "telegram"


class TelegramChannel:

    def __init__(self, token: str, bus: MessageBus, allow_from: list[str] | None = None) -> None:
        self._bus = bus
        self._allow_from: set[str] = set(allow_from) if allow_from else set()
        self._app = Application.builder().token(token).build()
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        bus.subscribe_outbound(_CHANNEL, self._on_response)
        # username.lower() → chat_id 映射，用户发过消息后自动记录
        self.user_map: dict[str, str] = {}

    @property
    def bot(self):
        return self._app.bot

    async def start(self) -> None:
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        logger.info("TelegramChannel 已启动")

    async def stop(self) -> None:
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        logger.info("TelegramChannel 已停止")

    # ── 私有方法 ──────────────────────────────────────────────────

    def _is_allowed(self, user) -> bool:
        """检查用户是否在白名单中，白名单为空则允许所有人"""
        if not self._allow_from:
            return True
        # 同时匹配数字 ID 和用户名（@username 去掉 @ 后）
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

        # 记录 username → chat_id，供主动推送工具使用
        if user.username:
            self.user_map[user.username.lower()] = str(chat.id)

        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=msg.text,
            metadata={"username": user.username or ""},
        ))

    async def _on_response(self, msg: OutboundMessage) -> None:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"[telegram] 发送回复  chat_id={msg.chat_id}  内容: {preview!r}")
        try:
            await self._app.bot.send_message(
                chat_id=int(msg.chat_id),
                text=msg.content,
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            logger.debug(f"[telegram] HTML 解析失败，降级为纯文本  chat_id={msg.chat_id}")
            await self._app.bot.send_message(
                chat_id=int(msg.chat_id),
                text=msg.content,
            )
