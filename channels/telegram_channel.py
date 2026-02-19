"""
Telegram Channel

将 Telegram Bot 接入 MessageBus，支持 allowFrom 白名单。
"""
import logging

from telegram import Update
from telegram.constants import ChatAction
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
            self._session_manager.save(session)

        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=str(user.id),
            chat_id=str(chat.id),
            content=msg.text,
            metadata={"username": user.username or ""},
        ))

    async def send(self, chat_id: str, message: str) -> None:
        """主动发送消息（供 MessagePushTool 调用）。
        chat_id 可以是数字 ID，也可以是用户名（不含 @），会自动查表转换。
        """
        resolved = chat_id.lstrip("@").lower()
        if not resolved.lstrip("-").isdigit():
            resolved = self.user_map.get(resolved)
            if not resolved:
                raise ValueError(
                    f"找不到用户 {chat_id!r} 的 chat_id，该用户需先给 bot 发一条消息。"
                    f"已知用户：{list(self.user_map.keys()) or '（无）'}"
                )
        await send_markdown(self._app.bot, resolved, message)

    async def _on_response(self, msg: OutboundMessage) -> None:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"[telegram] 发送回复  chat_id={msg.chat_id}  内容: {preview!r}")
        await send_markdown(self._app.bot, msg.chat_id, msg.content)
