"""
QQ Channel

通过 NcatBot（NapCat Python SDK）接入 QQ 私聊和群聊消息。
消息流向：QQ → NcatBot → MessageBus → AgentLoop → MessageBus → QQ

chat_id 约定：
  私聊："{user_id}"           （如 "987654321"）
  群聊："gqq:{group_id}"     （如 "gqq:111222333"）

摩擦点说明：
  1. run_backend() 是同步阻塞调用 → 用 run_in_executor 包裹
  2. NcatBot 事件回调运行在独立线程/loop → 用 run_coroutine_threadsafe 桥接到主 loop
  3. 出站消息需跨 loop 调用 API → 使用 run_coroutine_threadsafe 投递回 NcatBot loop
"""
import asyncio
import base64
import logging
from pathlib import Path

from agent.config import QQGroupConfig
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from channels.group_filter import DefaultGroupFilter, GroupMessageFilter, strip_at_segments
from session.manager import SessionManager

# NcatBot 运行时产物（plugins、logs）放到用户目录，不污染项目目录
_NCATBOT_DIR = Path.home() / ".akasic" / "ncatbot"

logger = logging.getLogger(__name__)

_CHANNEL = "qq"
_GROUP_PREFIX = "gqq:"


class QQChannel:

    def __init__(
        self,
        bot_uin: str,
        bus: MessageBus,
        session_manager: SessionManager,
        allow_from: list[str] | None = None,
        groups: list[QQGroupConfig] | None = None,
        group_filter: GroupMessageFilter | None = None,
    ) -> None:
        from ncatbot.core import BotClient
        from ncatbot.utils import ncatbot_config

        self._bus = bus
        self._session_manager = session_manager
        self._bot_uin = bot_uin
        self._allow_from: set[str] = set(allow_from) if allow_from else set()

        # group_id → QQGroupConfig
        self._groups: dict[str, QQGroupConfig] = {g.group_id: g for g in (groups or [])}

        # 消息过滤器，默认使用 DefaultGroupFilter
        self._group_filter: GroupMessageFilter = group_filter or DefaultGroupFilter(bot_uin)

        self._bot = BotClient()
        self._api = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._bot_loop: asyncio.AbstractEventLoop | None = None

        ncatbot_config.bt_uin = bot_uin
        # NapCat 由 Docker 容器管理，NcatBot 只负责连接 WebSocket
        ncatbot_config.check_ncatbot_update = False
        ncatbot_config.skip_ncatbot_install_check = True
        ncatbot_config.napcat.remote_mode = True
        ncatbot_config.enable_webui_interaction = False
        # 运行时产物重定向到 ~/.akasic/ncatbot/，不污染项目目录
        _NCATBOT_DIR.mkdir(parents=True, exist_ok=True)
        (_NCATBOT_DIR / "plugins").mkdir(exist_ok=True)
        ncatbot_config.plugin.plugins_dir = str(_NCATBOT_DIR / "plugins")

        # username（QQ 号字符串）→ chat_id 映射，供主动推送工具使用
        self.user_map: dict[str, str] = {}

    def _is_allowed(self, user_id: str) -> bool:
        if not self._allow_from:
            return True
        return user_id in self._allow_from

    async def start(self) -> None:
        self._main_loop = asyncio.get_running_loop()

        @self._bot.on_private_message()
        async def _(event) -> None:
            if self._bot_loop is None:
                self._bot_loop = asyncio.get_running_loop()
            user_id = str(event.user_id)

            if not self._is_allowed(user_id):
                logger.warning(f"[qq] 拒绝未授权用户  user_id={user_id}")
                return

            content: str = event.raw_message
            preview = content[:60] + "..." if len(content) > 60 else content
            logger.info(f"[qq] 私聊消息  user_id={user_id}  内容: {preview!r}")

            self.user_map[user_id] = user_id

            asyncio.run_coroutine_threadsafe(
                self._handle_private(user_id, content),
                self._main_loop,
            )

        @self._bot.on_group_message()
        async def _(event) -> None:
            if self._bot_loop is None:
                self._bot_loop = asyncio.get_running_loop()

            group_id = str(event.group_id)
            user_id = str(event.user_id)

            group_cfg = self._groups.get(group_id)
            if group_cfg is None:
                logger.debug(f"[qq] 忽略未配置群  group_id={group_id}")
                return

            # 过滤判断（同步包装异步 filter，在 bot loop 里执行）
            future = asyncio.run_coroutine_threadsafe(
                self._group_filter.should_process(event, group_cfg),
                self._main_loop,
            )
            if not future.result(timeout=5):
                return

            content = strip_at_segments(event.raw_message)
            preview = content[:60] + "..." if len(content) > 60 else content
            logger.info(f"[qq] 群聊消息  group_id={group_id}  user_id={user_id}  内容: {preview!r}")

            asyncio.run_coroutine_threadsafe(
                self._handle_group(group_id, user_id, content),
                self._main_loop,
            )

        @self._bot.on_startup()
        async def _(_event) -> None:
            self._bot_loop = asyncio.get_running_loop()

        logger.info("[qq] 正在启动 NcatBot（首次运行需要扫码登录）...")
        self._api = await self._main_loop.run_in_executor(None, self._bot.run_backend)
        logger.info("[qq] NcatBot 已启动")

        self._bus.subscribe_outbound(_CHANNEL, self._on_response)

    async def stop(self) -> None:
        if self._api:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._bot.exit)
            logger.info("[qq] QQChannel 已停止")

    # ── 入站处理 ──────────────────────────────────────────────────────

    async def _handle_private(self, user_id: str, content: str) -> None:
        """私聊入站：chat_id = user_id"""
        session = self._session_manager.get_or_create(f"{_CHANNEL}:{user_id}")
        if "user_id" not in session.metadata:
            session.metadata["user_id"] = user_id
            self._session_manager.save(session)
        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=user_id,
            chat_id=user_id,
            content=content,
            metadata={"chat_type": "private"},
        ))

    async def _handle_group(self, group_id: str, user_id: str, content: str) -> None:
        """群聊入站：chat_id = gqq:{group_id}，session 按群共享"""
        chat_id = f"{_GROUP_PREFIX}{group_id}"
        session = self._session_manager.get_or_create(f"{_CHANNEL}:{chat_id}")
        if "group_id" not in session.metadata:
            session.metadata["group_id"] = group_id
            self._session_manager.save(session)
        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=user_id,
            chat_id=chat_id,
            content=content,
            metadata={"chat_type": "group", "group_id": group_id, "sender_id": user_id},
        ))

    # ── 出站路由 ──────────────────────────────────────────────────────

    async def _on_response(self, msg: OutboundMessage) -> None:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        try:
            if msg.chat_id.startswith(_GROUP_PREFIX):
                group_id = msg.chat_id[len(_GROUP_PREFIX):]
                logger.info(f"[qq] 群聊回复  group_id={group_id}  内容: {preview!r}")
                await self._run_on_bot_loop(self._api.send_group_text(int(group_id), msg.content))
            else:
                logger.info(f"[qq] 私聊回复  user_id={msg.chat_id}  内容: {preview!r}")
                await self._run_on_bot_loop(self._api.send_private_text(int(msg.chat_id), msg.content))
        except Exception as e:
            logger.error(f"[qq] 发送失败  chat_id={msg.chat_id}  错误: {e}")

    # ── 主动推送（供 MessagePushTool 使用）────────────────────────────

    async def send(self, chat_id: str, message: str) -> None:
        """发送文本消息，自动区分私聊/群聊"""
        if not self._api:
            raise RuntimeError("QQChannel 尚未启动")
        if chat_id.startswith(_GROUP_PREFIX):
            group_id = chat_id[len(_GROUP_PREFIX):]
            await self._run_on_bot_loop(self._api.send_group_text(int(group_id), message))
        else:
            await self._run_on_bot_loop(self._api.send_private_text(int(chat_id), message))

    async def send_file(self, chat_id: str, file_path: str, name: str | None = None) -> None:
        """发送文件，自动区分私聊/群聊"""
        if not self._api:
            raise RuntimeError("QQChannel 尚未启动")
        uri = _local_to_base64(file_path) if _is_local(file_path) else file_path
        if chat_id.startswith(_GROUP_PREFIX):
            group_id = chat_id[len(_GROUP_PREFIX):]
            await self._run_on_bot_loop(self._api.send_group_file(int(group_id), uri, name))
        else:
            await self._run_on_bot_loop(self._api.send_private_file(int(chat_id), uri, name))

    async def send_image(self, chat_id: str, image: str) -> None:
        """发送图片，自动区分私聊/群聊"""
        if not self._api:
            raise RuntimeError("QQChannel 尚未启动")
        uri = _local_to_base64(image) if _is_local(image) else image
        if chat_id.startswith(_GROUP_PREFIX):
            group_id = chat_id[len(_GROUP_PREFIX):]
            await self._run_on_bot_loop(self._api.send_group_image(int(group_id), uri))
        else:
            await self._run_on_bot_loop(self._api.send_private_image(int(chat_id), uri))

    async def _run_on_bot_loop(self, coro):
        if self._bot_loop is None:
            raise RuntimeError("QQ bot loop 未就绪")
        future = asyncio.run_coroutine_threadsafe(coro, self._bot_loop)
        await asyncio.wrap_future(future)


def _is_local(path: str) -> bool:
    """判断是否为本地文件路径（非 URL、非 base64）"""
    return not path.startswith(("http://", "https://", "base64://", "file://"))


def _local_to_base64(path: str) -> str:
    """将本地文件编码为 NapCat 接受的 base64:// URI"""
    data = Path(path).read_bytes()
    return "base64://" + base64.b64encode(data).decode()
