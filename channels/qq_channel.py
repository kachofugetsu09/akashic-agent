"""
QQ Channel

通过 NcatBot（NapCat Python SDK）接入 QQ 私聊消息。
消息流向：QQ → NcatBot → MessageBus → AgentLoop → MessageBus → QQ

摩擦点说明：
  1. run_backend() 是同步阻塞调用 → 用 run_in_executor 包裹
  2. NcatBot 事件回调运行在独立线程/loop → 用 run_coroutine_threadsafe 桥接到主 loop
  3. 出站消息需跨 loop 调用 API → 使用 run_coroutine_threadsafe 投递回 NcatBot loop
"""
import asyncio
import base64
import logging
from pathlib import Path

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from session.manager import SessionManager

# NcatBot 运行时产物（plugins、logs）放到用户目录，不污染项目目录
_NCATBOT_DIR = Path.home() / ".akasic" / "ncatbot"

logger = logging.getLogger(__name__)

_CHANNEL = "qq"


class QQChannel:

    def __init__(self, bot_uin: str, bus: MessageBus, session_manager: SessionManager, allow_from: list[str] | None = None) -> None:
        from ncatbot.core import BotClient
        from ncatbot.utils import ncatbot_config

        self._bus = bus
        self._session_manager = session_manager
        self._allow_from: set[str] = set(allow_from) if allow_from else set()
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
        # QQ 私聊 chat_id == user_id，此 map 主要用于按 QQ 号检索
        self.user_map: dict[str, str] = {}

    def _is_allowed(self, user_id: str) -> bool:
        if not self._allow_from:
            return True
        return user_id in self._allow_from

    async def start(self) -> None:
        self._main_loop = asyncio.get_running_loop()

        # 在 start() 时注册回调，之后再启动 bot
        # 注意：回调运行在 NcatBot 内部 loop（独立线程），不能直接 await 主 loop 的协程
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
            logger.info(f"[qq] 收到消息  user_id={user_id}  内容: {preview!r}")

            self.user_map[user_id] = user_id

            # 桥接到主 asyncio loop（run_coroutine_threadsafe 是线程安全的）
            asyncio.run_coroutine_threadsafe(
                self._handle_inbound(user_id, content),
                self._main_loop,
            )

        @self._bot.on_startup()
        async def _(_event) -> None:
            self._bot_loop = asyncio.get_running_loop()

        # run_backend() 阻塞直到 NapCat 连接成功，放入线程池避免阻塞主 loop
        logger.info("[qq] 正在启动 NcatBot（首次运行需要扫码登录）...")
        self._api = await self._main_loop.run_in_executor(None, self._bot.run_backend)
        logger.info("[qq] NcatBot 已启动")

        self._bus.subscribe_outbound(_CHANNEL, self._on_response)

    async def stop(self) -> None:
        if self._api:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._bot.exit)
            logger.info("[qq] QQChannel 已停止")

    async def _handle_inbound(self, user_id: str, content: str) -> None:
        """在主 loop 里执行：持久化 sender metadata + 发布到 bus。"""
        session = self._session_manager.get_or_create(f"{_CHANNEL}:{user_id}")
        if "user_id" not in session.metadata:
            session.metadata["user_id"] = user_id
            self._session_manager.save(session)
        await self._bus.publish_inbound(InboundMessage(
            channel=_CHANNEL,
            sender=user_id,
            chat_id=user_id,
            content=content,
        ))

    async def send(self, chat_id: str, message: str) -> None:
        """发送文本消息"""
        if not self._api:
            raise RuntimeError("QQChannel 尚未启动")
        await self._run_on_bot_loop(self._api.send_private_text(int(chat_id), message))

    async def send_file(self, chat_id: str, file_path: str, name: str | None = None) -> None:
        """发送文件（base64 编码，绕过 Docker 路径限制）"""
        if not self._api:
            raise RuntimeError("QQChannel 尚未启动")
        uri = _local_to_base64(file_path) if _is_local(file_path) else file_path
        await self._run_on_bot_loop(self._api.send_private_file(int(chat_id), uri, name))

    async def send_image(self, chat_id: str, image: str) -> None:
        """发送图片（本地文件转 base64，绕过 Docker 路径限制；URL 直接传）"""
        if not self._api:
            raise RuntimeError("QQChannel 尚未启动")
        uri = _local_to_base64(image) if _is_local(image) else image
        await self._run_on_bot_loop(self._api.send_private_image(int(chat_id), uri))

    async def _on_response(self, msg: OutboundMessage) -> None:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"[qq] 发送回复  user_id={msg.chat_id}  内容: {preview!r}")
        try:
            await self._run_on_bot_loop(self._api.send_private_text(int(msg.chat_id), msg.content))
        except Exception as e:
            logger.error(f"[qq] 发送失败  user_id={msg.chat_id}  错误: {e}")

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
