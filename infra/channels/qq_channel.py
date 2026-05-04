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
import html
import importlib
import logging
import re
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, cast

from agent.config_models import QQGroupConfig
from agent.looping.interrupt import InterruptController
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from infra.channels.base import AttachmentStore, SessionIdentityIndex
from infra.channels.group_filter import (
    DefaultGroupFilter,
    GroupMessageFilter,
    strip_at_segments,
)
from core.net.http import HttpRequester, RequestBudget, get_default_http_requester
from session.manager import SessionManager

# NcatBot 运行时产物（plugins、logs）放到用户目录，不污染项目目录
_NCATBOT_DIR = Path.home() / ".akashic" / "ncatbot"

logger = logging.getLogger(__name__)

_CHANNEL = "qq"
_GROUP_PREFIX = "gqq:"

# 匹配 CQ:image 码中的 url 字段
_CQ_IMAGE_RE = re.compile(r"\[CQ:image[^\]]*?(?:,|\b)url=([^,\]]+)[^\]]*\]")


def _patch_ncatbot_ws_open_timeout(timeout_seconds: float) -> None:
    """覆盖 ncatbot 进程内写死的 1 秒 WebSocket 握手超时。"""
    if timeout_seconds <= 0:
        return

    adapter_mod = importlib.import_module("ncatbot.core.adapter.adapter")
    original_connect = getattr(
        adapter_mod,
        "_akashic_original_websockets_connect",
        None,
    )
    if original_connect is None:
        original_connect = adapter_mod.websockets.connect
        adapter_mod._akashic_original_websockets_connect = original_connect

        def _patched_connect(*args, **kwargs):
            configured_timeout = getattr(
                adapter_mod,
                "_akashic_websocket_open_timeout_seconds",
                None,
            )
            if configured_timeout is not None:
                kwargs["open_timeout"] = configured_timeout
            return adapter_mod._akashic_original_websockets_connect(*args, **kwargs)

        adapter_mod.websockets.connect = _patched_connect

    adapter_mod._akashic_websocket_open_timeout_seconds = timeout_seconds


def _extract_cq_images(raw: str) -> tuple[str, list[str]]:
    """从 CQ 码中提取图片 URL，返回 (纯文本, [url...])"""
    urls = _CQ_IMAGE_RE.findall(raw)
    text = re.sub(r"\[CQ:image[^\]]*\]", "", raw).strip()
    return text, urls


async def _download_to_temp(
    urls: list[str],
    requester: HttpRequester,
    attachments: AttachmentStore | None = None,
) -> list[str]:
    """下载图片到临时文件，返回本地路径列表"""
    if not urls:
        return []
    paths: list[str] = []
    attachment_store = attachments or AttachmentStore()
    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
    }
    for url in urls:
        try:
            url = html.unescape(url)  # 还原 &amp; 等 HTML 实体
            resp = await requester.get(
                url,
                follow_redirects=True,
                timeout_s=15.0,
                budget=RequestBudget(total_timeout_s=20.0),
            )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
            ext = ext_map.get(ct, ".jpg")
            path = attachment_store.write_bytes(
                resp.content,
                prefix="akashic_qq_",
                suffix=ext,
            )
            paths.append(str(path))
        except Exception as e:
            logger.warning(f"[qq] 图片下载失败  url={url[:80]}  错误: {e}")
    return paths


class QQChannel:

    def __init__(
        self,
        bot_uin: str,
        bus: MessageBus,
        session_manager: SessionManager,
        allow_from: list[str] | None = None,
        groups: list[QQGroupConfig] | None = None,
        websocket_open_timeout_seconds: float = 5.0,
        group_filter: GroupMessageFilter | None = None,
        http_requester: HttpRequester | None = None,
        interrupt_controller: InterruptController | None = None,
    ) -> None:
        from ncatbot.core import BotClient
        from ncatbot.utils import ncatbot_config

        self._bus = bus
        self._session_manager = session_manager
        self._bot_uin = bot_uin
        allowed_users = [str(user_id) for user_id in (allow_from or [])]
        self._allow_from: set[str] = set(allowed_users)
        self._websocket_open_timeout_seconds = float(websocket_open_timeout_seconds)
        self._interrupt_controller = interrupt_controller
        ws = getattr(session_manager, "workspace", None)
        self._attachments = AttachmentStore(Path(ws) / "uploads" if ws else None)
        self._identity_index = SessionIdentityIndex(
            session_manager,
            channel=_CHANNEL,
            metadata_key="user_id",
        )

        # group_id → QQGroupConfig
        self._groups: dict[str, QQGroupConfig] = {g.group_id: g for g in (groups or [])}

        # 消息过滤器，默认使用 DefaultGroupFilter
        self._group_filter: GroupMessageFilter = group_filter or DefaultGroupFilter(
            bot_uin
        )
        self._http_requester = http_requester or get_default_http_requester(
            "external_default"
        )

        self._bot = BotClient()
        self._api = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._bot_loop: asyncio.AbstractEventLoop | None = None

        _patch_ncatbot_ws_open_timeout(self._websocket_open_timeout_seconds)
        ncatbot_config.bt_uin = bot_uin
        ncatbot_config.root = allowed_users[0] if allowed_users else bot_uin
        # NapCat 由 Docker 容器管理，NcatBot 只负责连接 WebSocket
        ncatbot_config.check_ncatbot_update = False
        ncatbot_config.skip_ncatbot_install_check = True
        ncatbot_config.napcat.remote_mode = True
        # Akashic 只需要 NapCat 的 OneBot WebSocket，禁用 WebUI 避免启动时卡交互 token。
        ncatbot_config.napcat.enable_webui = False
        ncatbot_config.enable_webui_interaction = False
        # 运行时产物重定向到 ~/.akashic/ncatbot/，不污染项目目录
        _NCATBOT_DIR.mkdir(parents=True, exist_ok=True)
        (_NCATBOT_DIR / "plugins").mkdir(exist_ok=True)
        ncatbot_config.plugin.plugins_dir = str(_NCATBOT_DIR / "plugins")

        # username（QQ 号字符串）→ chat_id 映射，供主动推送工具使用
        self.user_map = self._identity_index.mapping

    def _is_allowed(self, user_id: str) -> bool:
        if not self._allow_from:
            return True
        return user_id in self._allow_from

    async def start(self) -> None:
        self._main_loop = asyncio.get_running_loop()
        self._identity_index.rebuild()

        @cast(Any, self._bot.on_private_message())
        async def _(event) -> None:
            if self._bot_loop is None:
                self._bot_loop = asyncio.get_running_loop()
            user_id = str(event.user_id)

            if not self._is_allowed(user_id):
                logger.warning(f"[qq] 拒绝未授权用户  user_id={user_id}")
                return

            raw: str = event.raw_message
            text, img_urls = _extract_cq_images(raw)
            if text.strip() == "/stop":
                self._submit_to_main_loop(self._handle_stop_private(user_id))
                return
            preview = text[:60] + "..." if len(text) > 60 else text
            logger.info(
                f"[qq] 私聊消息  user_id={user_id}  内容: {preview!r}  图片: {len(img_urls)}"
            )

            self.user_map[user_id] = user_id

            self._submit_to_main_loop(self._handle_private(user_id, text, img_urls))

        @cast(Any, self._bot.on_group_message())
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
                self._require_main_loop(),
            )
            if not future.result(timeout=5):
                return

            raw = strip_at_segments(event.raw_message)
            text, img_urls = _extract_cq_images(raw)
            if text.strip() == "/stop":
                self._submit_to_main_loop(self._handle_stop_group(group_id, user_id))
                return
            preview = text[:60] + "..." if len(text) > 60 else text
            logger.info(
                f"[qq] 群聊消息  group_id={group_id}  user_id={user_id}  内容: {preview!r}  图片: {len(img_urls)}"
            )

            self._submit_to_main_loop(
                self._handle_group(group_id, user_id, text, img_urls)
            )

        @cast(Any, self._bot.on_startup())
        async def _(_event) -> None:
            self._bot_loop = asyncio.get_running_loop()

        logger.info("[qq] 正在启动 NcatBot（首次运行需要扫码登录）...")
        self._api = await self._main_loop.run_in_executor(None, self._bot.run_backend)
        logger.info("[qq] NcatBot 已启动")

        self._bus.subscribe_outbound(_CHANNEL, self._on_response)

    async def stop(self) -> None:
        if self._api:
            loop = asyncio.get_running_loop()
            bot_exit = getattr(self._bot, "exit", None)
            if callable(bot_exit):
                await loop.run_in_executor(None, bot_exit)
            logger.info("[qq] QQChannel 已停止")

    # ── 入站处理 ──────────────────────────────────────────────────────

    async def _handle_private(
        self, user_id: str, content: str, img_urls: list[str] | None = None
    ) -> None:
        """私聊入站：chat_id = user_id"""
        await self._identity_index.remember(user_id, user_id)
        media = await _download_to_temp(
            img_urls or [],
            self._http_requester,
            self._attachments,
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=user_id,
                chat_id=user_id,
                content=content,
                media=media,
                metadata={"chat_type": "private"},
            )
        )

    async def _handle_stop_private(self, user_id: str) -> None:
        if self._interrupt_controller is None:
            await self.send(user_id, "当前未启用中断功能。")
            return
        result = self._interrupt_controller.request_interrupt(
            session_key=f"{_CHANNEL}:{user_id}",
            sender=user_id,
            command="/stop",
        )
        await self.send(user_id, result.message)

    async def _handle_group(
        self,
        group_id: str,
        user_id: str,
        content: str,
        img_urls: list[str] | None = None,
    ) -> None:
        """群聊入站：chat_id = gqq:{group_id}，session 按群共享"""
        chat_id = f"{_GROUP_PREFIX}{group_id}"
        session = self._session_manager.get_or_create(f"{_CHANNEL}:{chat_id}")
        if "group_id" not in session.metadata:
            session.metadata["group_id"] = group_id
            await self._session_manager.save_async(session)
        media = await _download_to_temp(
            img_urls or [],
            self._http_requester,
            self._attachments,
        )
        await self._bus.publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=user_id,
                chat_id=chat_id,
                content=content,
                media=media,
                metadata={
                    "chat_type": "group",
                    "group_id": group_id,
                    "sender_id": user_id,
                },
            )
        )

    async def _handle_stop_group(self, group_id: str, user_id: str) -> None:
        chat_id = f"{_GROUP_PREFIX}{group_id}"
        if self._interrupt_controller is None:
            await self.send(chat_id, "当前未启用中断功能。")
            return
        result = self._interrupt_controller.request_interrupt(
            session_key=f"{_CHANNEL}:{chat_id}",
            sender=user_id,
            command="/stop",
        )
        await self.send(chat_id, result.message)

    # ── 出站路由 ──────────────────────────────────────────────────────

    async def _on_response(self, msg: OutboundMessage) -> None:
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        api = self._api
        if api is None:
            raise RuntimeError("QQChannel 尚未启动")
        if msg.content.strip():
            try:
                if msg.chat_id.startswith(_GROUP_PREFIX):
                    group_id = msg.chat_id[len(_GROUP_PREFIX) :]
                    logger.info(f"[qq] 群聊回复  group_id={group_id}  内容: {preview!r}")
                    await self._run_on_bot_loop(
                        api.send_group_text(int(group_id), msg.content)
                    )
                else:
                    logger.info(f"[qq] 私聊回复  user_id={msg.chat_id}  内容: {preview!r}")
                    await self._run_on_bot_loop(
                        api.send_private_text(int(msg.chat_id), msg.content)
                    )
            except Exception as e:
                logger.error(f"[qq] 发送失败  chat_id={msg.chat_id}  错误: {e}")
        for image in (msg.media or []):
            try:
                await self.send_image(msg.chat_id, image)
            except Exception as e:
                logger.error(f"[qq] meme 图片发送失败  chat_id={msg.chat_id}  path={image}  err={e}")

    # ── 主动推送（供 MessagePushTool 使用）────────────────────────────

    async def send(self, chat_id: str, message: str) -> None:
        """发送文本消息，自动区分私聊/群聊"""
        api = self._api
        if api is None:
            raise RuntimeError("QQChannel 尚未启动")
        if chat_id.startswith(_GROUP_PREFIX):
            group_id = chat_id[len(_GROUP_PREFIX) :]
            await self._run_on_bot_loop(api.send_group_text(int(group_id), message))
        else:
            await self._run_on_bot_loop(api.send_private_text(int(chat_id), message))

    async def send_file(
        self, chat_id: str, file_path: str, name: str | None = None
    ) -> None:
        """发送文件，自动区分私聊/群聊"""
        api = self._api
        if api is None:
            raise RuntimeError("QQChannel 尚未启动")
        uri = _local_to_base64(file_path) if _is_local(file_path) else file_path
        if chat_id.startswith(_GROUP_PREFIX):
            group_id = chat_id[len(_GROUP_PREFIX) :]
            await self._run_on_bot_loop(api.send_group_file(int(group_id), uri, name))
        else:
            await self._run_on_bot_loop(api.send_private_file(int(chat_id), uri, name))

    async def send_image(self, chat_id: str, image: str) -> None:
        """发送图片，自动区分私聊/群聊"""
        api = self._api
        if api is None:
            raise RuntimeError("QQChannel 尚未启动")
        uri = _local_to_base64(image) if _is_local(image) else image
        if chat_id.startswith(_GROUP_PREFIX):
            group_id = chat_id[len(_GROUP_PREFIX) :]
            await self._run_on_bot_loop(api.send_group_image(int(group_id), uri))
        else:
            await self._run_on_bot_loop(api.send_private_image(int(chat_id), uri))

    def _require_main_loop(self) -> asyncio.AbstractEventLoop:
        if self._main_loop is None:
            raise RuntimeError("QQ main loop 未就绪")
        return self._main_loop

    def _submit_to_main_loop(self, coro: Coroutine[object, object, None]) -> None:
        asyncio.run_coroutine_threadsafe(coro, self._require_main_loop())

    async def _run_on_bot_loop(
        self, coro: Coroutine[object, object, object]
    ) -> None:
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
