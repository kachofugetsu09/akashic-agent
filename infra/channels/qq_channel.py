"""
QQ Channel

通过 NcatBot（NapCat Python SDK）接入 QQ 私聊和群聊消息。
消息流向：QQ → NcatBot → Core v3 Channel ingress → AgentLoop → Core v3 outbound → QQ

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
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import importlib
import logging
import re
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, cast

from agent.config_models import QQGroupConfig
from agent.looping.interrupt import InterruptController
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelAdapter,
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelRuntimePorts,
    InboundIdentity,
    ProviderDeliveryRequest,
    RawInbound,
    StopReceipt,
)
from bus.event_bus import EventBus
from bus.events import (
    ChannelMessage,
    DeliveryReceipt,
    OutboundMessage,
)
from bus.events_lifecycle import (
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from bus.queue import MessageBus
from infra.channels.contract import ChannelContext
from infra.channels.delivery import deliver_message_parts
from infra.channels.group_filter import (
    DefaultGroupFilter,
    GroupMessageFilter,
    strip_at_segments,
)
from infra.channels.native_delivery import NativeChannelDeliveryAdapter
from core.net.http import HttpRequester, RequestBudget, get_default_http_requester

# NcatBot 运行时产物（plugins、logs）放到用户目录，不污染项目目录
_NCATBOT_DIR = Path.home() / ".akashic" / "ncatbot"

logger = logging.getLogger(__name__)

_CHANNEL = "qq"
_GROUP_PREFIX = "gqq:"
_TRACE_THINKING_LIMIT = 500
_TRACE_TOOL_RESULT_LIMIT = 120
_TRACE_DEFAULT_ACTOR = "Akashic"
MAX_QQ_IMAGE_COUNT = 10
MAX_QQ_IMAGE_BYTES = 10 * 1024 * 1024
MAX_QQ_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024


def _normalize_v3_content(value: str) -> str:
    """Keep provider text while replacing Core-forbidden control characters."""

    return "".join(
        "\u2028" if ord(char) in {10, 13} else " " if ord(char) < 32 else char
        for char in value
    )


@dataclass
class _QQTraceLine:
    tool_name: str
    status: str = "started"
    intent: str = ""
    target: str = ""
    result_preview: str = ""


@dataclass
class _QQTraceState:
    user_message: str = ""
    tool_lines: list[_QQTraceLine] = field(default_factory=list)


class _QQInboundRuntime:
    """Gate main-loop QQ callbacks on one exact formal Core binding."""

    def __init__(self) -> None:
        self._ports: ChannelRuntimePorts | None = None
        self._open = False
        self._wake = asyncio.Event()
        self._futures: set[Any] = set()

    def attach(self, ports: ChannelRuntimePorts) -> None:
        if self._open:
            raise RuntimeError("QQ v3 ingress 已打开")
        if ports.ingress is None:
            raise RuntimeError("QQ v3 ingress 缺少 Core ingress")
        self._ports = ports
        self._open = False
        self._wake.clear()

    def open(self) -> None:
        if self._ports is None:
            raise RuntimeError("QQ v3 ingress 尚未 attach")
        self._open = True
        self._wake.set()

    def close(self) -> None:
        self._open = False
        self._ports = None
        self._wake.set()

    def track_future(self, future: Any) -> None:
        self._futures.add(future)
        future.add_done_callback(self._futures.discard)
        future.add_done_callback(self._report_failure)

    @staticmethod
    def _report_failure(future: Any) -> None:
        if future.cancelled():
            return
        try:
            error = future.exception()
        except BaseException as exc:
            logger.error("[qq] v3 入站任务状态读取失败: %s", exc)
            return
        if error is not None:
            logger.error("[qq] v3 入站任务失败", exc_info=error)

    async def wait_quiescent(self) -> None:
        futures = tuple(self._futures)
        if futures:
            await asyncio.gather(
                *(asyncio.wrap_future(future) for future in futures),
                return_exceptions=True,
            )

    async def wait_open(self) -> ChannelRuntimePorts:
        ports = self._ports
        if ports is None:
            raise RuntimeError("QQ v3 ingress 尚未 attach")
        await self._wake.wait()
        if not self._open or self._ports is not ports:
            raise RuntimeError("QQ v3 ingress admission 已关闭")
        return ports

    def require_open(self, ports: ChannelRuntimePorts) -> None:
        if not self._open or self._ports is not ports:
            raise RuntimeError("QQ v3 ingress admission 已关闭")

    async def admit(
        self,
        raw: RawInbound,
        *,
        ports: ChannelRuntimePorts | None = None,
    ) -> bool:
        if ports is None:
            ports = await self.wait_open()
        if ports.ingress is None:
            raise RuntimeError("QQ v3 ingress 缺少 Core ingress")
        if not self._open or self._ports is not ports:
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
        if ports.attachment_import is None:
            raise RuntimeError("QQ v3 attachment import 缺少 Core port")
        if not self._open or self._ports is not ports:
            raise RuntimeError("QQ v3 attachment import admission 已关闭")
        return await ports.attachment_import.import_bytes(
            data,
            kind=kind,
            filename=filename,
            media_type=media_type,
        )


def _truncate_trace_text(text: str, limit: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    omitted = len(raw) - limit
    head = max(0, limit // 2)
    tail = max(0, limit - head)
    return f"{raw[:head]} ...[{omitted} chars omitted]... {raw[-tail:]}"


def _format_tool_intent(arguments: dict[str, Any]) -> str:
    if not isinstance(arguments, dict):
        return ""
    for key in ("description", "query", "summary", "task", "action"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return _truncate_trace_text(value, 80)
    return ""


def _format_tool_target(arguments: dict[str, Any]) -> str:
    if not isinstance(arguments, dict):
        return ""
    if isinstance(arguments.get("path"), str) and arguments.get("path", "").strip():
        return _truncate_trace_text(str(arguments["path"]).strip(), 60)
    if isinstance(arguments.get("file_path"), str) and arguments.get("file_path", "").strip():
        return _truncate_trace_text(str(arguments["file_path"]).strip(), 60)
    for key in (
        "cmd",
        "command",
        "query",
        "url",
        "file",
        "text",
        "content",
        "prompt",
        "name",
    ):
        value = arguments.get(key)
        if isinstance(value, str | int | float) and str(value).strip():
            return _truncate_trace_text(str(value).strip(), 80)
    return ""


def _format_tool_trace_lines(lines: list[_QQTraceLine]) -> str:
    if not lines:
        return "No tool calls."
    rendered: list[str] = []
    for index, line in enumerate(lines, start=1):
        rendered.append(f"{index}. {_compress_tool_line(line)}")
    return "\n".join(rendered)


def _summarize_tool_result_preview(tool_name: str, preview: str) -> str:
    text = str(preview or "").strip()
    if not text:
        return ""
    name = tool_name.lower()
    if name == "fetch_messages":
        if '"matched_count"' in text or '"count"' in text:
            matched = re.search(r'"matched_count"\s*:\s*(\d+)', text)
            count = re.search(r'"count"\s*:\s*(\d+)', text)
            hit_text = matched.group(1) if matched else "?"
            total_text = count.group(1) if count else "?"
            return f"结果：命中 {hit_text} 条，返回上下文 {total_text} 条"
        return "结果：已返回消息上下文"
    if name == "list_dir":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return f"结果：列出 {len(lines)} 项"
        return "结果：已列出目录内容"
    if name == "read_file":
        line_no = re.search(r"(\d+)→", text)
        if line_no:
            return f"结果：已读取第 {line_no.group(1)} 行附近内容"
        if "字节" in text:
            return "结果：已读取文件片段"
        return "结果：已读取文件"
    if name == "shell":
        exit_code = re.search(r'"exit_code"\s*:\s*(-?\d+)', text)
        if exit_code:
            code = exit_code.group(1)
            if code == "0":
                return "结果：命令执行成功"
            return f"结果：命令退出码 {code}"
        command = re.search(r'"command"\s*:\s*"([^"]+)"', text)
        if command:
            snippet = _truncate_trace_text(command.group(1), 50)
            return f"结果：已执行命令 {snippet}"
        if "（无输出）" in text or "(无输出)" in text:
            return "结果：命令已执行（无输出）"
        return "结果：命令已执行"
    if name == "list_schedules":
        matched = re.search(r"(\d+)\s*个", text)
        if matched:
            return f"结果：当前有 {matched.group(1)} 个提醒"
        return "结果：已列出提醒"
    if name == "cancel_schedule":
        matched = re.search(r"(\d+)\s*个", text)
        if matched:
            return f"结果：已取消 {matched.group(1)} 个提醒"
        return "结果：已执行取消"
    if name == "schedule":
        return "结果：已创建提醒"
    return f"结果：{_truncate_trace_text(text, _TRACE_TOOL_RESULT_LIMIT)}"


def _tool_emoji(tool_name: str) -> str:
    name = tool_name.lower()
    if name.startswith("mcp"):
        return "📡"
    if "search" in name or "fetch" in name:
        return "🔍"
    if "schedule" in name or "cancel" in name:
        return "⏰"
    if "shell" in name:
        return "⚙"
    if "file" in name or "read" in name or "write" in name:
        return "📄"
    return "🔧"


def _compress_tool_line(line: _QQTraceLine) -> str:
    status = "已完成" if line.status == "done" else "失败" if line.status == "error" else "进行中"
    parts = [f"{_tool_emoji(line.tool_name)} {line.tool_name}", status]
    if line.intent:
        parts.append(f"意图：{line.intent}")
    elif line.target:
        parts.append(f"目标：{line.target}")
    if line.result_preview:
        parts.append(line.result_preview)
    return " | ".join(parts)

# 匹配 CQ:image 码中的 url 字段
_CQ_IMAGE_RE = re.compile(r"\[CQ:image[^\]]*?(?:,|\b)url=([^,\]]+)[^\]]*\]")


def _patch_ncatbot_ws_open_timeout(timeout_seconds: float) -> None:
    """覆盖 ncatbot 进程内写死的 1 秒 WebSocket 握手超时。"""
    if timeout_seconds <= 0:
        return

    try:
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
    except Exception as e:
        logger.warning("[qq] patch ncatbot WebSocket open_timeout 失败，沿用 SDK 默认值: %s", e)


def _extract_cq_images(raw: str) -> tuple[str, list[str]]:
    """从 CQ 码中提取图片 URL，返回 (纯文本, [url...])"""
    urls = _CQ_IMAGE_RE.findall(raw)
    text = re.sub(r"\[CQ:image[^\]]*\]", "", raw).strip()
    return text, urls


def _qq_message_id(event: object) -> str | None:
    """Extract the provider-owned message identity without fabricating one."""

    for name in ("message_id", "message_seq"):
        value = getattr(event, name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


async def _read_qq_image(
    url: str,
    requester: HttpRequester,
    *,
    max_bytes: int,
) -> tuple[bytes, str]:
    """流式读取单张 QQ 图片，分配内存前执行单项上限。"""

    stream_context = requester.stream(
        "GET",
        url,
        timeout_s=15.0,
        budget=RequestBudget(total_timeout_s=20.0),
        validate_redirects=True,
    )
    async with cast(AbstractAsyncContextManager[Any], stream_context) as response:
        if response.status_code < 200 or response.status_code >= 300:
            raise ValueError(f"HTTP {response.status_code}")
        content_type = (
            response.headers.get("content-type", "image/jpeg")
            .split(";", 1)[0]
            .strip()
        )
        content = bytearray()
        async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
            if len(content) + len(chunk) > max_bytes:
                raise ValueError(
                    f"图片超过 {max_bytes // (1024 * 1024)}MB 上限"
                )
            content.extend(chunk)
        return bytes(content), content_type


class QQChannel:
    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    name = _CHANNEL

    def __init__(
        self,
        bot_uin: str,
        bus: MessageBus,
        workspace: Path,
        allow_from: list[str] | None = None,
        groups: list[QQGroupConfig] | None = None,
        websocket_open_timeout_seconds: float = 5.0,
        group_filter: GroupMessageFilter | None = None,
        http_requester: HttpRequester | None = None,
        event_bus: EventBus | None = None,
        interrupt_controller: InterruptController | None = None,
    ) -> None:
        from ncatbot.core import BotClient
        from ncatbot.utils import ncatbot_config

        self._bus = bus
        self._bot_uin = bot_uin
        allowed_users = [str(user_id) for user_id in (allow_from or [])]
        self._allow_from: set[str] = set(allowed_users)
        self._websocket_open_timeout_seconds = float(websocket_open_timeout_seconds)
        self._interrupt_controller = interrupt_controller
        self._workspace = workspace
        self._trace_actor_name_cache: str | None = None
        # group_id → QQGroupConfig
        self._groups: dict[str, QQGroupConfig] = {g.group_id: g for g in (groups or [])}

        # 消息过滤器，默认使用 DefaultGroupFilter
        self._group_filter: GroupMessageFilter = group_filter or DefaultGroupFilter(
            bot_uin
        )
        self._http_requester = http_requester or get_default_http_requester(
            "external_default"
        )
        self._event_bus = event_bus
        self._events_bound = False
        self._trace_states: dict[str, _QQTraceState] = {}
        self._v3_inbound_runtime = _QQInboundRuntime()

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


    def _is_allowed(self, user_id: str) -> bool:
        if not self._allow_from:
            return True
        return user_id in self._allow_from

    async def start(self, ctx: ChannelContext | None = None) -> None:
        if ctx is not None:
            self._bus = ctx.bus
            self._event_bus = ctx.event_bus
            self._interrupt_controller = ctx.interrupt_controller
        self._main_loop = asyncio.get_running_loop()
        self._bind_events()

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
            preview = text[:60] + "..." if len(text) > 60 else text
            logger.info(
                f"[qq] 私聊消息  user_id={user_id}  内容: {preview!r}  图片: {len(img_urls)}"
            )


            self._submit_to_main_loop(
                self._handle_private(
                    user_id,
                    text,
                    img_urls,
                    message_id=_qq_message_id(event),
                    event=event,
                ),
                track=True,
            )

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
            preview = text[:60] + "..." if len(text) > 60 else text
            logger.info(
                f"[qq] 群聊消息  group_id={group_id}  user_id={user_id}  内容: {preview!r}  图片: {len(img_urls)}"
            )

            self._submit_to_main_loop(
                self._handle_group(
                    group_id,
                    user_id,
                    text,
                    img_urls,
                    message_id=_qq_message_id(event),
                    event=event,
                ),
                track=True,
            )

        @cast(Any, self._bot.on_startup())
        async def _(_event) -> None:
            self._bot_loop = asyncio.get_running_loop()

        logger.info("[qq] 正在启动 NcatBot（首次运行需要扫码登录）...")
        self._api = await self._main_loop.run_in_executor(None, self._bot.run_backend)
        logger.info("[qq] NcatBot 已启动")

    def _bind_events(self) -> None:
        if self._event_bus is None or self._events_bound:
            return
        self._event_bus.on(TurnStarted, self._on_turn_started)
        self._event_bus.on(ToolCallStarted, self._on_tool_call_started)
        self._event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
        self._events_bound = True

    async def stop(self) -> None:
        if self._api:
            loop = asyncio.get_running_loop()
            bot_exit = getattr(self._bot, "exit", None)
            if callable(bot_exit):
                await loop.run_in_executor(None, bot_exit)
            logger.info("[qq] QQChannel 已停止")

    async def _on_turn_started(self, event: TurnStarted) -> None:
        if event.channel != _CHANNEL:
            return
        self._trace_states[event.session_key] = _QQTraceState(
            user_message=event.content,
        )

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        if event.channel != _CHANNEL:
            return
        state = self._trace_states.setdefault(event.session_key, _QQTraceState())
        state.tool_lines.append(
            _QQTraceLine(
                tool_name=event.tool_name,
                intent=_format_tool_intent(event.arguments),
                target=_format_tool_target(event.arguments),
            )
        )

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        if event.channel != _CHANNEL:
            return
        state = self._trace_states.setdefault(event.session_key, _QQTraceState())
        line = next(
            (
                item
                for item in reversed(state.tool_lines)
                if item.tool_name == event.tool_name and item.status == "started"
            ),
            None,
        )
        if line is None:
            line = _QQTraceLine(
                tool_name=event.tool_name,
                intent=_format_tool_intent(event.final_arguments or event.arguments),
                target=_format_tool_target(event.final_arguments or event.arguments),
            )
            state.tool_lines.append(line)
        line.status = "error" if event.status == "error" else "done"
        preview = str(event.result_preview or "").strip()
        if preview:
            line.result_preview = _summarize_tool_result_preview(
                event.tool_name,
                preview,
            )

    # ── 入站处理 ──────────────────────────────────────────────────────

    async def _handle_private(
        self,
        user_id: str,
        content: str,
        img_urls: list[str] | None = None,
        *,
        message_id: str | None = None,
        event: object | None = None,
    ) -> None:
        """私聊入站：chat_id = user_id"""
        await self._handle_private_v3(
            user_id,
            content,
            img_urls or [],
            message_id=message_id,
            event=event,
        )

    @staticmethod
    def _v3_timestamp(event: object | None) -> datetime:
        value = None if event is None else getattr(event, "time", None)
        if value is None and event is not None:
            value = getattr(event, "timestamp", None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        return datetime.now(timezone.utc)

    async def _download_v3_images(
        self,
        img_urls: list[str],
        *,
        ports: ChannelRuntimePorts,
    ) -> tuple[AttachmentRef, ...]:
        """Import provider images through the exact Core attachment port."""

        if not img_urls:
            return ()
        runtime = self._v3_inbound_runtime
        refs: list[AttachmentRef] = []
        total = 0
        for index, raw_url in enumerate(img_urls[:MAX_QQ_IMAGE_COUNT]):
            runtime.require_open(ports)
            if total >= MAX_QQ_TOTAL_IMAGE_BYTES:
                raise ValueError("QQ 图片总大小超过 v3 上限")
            url = html.unescape(raw_url)
            remaining = MAX_QQ_TOTAL_IMAGE_BYTES - total
            payload, media_type = await _read_qq_image(
                url,
                self._http_requester,
                max_bytes=min(MAX_QQ_IMAGE_BYTES, remaining),
            )
            if not media_type.startswith("image/"):
                raise ValueError(f"不支持的媒体类型: {media_type}")
            suffix = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/gif": ".gif",
                "image/webp": ".webp",
            }.get(media_type, ".img")
            refs.append(
                await runtime.import_bytes(
                    payload,
                    kind=AttachmentKind.IMAGE,
                    filename=f"qq_image_{index + 1}{suffix}",
                    media_type=media_type,
                    ports=ports,
                )
            )
            total += len(payload)
        if len(img_urls) > MAX_QQ_IMAGE_COUNT:
            logger.warning(
                "[qq] v3 入站图片已截断 count=%d limit=%d",
                len(img_urls),
                MAX_QQ_IMAGE_COUNT,
            )
        return tuple(refs)

    async def _admit_v3(
        self,
        *,
        message_id: str | None,
        event: object | None,
        sender: str,
        chat_id: str,
        content: str,
        metadata: dict[str, Any],
        attachments: tuple[AttachmentRef, ...],
        ports: ChannelRuntimePorts,
    ) -> None:
        if not message_id:
            raise ValueError("QQ v3 入站缺少 provider message_id")
        raw = RawInbound(
            message_id=message_id,
            provider_identity=sender,
            recipient=chat_id,
            message=ChannelInboundMessage(
                channel=self.name,
                sender=sender,
                chat_id=chat_id,
                content=_normalize_v3_content(content),
                timestamp=self._v3_timestamp(event),
                metadata=metadata,
                attachments=attachments,
            ),
        )
        accepted = await self._v3_inbound_runtime.admit(raw, ports=ports)
        if not accepted:
            logger.debug("[qq] v3 ingress closed or duplicate message_id=%s", message_id)

    async def _handle_private_v3(
        self,
        user_id: str,
        content: str,
        img_urls: list[str],
        *,
        message_id: str | None,
        event: object | None,
    ) -> None:
        """Admit one QQ private message after exact Core attachment import."""

        ports = await self._v3_inbound_runtime.wait_open()
        attachments = await self._download_v3_images(img_urls, ports=ports)
        await self._admit_v3(
            message_id=message_id,
            event=event,
            sender=user_id,
            chat_id=user_id,
            content=content,
            metadata={"chat_type": "private"},
            attachments=attachments,
            ports=ports,
        )

    async def _handle_group_v3(
        self,
        group_id: str,
        user_id: str,
        content: str,
        img_urls: list[str],
        *,
        message_id: str | None,
        event: object | None,
    ) -> None:
        """Admit one QQ group message after exact Core attachment import."""

        ports = await self._v3_inbound_runtime.wait_open()
        chat_id = f"{_GROUP_PREFIX}{group_id}"
        attachments = await self._download_v3_images(img_urls, ports=ports)
        await self._admit_v3(
            message_id=message_id,
            event=event,
            sender=user_id,
            chat_id=chat_id,
            content=content,
            metadata={
                "chat_type": "group",
                "group_id": group_id,
                "sender_id": user_id,
            },
            attachments=attachments,
            ports=ports,
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
        *,
        message_id: str | None = None,
        event: object | None = None,
    ) -> None:
        """群聊入站：chat_id = gqq:{group_id}，session 按群共享"""
        await self._handle_group_v3(
            group_id,
            user_id,
            content,
            img_urls or [],
            message_id=message_id,
            event=event,
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

    async def _send_private_trace(
        self,
        chat_id: str,
        session_key: str,
        msg: OutboundMessage,
    ) -> None:
        api = self._api
        if api is None:
            raise RuntimeError("QQChannel 尚未启动")
        trace = self._trace_states.get(session_key)
        if trace is None:
            return
        thinking_source = str(msg.thinking or "")
        thinking = _truncate_trace_text(thinking_source, _TRACE_THINKING_LIMIT)
        tool_text = _format_tool_trace_lines(trace.tool_lines)
        if not thinking and not trace.tool_lines:
            return
        from ncatbot.core import ForwardConstructor

        info = await self._run_on_bot_loop(api.get_login_info())
        actor_name = self._trace_actor_name()
        constructor = ForwardConstructor(str(info.user_id), actor_name)
        constructor.attach_text(
            f"【模型思路】\n{thinking or '（无 thinking）'}",
            nickname=actor_name,
        )
        constructor.attach_text(
            f"【工具链】\n{tool_text}",
            nickname=actor_name,
        )
        forward = constructor.to_forward()
        payload = forward.to_forward_dict()
        payload["source"] = f"{actor_name} 的过程记录"
        payload["summary"] = "查看本轮过程记录"
        payload["prompt"] = f"{actor_name} 过程记录"
        payload["news"] = [
            {"text": f"{actor_name}：【模型思路】"},
            {"text": f"{actor_name}：【工具链】"},
        ]
        await self._run_on_bot_loop(
            api.send_private_forward_msg(int(chat_id), **payload)
        )

    def _trace_actor_name(self) -> str:
        cached = self._trace_actor_name_cache
        if cached:
            return cached
        workspace = self._workspace
        if workspace is None:
            self._trace_actor_name_cache = _TRACE_DEFAULT_ACTOR
            return _TRACE_DEFAULT_ACTOR
        self_path = workspace / "memory" / "SELF.md"
        try:
            text = self_path.read_text(encoding="utf-8")
        except Exception:
            self._trace_actor_name_cache = _TRACE_DEFAULT_ACTOR
            return _TRACE_DEFAULT_ACTOR
        body_match = re.search(
            r"(?m)^-\s*我是\s+([A-Za-z][A-Za-z0-9_-]{1,40})\b",
            text,
        )
        if body_match:
            name = body_match.group(1).strip()
            if name:
                self._trace_actor_name_cache = name
                return name
        match = re.search(r"(?m)^#\s*(.+?)\s+的自我认知\s*$", text)
        if match:
            name = match.group(1).strip()
            if name:
                self._trace_actor_name_cache = name
                return name
        self._trace_actor_name_cache = _TRACE_DEFAULT_ACTOR
        return _TRACE_DEFAULT_ACTOR

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

    async def _deliver_message(self, message: ChannelMessage) -> DeliveryReceipt:
        """以 QQ 原生调用提交完整消息并报告部分送达。"""

        return await deliver_message_parts(
            message,
            send_text=self.send,
            send_file=self.send_file,
            send_image=self.send_image,
        )

    def build_v3_adapter(self, context: ChannelFactoryContext) -> ChannelAdapter:
        """Build a Core adapter over this already-started QQ provider owner."""

        return QQV3ChannelAdapter(self, context)

    def _attach_v3_inbound(self, ports: ChannelRuntimePorts) -> None:
        self._v3_inbound_runtime.attach(ports)

    def _open_v3_inbound(self) -> None:
        self._v3_inbound_runtime.open()

    def _close_v3_inbound(self) -> None:
        self._v3_inbound_runtime.close()

    async def _drain_v3_inbound(self) -> None:
        await self._v3_inbound_runtime.wait_quiescent()

    def _require_main_loop(self) -> asyncio.AbstractEventLoop:
        if self._main_loop is None:
            raise RuntimeError("QQ main loop 未就绪")
        return self._main_loop

    def _submit_to_main_loop(
        self,
        coro: Coroutine[object, object, None],
        *,
        track: bool,
    ) -> None:
        try:
            future = asyncio.run_coroutine_threadsafe(
                coro,
                self._require_main_loop(),
            )
        except BaseException:
            coro.close()
            raise
        if track:
            try:
                self._v3_inbound_runtime.track_future(future)
            except BaseException:
                future.cancel()
                raise

    async def _run_on_bot_loop(
        self, coro: Coroutine[object, object, object]
    ) -> object:
        if self._bot_loop is None:
            raise RuntimeError("QQ bot loop 未就绪")
        future = asyncio.run_coroutine_threadsafe(coro, self._bot_loop)
        return await asyncio.wrap_future(future)


class QQV3ChannelAdapter(NativeChannelDeliveryAdapter):
    """Deliver Core requests through an already-started QQChannel."""

    def __init__(self, channel: QQChannel, context: ChannelFactoryContext) -> None:
        self._channel = channel
        super().__init__(
            context,
            channel_name=channel.name,
            validate_recipient=self._validate_recipient,
            send_text=self._send_text,
            send_attachment=self._send_attachment,
        )

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        """Bind provider callbacks to one exact formal Core ingress."""

        self._channel._attach_v3_inbound(ports)

    def open_admission(self) -> None:
        """Release provider callbacks after formal snapshot publication."""

        self._channel._open_v3_inbound()

    def close_admission(self) -> None:
        """Stop provider callbacks before Host drains Core operations."""

        self._channel._close_v3_inbound()

    async def stop(self) -> StopReceipt:
        self._channel._close_v3_inbound()
        await self._channel._drain_v3_inbound()
        return await super().stop()

    def _validate_recipient(self, recipient: str) -> None:
        if recipient.startswith(_GROUP_PREFIX):
            group_id = recipient[len(_GROUP_PREFIX) :]
            if not group_id.isdigit() or int(group_id) <= 0:
                raise ValueError(f"QQ 群聊 recipient 无效: {recipient}")
            return
        if not recipient.isdigit() or int(recipient) <= 0:
            raise ValueError(f"QQ 私聊 recipient 无效: {recipient}")

    async def _send_text(self, request: ProviderDeliveryRequest) -> None:
        await self._channel.send(request.recipient, request.body)

    async def _send_attachment(
        self,
        request: ProviderDeliveryRequest,
        ref: AttachmentRef,
        payload: bytes,
    ) -> None:
        api = self._channel._api
        if api is None:
            raise RuntimeError("QQChannel 尚未启动")
        uri = "base64://" + base64.b64encode(payload).decode("ascii")
        recipient = request.recipient
        if ref.kind.value == "image":
            if recipient.startswith(_GROUP_PREFIX):
                group_id = int(recipient[len(_GROUP_PREFIX) :])
                await self._channel._run_on_bot_loop(
                    api.send_group_image(group_id, uri)
                )
            else:
                await self._channel._run_on_bot_loop(
                    api.send_private_image(int(recipient), uri)
                )
            return
        filename = ref.filename or ref.artifact_id
        if recipient.startswith(_GROUP_PREFIX):
            group_id = int(recipient[len(_GROUP_PREFIX) :])
            await self._channel._run_on_bot_loop(
                api.send_group_file(group_id, uri, filename)
            )
        else:
            await self._channel._run_on_bot_loop(
                api.send_private_file(int(recipient), uri, filename)
            )


def _is_local(path: str) -> bool:
    """判断是否为本地文件路径（非 URL、非 base64）"""
    return not path.startswith(("http://", "https://", "base64://", "file://"))


def _local_to_base64(path: str) -> str:
    """将本地文件编码为 NapCat 接受的 base64:// URI"""
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_QQ_IMAGE_BYTES + 1)
    if len(data) > MAX_QQ_IMAGE_BYTES:
        raise ValueError(f"QQ 图片不能超过 {MAX_QQ_IMAGE_BYTES} 字节")
    return "base64://" + base64.b64encode(data).decode()
