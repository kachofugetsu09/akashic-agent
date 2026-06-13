"""
飞书私聊渠道。

- lark_oapi 长连接接收私聊事件（文本 / 图片 / 文件 / 富文本 post）
- REST 发送：最终回复与主动推送走 interactive 卡片（lark_md 渲染 markdown）
- 流式 live 预览：订阅 TurnStarted/StreamDelta/ToolCall 事件，创建并 PATCH 一张卡片
- /stop 中断、白名单、身份索引、reply 引用上下文，与 Telegram 渠道对齐
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from agent.looping.interrupt import InterruptController
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from bus.queue import MessageBus
from infra.channels.base import AttachmentStore, MessageDeduper, SessionIdentityIndex
from infra.channels.contract import ChannelContext
from plugins.feishu.cards import (
    ToolLiveLine,
    build_live_card,
    build_markdown_card,
    build_summary_card,
    format_tool_intent,
    format_tool_target,
)

logger = logging.getLogger(__name__)

_CHANNEL = "feishu"
_SEEN_MSG_MAXSIZE = 500
_LIVE_STREAM_MIN_CHARS = 200
_LIVE_STREAM_MIN_INTERVAL_S = 2.0
_LIVE_MAX_FAILURES = 3
_LIVE_MAX_BACKOFF_S = 16.0
_WS_RECONNECT_DELAY_S = 5.0
_CARD_TEXT_LIMIT = 4000
_MESSAGE_MAX_ATTEMPTS = 4
_RETRY_BASE_DELAY_S = 0.5
_RETRY_MAX_DELAY_S = 8.0
# 飞书频控：HTTP 429 一定是限流；以下为常见频控业务码（尽力覆盖，主要仍依赖 429）。
_RATE_LIMIT_CODES = frozenset({99991400, 99991661, 230020, 230027, 11232})


@dataclass
class _TokenCache:
    token: str
    expires_at: float


# 飞书业务错误（code != 0），携带 code 以便判定频控。
class FeishuApiError(RuntimeError):
    def __init__(self, code: int, msg: str) -> None:
        super().__init__(f"飞书 API 失败 code={code} msg={msg}")
        self.code = code


def _is_rate_limited(err: Exception) -> bool:
    if isinstance(err, httpx.HTTPStatusError):
        return err.response.status_code == 429
    if isinstance(err, FeishuApiError):
        return err.code in _RATE_LIMIT_CODES
    return False


def _retry_after_seconds(err: Exception, default: float) -> float:
    if isinstance(err, httpx.HTTPStatusError):
        header = err.response.headers.get("Retry-After")
        if header:
            try:
                return max(float(header), default)
            except ValueError:
                return default
    return default


class FeishuChannel:
    name = _CHANNEL

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        allow_from: list[str] | None = None,
        domain: str = "https://open.feishu.cn",
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._allow_from = set(allow_from or [])
        self._domain = domain.rstrip("/")
        self._bus: MessageBus | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._interrupt_controller: InterruptController | None = None
        self._attachments: AttachmentStore | None = None
        self._identity_index: SessionIdentityIndex | None = None
        self._client = httpx.AsyncClient(timeout=30.0)
        self._token: _TokenCache | None = None
        self._message_deduper = MessageDeduper(_SEEN_MSG_MAXSIZE)
        self._outbound_bound = False
        self._events_bound = False
        # 长连接线程
        self._ws_client: Any | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._ws_stopped = threading.Event()
        # live 预览状态
        self._live_messages: dict[str, str] = {}
        self._reply_buffers: dict[str, str] = {}
        self._thinking_buffers: dict[str, str] = {}
        self._tool_lines: dict[str, list[ToolLiveLine]] = {}
        self._live_next_at: dict[str, float] = {}
        self._live_last_lengths: dict[str, int] = {}
        self._live_failures: dict[str, int] = {}
        self._live_interval: dict[str, float] = {}
        self._live_backoff_until: dict[str, float] = {}
        self._live_disabled: set[str] = set()
        self._live_locks: dict[str, asyncio.Lock] = {}
        self._live_tasks: set[asyncio.Task[None]] = set()
        self._live_tasks_by_session: dict[str, set[asyncio.Task[None]]] = {}

    async def start(self, ctx: ChannelContext) -> None:
        self._bus = ctx.bus
        self._loop = asyncio.get_running_loop()
        self._interrupt_controller = ctx.interrupt_controller
        self._attachments = ctx.attachment_store
        self._identity_index = SessionIdentityIndex(
            ctx.session_manager,
            channel=_CHANNEL,
            metadata_key="feishu_open_id",
        )
        _ = self._identity_index.rebuild()
        if not self._events_bound:
            ctx.event_bus.on(TurnStarted, self._on_turn_started)
            ctx.event_bus.on(StreamDeltaReady, self._on_stream_delta)
            ctx.event_bus.on(ToolCallStarted, self._on_tool_call_started)
            ctx.event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
            self._events_bound = True
        ctx.push_tool.register_channel(
            self.name,
            text=self.send,
            stream_text=self.send_stream,
            file=self.send_file,
            image=self.send_image,
        )
        if not self._outbound_bound:
            ctx.bus.subscribe_outbound(_CHANNEL, self._on_response)
            self._outbound_bound = True
        self._ws_stopped.clear()
        self._ws_thread = threading.Thread(
            target=self._run_ws_client,
            name="feishu-ws",
            daemon=True,
        )
        self._ws_thread.start()
        logger.info("[feishu] 飞书私聊渠道已启动")

    async def stop(self) -> None:
        self._ws_stopped.set()
        await self._disconnect_ws()
        await self._drain_live_tasks()
        await self._client.aclose()
        logger.info("[feishu] 飞书私聊渠道已停止")

    def _require_bus(self) -> MessageBus:
        if self._bus is None:
            raise RuntimeError("FeishuChannel 尚未启动")
        return self._bus

    # ── 长连接 ────────────────────────────────────────────────

    # 在独立线程跑 lark 长连接，外层包重连循环，避免单次异常后永久失联。
    def _run_ws_client(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._ws_loop = loop
        try:
            while not self._ws_stopped.is_set():
                try:
                    self._build_ws_client().start()
                except Exception as e:
                    logger.warning("[feishu] 长连接退出，准备重连: %s", e)
                if self._ws_stopped.is_set():
                    break
                time.sleep(_WS_RECONNECT_DELAY_S)
        finally:
            loop.close()

    def _build_ws_client(self) -> Any:
        from lark_oapi.core.enum import LogLevel
        from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
        from lark_oapi.ws import Client as WsClient

        handler = (
            EventDispatcherHandler.builder("", "")
            .register_p2_im_message_receive_v1(self._on_sdk_message)
            .build()
        )
        ws_client = WsClient(
            self._app_id,
            self._app_secret,
            log_level=LogLevel.INFO,
            event_handler=handler,
            domain=self._domain,
        )
        self._ws_client = ws_client
        return ws_client

    async def _disconnect_ws(self) -> None:
        ws_client = self._ws_client
        ws_loop = self._ws_loop
        if ws_client is None or ws_loop is None:
            return
        raw_disconnect = getattr(ws_client, "_disconnect", None)
        disconnect = cast(
            Callable[[], Coroutine[Any, Any, None]] | None,
            raw_disconnect if callable(raw_disconnect) else None,
        )
        if disconnect is None:
            return
        future = asyncio.run_coroutine_threadsafe(disconnect(), ws_loop)
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=5)
        except (TimeoutError, Exception) as e:
            logger.warning("[feishu] 长连接停止异常: %s", e)

    def _on_sdk_message(self, event: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        _ = asyncio.run_coroutine_threadsafe(self._handle_message_event(event), loop)

    # ── 入站 ──────────────────────────────────────────────────

    async def _handle_message_event(self, event: Any) -> None:
        data = getattr(event, "event", None)
        message = getattr(data, "message", None)
        sender = getattr(data, "sender", None)
        if message is None or sender is None:
            return
        if str(getattr(message, "chat_type", "") or "") != "p2p":
            return
        message_id = str(getattr(message, "message_id", "") or "")
        if message_id and self._message_deduper.seen(message_id):
            return
        sender_id = getattr(sender, "sender_id", None)
        open_id = str(getattr(sender_id, "open_id", "") or "")
        user_id = str(getattr(sender_id, "user_id", "") or "")
        union_id = str(getattr(sender_id, "union_id", "") or "")
        if self._allow_from and not ({open_id, user_id, union_id} & self._allow_from):
            logger.warning("[feishu] 拒绝未授权私聊用户 open_id=%s user_id=%s", open_id, user_id)
            return
        chat_id = str(getattr(message, "chat_id", "") or "")
        if not chat_id:
            return
        await self._ingest_message(message, message_id, chat_id, open_id, user_id, union_id)

    async def _ingest_message(
        self,
        message: Any,
        message_id: str,
        chat_id: str,
        open_id: str,
        user_id: str,
        union_id: str,
    ) -> None:
        text, media = await self._extract_message_payload(message, message_id)
        sender = open_id or user_id or union_id
        if text == "/stop":
            await self._handle_stop(chat_id, sender)
            return
        if not text and not media:
            return
        inbound_text, reply_meta = await self._merge_reply_context(message, text)
        if self._identity_index is not None and open_id:
            await self._identity_index.remember(open_id, chat_id)
        await self._require_bus().publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=sender,
                chat_id=chat_id,
                content=inbound_text,
                media=media,
                metadata={
                    "chat_type": "private",
                    "message_id": message_id,
                    "open_id": open_id,
                    "user_id": user_id,
                    "union_id": union_id,
                    **reply_meta,
                },
            )
        )

    # 按消息类型解析文本与媒体；图片/文件会下载落盘到 AttachmentStore。
    async def _extract_message_payload(
        self,
        message: Any,
        message_id: str,
    ) -> tuple[str, list[str]]:
        msg_type = str(getattr(message, "message_type", "") or "")
        content_raw = str(getattr(message, "content", "") or "")
        if msg_type == "text":
            return _extract_text(content_raw), []
        if msg_type == "image":
            image_key = _extract_key(content_raw, "image_key")
            path = await self._download_resource(message_id, image_key, "image", ".jpg")
            return "[图片]", [path] if path else []
        if msg_type == "file":
            file_key = _extract_key(content_raw, "file_key")
            file_name = _extract_key(content_raw, "file_name") or "file"
            suffix = "." + file_name.rsplit(".", 1)[-1] if "." in file_name else ""
            path = await self._download_resource(message_id, file_key, "file", suffix)
            return f"[文件: {file_name}]", [path] if path else []
        if msg_type == "post":
            text, image_keys = _extract_post(content_raw)
            media: list[str] = []
            for key in image_keys:
                path = await self._download_resource(message_id, key, "image", ".jpg")
                if path:
                    media.append(path)
            return text or "[富文本]", media
        logger.debug("[feishu] 暂不支持的消息类型 msg_type=%s", msg_type)
        return "", []

    # 若消息回复了历史消息，拉取父消息文本并合并入站，避免 agent 丢失引用。
    async def _merge_reply_context(
        self,
        message: Any,
        text: str,
    ) -> tuple[str, dict[str, str]]:
        parent_id = str(getattr(message, "parent_id", "") or "")
        if not parent_id:
            return text, {}
        reply_text = await self._fetch_message_text(parent_id)
        if not reply_text:
            return text, {"reply_to_message_id": parent_id}
        merged = (
            "【你正在回复一条历史消息】\n"
            f"被回复消息：\n{reply_text}\n\n"
            "【你当前新消息】\n"
            f"{text}"
        ).strip()
        return merged, {"reply_to_message_id": parent_id}

    async def _handle_stop(self, chat_id: str, sender: str) -> None:
        if self._interrupt_controller is None:
            await self.send(chat_id, "当前未启用中断功能。")
            return
        result = self._interrupt_controller.request_interrupt(
            session_key=f"{_CHANNEL}:{chat_id}",
            sender=sender,
            command="/stop",
        )
        await self.send(chat_id, result.message)

    # ── live 预览（卡片）────────────────────────────────────────

    async def _on_turn_started(self, event: TurnStarted) -> None:
        if event.channel != _CHANNEL:
            return
        await self._cancel_live_tasks(event.session_key)
        self._clear_live_session(event.session_key)

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        if event.channel != _CHANNEL:
            return
        if not event.content_delta and not event.thinking_delta:
            return
        if event.content_delta:
            self._reply_buffers[event.session_key] = (
                self._reply_buffers.get(event.session_key, "") + event.content_delta
            )
        if event.thinking_delta:
            self._thinking_buffers[event.session_key] = (
                self._thinking_buffers.get(event.session_key, "") + event.thinking_delta
            )
        live_len = len(self._reply_buffers.get(event.session_key, "")) + len(
            self._thinking_buffers.get(event.session_key, "")
        )
        last_len = self._live_last_lengths.get(event.session_key, 0)
        now = asyncio.get_running_loop().time()
        next_at = self._live_next_at.get(event.session_key, 0.0)
        if now < next_at and live_len - last_len < _LIVE_STREAM_MIN_CHARS:
            return
        self._live_next_at[event.session_key] = now + _LIVE_STREAM_MIN_INTERVAL_S
        self._live_last_lengths[event.session_key] = live_len
        self._start_live_task(
            event.session_key,
            self._sync_live_card(event.session_key, event.chat_id),
        )

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        if event.channel != _CHANNEL:
            return
        lines = self._tool_lines.setdefault(event.session_key, [])
        lines.append(
            ToolLiveLine(
                call_id=event.call_id,
                tool_name=event.tool_name,
                intent=format_tool_intent(event.arguments),
                target=format_tool_target(event.arguments),
            )
        )
        self._start_live_task(
            event.session_key,
            self._sync_live_card(event.session_key, event.chat_id),
        )

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        if event.channel != _CHANNEL:
            return
        lines = self._tool_lines.setdefault(event.session_key, [])
        line = next((item for item in lines if item.call_id == event.call_id), None)
        if line is None:
            line = ToolLiveLine(
                call_id=event.call_id,
                tool_name=event.tool_name,
                intent=format_tool_intent(event.final_arguments or event.arguments),
                target=format_tool_target(event.final_arguments or event.arguments),
            )
            lines.append(line)
        line.status = "error" if event.status == "error" else "done"
        self._start_live_task(
            event.session_key,
            self._sync_live_card(event.session_key, event.chat_id),
        )

    # 创建或 PATCH live 卡片；失败累计后禁用该会话的 live，回退一次性发送。
    async def _sync_live_card(self, session_key: str, chat_id: str) -> None:
        if session_key in self._live_disabled:
            return
        if asyncio.get_running_loop().time() < self._live_backoff_until.get(session_key, 0.0):
            return
        card = build_live_card(
            self._thinking_buffers.get(session_key, ""),
            self._tool_lines.get(session_key, []),
            self._reply_buffers.get(session_key, ""),
        )
        lock = self._live_locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            if session_key in self._live_disabled:
                return
            try:
                await self._upsert_live_card(session_key, chat_id, card)
            except Exception as e:
                self._record_live_failure(session_key, e)

    async def _upsert_live_card(self, session_key: str, chat_id: str, card: str) -> None:
        message_id = self._live_messages.get(session_key)
        if message_id is None:
            data = await self._post_message_once(chat_id, "interactive", card)
            new_id = str(data.get("message_id") or "")
            if new_id:
                self._live_messages[session_key] = new_id
        else:
            _ = await self._patch_message_once(message_id, card)
        # 成功即重置失败计数与自适应间隔
        self._live_failures[session_key] = 0
        self._live_interval[session_key] = _LIVE_STREAM_MIN_INTERVAL_S

    # live 刷新失败处理：频控走自适应退避降频（间隔翻倍），其余错误累计后禁用 live 回退。
    def _record_live_failure(self, session_key: str, err: Exception) -> None:
        if _is_rate_limited(err):
            interval = min(
                self._live_interval.get(session_key, _LIVE_STREAM_MIN_INTERVAL_S) * 2,
                _LIVE_MAX_BACKOFF_S,
            )
            self._live_interval[session_key] = interval
            self._live_backoff_until[session_key] = asyncio.get_running_loop().time() + interval
            logger.warning("[feishu] live 命中频控，退避降频 session=%s 下次间隔=%.1fs", session_key, interval)
            return
        failures = self._live_failures.get(session_key, 0) + 1
        self._live_failures[session_key] = failures
        if failures >= _LIVE_MAX_FAILURES:
            self._live_disabled.add(session_key)
        logger.warning(
            "[feishu] live 卡片刷新失败 session=%s failures=%d disabled=%s err=%s",
            session_key,
            failures,
            session_key in self._live_disabled,
            err,
        )

    # ── 出站 ──────────────────────────────────────────────────

    async def _on_response(self, msg: OutboundMessage) -> None:
        session_key = f"{_CHANNEL}:{msg.chat_id}"
        content = msg.content.strip()
        thinking = self._final_thinking_text(session_key, msg.thinking)
        tool_lines = self._tool_lines.get(session_key, [])
        if session_key in self._live_messages:
            await self._cancel_live_tasks(session_key)
        # 1. 把实时预览卡原地定格为"过程"卡（思考折叠 + 工具），不撤回
        await self._freeze_live_card(session_key, msg.chat_id, thinking, tool_lines)
        # 2. 最终结果单独发一条（超长分块、失败降级纯文本）
        if content:
            for chunk in _split_markdown(content, _CARD_TEXT_LIMIT):
                await self._post_card_or_text(msg.chat_id, build_markdown_card(chunk), chunk)
        self._clear_live_session(session_key)
        for image in (msg.media or []):
            try:
                await self.send_image(msg.chat_id, image)
            except Exception as e:
                logger.warning("[feishu] 媒体图片发送失败 chat_id=%s path=%s err=%s", msg.chat_id, image, e)

    # 把实时预览卡 PATCH 成过程卡（思考折叠 + 工具时间线）；无预览卡但有过程则新发一张。不撤回。
    async def _freeze_live_card(
        self,
        session_key: str,
        chat_id: str,
        thinking: str,
        tool_lines: list[ToolLiveLine],
    ) -> None:
        if not thinking.strip() and not tool_lines:
            return
        summary = build_summary_card(thinking, tool_lines)
        message_id = self._live_messages.get(session_key)
        if message_id is not None:
            try:
                _ = await self._patch_message_once(message_id, summary)
                return
            except Exception as e:
                logger.warning("[feishu] 过程卡定格失败，改为新发: %s", e)
        await self._post_card_or_text(chat_id, summary, thinking)

    def _final_thinking_text(self, session_key: str, thinking: str | None) -> str:
        streamed = self._thinking_buffers.get(session_key, "").strip()
        final = (thinking or "").strip()
        if streamed and final:
            if final in streamed:
                return streamed
            if streamed in final:
                return final
            return f"{streamed}\n\n{final}"
        return streamed or final

    # 文本消息（供 MessagePushTool 调用）：走卡片渲染 markdown，超长分块、失败降级纯文本。
    async def send(self, chat_id: str, text: str) -> None:
        if not text.strip():
            return
        for chunk in _split_markdown(text, _CARD_TEXT_LIMIT):
            await self._post_card_or_text(chat_id, build_markdown_card(chunk), chunk)

    # 发送卡片；渲染/大小异常时降级为 msg_type text，保证消息不丢（对齐 Telegram 降级哲学）。
    async def _post_card_or_text(self, chat_id: str, card: str, fallback_text: str) -> None:
        try:
            _ = await self._post_message(chat_id, "interactive", card)
        except Exception as e:
            logger.warning("[feishu] 卡片发送失败，降级纯文本: %s", e)
            if fallback_text.strip():
                content = json.dumps({"text": fallback_text}, ensure_ascii=False)
                _ = await self._post_message(chat_id, "text", content)

    async def send_stream(self, chat_id: str, text: str) -> None:
        await self.send(chat_id, text)

    async def send_image(self, chat_id: str, image: str) -> None:
        if image.startswith(("http://", "https://")):
            resp = await self._client.get(image)
            _ = resp.raise_for_status()
            data = resp.content
        else:
            data = Path(image).read_bytes()
        image_key = await self._upload_image(data)
        content = json.dumps({"image_key": image_key}, ensure_ascii=False)
        _ = await self._post_message(chat_id, "image", content)

    async def send_file(
        self,
        chat_id: str,
        file_path: str,
        name: str | None = None,
        caption: str | None = None,
    ) -> None:
        path = Path(file_path)
        file_name = name or path.name
        file_key = await self._upload_file(path.read_bytes(), file_name)
        content = json.dumps({"file_key": file_key}, ensure_ascii=False)
        _ = await self._post_message(chat_id, "file", content)
        if caption and caption.strip():
            await self.send(chat_id, caption)

    # ── live 任务管理 ──────────────────────────────────────────

    def _start_live_task(self, session_key: str, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._live_tasks.add(task)
        self._live_tasks_by_session.setdefault(session_key, set()).add(task)
        task.add_done_callback(lambda done: self._on_live_task_done(session_key, done))

    def _on_live_task_done(self, session_key: str, task: asyncio.Task[None]) -> None:
        self._live_tasks.discard(task)
        tasks = self._live_tasks_by_session.get(session_key)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                _ = self._live_tasks_by_session.pop(session_key, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug("[feishu] live 任务异常: %s", exc)

    async def _cancel_live_tasks(self, session_key: str) -> None:
        tasks = list(self._live_tasks_by_session.get(session_key, set()))
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    async def _drain_live_tasks(self) -> None:
        tasks = [task for task in self._live_tasks if not task.done()]
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    def _clear_live_session(self, session_key: str) -> None:
        _ = self._live_messages.pop(session_key, None)
        _ = self._reply_buffers.pop(session_key, None)
        _ = self._thinking_buffers.pop(session_key, None)
        _ = self._tool_lines.pop(session_key, None)
        _ = self._live_next_at.pop(session_key, None)
        _ = self._live_last_lengths.pop(session_key, None)
        _ = self._live_failures.pop(session_key, None)
        _ = self._live_interval.pop(session_key, None)
        _ = self._live_backoff_until.pop(session_key, None)
        self._live_disabled.discard(session_key)
        _ = self._live_locks.pop(session_key, None)

    # ── REST ──────────────────────────────────────────────────

    def _resolve_receive(self, chat_id: str) -> tuple[str, str]:
        value = chat_id.strip()
        if value.startswith(f"{_CHANNEL}:"):
            value = value[len(_CHANNEL) + 1:]
        if value.startswith("oc_"):
            return value, "chat_id"
        if self._identity_index is not None:
            resolved = self._identity_index.resolve(value)
            if resolved:
                return resolved, "chat_id"
        if value.startswith("ou_"):
            return value, "open_id"
        if value.startswith("on_"):
            return value, "union_id"
        return value, "chat_id"

    # 单次发送，无重试：供 live 卡片使用，撞频控时丢帧保实时（对齐 Telegram live max_attempts=1）。
    async def _post_message_once(self, chat_id: str, msg_type: str, content: str) -> dict[str, Any]:
        receive_id, receive_id_type = self._resolve_receive(chat_id)
        token = await self._get_access_token()
        resp = await self._client.post(
            f"{self._domain}/open-apis/im/v1/messages",
            params={"receive_id_type": receive_id_type},
            headers={"Authorization": f"Bearer {token}"},
            json={"receive_id": receive_id, "msg_type": msg_type, "content": content},
        )
        return self._check_response(resp)

    async def _patch_message_once(self, message_id: str, content: str) -> dict[str, Any]:
        token = await self._get_access_token()
        resp = await self._client.patch(
            f"{self._domain}/open-apis/im/v1/messages/{message_id}",
            headers={"Authorization": f"Bearer {token}"},
            json={"content": content},
        )
        return self._check_response(resp)

    async def _post_message(self, chat_id: str, msg_type: str, content: str) -> dict[str, Any]:
        return await self._with_rate_limit_retry(
            lambda: self._post_message_once(chat_id, msg_type, content),
            label="post_message",
        )

    async def _fetch_message_text(self, message_id: str) -> str:
        try:
            token = await self._get_access_token()
            resp = await self._client.get(
                f"{self._domain}/open-apis/im/v1/messages/{message_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            data = self._check_response(resp)
        except Exception as e:
            logger.debug("[feishu] 拉取父消息失败 id=%s err=%s", message_id, e)
            return ""
        items = data.get("items")
        if not isinstance(items, list) or not items:
            return ""
        first = cast(dict[str, Any], items[0]) if isinstance(items[0], dict) else {}
        body = first.get("body")
        if not isinstance(body, dict):
            return ""
        return _extract_text(str(cast(dict[str, Any], body).get("content") or ""))

    async def _download_resource(
        self,
        message_id: str,
        file_key: str,
        resource_type: str,
        suffix: str,
    ) -> str | None:
        if not file_key or self._attachments is None:
            return None
        try:
            token = await self._get_access_token()
            resp = await self._client.get(
                f"{self._domain}/open-apis/im/v1/messages/{message_id}/resources/{file_key}",
                params={"type": resource_type},
                headers={"Authorization": f"Bearer {token}"},
            )
            _ = resp.raise_for_status()
        except Exception as e:
            logger.warning("[feishu] 资源下载失败 key=%s err=%s", file_key, e)
            return None
        path = self._attachments.write_bytes(
            resp.content,
            prefix=f"feishu_{resource_type}_",
            suffix=suffix,
        )
        return str(path)

    async def _upload_image(self, data: bytes) -> str:
        token = await self._get_access_token()
        resp = await self._client.post(
            f"{self._domain}/open-apis/im/v1/images",
            headers={"Authorization": f"Bearer {token}"},
            data={"image_type": "message"},
            files={"image": ("image", data)},
        )
        payload = self._check_response(resp)
        return str(payload.get("image_key") or "")

    async def _upload_file(self, data: bytes, file_name: str) -> str:
        token = await self._get_access_token()
        resp = await self._client.post(
            f"{self._domain}/open-apis/im/v1/files",
            headers={"Authorization": f"Bearer {token}"},
            data={"file_type": "stream", "file_name": file_name},
            files={"file": (file_name, data)},
        )
        payload = self._check_response(resp)
        return str(payload.get("file_key") or "")

    def _check_response(self, resp: httpx.Response) -> dict[str, Any]:
        _ = resp.raise_for_status()
        payload = cast(dict[str, Any], resp.json())
        code = int(payload.get("code") or 0)
        if code != 0:
            raise FeishuApiError(code, str(payload.get("msg") or ""))
        data = payload.get("data")
        return cast(dict[str, Any], data) if isinstance(data, dict) else {}

    # 带频控退避重试的消息发送（最终回复 / 主动推送，不能丢）。对齐 Telegram 的 RetryAfter 处理。
    async def _with_rate_limit_retry(
        self,
        factory: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
        *,
        label: str,
    ) -> dict[str, Any]:
        delay = _RETRY_BASE_DELAY_S
        for attempt in range(1, _MESSAGE_MAX_ATTEMPTS + 1):
            try:
                return await factory()
            except (httpx.HTTPStatusError, FeishuApiError) as e:
                if attempt >= _MESSAGE_MAX_ATTEMPTS or not _is_rate_limited(e):
                    raise
                wait = _retry_after_seconds(e, delay)
                logger.warning(
                    "[feishu] %s 命中频控，退避重试 attempt=%d/%d delay=%.1fs",
                    label,
                    attempt,
                    _MESSAGE_MAX_ATTEMPTS,
                    wait,
                )
                await asyncio.sleep(wait)
                delay = min(delay * 2, _RETRY_MAX_DELAY_S)
        raise RuntimeError(f"{label} 重试耗尽")

    async def _get_access_token(self) -> str:
        if self._token and self._token.expires_at > time.time() + 60:
            return self._token.token
        resp = await self._client.post(
            f"{self._domain}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        _ = resp.raise_for_status()
        payload = cast(dict[str, Any], resp.json())
        code = int(payload.get("code") or 0)
        if code != 0:
            raise RuntimeError(f"飞书 token 获取失败 code={code} msg={payload.get('msg')}")
        token = str(payload.get("tenant_access_token") or "")
        expire = int(payload.get("expire") or 0)
        self._token = _TokenCache(token=token, expires_at=time.time() + expire)
        return token


def _extract_text(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content.strip()
    if not isinstance(parsed, dict):
        return content.strip()
    return str(cast(dict[str, object], parsed).get("text") or "").strip()


# 按行切分超长文本，单段不超过 limit（对齐 Telegram 的分块发送，避免超卡片大小上限）。
def _split_markdown(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks


def _extract_key(content: str, key: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    return str(cast(dict[str, object], parsed).get(key) or "").strip()


# 解析富文本 post：拼接文本段，收集内嵌图片 image_key。
def _extract_post(content: str) -> tuple[str, list[str]]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content.strip(), []
    if not isinstance(parsed, dict):
        return "", []
    body = cast(dict[str, Any], parsed)
    if "content" not in body:
        for value in body.values():
            if isinstance(value, dict) and "content" in value:
                body = cast(dict[str, Any], value)
                break
    texts: list[str] = []
    images: list[str] = []
    title = str(body.get("title") or "").strip()
    if title:
        texts.append(title)
    paragraphs = body.get("content")
    if isinstance(paragraphs, list):
        for paragraph in cast(list[Any], paragraphs):
            line = _extract_post_line(paragraph, images)
            if line:
                texts.append(line)
    return "\n".join(texts).strip(), images


def _extract_post_line(paragraph: Any, images: list[str]) -> str:
    if not isinstance(paragraph, list):
        return ""
    parts: list[str] = []
    for segment in cast(list[Any], paragraph):
        if not isinstance(segment, dict):
            continue
        seg = cast(dict[str, Any], segment)
        tag = str(seg.get("tag") or "")
        if tag == "text":
            parts.append(str(seg.get("text") or ""))
        elif tag in ("a", "link"):
            parts.append(str(seg.get("text") or seg.get("href") or ""))
        elif tag == "at":
            parts.append("@" + str(seg.get("user_name") or seg.get("user_id") or ""))
        elif tag == "img":
            key = str(seg.get("image_key") or "")
            if key:
                images.append(key)
    return "".join(parts)
