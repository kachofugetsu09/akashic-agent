from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, cast

import httpx

from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from infra.channels.base import MessageDeduper
from infra.channels.contract import ChannelContext

logger = logging.getLogger(__name__)

_CHANNEL = "feishu"
_SEEN_MSG_MAXSIZE = 500


@dataclass
class _TokenCache:
    token: str
    expires_at: float


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
        self._client = httpx.AsyncClient(timeout=30.0)
        self._token: _TokenCache | None = None
        self._ws_client: Any | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._message_deduper = MessageDeduper(_SEEN_MSG_MAXSIZE)
        self._outbound_bound = False

    async def start(self, ctx: ChannelContext) -> None:
        self._bus = ctx.bus
        self._loop = asyncio.get_running_loop()
        ctx.push_tool.register_channel(self.name, text=self.send)
        if not self._outbound_bound:
            ctx.bus.subscribe_outbound(_CHANNEL, self._on_response)
            self._outbound_bound = True
        self._ws_thread = threading.Thread(
            target=self._run_ws_client,
            name="feishu-ws",
            daemon=True,
        )
        self._ws_thread.start()
        logger.info("[feishu] 飞书私聊渠道已启动")

    async def stop(self) -> None:
        ws_client = self._ws_client
        ws_loop = self._ws_loop
        if ws_client is not None and ws_loop is not None:
            raw_disconnect = getattr(ws_client, "_disconnect", None)
            disconnect = cast(
                Callable[[], Coroutine[Any, Any, None]] | None,
                raw_disconnect if callable(raw_disconnect) else None,
            )
            if disconnect is not None:
                future = asyncio.run_coroutine_threadsafe(disconnect(), ws_loop)
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=5)
                except TimeoutError:
                    logger.warning("[feishu] 长连接停止超时")
        await self._client.aclose()
        logger.info("[feishu] 飞书私聊渠道已停止")

    async def send(self, chat_id: str, text: str) -> None:
        token = await self._get_access_token()
        resp = await self._client.post(
            f"{self._domain}/open-apis/im/v1/messages",
            params={"receive_id_type": "chat_id"},
            headers={"Authorization": f"Bearer {token}"},
            json={
                "receive_id": chat_id,
                "msg_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
            },
        )
        _ = resp.raise_for_status()
        payload = resp.json()
        code = int(payload.get("code") or 0)
        if code != 0:
            msg = str(payload.get("msg") or "")
            raise RuntimeError(f"飞书发送失败 code={code} msg={msg}")

    async def _on_response(self, msg: OutboundMessage) -> None:
        content = msg.content.strip()
        if content:
            await self.send(msg.chat_id, content)

    def _run_ws_client(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._ws_loop = loop
        try:
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
            ws_client.start()
        except Exception as e:
            logger.warning("[feishu] 长连接退出: %s", e)
        finally:
            loop.close()

    def _on_sdk_message(self, event: Any) -> None:
        loop = self._loop
        if loop is None:
            return
        _ = asyncio.run_coroutine_threadsafe(self._handle_message_event(event), loop)

    async def _handle_message_event(self, event: Any) -> None:
        data = getattr(event, "event", None)
        message = getattr(data, "message", None)
        sender = getattr(data, "sender", None)
        if message is None or sender is None:
            return
        if str(getattr(message, "chat_type", "") or "") != "p2p":
            return
        if str(getattr(message, "message_type", "") or "") != "text":
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
        content = _extract_text(str(getattr(message, "content", "") or ""))
        if not chat_id or not content:
            return
        await self._require_bus().publish_inbound(
            InboundMessage(
                channel=_CHANNEL,
                sender=open_id or user_id or union_id,
                chat_id=chat_id,
                content=content,
                metadata={
                    "chat_type": "private",
                    "message_id": message_id,
                    "open_id": open_id,
                    "user_id": user_id,
                    "union_id": union_id,
                },
            )
        )

    def _require_bus(self) -> MessageBus:
        if self._bus is None:
            raise RuntimeError("FeishuChannel 尚未启动")
        return self._bus

    async def _get_access_token(self) -> str:
        if self._token and self._token.expires_at > time.time() + 60:
            return self._token.token
        resp = await self._client.post(
            f"{self._domain}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": self._app_id, "app_secret": self._app_secret},
        )
        _ = resp.raise_for_status()
        payload = resp.json()
        code = int(payload.get("code") or 0)
        if code != 0:
            msg = str(payload.get("msg") or "")
            raise RuntimeError(f"飞书 token 获取失败 code={code} msg={msg}")
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
