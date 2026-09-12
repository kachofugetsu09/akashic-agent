from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import logging
import re
import threading
from pathlib import Path
from typing import Any, cast

import httpx

from agent.plugin_composition import (
    AttachmentKind,
    AttachmentRef,
    ChannelAdapter,
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelReady,
    DeliveryStatus,
    InboundIdentity,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    RawInbound,
    StopReceipt,
)
from agent.plugin_composition.channels import ChannelRuntimePorts

from .config import QQChannelConfig, QQGroupConfig
from .group_filter import DefaultGroupFilter, strip_at_segments

logger = logging.getLogger(__name__)

_CHANNEL = "qq"
_GROUP_PREFIX = "gqq:"
_NCATBOT_DIR = Path.home() / ".akashic" / "ncatbot"
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_COUNT = 10
_CQ_IMAGE_RE = re.compile(r"\[CQ:image[^\]]*?(?:,|\b)url=([^,\]]+)[^\]]*\]")


def build_qq_channel(context: ChannelFactoryContext) -> ChannelAdapter:
    """Build a side-effect-free QQ adapter; NcatBot starts only in start()."""

    if not isinstance(context, ChannelFactoryContext):
        raise TypeError("QQ channel factory 只接受 ChannelFactoryContext")
    if context.ingress is None or context.identity is None:
        raise RuntimeError("QQ channel 需要 Core ingress 和 identity")
    if context.attachment_import is None:
        raise RuntimeError("QQ channel 需要 Core attachment import")
    return QQChannelAdapter(context)


class QQChannelAdapter:
    """Own NapCat callbacks and API calls while Core owns admission and state."""

    name = _CHANNEL
    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    def __init__(self, context: ChannelFactoryContext) -> None:
        self._context = context
        self._binding_token = context.binding_token
        self._ingress = context.ingress
        self._identity = context.identity
        self._attachment_import = context.attachment_import
        self._attachment_read = context.attachment_read
        self._config = QQChannelConfig.model_validate(dict(context.config))
        self._groups: dict[str, QQGroupConfig] = {
            item.group_id: item for item in self._config.groups
        }
        self._runtime: ChannelRuntimePorts | None = None
        self._admission_open = False
        self._stopping = False
        self._stop_task: asyncio.Task[StopReceipt] | None = None
        self._started = False
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._bot_loop: asyncio.AbstractEventLoop | None = None
        self._bot: Any | None = None
        self._api: Any | None = None
        self._backend_task: asyncio.Task[Any] | None = None
        self._connection_task: asyncio.Task[Any] | None = None
        self._backend_thread: threading.Thread | None = None
        self._inbound_futures: set[Any] = set()
        self._http: httpx.AsyncClient | None = None
        self._group_filter = DefaultGroupFilter(self._config.bot_uin)

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        """Bind one exact generation before provider callbacks are released."""

        if self._runtime is not None:
            raise RuntimeError("QQ runtime 不能重复绑定")
        if ports.ingress is None or ports.identity is None:
            raise RuntimeError("QQ runtime 缺少 ingress 或 identity")
        self._runtime = ports

    def open_admission(self) -> None:
        """Release provider callbacks only after snapshot publication."""

        if self._stopping or self._runtime is None:
            raise RuntimeError("QQ channel 尚未准备好 admission")
        self._admission_open = True

    def close_admission(self) -> None:
        """Reject callbacks before the Host drains this generation."""

        self._admission_open = False

    async def start(self) -> ChannelReady:
        """Configure and schedule NcatBot without opening Core admission."""

        if self._started or self._stopping:
            raise RuntimeError("QQ channel 已启动或正在停止")
        self._main_loop = asyncio.get_running_loop()
        self._http = httpx.AsyncClient(timeout=15.0, follow_redirects=True)
        try:
            from ncatbot.core import BotClient
            from ncatbot.utils import ncatbot_config

            _configure_ncatbot(ncatbot_config, self._config)
            self._bot = BotClient()
            self._bind_backend_lifetime()
            self._register_callbacks()
            self._backend_task = asyncio.create_task(
                self._run_backend(),
                name="qq-channel-backend",
            )
            # run_backend 只有收到 provider startup 后才返回 API。
            self._api = await asyncio.shield(self._backend_task)
            if self._api is None or self._connection_task is None or self._backend_thread is None:
                raise RuntimeError("QQ backend 没有发布实际连接与 API")
            self._started = True
            return ChannelReady(
                binding_token=self._binding_token,
                subscriptions=("qq.ncatbot",),
                admission_open=False,
            )
        except BaseException:
            await self.stop()
            raise

    async def _run_backend(self) -> object:
        bot = self._bot
        if bot is None:
            raise RuntimeError("QQ BotClient 未创建")
        return await asyncio.to_thread(bot.run_backend)

    def _bind_backend_lifetime(self) -> None:
        """记录 SDK 真正的连接任务和线程；事件回调本身是短命子任务。"""
        bot = self._bot
        assert bot is not None
        start = bot.start
        connect = bot.adapter.connect_websocket

        def start_owned() -> object:
            self._backend_thread = threading.current_thread()
            return start()

        async def connect_owned() -> object:
            self._bot_loop = asyncio.get_running_loop()
            self._connection_task = asyncio.current_task()
            return await connect()

        bot.start = start_owned
        bot.adapter.connect_websocket = connect_owned

    def _register_callbacks(self) -> None:
        bot = self._bot
        if bot is None:
            raise RuntimeError("QQ BotClient 未创建")

        @cast(Any, bot.on_private_message())
        async def _private(event: object) -> None:
            if self._bot_loop is None:
                self._bot_loop = asyncio.get_running_loop()
            user_id = str(getattr(event, "user_id", ""))
            if self._config.allow_from and user_id not in self._config.allow_from:
                return
            self._submit_to_main_loop(self._handle_private(event, user_id), track=True)

        @cast(Any, bot.on_group_message())
        async def _group(event: object) -> None:
            if self._bot_loop is None:
                self._bot_loop = asyncio.get_running_loop()
            group_id = str(getattr(event, "group_id", ""))
            config = self._groups.get(group_id)
            if config is None or not await self._group_filter.should_process(event, config):
                return
            user_id = str(getattr(event, "user_id", ""))
            self._submit_to_main_loop(self._handle_group(event, group_id, user_id), track=True)

    def _submit_to_main_loop(self, coroutine: Any, *, track: bool) -> None:
        loop = self._main_loop
        if loop is None:
            coroutine.close()
            raise RuntimeError("QQ main loop 未就绪")
        if loop is self._bot_loop:
            task = asyncio.create_task(coroutine)
            if track:
                self._inbound_futures.add(task)
                task.add_done_callback(self._inbound_futures.discard)
            return
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        if track:
            self._inbound_futures.add(future)
            future.add_done_callback(self._inbound_futures.discard)

    async def _handle_private(self, event: object, user_id: str) -> None:
        raw_message = str(getattr(event, "raw_message", "") or "")
        text, image_urls = _extract_cq_images(raw_message)
        await self._admit_event(
            event,
            sender=user_id,
            chat_id=user_id,
            content=text,
            image_urls=image_urls,
        )

    async def _handle_group(self, event: object, group_id: str, user_id: str) -> None:
        raw_message = strip_at_segments(str(getattr(event, "raw_message", "") or ""))
        text, image_urls = _extract_cq_images(raw_message)
        await self._admit_event(
            event,
            sender=user_id,
            chat_id=_GROUP_PREFIX + group_id,
            content=text,
            image_urls=image_urls,
        )

    async def _admit_event(
        self,
        event: object,
        *,
        sender: str,
        chat_id: str,
        content: str,
        image_urls: list[str],
    ) -> None:
        if not self._admission_open:
            return
        message_id = _message_id(event)
        if message_id is None:
            logger.warning("[qq] 丢弃缺少 provider message id 的事件")
            return
        attachments = await self._download_images(image_urls)
        if not content and not attachments:
            return
        raw = RawInbound(
            message_id=message_id,
            provider_identity=sender,
            recipient=chat_id,
            message=ChannelInboundMessage(
                channel=_CHANNEL,
                sender=sender,
                chat_id=chat_id,
                content="".join("\u2028" if ord(char) in {10, 13} else " " if ord(char) < 32 else char for char in (content or "[图片]")),
                timestamp=_event_timestamp(event),
                metadata={"provider_message_id": message_id},
                attachments=attachments,
            ),
        )
        ingress = self._ingress
        if ingress is None:
            raise RuntimeError("QQ ingress 未绑定")
        await ingress.admit(raw)

    async def _download_images(self, urls: list[str]) -> tuple[AttachmentRef, ...]:
        if not urls:
            return ()
        importer = self._attachment_import
        client = self._http
        if importer is None or client is None:
            raise RuntimeError("QQ attachment resources 未绑定")
        result: list[AttachmentRef] = []
        total = 0
        for raw_url in urls[:_MAX_IMAGE_COUNT]:
            url = html.unescape(raw_url)
            async with client.stream("GET", url) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise ValueError(f"QQ 图片 HTTP {response.status_code}")
                media_type = response.headers.get("content-type", "image/jpeg").split(";", 1)[0]
                data = bytearray()
                async for chunk in response.aiter_bytes(64 * 1024):
                    if len(data) + len(chunk) > _MAX_IMAGE_BYTES:
                        raise ValueError("QQ 图片超过单项大小上限")
                    if total + len(data) + len(chunk) > _MAX_TOTAL_IMAGE_BYTES:
                        raise ValueError("QQ 图片超过批次大小上限")
                    data.extend(chunk)
            total += len(data)
            result.append(
                await importer.import_bytes(
                    bytes(data),
                    kind=AttachmentKind.IMAGE,
                    filename="qq-image.jpg",
                    media_type=media_type,
                )
            )
        return tuple(result)

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        """Read exact Core attachment leases and call the active NapCat API."""

        if request.binding_token != self._binding_token:
            raise RuntimeError("QQ delivery binding token 不匹配")
        try:
            self._validate_recipient(request.recipient)
            if not request.body.strip() and not request.attachments:
                return ProviderDeliveryReceipt(
                    request.delivery_id,
                    DeliveryStatus.REJECTED,
                    error="消息没有可发送正文或附件",
                )
            attachments = await self._read_attachments(request.attachments)
        except (TypeError, ValueError, RuntimeError) as error:
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.REJECTED,
                error=f"QQ 发送准备失败: {error}",
            )
        api = self._api or (getattr(self._bot, "api", None) if self._bot else None)
        if api is None:
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.FAILED,
                error="QQ provider 尚未建立 API",
            )
        try:
            provider_ids: list[str] = []
            if request.body.strip():
                result = await self._send_text(api, request.recipient, request.body)
                _append_provider_id(provider_ids, result)
            for ref, data in attachments:
                result = await self._send_attachment(api, request.recipient, ref, data)
                _append_provider_id(provider_ids, result)
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.DELIVERED,
                tuple(provider_ids),
            )
        except (OSError, TimeoutError, RuntimeError) as error:
            return ProviderDeliveryReceipt(
                request.delivery_id,
                DeliveryStatus.FAILED,
                error=f"QQ provider 回执未确认: {type(error).__name__}",
            )

    async def _send_text(self, api: Any, recipient: str, text: str) -> object:
        if recipient.startswith(_GROUP_PREFIX):
            return await self._run_on_bot_loop(
                api.send_group_text(int(recipient[len(_GROUP_PREFIX) :]), text)
            )
        return await self._run_on_bot_loop(api.send_private_text(int(recipient), text))

    async def _send_attachment(
        self,
        api: Any,
        recipient: str,
        ref: AttachmentRef,
        data: bytes,
    ) -> object:
        uri = "base64://" + base64.b64encode(data).decode("ascii")
        group = recipient.startswith(_GROUP_PREFIX)
        identifier = int(recipient[len(_GROUP_PREFIX) :]) if group else int(recipient)
        if ref.kind is AttachmentKind.IMAGE:
            call = api.send_group_image(identifier, uri) if group else api.send_private_image(identifier, uri)
        else:
            filename = ref.filename or ref.artifact_id
            call = api.send_group_file(identifier, uri, filename) if group else api.send_private_file(identifier, uri, filename)
        return await self._run_on_bot_loop(call)

    async def _run_on_bot_loop(self, coroutine: Any) -> object:
        loop = self._bot_loop
        if loop is None:
            loop = self._main_loop
        if loop is None:
            coroutine.close()
            raise RuntimeError("QQ provider loop 未就绪")
        if loop is asyncio.get_running_loop():
            return await coroutine
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, loop))

    async def stop(self) -> StopReceipt:
        """关闭共用一次真实清理；取消等待者不能伪造资源已释放。"""
        self._stopping = True
        self._admission_open = False
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop(), name=self.name + "-channel-stop")
        return await asyncio.shield(self._stop_task)

    async def _stop(self) -> StopReceipt:
        self._stopping = True
        self._admission_open = False
        # 1. 不取消 to_thread 的等待者来冒充 provider 已经退出。
        starting = self._backend_task
        if starting is not None:
            try:
                await asyncio.shield(starting)
            except Exception:
                # 启动错误由 start 原样传播；此处继续清理已取得的资源。
                logger.exception("QQ backend 启动失败，继续清理")
        loop = self._bot_loop
        connection = self._connection_task
        bot = self._bot
        if loop is not None and connection is not None and not connection.done():
            async def close_connection() -> None:
                # 2. 在 provider 自己的 loop 卸载插件，再取消长连接以执行其 finally。
                try:
                    assert bot is not None
                    await bot.plugin_loader.unload_all()
                finally:
                    connection.cancel()
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(close_connection(), loop))
        thread = self._backend_thread
        if thread is not None:
            await asyncio.to_thread(thread.join)
        futures = tuple(self._inbound_futures)
        if futures:
            await asyncio.gather(
                *(asyncio.wrap_future(item) if not isinstance(item, asyncio.Task) else item for item in futures),
                return_exceptions=True,
            )
        await self._close_http()
        self._bot = None
        self._api = None
        self._started = False
        return StopReceipt(self._binding_token, resources_closed=True)

    async def _close_http(self) -> None:
        client = self._http
        self._http = None
        if client is not None:
            await client.aclose()

    def _validate_recipient(self, recipient: str) -> None:
        value = recipient[len(_GROUP_PREFIX) :] if recipient.startswith(_GROUP_PREFIX) else recipient
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"QQ recipient 无效: {recipient}")

    async def _read_attachments(self, refs: tuple[AttachmentRef, ...]) -> list[tuple[AttachmentRef, bytes]]:
        if not refs:
            return []
        reader = self._attachment_read
        if reader is None:
            raise RuntimeError("QQ outbound 缺少 attachment_read")
        result: list[tuple[AttachmentRef, bytes]] = []
        for ref in refs:
            lease = await reader.acquire(ref)
            try:
                if lease.ref != ref:
                    raise RuntimeError("QQ attachment lease ref 不匹配")
                data = await lease.read_bytes(max_bytes=max(ref.size_bytes, 1))
                if len(data) != ref.size_bytes:
                    raise ValueError("QQ attachment size 不匹配")
                if hashlib.sha256(data).hexdigest() != ref.sha256:
                    raise ValueError("QQ attachment sha256 不匹配")
                result.append((ref, data))
            finally:
                await lease.aclose()
        return result


def _configure_ncatbot(config: Any, values: QQChannelConfig) -> None:
    """Apply only provider-owned NapCat settings before the backend starts."""

    config.bt_uin = values.bot_uin
    config.root = values.allow_from[0] if values.allow_from else values.bot_uin
    config.check_ncatbot_update = False
    config.skip_ncatbot_install_check = True
    config.napcat.remote_mode = True
    config.napcat.enable_webui = False
    config.enable_webui_interaction = False
    _NCATBOT_DIR.mkdir(parents=True, exist_ok=True)
    (_NCATBOT_DIR / "plugins").mkdir(exist_ok=True)
    config.plugin.plugins_dir = str(_NCATBOT_DIR / "plugins")


def _extract_cq_images(raw: str) -> tuple[str, list[str]]:
    urls = _CQ_IMAGE_RE.findall(raw)
    text = re.sub(r"\[CQ:image[^\]]*\]", "", raw).strip()
    return text, urls


def _message_id(event: object) -> str | None:
    for name in ("message_id", "message_seq"):
        value = getattr(event, name, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _event_timestamp(event: object) -> Any:
    from datetime import datetime, timezone

    value = getattr(event, "time", None) or getattr(event, "timestamp", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    return datetime.now(timezone.utc)


def _append_provider_id(target: list[str], result: object) -> None:
    value = getattr(result, "message_id", None)
    if value is None and isinstance(result, dict):
        value = result.get("message_id")
    if value is not None:
        target.append(str(value))


__all__ = ["QQChannelAdapter", "build_qq_channel"]
