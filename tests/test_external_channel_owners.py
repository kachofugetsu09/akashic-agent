from __future__ import annotations

import websockets

import asyncio
import hashlib
import sys
import threading
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agent.plugin_composition import (
    AttachmentKind,
    ChannelFactoryContext,
    AttachmentRef,
    CredentialRef,
    DeliveryStatus,
    ProviderDeliveryRequest,
)
from agent.plugin_composition.channels import ChannelRuntimePorts, ChannelAttachmentReadPort
from plugins.qq_channel import channel as qq_channel
from plugins.telegram_channel import channel as telegram_channel


class _Ingress:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def admit(self, raw: Any) -> bool:
        self.messages.append(raw)
        return True


class _Identity:
    def resolve(self, provider_identity: str) -> str:
        return provider_identity


class _AttachmentImport:
    async def import_bytes(self, data: bytes, *, kind, filename, media_type) -> AttachmentRef:
        return AttachmentRef("fixture-attachment", kind, filename, media_type,
                             len(data), hashlib.sha256(data).hexdigest())


class _ProviderClient:
    def __init__(self) -> None:
        self.closed = False
        self.resolved: list[CredentialRef] = []

    def credential(self, ref: CredentialRef) -> str:
        self.resolved.append(ref)
        return "telegram-token"

    async def aclose(self) -> None:
        self.closed = True


class _ProviderFactory:
    def __init__(self) -> None:
        self.client = _ProviderClient()
        self.create_calls = 0

    async def create(self, credentials: object) -> _ProviderClient:
        self.create_calls += 1
        return self.client

    async def aclose(self) -> None:
        return None


class _AttachmentLease:
    def __init__(self, ref: AttachmentRef, data: bytes) -> None:
        self.ref = ref
        self._data = data
        self.closed = False

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        if len(self._data) > max_bytes:
            raise ValueError("fixture lease max_bytes")
        return self._data

    async def aclose(self) -> None:
        self.closed = True


class _AttachmentRead:
    def __init__(self, allowed: AttachmentRef, data: bytes) -> None:
        self.allowed = allowed
        self.data = data
        self.acquired: list[AttachmentRef] = []
        self.leases: list[_AttachmentLease] = []

    async def acquire(self, ref: AttachmentRef) -> _AttachmentLease:
        self.acquired.append(ref)
        if ref != self.allowed:
            raise PermissionError("fixture attachment lease denied")
        lease = _AttachmentLease(ref, self.data)
        self.leases.append(lease)
        return lease


def _context(
    *,
    config: dict[str, object],
    credentials: dict[str, CredentialRef] | None = None,
    attachment_read: ChannelAttachmentReadPort | None = None,
) -> tuple[ChannelFactoryContext, _Ingress, _ProviderFactory]:
    ingress = _Ingress()
    provider = _ProviderFactory()
    context = ChannelFactoryContext(
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        boot_id="test-boot",
        binding_token="binding-1",
        config=config,
        credentials=credentials or {},
        provider_client_factory=provider,
        ingress=ingress,
        identity=_Identity(),
        attachment_import=_AttachmentImport(),
        attachment_read=attachment_read,
    )
    return context, ingress, provider


class _TelegramUpdater:
    def __init__(self) -> None:
        self.running = False

    async def start_polling(self, **_: object) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False


class _TelegramBot:
    def __init__(self) -> None:
        self.sent: list[tuple[str, int, str]] = []

    async def set_my_commands(self, _commands: object) -> None:
        return None

    async def send_message(self, *, chat_id: int, text: str, **_: object) -> Any:
        self.sent.append(("text", chat_id, text))
        return SimpleNamespace(message_id=11)

    async def send_photo(self, *, chat_id: int, **_: object) -> Any:
        self.sent.append(("photo", chat_id, ""))
        return SimpleNamespace(message_id=12)

    async def send_document(self, *, chat_id: int, **_: object) -> Any:
        self.sent.append(("document", chat_id, ""))
        return SimpleNamespace(message_id=13)


class _TelegramApplication:
    def __init__(self) -> None:
        self.bot = _TelegramBot()
        self.updater = _TelegramUpdater()
        self.running = False
        self.handlers: list[object] = []

    def add_handler(self, handler: object) -> None:
        self.handlers.append(handler)

    async def initialize(self) -> None:
        return None

    async def start(self) -> None:
        self.running = True

    async def stop(self) -> None:
        self.running = False

    async def shutdown(self) -> None:
        return None


class _TelegramApplicationBuilder:
    def __init__(self) -> None:
        self.application = _TelegramApplication()

    def token(self, _token: str) -> _TelegramApplicationBuilder:
        return self

    def connect_timeout(self, _value: float) -> _TelegramApplicationBuilder:
        return self

    def read_timeout(self, _value: float) -> _TelegramApplicationBuilder:
        return self

    def write_timeout(self, _value: float) -> _TelegramApplicationBuilder:
        return self

    def pool_timeout(self, _value: float) -> _TelegramApplicationBuilder:
        return self

    def build(self) -> _TelegramApplication:
        return self.application


class _TelegramApplicationFactory:
    builder_instance: _TelegramApplicationBuilder | None = None

    @classmethod
    def builder(cls) -> _TelegramApplicationBuilder:
        cls.builder_instance = _TelegramApplicationBuilder()
        return cls.builder_instance


@pytest.mark.asyncio
async def test_telegram_external_owner_handles_inbound_delivery_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_channel, "Application", _TelegramApplicationFactory)
    outbound_data = b"outbound-image"
    outbound_ref = AttachmentRef(
        "outbound-attachment",
        AttachmentKind.IMAGE,
        "reply.jpg",
        "image/jpeg",
        len(outbound_data),
        hashlib.sha256(outbound_data).hexdigest(),
    )
    attachment_read = _AttachmentRead(outbound_ref, outbound_data)
    context, ingress, provider = _context(
        config={"allow_from": ["alice"]},
        credentials={"token": CredentialRef(("token",))},
        attachment_read=attachment_read,
    )
    adapter = telegram_channel.build_telegram_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    ready = await adapter.start()
    assert not ready.admission_open
    assert provider.create_calls == 1
    assert provider.client.resolved == [CredentialRef(("token",))]

    update = SimpleNamespace(
        effective_message=SimpleNamespace(
            message_id=7,
            text="hello",
            caption=None,
            photo=[],
            document=None,
            reply_to_message=None,
            date=datetime.now(timezone.utc),
        ),
        effective_chat=SimpleNamespace(id=1001),
        effective_user=SimpleNamespace(id=42, username="alice"),
    )
    await adapter._on_update(update, SimpleNamespace(bot=adapter._app.bot))
    assert ingress.messages == []

    adapter.open_admission()
    await adapter._on_update(update, SimpleNamespace(bot=adapter._app.bot))
    assert len(ingress.messages) == 1
    assert ingress.messages[0].message.content == "hello"
    assert ingress.messages[0].recipient == "1001"
    await adapter._on_update(update, SimpleNamespace(bot=adapter._app.bot))
    assert len(ingress.messages) == 1

    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token=context.binding_token,
            delivery_id="delivery-1",
            recipient="1001",
            body="reply",
        )
    )
    assert receipt.status is DeliveryStatus.DELIVERED
    assert adapter._app.bot.sent == [("text", 1001, "reply")]

    attachment_receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token=context.binding_token,
            delivery_id="delivery-attachment",
            recipient="1001",
            body="",
            attachments=(outbound_ref,),
        )
    )
    assert attachment_receipt.status is DeliveryStatus.DELIVERED
    assert adapter._app.bot.sent[-1] == ("photo", 1001, "")
    assert attachment_read.leases[0].closed

    async def get_file(_file_id):
        async def download():
            return b"reply-image"
        return SimpleNamespace(download_as_bytearray=download)
    adapter._app.bot.get_file = get_file
    update.effective_message.message_id = 8
    update.effective_message.text = "line one\nline two"
    update.effective_message.reply_to_message = SimpleNamespace(
        photo=[SimpleNamespace(file_id="image")], document=None, text=None, caption=None,
    )
    await adapter._on_update(update, SimpleNamespace(bot=adapter._app.bot))
    assert ingress.messages[-1].message.content == "line one\u2028line two"
    assert ingress.messages[-1].message.attachments[0].sha256 == hashlib.sha256(b"reply-image").hexdigest()
    adapter.close_admission()
    first, second = await asyncio.gather(adapter.stop(), adapter.stop())
    assert first is second and first.resources_closed
    assert provider.client.closed
    assert adapter._app is None


@pytest.mark.asyncio
async def test_telegram_start_failure_preserves_error_and_cleanup_can_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_channel, "Application", _TelegramApplicationFactory)

    async def fail_initialize(self) -> None:
        raise RuntimeError("fixture initialize failure")

    monkeypatch.setattr(_TelegramApplication, "initialize", fail_initialize)
    context, _, provider = _context(
        config={"allow_from": []},
        credentials={"token": CredentialRef(("token",))},
    )
    adapter = telegram_channel.build_telegram_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    with pytest.raises(RuntimeError, match="fixture initialize failure"):
        await adapter.start()
    failed = await adapter.stop()
    assert failed.resources_closed
    assert provider.client.closed

    monkeypatch.undo()
    monkeypatch.setattr(telegram_channel, "Application", _TelegramApplicationFactory)
    context, _, provider = _context(
        config={"allow_from": []},
        credentials={"token": CredentialRef(("token",))},
    )
    adapter = telegram_channel.build_telegram_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    await adapter.start()

    async def fail_updater_stop() -> None:
        raise RuntimeError("fixture updater stop failure")

    original_stop = adapter._app.updater.stop
    adapter._app.updater.stop = fail_updater_stop
    failed = await adapter.stop()
    assert not failed.resources_closed
    assert any(item.resource == "updater" for item in failed.failures)
    assert adapter._app is not None
    assert provider.client.closed
    adapter._app.updater.stop = original_stop
    assert (await adapter.stop()).resources_closed


@pytest.mark.asyncio
async def test_telegram_cancelled_stop_waiter_does_not_cancel_shared_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telegram_channel, "Application", _TelegramApplicationFactory)
    context, _, _ = _context(
        config={"allow_from": []},
        credentials={"token": CredentialRef(("token",))},
    )
    adapter = telegram_channel.build_telegram_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    await adapter.start()
    started = asyncio.Event()
    release = asyncio.Event()

    async def stop_waiter() -> None:
        started.set()
        await release.wait()
        adapter._app.running = False

    adapter._app.stop = stop_waiter
    cleanup = asyncio.create_task(adapter.stop())
    await started.wait()
    waiter = asyncio.create_task(adapter.stop())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not cleanup.done()
    release.set()
    receipt = await cleanup
    assert receipt.resources_closed
    assert adapter._app is None


class _QQApi:
    async def send_private_text(self, user_id: int, text: str) -> Any:
        return SimpleNamespace(message_id=f"private:{user_id}:{text}")


class _QQBot:
    def __init__(
        self,
        api: _QQApi,
        *,
        backend_error: BaseException | None = None,
        hold_backend: bool = False,
        unload_error: BaseException | None = None,
    ) -> None:
        self.api = api
        self.backend_error = backend_error
        self.hold_backend = hold_backend
        self.backend_release = threading.Event()
        self.unload_error = unload_error
        self.callbacks: dict[str, Any] = {}
        self.exited = False
        self.ready = threading.Event()
        self.unloaded = False
        self.plugin_loader = SimpleNamespace(unload_all=self.unload_all)
        self.thread: threading.Thread | None = None
        self.adapter = SimpleNamespace(connect_websocket=self.connect_websocket)

    async def unload_all(self) -> None:
        if self.unload_error is not None:
            raise self.unload_error
        self.unloaded = True

    def _decorator(self, name: str):
        def register(callback: Any) -> Any:
            self.callbacks[name] = callback
            return callback

        return register

    def on_startup(self) -> Any:
        return self._decorator("startup")

    def on_private_message(self) -> Any:
        return self._decorator("private")

    def on_group_message(self) -> Any:
        return self._decorator("group")

    async def connect_websocket(self) -> None:
        try:
            self.ready.set()
            await asyncio.Future()
        finally:
            self.exited = True

    def start(self) -> None:
        try:
            asyncio.run(self.adapter.connect_websocket())
        except asyncio.CancelledError:
            pass

    def run_backend(self) -> _QQApi:
        """模拟 SDK：连接在独立线程运行，startup 后才返回 API。"""
        if self.backend_error is not None:
            raise self.backend_error
        self.thread = threading.Thread(target=self.start)
        self.thread.start()
        if not self.ready.wait(5):
            raise RuntimeError("fixture startup 未就绪")
        if self.hold_backend and not self.backend_release.wait(5):
            raise RuntimeError("fixture backend release timeout")
        return self.api


def _install_qq_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    bot: _QQBot,
    *,
    config: SimpleNamespace | None = None,
) -> SimpleNamespace:
    ncatbot = ModuleType("ncatbot")
    ncatbot_core = ModuleType("ncatbot.core")
    ncatbot_utils = ModuleType("ncatbot.utils")
    ncatbot_core.BotClient = lambda: bot  # type: ignore[attr-defined]
    config = config or SimpleNamespace(
        bt_uin="original",
        root="original-root",
        check_ncatbot_update=True,
        skip_ncatbot_install_check=False,
        websocket_timeout=15,
        napcat=SimpleNamespace(remote_mode=False, enable_webui=True),
        enable_webui_interaction=True,
        plugin=SimpleNamespace(plugins_dir="original-plugins"),
    )
    ncatbot_utils.ncatbot_config = config
    monkeypatch.setitem(sys.modules, "ncatbot", ncatbot)
    monkeypatch.setitem(sys.modules, "ncatbot.core", ncatbot_core)
    monkeypatch.setitem(sys.modules, "ncatbot.utils", ncatbot_utils)
    return config


@pytest.mark.asyncio
async def test_qq_external_owner_handles_inbound_delivery_and_stop(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _QQApi()
    bot = _QQBot(api)
    ncatbot = ModuleType("ncatbot")
    ncatbot_core = ModuleType("ncatbot.core")
    ncatbot_utils = ModuleType("ncatbot.utils")
    ncatbot_core.BotClient = lambda: bot  # type: ignore[attr-defined]
    ncatbot_utils.ncatbot_config = SimpleNamespace(
        bt_uin="",
        root="",
        check_ncatbot_update=True,
        skip_ncatbot_install_check=False,
        napcat=SimpleNamespace(remote_mode=False, enable_webui=True),
        enable_webui_interaction=True,
        plugin=SimpleNamespace(plugins_dir=None),
    )
    monkeypatch.setitem(sys.modules, "ncatbot", ncatbot)
    monkeypatch.setitem(sys.modules, "ncatbot.core", ncatbot_core)
    monkeypatch.setitem(sys.modules, "ncatbot.utils", ncatbot_utils)

    context, ingress, _provider = _context(
        config={"bot_uin": "9001", "allow_from": ["42"], "groups": []}
    )
    adapter = qq_channel.build_qq_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    ready = await adapter.start()
    assert not ready.admission_open
    assert adapter._api is api
    adapter.open_admission()

    event = SimpleNamespace(user_id=42, raw_message="hello", message_id=17)
    admitted = asyncio.Event()
    original_admit = ingress.admit
    async def admit(raw):
        result = await original_admit(raw)
        admitted.set()
        return result
    ingress.admit = admit
    await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(bot.callbacks["private"](event), adapter._bot_loop))
    await asyncio.wait_for(admitted.wait(), 2)
    assert len(ingress.messages) == 1
    assert ingress.messages[0].message.content == "hello"
    assert ingress.messages[0].recipient == "42"

    receipt = await adapter.deliver(
        ProviderDeliveryRequest(
            binding_token=context.binding_token,
            delivery_id="delivery-2",
            recipient="42",
            body="reply",
        )
    )
    assert receipt.status is DeliveryStatus.DELIVERED
    assert receipt.provider_ids == ("private:42:reply",)

    adapter.close_admission()
    stopped = await adapter.stop()
    assert stopped.resources_closed
    assert bot.exited
    assert bot.unloaded
    assert bot.thread is not None and not bot.thread.is_alive()


def test_qq_sdk_timeout_is_bound_per_adapter_without_global_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK 两次 websocket.connect 都从这个实例的 globals 读取配置。"""
    import ncatbot.core.adapter.adapter as sdk_adapter

    calls: list[dict[str, object]] = []

    async def connect(_uri: str, **kwargs: object) -> object:
        calls.append(kwargs)
        raise RuntimeError("fixture connect")

    monkeypatch.setattr(sdk_adapter.websockets, "connect", connect)
    adapter = sdk_adapter.Adapter()
    scoped = qq_channel._bind_websocket_timeout(adapter.connect_websocket, 7.25)
    with pytest.raises(RuntimeError, match="fixture connect"):
        asyncio.run(scoped())
    assert calls == [{"close_timeout": 0.2, "max_size": 2**30, "open_timeout": 7.25}]
    assert sdk_adapter.Adapter.connect_websocket.__globals__["websockets"].connect is connect


@pytest.mark.asyncio
async def test_qq_start_failure_and_cancellation_close_actual_resources(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_bot = _QQBot(_QQApi(), backend_error=RuntimeError("fixture startup failure"))
    config = _install_qq_fixture(monkeypatch, tmp_path, failed_bot)
    context, _, _ = _context(config={"bot_uin": "9001"})
    adapter = qq_channel.build_qq_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    with pytest.raises(RuntimeError, match="fixture startup failure"):
        await adapter.start()
    failed_receipt = await adapter.stop()
    assert failed_receipt.resources_closed
    assert config.bt_uin == "original"
    assert config.plugin.plugins_dir == "original-plugins"

    cancelled_bot = _QQBot(_QQApi(), hold_backend=True)
    config = _install_qq_fixture(monkeypatch, tmp_path, cancelled_bot, config=config)
    context, _, _ = _context(config={"bot_uin": "9002"})
    adapter = qq_channel.build_qq_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    starting = asyncio.create_task(adapter.start())
    await asyncio.to_thread(cancelled_bot.ready.wait, 2)
    starting.cancel()
    cancelled_bot.backend_release.set()
    with pytest.raises(asyncio.CancelledError):
        await starting
    cancelled_receipt = await adapter.stop()
    assert cancelled_receipt.resources_closed
    assert cancelled_bot.thread is not None and not cancelled_bot.thread.is_alive()
    assert config.bt_uin == "original"


@pytest.fixture
def separate_qq_generation():
    """用正式 fresh importer 创建另一份插件模块，不能共享业务模块里的锁。"""
    import importlib.util
    from pathlib import Path
    from agent.plugins.importer import FreshPluginImporter

    name = "fixture_qq_external_generation"
    source = Path(__file__).parents[1] / "plugins" / "qq_channel"
    importer = FreshPluginImporter()
    importer.register(name, source)
    spec = importer.root_spec(name, source / "plugin.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield sys.modules[name + ".channel"]
    finally:
        importer.unregister(name)
        for key in tuple(sys.modules):
            if key == name or key.startswith(name + "."):
                del sys.modules[key]


@pytest.mark.asyncio
async def test_qq_ncatbot_config_is_exclusive_and_restored_between_generations(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    separate_qq_generation,
) -> None:
    first_bot = _QQBot(_QQApi())
    config = _install_qq_fixture(monkeypatch, tmp_path, first_bot)
    context, _, _ = _context(config={"bot_uin": "7001"})
    first = qq_channel.build_qq_channel(context)
    first.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    await first.start()
    assert config.bt_uin == "7001"
    first_runtime = first._ncatbot_dir
    assert first_runtime is not None and first_runtime.is_dir()

    second_bot = _QQBot(_QQApi())
    sys.modules["ncatbot.core"].BotClient = lambda: second_bot  # type: ignore[attr-defined]
    context, _, _ = _context(config={"bot_uin": "7002"})
    second = separate_qq_generation.build_qq_channel(context)
    assert type(first) is not type(second)
    second.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    with pytest.raises(RuntimeError, match="另一 owner"):
        await second.start()
    assert config.bt_uin == "7001"
    assert (await first.stop()).resources_closed
    assert not first_runtime.exists()
    assert config.bt_uin == "original"

    context, _, _ = _context(config={"bot_uin": "7002"})
    third = separate_qq_generation.build_qq_channel(context)
    third.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    await third.start()
    assert config.bt_uin == "7002"
    assert (await third.stop()).resources_closed
    assert config.bt_uin == "original"


@pytest.mark.asyncio
async def test_qq_stop_failure_and_cancelled_waiter_share_unconfirmed_cleanup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot = _QQBot(_QQApi(), unload_error=RuntimeError("fixture unload failure"))
    _install_qq_fixture(monkeypatch, tmp_path, bot)
    context, _, _ = _context(config={"bot_uin": "9001"})
    adapter = qq_channel.build_qq_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    await adapter.start()
    first, second = await asyncio.gather(adapter.stop(), adapter.stop())
    assert first is second
    assert not first.resources_closed
    assert any(item.resource == "connection" for item in first.failures)
    assert bot.thread is not None and bot.thread.is_alive()
    bot.unload_error = None
    assert (await adapter.stop()).resources_closed
    assert bot.unloaded and not bot.thread.is_alive()

    bot = _QQBot(_QQApi())
    _install_qq_fixture(monkeypatch, tmp_path, bot)
    context, _, _ = _context(config={"bot_uin": "9001"})
    adapter = qq_channel.build_qq_channel(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    await adapter.start()
    started = threading.Event()
    release = threading.Event()

    async def unload_waiter() -> None:
        started.set()
        await asyncio.to_thread(release.wait, 2)
        bot.unloaded = True

    bot.plugin_loader.unload_all = unload_waiter
    cleanup = asyncio.create_task(adapter.stop())
    await asyncio.to_thread(started.wait, 2)
    waiter = asyncio.create_task(adapter.stop())
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not cleanup.done()
    release.set()
    receipt = await cleanup
    assert receipt.resources_closed
    assert bot.thread is not None and not bot.thread.is_alive()


def test_channel_config_migration_resumes_before_removing_old_input(tmp_path, monkeypatch):
    """任一 channel/sender 目标发布失败时保留原配置，并可按原备份重试。"""
    import tomllib
    from scripts import migrate_legacy_channels as migration
    from agent.plugins.manifest import workspace_plugin_data_dir

    config = tmp_path / "config.toml"
    source = (
        '[channels.telegram]\ntoken="fixture-token"\n'
        '[channels.qq]\nbot_uin="9001"\nsender_endpoint="ws://127.0.0.1:3001/api"\n'
    )
    config.write_text(source)
    workspace = tmp_path / "workspace"
    qq = workspace_plugin_data_dir(workspace, "qq_channel", "external") / "config.local.toml"
    replace = migration.os.replace
    def fail_second(source_path, destination):
        if destination == qq:
            raise OSError("fixture channel target failure")
        replace(source_path, destination)
    with monkeypatch.context() as patch:
        patch.setattr(migration.os, "replace", fail_second)
        with pytest.raises(OSError, match="channel target failure"):
            migration.migrate_legacy_channels(config, workspace, marketplace="external")
    assert config.read_text() == source
    backup = config.with_name(config.name + ".before-channel-plugin-migration.bak")
    assert backup.read_text() == source
    assert migration.migrate_legacy_channels(config, workspace, marketplace="external") == (
        "telegram_channel", "qq_channel",
    )
    assert tomllib.loads(qq.read_text())["bot_uin"] == "9001"
    telegram = workspace_plugin_data_dir(workspace, "telegram_channel", "external") / "config.local.toml"
    assert tomllib.loads(telegram.read_text())["token"] == "fixture-token"
    telegram_sender = workspace_plugin_data_dir(workspace, "telegram_sender", "external") / "config.local.toml"
    assert tomllib.loads(telegram_sender.read_text()) == {
        "enabled": True,
        "channel": "telegram",
        "token": "fixture-token",
    }
    qq_sender = workspace_plugin_data_dir(workspace, "qq_sender", "external") / "config.local.toml"
    assert tomllib.loads(qq_sender.read_text()) == {
        "enabled": True,
        "channel": "qq",
        "endpoint": "ws://127.0.0.1:3001/api",
    }
    assert tomllib.loads(config.read_text()) == {}
    assert backup.read_text() == source


def test_channel_config_migration_requires_qq_sender_endpoint_before_writing(tmp_path):
    """QQ receiver identity cannot silently become a sender with a guessed endpoint."""
    from scripts import migrate_legacy_channels as migration

    config = tmp_path / "config.toml"
    source = '[channels.qq]\nbot_uin="9001"\n'
    config.write_text(source)
    workspace = tmp_path / "workspace"

    with pytest.raises(ValueError, match="sender_endpoint"):
        migration.migrate_legacy_channels(config, workspace, marketplace="external")

    assert config.read_text() == source
    assert not workspace.exists()
    assert not config.with_name(config.name + ".before-channel-plugin-migration.bak").exists()


@pytest.mark.parametrize("invalid", [
    "websocket_open_timeout_seconds=nan\n",
    "websocket_open_timeout_seconds=inf\n",
    'groups=[{group_id="42"}, {group_id="42"}]\n',
])
def test_channel_config_migration_rejects_invalid_receiver_before_writing(tmp_path, invalid):
    """无效接收配置不能在迁移后才报错并丢掉旧入口。"""
    from scripts.migrate_legacy_channels import migrate_legacy_channels

    config = tmp_path / "config.toml"
    source = '[channels.qq]\nbot_uin="9001"\nsender_endpoint="ws://127.0.0.1/api"\n' + invalid
    config.write_text(source)
    workspace = tmp_path / "workspace"
    with pytest.raises(ValueError, match="timeout|重复"):
        migrate_legacy_channels(config, workspace, marketplace="external")
    assert config.read_text() == source
    assert not workspace.exists()
    assert not config.with_name(config.name + ".before-channel-plugin-migration.bak").exists()
