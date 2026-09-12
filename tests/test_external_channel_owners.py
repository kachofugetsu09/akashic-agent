from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from agent.plugin_composition import (
    ChannelFactoryContext,
    CredentialRef,
    DeliveryStatus,
    ProviderDeliveryRequest,
)
from agent.plugin_composition.channels import ChannelRuntimePorts
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
    async def import_bytes(self, data: bytes, **_: object) -> Any:
        return data


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

    async def create(self, _credentials: object) -> _ProviderClient:
        self.create_calls += 1
        return self.client

    async def aclose(self) -> None:
        return None


def _context(
    *,
    config: dict[str, object],
    credentials: dict[str, CredentialRef] | None = None,
) -> tuple[ChannelFactoryContext, _Ingress, _ProviderFactory]:
    ingress = _Ingress()
    provider = _ProviderFactory()
    context = ChannelFactoryContext(
        snapshot_id="snapshot-1",
        generation_id="generation-1",
        binding_token="binding-1",
        config=config,
        credentials=credentials or {},
        provider_client_factory=provider,
        ingress=ingress,
        identity=_Identity(),
        attachment_import=_AttachmentImport(),
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
    context, ingress, provider = _context(
        config={"allow_from": ["alice"]},
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

    adapter.close_admission()
    stopped = await adapter.stop()
    assert stopped.resources_closed
    assert provider.client.closed
    assert adapter._app is None


class _QQApi:
    async def send_private_text(self, user_id: int, text: str) -> Any:
        return SimpleNamespace(message_id=f"private:{user_id}:{text}")


class _QQBot:
    def __init__(self, api: _QQApi) -> None:
        self.api = api
        self.callbacks: dict[str, Any] = {}
        self.exited = False

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

    def run_backend(self) -> None:
        return None

    def exit(self) -> None:
        self.exited = True


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
    monkeypatch.setattr(qq_channel, "_NCATBOT_DIR", tmp_path / "ncatbot")

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
    adapter._api = api
    adapter.open_admission()

    event = SimpleNamespace(user_id=42, raw_message="hello", message_id=17)
    await bot.callbacks["private"](event)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
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
