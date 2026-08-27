from __future__ import annotations

import asyncio
import json
from contextlib import suppress

import pytest

from agent.plugin_composition.channels import (
    ChannelCapability,
    ChannelFactoryContext,
    ChannelReady,
    ChannelRuntimePorts,
    DeliveryStatus,
    InboundIdentity,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    StopReceipt,
)
from agent.plugins.manager import PluginManager
from agent.tools.message_push import MessagePushTool
from bootstrap.core_channel_adapter import build_core_channel_definition
from bootstrap.tools import _dispatch_v3_channel_push
from bus.event_bus import EventBus
from bus.queue import MessageBus
from session.manager import SessionManager


class _NativeAdapter:
    def __init__(
        self,
        context: ChannelFactoryContext,
        received: list[ProviderDeliveryRequest],
    ) -> None:
        self._binding_token = context.binding_token
        self._received = received

    async def start(self) -> ChannelReady:
        return ChannelReady(self._binding_token)

    async def deliver(
        self,
        request: ProviderDeliveryRequest,
    ) -> ProviderDeliveryReceipt:
        self._received.append(request)
        return ProviderDeliveryReceipt(request.delivery_id, DeliveryStatus.DELIVERED)

    async def stop(self) -> StopReceipt:
        return StopReceipt(self._binding_token, resources_closed=True)


class _NativeChannel:
    name = "web"

    def __init__(self) -> None:
        self.received: list[ProviderDeliveryRequest] = []
        self.contexts: list[ChannelFactoryContext] = []

    def build_v3_adapter(self, context: ChannelFactoryContext) -> _NativeAdapter:
        self.contexts.append(context)
        return _NativeAdapter(context, self.received)


class _AkashicNativeChannel(_NativeChannel):
    name = "akashic"


class _InboundNativeAdapter(_NativeAdapter):
    def __init__(
        self,
        context: ChannelFactoryContext,
        received: list[ProviderDeliveryRequest],
    ) -> None:
        super().__init__(context, received)
        self.runtime: ChannelRuntimePorts | None = None
        self.open = False

    def attach_runtime(self, ports: ChannelRuntimePorts) -> None:
        self.runtime = ports

    def open_admission(self) -> None:
        self.open = True

    def close_admission(self) -> None:
        self.open = False


class _InboundNativeChannel(_NativeChannel):
    name = "telegram"
    v3_inbound_identity = InboundIdentity.PROVIDER_MESSAGE_ID

    def __init__(self) -> None:
        super().__init__()
        self.adapter: _InboundNativeAdapter | None = None

    def build_v3_adapter(self, context: ChannelFactoryContext) -> _InboundNativeAdapter:
        self.contexts.append(context)
        self.adapter = _InboundNativeAdapter(context, self.received)
        return self.adapter


@pytest.mark.asyncio
async def test_manager_publishes_native_core_catalog_without_plugins(tmp_path) -> None:
    """A Core channel is materialized only through its native v3 factory."""

    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    channel = _NativeChannel()

    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )

    snapshot = manager.current_snapshot
    runtime = manager.active_channel_generation
    assert snapshot is not None
    assert snapshot.channel_catalog is not None
    assert snapshot.channel_catalog.definition("web") is not None
    assert runtime is not None
    assert runtime.snapshot_id == snapshot.snapshot_id
    assert runtime.channel("web").admission_open is True
    assert len(channel.contexts) == 1
    assert channel.contexts[0].binding_token == runtime.channel("web").binding_token

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_native_inbound_definition_attaches_before_opening_provider_callbacks(
    tmp_path,
) -> None:
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    channel = _InboundNativeChannel()

    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )

    catalog = manager.stable_committed_channel_catalog()
    assert catalog is not None
    definition = catalog.definition("telegram")
    assert definition is not None
    assert definition.capabilities == frozenset(
        {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
    )
    assert definition.inbound_identity is InboundIdentity.PROVIDER_MESSAGE_ID
    assert channel.adapter is not None
    assert channel.adapter.runtime is not None
    assert channel.adapter.open is True

    await manager.terminate_all()
    assert channel.adapter.open is False


@pytest.mark.asyncio
async def test_native_core_catalog_routes_message_push_without_legacy_fallback(
    tmp_path,
) -> None:
    """MessagePush reaches the native adapter with one exact committed request."""

    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    channel = _NativeChannel()
    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )

    bus = MessageBus()
    bus.bind_channel_outbound_dispatcher(
        manager.channel_generation_host.dispatch_outbound
    )
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    tool = MessagePushTool(chat_lane=bus.chat_lane)
    tool.bind_v3_channel_dispatcher(
        lambda message, passive: _dispatch_v3_channel_push(
            manager,
            bus,
            message,
            passive,
        )
    )

    try:
        result = json.loads(
            await tool.execute(
                target_channel="web",
                target_chat_id="chat-1",
                message="report",
            )
        )
    finally:
        await bus.aclose()
        if not dispatch_task.done():
            dispatch_task.cancel()
        with suppress(asyncio.CancelledError):
            await dispatch_task
        await manager.terminate_all()

    assert result["status"] == "delivered"
    assert result["retryable"] is False
    assert len(channel.received) == 1
    request = channel.received[0]
    assert request.recipient == "chat-1"
    assert request.body == "report"
    assert request.binding_token == channel.contexts[0].binding_token


@pytest.mark.asyncio
async def test_akashic_direct_push_commits_session_before_client_notification(
    tmp_path,
) -> None:
    sessions = SessionManager(tmp_path / "workspace")
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    channel = _AkashicNativeChannel()
    await manager.bind_core_channel_definitions(
        (build_core_channel_definition(channel),)
    )
    bus = MessageBus()
    bus.bind_channel_outbound_dispatcher(
        manager.channel_generation_host.dispatch_outbound
    )
    dispatch_task = asyncio.create_task(bus.dispatch_outbound())
    tool = MessagePushTool(chat_lane=bus.chat_lane)
    tool.bind_v3_channel_dispatcher(
        lambda message, passive: _dispatch_v3_channel_push(
            manager,
            bus,
            message,
            passive,
            session_manager=sessions,
        )
    )

    try:
        result = json.loads(
            await tool.execute(
                target_channel="akashic",
                target_chat_id="chat-1",
                message="scheduled result",
            )
        )
    finally:
        await bus.aclose()
        if not dispatch_task.done():
            dispatch_task.cancel()
        with suppress(asyncio.CancelledError):
            await dispatch_task
        await manager.terminate_all()

    messages = sessions.control_store.fetch_session_messages("akashic:chat-1")
    assert result["status"] == "delivered"
    assert len(messages) == 1
    assert messages[0]["id"] == "akashic:chat-1:0"
    assert messages[0]["seq"] == 0
    assert messages[0]["content"] == "scheduled result"
    assert messages[0]["effects"] == {"post_commit": "suppress"}
    assert channel.received[0].session_message_id == messages[0]["id"]
    sessions.close()


def test_core_catalog_rejects_channel_without_native_v3_factory() -> None:
    class _LegacyOnly:
        name = "legacy"

        async def _deliver_message(self, _message: object) -> object:
            return object()

    with pytest.raises(TypeError, match="build_v3_adapter"):
        build_core_channel_definition(_LegacyOnly())
