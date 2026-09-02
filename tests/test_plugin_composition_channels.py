from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.plugin_composition import (
    CHANNELS,
    AttachmentKind,
    AttachmentReadLease,
    AttachmentRef,
    ChannelAttachmentImportPort,
    ChannelAttachmentReadPort,
    ChannelCapability,
    ChannelCommitRole,
    ChannelDeliveryReceipt,
    ChannelInboundMessage,
    ChannelCleanupFailure,
    ChannelDefinition,
    ChannelFactoryContext,
    ChannelReady,
    ChannelTerminalStatus,
    CompositionError,
    CompositionRoot,
    CredentialRef,
    DeliveryStatus,
    InboundIdentity,
    InboundEnvelope,
    InboundOwner,
    InboundState,
    OutboundEnvelope,
    PluginChannels,
    PluginRuntime,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    PushToolRequest,
    QueuedReceipt,
    RawInbound,
    StopReceipt,
)
from agent.plugin_composition.channels import (
    ChannelDescriptor,
    ChannelFactoryFreezeInput,
    ChannelFactoryProvenance,
    ChannelRegistrySnapshot,
    _freeze_plugin_channels,
    _registry_identity,
    channel_config_revision,
)
from agent.plugins.channel_generation_host import ChannelBindingLease


def _runtime(plugin_id: str, root: Path, *, generation: str = "plugin-generation") -> PluginRuntime:
    plugin_dir = root / plugin_id
    plugin_dir.mkdir(parents=True, exist_ok=True)
    return PluginRuntime(
        plugin_id=plugin_id,
        generation_id=generation,
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=None,
    )


def _definition(name: str = "feishu") -> ChannelDefinition:
    return ChannelDefinition(
        name=name,
        capabilities=frozenset(ChannelCapability),
        factory_export=f"{name}:build_channel",
        inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
        credential_paths=("app_id", "app_secret"),
    )


def _provenance(name: str, *, generation: str = "plugin-generation") -> ChannelFactoryProvenance:
    definition = _definition(name)
    return ChannelFactoryProvenance(
        plugin_id="plugin",
        generation_id=generation,
        channel_name=name,
        source_revision="source-1",
        config_revision="config-1",
        factory_export=definition.factory_export,
    )


def _attachment(
    *,
    artifact_id: str = "artifact-1",
    kind: AttachmentKind = AttachmentKind.FILE,
    filename: str | None = "report.txt",
    media_type: str | None = "text/plain",
    size_bytes: int = 3,
    sha256: str = "a" * 64,
) -> AttachmentRef:
    return AttachmentRef(
        artifact_id=artifact_id,
        kind=kind,
        filename=filename,
        media_type=media_type,
        size_bytes=size_bytes,
        sha256=sha256,
    )


class _Lease(ChannelBindingLease):
    snapshot_lease = object()

    def __init__(self) -> None:
        self.close_calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def snapshot_id(self) -> str:
        return "snapshot"

    @property
    def generation_id(self) -> str:
        return "generation"

    @property
    def channel_name(self) -> str:
        return "feishu"

    @property
    def binding_token(self) -> str:
        return "binding"

    @property
    def active(self) -> bool:
        return self.close_calls == 0

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        await self.release.wait()


class _Ingress:
    async def admit(self, raw: RawInbound) -> bool:
        return True


def _inbound_envelope(lease: _Lease | None = None) -> InboundEnvelope:
    actual_lease = lease or _Lease()
    message = ChannelInboundMessage(
        channel="feishu",
        sender="sender",
        chat_id="chat",
        content="hello",
        timestamp=datetime.now(timezone.utc),
        metadata=json.loads('{"nested": {"items": [1, "two"]}}'),
    )
    return InboundEnvelope(
        message_id="provider-message",
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        message=message,
        lease=actual_lease,
    )


@pytest.mark.asyncio
async def test_channel_registry_registration_health_freeze_and_effect_cleanup(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("root-generation")
    channels = PluginChannels(root.instance_token)
    await root.context.provide(CHANNELS, channels)

    async def apply(ctx) -> None:
        await ctx.require(CHANNELS).register(ctx, _definition())

    fiber = await root.mount(
        apply,
        name="plugin",
        runtime=_runtime("plugin", tmp_path),
        inject=(CHANNELS,),
    )
    snapshot = _freeze_plugin_channels(
        channels,
        root.instance_token,
        factory_provenance_by_owner={
            "plugin": ChannelFactoryFreezeInput(
                "plugin-generation",
                source_revision="source-1",
                config_revision="config-1",
            )
        },
    )
    assert snapshot.descriptors[0].owner == "plugin"
    assert snapshot.descriptors[0].capabilities == tuple(
        sorted(ChannelCapability, key=lambda item: item.value)
    )
    assert snapshot.factories[0].source_revision == "source-1"
    assert root.receipt().health[0].required is True
    assert root.receipt().effects == (
        "root:service:core.channels",
        "plugin:channel:feishu",
        "plugin:health:channel:feishu",
    )

    await fiber.dispose()
    assert _freeze_plugin_channels(channels, root.instance_token) is snapshot
    assert root.receipt().effects == ("root:service:core.channels",)
    await root.dispose()


@pytest.mark.asyncio
async def test_channel_registry_rejects_duplicate_frozen_and_wrong_root(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("root-generation")
    channels = PluginChannels(root.instance_token)
    await root.context.provide(CHANNELS, channels)
    captured = None

    async def duplicate(ctx) -> None:
        nonlocal captured
        captured = ctx
        service = ctx.require(CHANNELS)
        await service.register(ctx, _definition())
        await service.register(ctx, _definition())

    _ = await root.mount(
        duplicate,
        name="plugin",
        runtime=_runtime("plugin", tmp_path),
        inject=(CHANNELS,),
    )
    assert not root.receipt().ready
    assert root.receipt().health == ()
    assert len(_freeze_plugin_channels(channels, root.instance_token).descriptors) == 0
    assert captured is not None

    other = CompositionRoot("other-generation")
    other_channels = PluginChannels(other.instance_token)
    await other.context.provide(CHANNELS, other_channels)
    with pytest.raises(CompositionError, match="不属于当前 Root"):
        await channels.register(other.context, _definition())

    await other.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_channel_registry_identity_is_root_independent_and_ordered(
    tmp_path: Path,
) -> None:
    identities: list[str] = []
    for suffix, names in (("candidate", ("qqbot", "feishu")), ("formal", ("feishu", "qqbot"))):
        root = CompositionRoot(f"{suffix}-root")
        channels = PluginChannels(root.instance_token)
        await root.context.provide(CHANNELS, channels)

        async def apply(ctx) -> None:
            service = ctx.require(CHANNELS)
            for name in names:
                await service.register(ctx, _definition(name))

        _ = await root.mount(
            apply,
            name="plugin",
            runtime=_runtime("plugin", tmp_path / suffix),
            inject=(CHANNELS,),
        )
        snapshot = _freeze_plugin_channels(
            channels,
            root.instance_token,
            factory_provenance_by_owner={
                "plugin": ChannelFactoryFreezeInput(
                    generation_id="same-generation",
                    source_revision="same-source",
                    config_revision="same-config",
                )
            },
        )
        identities.append(snapshot.identity)
        assert tuple(item.channel_name for item in snapshot.factories) == (
            "feishu",
            "qqbot",
        )
        assert all(item.plugin_id == "plugin" for item in snapshot.factories)
        assert snapshot.root_instance_token is root.instance_token
        await root.dispose()
    assert identities[0] == identities[1]


def test_channel_config_revision_uses_redacted_projection() -> None:
    first = {
        "app_id": "app-1",
        "app_secret": CredentialRef(("app_secret",)),
        "options": {"retry": 2, "delay": 0.25},
    }
    reordered = {
        "options": {"delay": 0.25, "retry": 2},
        "app_secret": CredentialRef(("app_secret",)),
        "app_id": "app-1",
    }
    changed = {**first, "app_id": "app-2"}

    assert channel_config_revision(first) == channel_config_revision(reordered)
    assert channel_config_revision(first) != channel_config_revision(changed)


def test_channel_declarations_and_provenance_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        _ = _definition("BadName")
    with pytest.raises(ValueError):
        _ = ChannelDefinition(
            name="feishu",
            capabilities=frozenset({"inbound"}),  # type: ignore[arg-type]
            factory_export=lambda: None,  # type: ignore[arg-type]
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            credential_paths=("app_id", "app_id"),
        )
    with pytest.raises(ValueError):
        _ = CredentialRef(("app_secret", ".."))


def test_channel_inbound_identity_matches_declared_capability() -> None:
    outbound = ChannelDefinition(
        name="push",
        capabilities=frozenset({ChannelCapability.OUTBOUND}),
        factory_export="push:build_channel",
        inbound_identity=None,
        credential_paths=("token",),
    )
    assert outbound.inbound_identity is None

    with pytest.raises(ValueError, match="必须声明 inbound_identity"):
        _ = ChannelDefinition(
            name="inbound",
            capabilities=frozenset({ChannelCapability.INBOUND}),
            factory_export="inbound:build_channel",
            inbound_identity=None,
            credential_paths=("token",),
        )
    with pytest.raises(ValueError, match="不得声明 inbound_identity"):
        _ = ChannelDefinition(
            name="push",
            capabilities=frozenset({ChannelCapability.OUTBOUND}),
            factory_export="push:build_channel",
            inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
            credential_paths=("token",),
        )


def test_channel_snapshot_identity_is_content_addressed() -> None:
    descriptor = _definition()
    frozen_descriptor = ChannelDescriptor(
        owner="plugin",
        name=descriptor.name,
        capabilities=tuple(sorted(descriptor.capabilities, key=lambda item: item.value)),
        factory_export=descriptor.factory_export,
        inbound_identity=descriptor.inbound_identity,
        credential_paths=descriptor.credential_paths,
    )
    provenance = _provenance("feishu")
    snapshot = ChannelRegistrySnapshot(
        descriptors=(frozen_descriptor,),
        factories=(provenance,),
        identity=_registry_identity((frozen_descriptor,), (provenance,)),
        root_instance_token=object(),
    )
    assert snapshot.identity
    with pytest.raises(ValueError, match="identity"):
        _ = ChannelRegistrySnapshot(
            descriptors=snapshot.descriptors,
            factories=snapshot.factories,
            identity="not-the-digest",
            root_instance_token=object(),
        )

    with pytest.raises(ValueError, match="名称重复"):
        _ = ChannelRegistrySnapshot(
            descriptors=(frozen_descriptor, frozen_descriptor),
            factories=(provenance, provenance),
            identity="unused",
            root_instance_token=object(),
        )


def test_channel_factory_context_freezes_config_and_credential_refs() -> None:
    class ProviderFactory:
        async def create(self, credentials):  # type: ignore[no-untyped-def]
            raise AssertionError(credentials)

        async def aclose(self) -> None:
            return None

    raw = {"options": {"retry": [1, 2]}, "token": CredentialRef(("token",))}
    context = ChannelFactoryContext(
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        config=raw,
        credentials={"token": CredentialRef(("token",))},
        provider_client_factory=ProviderFactory(),
        ingress=_Ingress(),
        identity=None,
    )

    raw["options"] = {"retry": [99]}
    assert context.config["options"]["retry"] == (1, 2)  # type: ignore[index]
    assert context.credentials["token"] == CredentialRef(("token",))
    with pytest.raises(TypeError):
        context.config["new"] = "value"  # type: ignore[index]
    with pytest.raises(ValueError, match="path 与 ref"):
        _ = ChannelFactoryContext(
            snapshot_id="snapshot",
            generation_id="generation",
            binding_token="binding",
            config={},
            credentials={"token": CredentialRef(("other",))},
            provider_client_factory=ProviderFactory(),
            ingress=_Ingress(),
            identity=None,
        )


def test_channel_provider_delivery_and_cleanup_receipts_are_typed() -> None:
    request = ProviderDeliveryRequest(
        binding_token="binding",
        delivery_id="delivery",
        recipient="recipient",
        body="",
        commit_role=ChannelCommitRole.PASSIVE,
        thinking="thinking",
        reply_to="reply",
        session_message_id="message",
        control_turn_id="turn",
        execution_attempt_id="attempt",
        terminal_status=ChannelTerminalStatus.COMPLETED,
    )
    receipt = ProviderDeliveryReceipt(
        delivery_id=request.delivery_id,
        status=DeliveryStatus.DELIVERED,
        provider_ids=("remote-1",),
    )
    failure = ChannelCleanupFailure(
        stage="stop",
        plugin_id="plugin",
        generation_id="generation",
        binding_token=request.binding_token,
        resource="adapter",
        error_type="RuntimeError",
        message="stop failed",
        retry_action="retry_generation_cleanup",
    )

    assert ChannelReady(request.binding_token).admission_open is False
    assert request.commit_role is ChannelCommitRole.PASSIVE
    assert request.terminal_status is ChannelTerminalStatus.COMPLETED
    assert receipt.status is DeliveryStatus.DELIVERED
    assert StopReceipt(
        request.binding_token,
        resources_closed=False,
        failures=(failure,),
    ).failures == (failure,)


def test_attachment_ref_and_channel_payloads_are_frozen_and_typed() -> None:
    attachment = _attachment(
        kind=AttachmentKind.IMAGE,
        filename=None,
        media_type="image/png",
    )
    message = ChannelInboundMessage(
        channel="feishu",
        sender="sender",
        chat_id="chat",
        content="hello",
        timestamp=datetime.now(timezone.utc),
        metadata={},
        attachments=(attachment,),
    )
    outbound = OutboundEnvelope(
        logical_delivery_id="delivery",
        delivery_id="delivery",
        attempt_sequence=1,
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        channel="feishu",
        recipient="chat",
        body="hello",
        metadata={},
        attachments=(attachment,),
    )
    request = ProviderDeliveryRequest(
        binding_token="binding",
        delivery_id="delivery",
        recipient="chat",
        body="hello",
        attachments=(attachment,),
    )
    push = PushToolRequest(
        channel="feishu",
        recipient="chat",
        body="hello",
        metadata={},
        attachments=(attachment,),
    )

    assert message.attachments == (attachment,)
    assert outbound.attachments == (attachment,)
    assert request.attachments == (attachment,)
    assert push.attachments == (attachment,)
    with pytest.raises((AttributeError, TypeError)):
        attachment.artifact_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError, match="attachments 必须是 tuple"):
        _ = PushToolRequest(
            channel="feishu",
            recipient="chat",
            body="hello",
            metadata={},
            attachments=[attachment],  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="AttachmentRef"):
        _ = ProviderDeliveryRequest(
            binding_token="binding",
            delivery_id="delivery",
            recipient="chat",
            body="hello",
            attachments=("not-a-ref",),  # type: ignore[arg-type]
        )


def test_attachment_ref_rejects_unsafe_identity_metadata_and_digest() -> None:
    cases = (
        {"artifact_id": "../escape"},
        {"artifact_id": "/absolute"},
        {"kind": "file"},
        {"filename": "../report.txt"},
        {"filename": ""},
        {"media_type": "text"},
        {"size_bytes": -1},
        {"size_bytes": True},
        {"sha256": "A" * 64},
        {"sha256": "a" * 63},
    )
    for overrides in cases:
        values = {
            "artifact_id": "artifact-1",
            "kind": AttachmentKind.FILE,
            "filename": "report.txt",
            "media_type": "text/plain",
            "size_bytes": 3,
            "sha256": "a" * 64,
        }
        values.update(overrides)
        with pytest.raises((TypeError, ValueError)):
            _ = AttachmentRef(**values)  # type: ignore[arg-type]


def test_attachment_ports_are_exported_and_factory_context_validates_them() -> None:
    class ImportPort:
        async def import_bytes(
            self,
            data,
            *,
            kind,
            filename,
            media_type,
        ):  # type: ignore[no-untyped-def]
            raise AssertionError((data, kind, filename, media_type))

    class ReadPort:
        async def acquire(self, ref):  # type: ignore[no-untyped-def]
            raise AssertionError(ref)

    class ProviderFactory:
        async def create(self, credentials):  # type: ignore[no-untyped-def]
            raise AssertionError(credentials)

        async def aclose(self) -> None:
            return None

    context = ChannelFactoryContext(
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        config={},
        credentials={},
        provider_client_factory=ProviderFactory(),
        ingress=None,
        identity=None,
        attachment_import=ImportPort(),
        attachment_read=ReadPort(),
    )
    assert context.attachment_import is not None
    assert callable(context.attachment_import.import_bytes)
    assert context.attachment_read is not None
    assert callable(context.attachment_read.acquire)
    assert ChannelAttachmentImportPort
    assert ChannelAttachmentReadPort
    assert AttachmentReadLease

    with pytest.raises(TypeError, match="attachment_import"):
        _ = ChannelFactoryContext(
            snapshot_id="snapshot",
            generation_id="generation",
            binding_token="binding",
            config={},
            credentials={},
            provider_client_factory=ProviderFactory(),
            ingress=None,
            identity=None,
            attachment_import=object(),  # type: ignore[arg-type]
        )


def test_c14c_metadata_is_recursively_frozen_and_rejects_unsafe_values() -> None:
    metadata = json.loads('{"nested": {"items": [1, "two"]}}')
    message = ChannelInboundMessage(
        channel="feishu",
        sender="sender",
        chat_id="chat",
        content="hello",
        timestamp=datetime.now(timezone.utc),
        metadata=metadata,
    )
    outbound = OutboundEnvelope(
        logical_delivery_id="delivery",
        delivery_id="delivery",
        attempt_sequence=1,
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        channel="feishu",
        recipient="chat",
        body="hello",
        metadata=metadata,
    )
    push = PushToolRequest(
        channel="feishu",
        recipient="chat",
        body="hello",
        metadata=metadata,
    )
    metadata["nested"]["items"].append("source mutation")
    assert message.metadata["nested"]["items"] == (1, "two")  # type: ignore[index]
    assert outbound.metadata["nested"]["items"] == (1, "two")  # type: ignore[index]
    assert push.metadata["nested"]["items"] == (1, "two")  # type: ignore[index]

    with pytest.raises(TypeError):
        message.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(ValueError, match="非有限"):
        _ = ChannelInboundMessage(
            channel="feishu",
            sender="sender",
            chat_id="chat",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={"bad": float("nan")},
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        _ = ChannelInboundMessage(
            channel="feishu",
            sender="sender",
            chat_id="chat",
            content="hello",
            timestamp=datetime.now(),
            metadata={},
        )
    with pytest.raises(TypeError, match="值类型无效"):
        _ = PushToolRequest(
            channel="feishu",
            recipient="chat",
            body="hello",
            metadata={"bad": {"not", "json"}},  # type: ignore[dict-item]
        )


def test_channel_text_accepts_layout_controls_and_rejects_nul() -> None:
    body = "line one\nline two\tvalue\r\n"
    message = ChannelInboundMessage(
        channel="feishu",
        sender="sender",
        chat_id="chat",
        content=body,
        timestamp=datetime.now(timezone.utc),
        metadata={},
    )
    outbound = OutboundEnvelope(
        logical_delivery_id="delivery",
        delivery_id="delivery",
        attempt_sequence=1,
        snapshot_id="snapshot",
        generation_id="generation",
        binding_token="binding",
        channel="feishu",
        recipient="chat",
        body=body,
        metadata={},
    )

    assert message.content == body
    assert outbound.body == body
    with pytest.raises(ValueError, match="控制字符"):
        _ = PushToolRequest(
            channel="feishu",
            recipient="chat",
            body="unsafe\x00body",
            metadata={},
        )


def test_raw_inbound_and_outbound_receipts_enforce_identity_contract() -> None:
    envelope = _inbound_envelope()
    raw = RawInbound(message_id="provider-message", message=envelope.message)
    assert raw.message is envelope.message
    with pytest.raises(ValueError, match="1～256"):
        _ = RawInbound(message_id="x" * 257, message=envelope.message)
    assert ChannelDeliveryReceipt(
        delivery_id="delivery",
        status=DeliveryStatus.UNKNOWN,
        error="provider effect uncertain",
    ).status is DeliveryStatus.UNKNOWN
    assert QueuedReceipt(delivery_id="delivery", queued=True).queued is True

    with pytest.raises(ValueError, match="首次 delivery"):
        _ = OutboundEnvelope(
            logical_delivery_id="logical",
            delivery_id="delivery",
            attempt_sequence=1,
            snapshot_id="snapshot",
            generation_id="generation",
            binding_token="binding",
            channel="feishu",
            recipient="chat",
            body="hello",
            metadata={},
        )
    with pytest.raises(ValueError, match="新的 delivery_id"):
        _ = OutboundEnvelope(
            logical_delivery_id="delivery",
            delivery_id="delivery",
            attempt_sequence=2,
            snapshot_id="snapshot",
            generation_id="generation",
            binding_token="binding",
            channel="feishu",
            recipient="chat",
            body="hello",
            metadata={},
        )


@pytest.mark.asyncio
async def test_inbound_handoff_rejects_owner_jump_and_old_owner_close() -> None:
    envelope = _inbound_envelope()
    with pytest.raises(CompositionError, match="不能从"):
        envelope.handoff(InboundOwner.INGRESS, InboundOwner.LANE)

    assert envelope.handoff(InboundOwner.INGRESS, InboundOwner.BUS) is envelope
    with pytest.raises(CompositionError, match="当前 owner"):
        await envelope.close(InboundOwner.INGRESS)

    assert envelope.handoff(InboundOwner.BUS, InboundOwner.LANE) is envelope
    assert envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP) is envelope
    with pytest.raises(CompositionError, match="当前 owner"):
        await envelope.close(InboundOwner.BUS)
    envelope.lease.release.set()  # type: ignore[attr-defined]
    await envelope.close(InboundOwner.LOOP)
    with pytest.raises(CompositionError, match="terminal"):
        envelope.handoff(InboundOwner.LOOP, InboundOwner.BUS)


@pytest.mark.asyncio
async def test_inbound_close_is_idempotent_only_for_exact_owner() -> None:
    lease = _Lease()
    envelope = _inbound_envelope(lease)
    lease.release.set()
    await envelope.close(InboundOwner.INGRESS)
    await envelope.close(InboundOwner.INGRESS)
    assert lease.close_calls == 1
    assert envelope.owner is InboundOwner.CLOSED
    assert envelope.state is InboundState.TERMINAL
    with pytest.raises(CompositionError, match="另一 owner"):
        await envelope.close(InboundOwner.BUS)


@pytest.mark.asyncio
async def test_inbound_close_completes_lease_before_propagating_cancellation() -> None:
    lease = _Lease()
    envelope = _inbound_envelope(lease)
    close_task = asyncio.create_task(envelope.close(InboundOwner.INGRESS))
    await lease.started.wait()
    close_task.cancel()
    lease.release.set()

    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert lease.close_calls == 1
    assert envelope.owner is InboundOwner.CLOSED
    assert envelope.state is InboundState.TERMINAL
    await envelope.close(InboundOwner.INGRESS)
