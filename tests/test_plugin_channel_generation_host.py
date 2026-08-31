from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, cast

import pytest

from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    AttachmentReadLease,
    ChannelCapability,
    ChannelCommitRole,
    ChannelDeliveryReceipt,
    ChannelDefinition,
    ChannelFactoryFreezeInput,
    ChannelReady,
    ChannelTerminalStatus,
    ChannelInboundMessage,
    CommittedChannelCatalog,
    ControlResponseBodies,
    CoreChannelDefinition,
    InboundIdentity,
    CredentialRef,
    DeliveryStatus,
    InboundEnvelope,
    InboundOwner,
    OutboundEnvelope,
    PluginChannels,
    ProviderClient,
    ProviderClientFactory,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    PresentationReceipt,
    RawInbound,
    StreamDeltaPresentation,
    StopReceipt,
    TurnStartedPresentation,
    TurnStreamEvent,
    TurnStreamEventKind,
    _freeze_plugin_channels,
    channel_config_revision,
)
from agent.plugin_composition.model import CompositionError, ServiceKey
from agent.plugins.channel_generation_host import (
    ChannelBindingLease,
    ChannelGenerationHost,
    ChannelStartRecord,
    bind_channel_turn_binding,
    get_current_channel_turn_binding,
    reset_channel_turn_binding,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.generation import GateResult, PluginContributions, PluginGeneration
from agent.plugins.snapshot import RuntimeSnapshotCompiler, RuntimeSnapshotLease, RuntimeSnapshotStore
from bus.queue import MessageBus
from bus.event_bus import EventBus
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnOutputCompleted,
    TurnStarted,
)
from bootstrap.channel_presentation import ChannelTurnPresentationBridge
from session.manager import SessionManager


def _diagnostic_fields(record: logging.LogRecord) -> dict[str, object]:
    return cast(dict[str, object], getattr(record, "akashic_fields"))


@dataclass
class ClientFactory:
    created: int = 0
    closed: int = 0
    fail_close: bool = False

    async def create(
        self,
        credentials: Mapping[str, CredentialRef],
    ) -> ProviderClient:
        self.created += 1
        return _Client(credentials)

    async def aclose(self) -> None:
        self.closed += 1
        if self.fail_close:
            raise RuntimeError("factory close failed")


class _Client:
    def __init__(self, credentials: Mapping[str, CredentialRef]) -> None:
        self._credentials = credentials

    def credential(self, ref: CredentialRef) -> str:
        if ref not in self._credentials.values():
            raise AssertionError(f"unexpected credential: {ref.path}")
        return "test-credential"

    async def aclose(self) -> None:
        return None


class Adapter:
    def __init__(
        self,
        context: Any,
        *,
        fail_start: bool = False,
        fail_stop: bool = False,
        block_stop: bool = False,
        wrong_receipt: bool = False,
        cancel_stop: bool = False,
        cancel_start: bool = False,
    ) -> None:
        self.context = context
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.block_stop = block_stop
        self.wrong_receipt = wrong_receipt
        self.cancel_stop = cancel_stop
        self.cancel_start = cancel_start
        self.started = 0
        self.stopped = 0
        self.deliveries: list[str] = []
        self.requests: list[ProviderDeliveryRequest] = []
        self.release = asyncio.Event()
        self.stop_started = asyncio.Event()
        self.stop_release = asyncio.Event()
        self.runtime_events: list[str] = []

    def attach_runtime(self, ports: Any) -> None:
        self.runtime_events.append("attach")
        self.runtime_ports = ports

    def open_admission(self) -> None:
        self.runtime_events.append("open")

    def close_admission(self) -> None:
        self.runtime_events.append("close")

    async def start(self) -> ChannelReady:
        self.started += 1
        if self.cancel_start:
            raise asyncio.CancelledError
        if self.fail_start:
            raise RuntimeError("start failed")
        return ChannelReady(self.context.binding_token)

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        self.deliveries.append(request.delivery_id)
        self.requests.append(request)
        if not self.release.is_set():
            await self.release.wait()
        delivery_id = "wrong" if self.wrong_receipt else request.delivery_id
        return ProviderDeliveryReceipt(delivery_id, DeliveryStatus.DELIVERED, ("p1",))

    async def stop(self) -> StopReceipt:
        self.stopped += 1
        self.stop_started.set()
        if self.block_stop:
            await self.stop_release.wait()
        if self.cancel_stop:
            raise asyncio.CancelledError
        if self.fail_stop:
            raise RuntimeError("stop failed")
        return StopReceipt(self.context.binding_token, True)


class PresentationAdapter(Adapter):
    def attach_presentation(self, ports: Any) -> None:
        self.presentation_ports = ports


class IncompleteStopAdapter(Adapter):
    def __init__(self, context: Any, **kwargs: Any) -> None:
        super().__init__(context, **kwargs)
        self.resources_closed = False

    async def stop(self) -> StopReceipt:
        self.stopped += 1
        self.stop_started.set()
        return StopReceipt(self.context.binding_token, self.resources_closed)


async def _noop_record(record: ChannelStartRecord) -> None:
    return None


async def _noop_failure(failure: Any) -> None:
    return None


def _host(**kwargs: Any) -> ChannelGenerationHost:
    kwargs.setdefault("on_before_start", _noop_record)
    kwargs.setdefault("config_revision_checker", _noop_record)
    kwargs.setdefault("on_failure", _noop_failure)
    return ChannelGenerationHost(**kwargs)


class _FakeSnapshotLease(RuntimeSnapshotLease):
    def __init__(
        self,
        snapshot: Any,
        *,
        release_gate: asyncio.Event | None = None,
    ) -> None:
        self.snapshot = snapshot
        self._active = True
        self.release_gate = release_gate
        self.forks: list[_FakeSnapshotLease] = []

    @property
    def active(self) -> bool:
        return self._active

    def fork(self) -> _FakeSnapshotLease:
        if not self.active:
            raise RuntimeError("lease closed")
        child = _FakeSnapshotLease(
            self.snapshot,
            release_gate=self.release_gate,
        )
        self.forks.append(child)
        return child

    async def release(self) -> None:
        if not self.active:
            return
        if self.release_gate is not None:
            await self.release_gate.wait()
        self._active = False


def _attachment_ref() -> AttachmentRef:
    return AttachmentRef(
        artifact_id="artifact-1",
        kind=AttachmentKind.FILE,
        filename="report.txt",
        media_type="text/plain",
        size_bytes=5,
        sha256="a" * 64,
    )


class _FakeAttachmentReadLease:
    def __init__(
        self,
        ref: AttachmentRef,
        *,
        close_started: asyncio.Event | None = None,
        close_release: asyncio.Event | None = None,
    ) -> None:
        self.ref = ref
        self.close_started = close_started
        self.close_release = close_release
        self.close_calls = 0

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        assert max_bytes >= 5
        return b"hello"

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_started is not None:
            self.close_started.set()
        if self.close_release is not None:
            await self.close_release.wait()


class _FakeAttachmentImportPort:
    def __init__(self, ref: AttachmentRef) -> None:
        self.ref = ref
        self.calls = 0
        self.fail = False
        self.gate: asyncio.Event | None = None

    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        self.calls += 1
        assert data == b"hello"
        assert kind is AttachmentKind.FILE
        assert filename == "report.txt"
        assert media_type == "text/plain"
        if self.fail:
            raise OSError("import failed")
        if self.gate is not None:
            await self.gate.wait()
        return self.ref


class _FakeAttachmentReadPort:
    def __init__(self, lease: AttachmentReadLease) -> None:
        self.lease = lease
        self.calls = 0
        self.fail = False
        self.gate: asyncio.Event | None = None

    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease:
        self.calls += 1
        assert ref == self.lease.ref
        if self.fail:
            raise OSError("acquire failed")
        if self.gate is not None:
            await self.gate.wait()
        return self.lease


def _module(
    *,
    name: str = "feishu",
    factory_name: str = "make_adapter",
    adapter_cls: type[Adapter] = Adapter,
) -> ModuleType:
    module = ModuleType(f"plugins.{name}")
    module.api_version = 3  # type: ignore[attr-defined]
    module.name = name  # type: ignore[attr-defined]
    module.version = "1"  # type: ignore[attr-defined]
    module.inject = (ServiceKey("core.channels"),)  # type: ignore[attr-defined]
    async def apply(ctx: Any, config: Any) -> None:
        return None

    module.apply = apply  # type: ignore[attr-defined]
    setattr(module, factory_name, lambda context: adapter_cls(context))
    return module


async def _make_snapshot(
    *,
    module: ModuleType | None = None,
    adapter_cls: type[Adapter] = Adapter,
    fail_start: bool = False,
    fail_stop: bool = False,
    block_stop: bool = False,
    wrong_receipt: bool = False,
    cancel_stop: bool = False,
    cancel_start: bool = False,
    cancel_factory: bool = False,
    fail_after: int | None = None,
    factory_events: list[str] | None = None,
    capabilities: frozenset[ChannelCapability] = frozenset(
        {ChannelCapability.OUTBOUND}
    ),
) -> tuple[Any, dict[str, ClientFactory], dict[str, Adapter]]:
    module = module or _module(adapter_cls=adapter_cls)
    plugin = ComposablePlugin.from_module(module)
    root_token = object()
    channels = PluginChannels(root_token)
    from agent.plugin_composition.channels import CredentialRef

    class Fiber:
        activation_token = object()

    class Runtime:
        plugin_id = "plugin.feishu"
        config = {"app_secret": CredentialRef(("app_secret",))}

    class Context:
        fiber = Fiber()
        runtime = Runtime()
        generation_id = "gen-1"

        def report_incident(self, *args: Any) -> Any:
            return None

        def require(self, key: Any) -> Any:
            return channels

        def _root_instance_token(self) -> object:
            return root_token

        async def effect(self, setup: Any, label: str) -> Any:
            setup()
            return SimpleNamespace(aclose=lambda: None)

        async def health(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace()

    channel_names = tuple(getattr(module, "channel_names", ("feishu",)))
    config_projection: dict[str, object] = {
        "app_secret": CredentialRef(("app_secret",))
    }
    for channel_name in channel_names:
        await channels.register(
            cast(Any, Context()),
            ChannelDefinition(
                name=channel_name,
                capabilities=capabilities,
                factory_export="make_adapter",
                inbound_identity=(
                    InboundIdentity.PROVIDER_MESSAGE_ID
                    if ChannelCapability.INBOUND in capabilities
                    else None
                ),
                credential_paths=("app_secret",),
            ),
        )
    registry = _freeze_plugin_channels(
        channels,
        root_token,
        factory_provenance_by_owner={
            "plugin.feishu": ChannelFactoryFreezeInput(
                "gen-1",
                "source-1",
                channel_config_revision(config_projection),
            )
        },
    )
    generation = PluginGeneration(
        plugin_id="plugin.feishu",
        generation_id="gen-1",
        module_path="plugins/feishu/plugin.py",
        source_revision="source-1",
        config_revision="raw-config-1",
        plugin_dir=__import__("pathlib").Path("/tmp/plugin"),
        data_dir=__import__("pathlib").Path("/tmp/plugin-data"),
        config={"app_secret": "raw-secret"},
        instance=plugin,
        scope=cast(Any, object()),
        contributions=PluginContributions(manifest={}),
        gate_result=GateResult("test", "plugin.feishu", "rev", "passed", ()),
        config_projection=config_projection,
    )
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-1",
        state="committed",
        composition_root=SimpleNamespace(instance_token=root_token),
        channel_registry=registry,
        channel_registry_identity=registry.identity,
        generations={"plugin.feishu": generation},
    )
    adapters: dict[str, Adapter] = {}
    factory_count = 0

    def factory(context: Any) -> Adapter:
        nonlocal factory_count
        factory_count += 1
        if factory_events is not None:
            factory_events.append("factory")
        if cancel_factory:
            raise asyncio.CancelledError
        adapter = adapter_cls(
            context,
            fail_start=fail_start or (fail_after is not None and factory_count >= fail_after),
            fail_stop=fail_stop,
            block_stop=block_stop,
            wrong_receipt=wrong_receipt,
            cancel_stop=cancel_stop,
            cancel_start=cancel_start,
        )
        adapters[context.binding_token] = adapter
        return adapter

    setattr(module, "make_adapter", factory)
    return snapshot, {channel_name: ClientFactory() for channel_name in channel_names}, adapters


@pytest.mark.asyncio
async def test_formal_binding_starts_closed_and_delivers_after_open() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    binding = generation.channel("feishu")
    assert not binding.admission_open
    with pytest.raises(RuntimeError, match="关闭"):
        await binding.deliver(ProviderDeliveryRequest(binding.binding_token, "d1", "u", "hi"))
    binding.open_admission()
    request = ProviderDeliveryRequest(binding.binding_token, "d1", "u", "hi")
    task = asyncio.create_task(binding.deliver(request))
    await asyncio.sleep(0)
    assert binding.in_flight == 1
    binding.close_admission()
    assert binding.in_flight == 1
    next_request = ProviderDeliveryRequest(binding.binding_token, "d2", "u", "hi")
    with pytest.raises(RuntimeError, match="关闭"):
        await binding.deliver(next_request)
    for adapter in adapters.values():
        adapter.release.set()
    receipt = await task
    assert receipt.delivery_id == "d1"
    await generation.stop()
    assert factories["feishu"].closed == 1


@pytest.mark.asyncio
async def test_plugin_channel_callbacks_share_generic_diagnostic_boundary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    binding = generation.channel("feishu")
    binding.open_admission()
    adapters[binding.binding_token].release.set()
    _ = await binding.deliver(
        ProviderDeliveryRequest(binding.binding_token, "d1", "u", "hi")
    )
    binding.close_admission()
    _ = await generation.stop()

    terminals = [
        _diagnostic_fields(record)
        for record in caplog.records
        if _diagnostic_fields(record).get("event") == "plugin.operation.done"
    ]
    assert {item["operation"] for item in terminals} == {
        "channel.factory",
        "channel.attach_runtime",
        "channel.attach_presentation",
        "channel.start",
        "channel.open_admission",
        "channel.deliver",
        "channel.close_admission",
        "channel.stop",
    }
    assert all(item["plugin_id"] == "plugin.feishu" for item in terminals)
    assert all(item["generation_id"] == "gen-1" for item in terminals)
    assert all(item["plugin_entrypoint"] == "feishu" for item in terminals)


@pytest.mark.asyncio
async def test_inbound_runtime_attaches_closed_then_opens_and_closes_before_drain() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset({ChannelCapability.INBOUND})
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    adapter = adapters[next(iter(adapters))]
    assert adapter.runtime_events == ["attach"]

    binding = generation.channel("feishu")
    assert not binding.admission_open
    binding.open_admission()
    assert adapter.runtime_events == ["attach", "open"]
    binding.close_admission()
    assert adapter.runtime_events == ["attach", "open", "close"]

    await generation.stop()
    assert adapter.runtime_events == ["attach", "open", "close"]
    assert factories["feishu"].closed == 1


@pytest.mark.asyncio
async def test_c14d_control_uses_exact_binding_and_bounded_dedupe() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    sources: list[_FakeSnapshotLease] = []
    requested_snapshot_ids: list[str] = []

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        requested_snapshot_ids.append(snapshot_id)
        source = _FakeSnapshotLease(snapshot)
        sources.append(source)
        return source

    interrupted: list[str] = []
    dispatched: list[tuple[str, str]] = []

    async def interrupt(raw: RawInbound) -> bool:
        interrupted.append(raw.message_id)
        return raw.message_id != "stop-idle"

    async def dispatch(envelope: OutboundEnvelope, binding: Any) -> ChannelDeliveryReceipt:
        dispatched.append((envelope.binding_token, binding.binding_token))
        return ChannelDeliveryReceipt(envelope.delivery_id, DeliveryStatus.DELIVERED)

    host = _host(
        snapshot_lease_acquirer=acquire,
        control_interrupter=interrupt,
        control_response_dispatcher=dispatch,
    )
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    ports = adapter.presentation_ports
    assert ports.control is not None
    assert ports.turn_stream is not None
    raw = RawInbound(
        message_id="stop-1",
        provider_identity="account-1",
        recipient="chat-1",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="user",
            chat_id="chat-1",
            content="/stop",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )

    first = await ports.control.interrupt(
        raw,
        response_bodies=ControlResponseBodies("interrupted", "idle"),
    )
    duplicate = await ports.control.interrupt(
        raw,
        response_bodies=ControlResponseBodies("interrupted", "idle"),
    )
    assert first.accepted is True
    assert first.reason == "interrupted"
    assert first.response is not None
    assert duplicate.accepted is False and duplicate.reason == "duplicate"
    assert interrupted == ["stop-1"]
    assert len(dispatched) == 1
    assert dispatched[0][0] == dispatched[0][1]
    assert len(sources) == 1
    assert not sources[0].active and not sources[0].forks[0].active
    idle = RawInbound(
        message_id="stop-idle",
        provider_identity="account-1",
        recipient="chat-1",
        message=raw.message,
    )
    idle_receipt = await ports.control.interrupt(
        idle,
        response_bodies=ControlResponseBodies("interrupted", "idle"),
    )
    assert idle_receipt.accepted is False and idle_receipt.reason == "idle"
    assert idle_receipt.response is not None
    assert requested_snapshot_ids == [snapshot.snapshot_id, snapshot.snapshot_id]
    await generation.stop()


@pytest.mark.asyncio
async def test_c14d_control_cancelled_during_source_release_closes_binding() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    release_gate = asyncio.Event()
    sources: list[_FakeSnapshotLease] = []

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        source = _FakeSnapshotLease(snapshot, release_gate=release_gate)
        sources.append(source)
        return source

    host = _host(snapshot_lease_acquirer=acquire)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    control = tuple(adapters.values())[0].presentation_ports.control
    assert control is not None
    task = asyncio.create_task(
        control.interrupt(
            RawInbound(
                message_id="cancel-source-release",
                message=ChannelInboundMessage(
                    channel="feishu",
                    sender="sender",
                    chat_id="chat",
                    content="/stop",
                    timestamp=datetime.now(timezone.utc),
                    metadata={},
                ),
            ),
            response_bodies=ControlResponseBodies("stopped", "idle"),
        )
    )
    while not sources or not sources[0].forks:
        await asyncio.sleep(0)

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not sources[0].active
    assert not sources[0].forks[0].active
    assert generation.channel("feishu").in_flight == 0
    assert not host._binding_leases
    await generation.stop()


@pytest.mark.asyncio
async def test_c14d_existing_turn_lease_can_finish_stream_after_close_admission() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host(snapshot_lease_acquirer=lambda snapshot_id: _lease_for(snapshot))
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    events: list[str] = []
    callback_started = asyncio.Event()
    callback_release = asyncio.Event()

    async def callback(event: TurnStreamEvent) -> PresentationReceipt:
        events.append(event.kind.value)
        if event.kind is TurnStreamEventKind.STREAM_DELTA:
            callback_started.set()
            await callback_release.wait()
        return PresentationReceipt(event.presentation_id, DeliveryStatus.DELIVERED)

    ports = adapter.presentation_ports
    assert ports.turn_stream is not None
    subscription = ports.turn_stream.subscribe(callback)
    binding = generation.channel("feishu")
    source = _FakeSnapshotLease(snapshot)
    lease = host.acquire_binding(source, "feishu")
    await binding.publish_turn_event(
        TurnStreamEvent(
            "preview-1",
            TurnStreamEventKind.TURN_STARTED,
            TurnStartedPresentation("turn-1", "client-1"),
        )
    )
    binding.close_admission()
    blocked = asyncio.create_task(
        lease.publish_turn_event(
            TurnStreamEvent(
                "preview-1",
                TurnStreamEventKind.STREAM_DELTA,
                StreamDeltaPresentation("turn-1", 1, "hello", ""),
            )
        )
    )
    await callback_started.wait()
    stop = asyncio.create_task(generation.stop())
    await asyncio.sleep(0)
    assert not stop.done()
    callback_release.set()
    assert len(await blocked) == 1
    await lease.aclose()
    await stop
    await subscription.close()
    assert events == ["turn.started", "stream.delta"]


@pytest.mark.asyncio
async def test_c14d_control_claim_blocks_old_binding_drain_before_lease_acquire() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    interrupt_started = asyncio.Event()
    interrupt_release = asyncio.Event()

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        return _FakeSnapshotLease(snapshot)

    async def interrupt(_raw: RawInbound) -> str:
        interrupt_started.set()
        await interrupt_release.wait()
        return "interrupted"

    async def dispatch(
        envelope: OutboundEnvelope,
        _binding: object,
    ) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(envelope.delivery_id, DeliveryStatus.DELIVERED)

    host = _host(
        snapshot_lease_acquirer=acquire,
        control_interrupter=interrupt,
        control_response_dispatcher=dispatch,
    )
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    control = tuple(adapters.values())[0].presentation_ports.control
    assert control is not None
    raw = RawInbound(
        message_id="control-race",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="sender",
            chat_id="chat",
            content="/stop",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )
    task = asyncio.create_task(
        control.interrupt(
            raw,
            response_bodies=ControlResponseBodies("stopped", "idle"),
        )
    )
    await interrupt_started.wait()
    generation.channel("feishu").close_admission()
    drain = asyncio.create_task(generation.channel("feishu").drain())
    await asyncio.sleep(0)
    assert not drain.done()
    interrupt_release.set()
    assert (await task).accepted is True
    await drain
    await generation.stop()


@pytest.mark.asyncio
async def test_c14d_control_claim_survives_real_store_provisional_pause() -> None:
    prototype, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    compiler = RuntimeSnapshotCompiler()
    stable = compiler.compile(prototype.generations, snapshot_revision="stable")
    latest = compiler.compile(prototype.generations, snapshot_revision="latest")
    store = RuntimeSnapshotStore()
    store.install(stable)
    stable.composition_root = prototype.composition_root
    stable.channel_registry = prototype.channel_registry
    stable.channel_registry_identity = prototype.channel_registry_identity
    interrupted = asyncio.Event()
    release = asyncio.Event()

    async def interrupt(_raw: RawInbound) -> str:
        interrupted.set()
        await release.wait()
        return "interrupted"

    async def dispatch(
        envelope: OutboundEnvelope,
        _binding: object,
    ) -> ChannelDeliveryReceipt:
        return ChannelDeliveryReceipt(envelope.delivery_id, DeliveryStatus.DELIVERED)

    host = _host(
        snapshot_lease_acquirer=store.lease,
        control_interrupter=interrupt,
        control_response_dispatcher=dispatch,
    )
    generation = await host.start_formal(stable, factories)
    generation.open_admission()
    control = tuple(adapters.values())[0].presentation_ports.control
    assert control is not None
    task = asyncio.create_task(
        control.interrupt(
            RawInbound(
                message_id="control-provisional",
                message=ChannelInboundMessage(
                    channel="feishu",
                    sender="sender",
                    chat_id="chat",
                    content="/stop",
                    timestamp=datetime.now(timezone.utc),
                    metadata={},
                ),
            ),
            response_bodies=ControlResponseBodies("stopped", "idle"),
        )
    )
    await interrupted.wait()
    transaction = store.begin_publish(latest)
    await store.commit_provisional(transaction)
    generation.channel("feishu").close_admission()
    drain = asyncio.create_task(generation.channel("feishu").drain())
    await asyncio.sleep(0)
    assert not drain.done()
    release.set()
    assert (await task).accepted is True
    await drain
    await generation.stop()
    await store.rollback_provisional(transaction, keep_candidate_latest=False)
    await store.abort(transaction)
    await store.close()


@pytest.mark.asyncio
async def test_c14d_production_bridge_preserves_old_binding_and_typed_sequence() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    received: list[TurnStreamEvent] = []

    async def callback(event: TurnStreamEvent) -> PresentationReceipt:
        received.append(event)
        return PresentationReceipt(event.presentation_id, DeliveryStatus.DELIVERED)

    assert adapter.presentation_ports.turn_stream is not None
    adapter.presentation_ports.turn_stream.subscribe(callback)
    source = _FakeSnapshotLease(snapshot)
    lease = host.acquire_binding(source, "feishu")
    bus = EventBus()
    bridge = ChannelTurnPresentationBridge(bus)
    token = bind_channel_turn_binding(lease)
    try:
        await bus.observe(
            TurnStarted(
                session_key="feishu:chat",
                channel="feishu",
                chat_id="chat",
                content="hello",
                timestamp=datetime.now(timezone.utc),
                turn_id="turn-bridge",
                client_message_id="provider-message",
            )
        )
        generation.channel("feishu").close_admission()
        await bus.observe(
            StreamDeltaReady(
                "feishu:chat",
                "feishu",
                "chat",
                "turn-bridge",
                "delta",
                "thinking",
            )
        )
        await bus.observe(
            ToolCallStarted(
                "feishu:chat",
                "feishu",
                "chat",
                1,
                "call-1",
                "search",
                {},
                "turn-bridge",
            )
        )
        await bus.observe(
            ToolCallCompleted(
                "feishu:chat",
                "feishu",
                "chat",
                1,
                "call-1",
                "search",
                {},
                {},
                "success",
                "ok",
                {},
                "turn-bridge",
            )
        )
        await bus.observe(
            TurnOutputCompleted(
                "feishu:chat",
                "feishu",
                "chat",
                "turn-bridge",
                "provider-message",
            )
        )
    finally:
        reset_channel_turn_binding(token)
        await bridge.aclose()
        await lease.aclose()
        await generation.stop()

    assert [event.kind for event in received] == list(TurnStreamEventKind)
    assert received[0].payload.client_message_id == "provider-message"
    assert [event.payload.sequence for event in received[1:]] == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_c14d_turn_binding_does_not_leak_into_child_task() -> None:
    snapshot, factories, _adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    lease = host.acquire_binding(_FakeSnapshotLease(snapshot), "feishu")
    token = bind_channel_turn_binding(lease)
    try:
        assert get_current_channel_turn_binding() is lease
        async def inherited_binding() -> object:
            return get_current_channel_turn_binding()

        child = asyncio.create_task(inherited_binding())
        assert await child is None
    finally:
        reset_channel_turn_binding(token)
        await lease.aclose()
        await generation.stop()


@pytest.mark.asyncio
async def test_c14d_callback_failure_settles_unknown_and_stops_presentation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    ports = adapter.presentation_ports
    assert ports.turn_stream is not None

    async def broken(_event: TurnStreamEvent) -> PresentationReceipt:
        raise RuntimeError("provider after-effect failure")

    ports.turn_stream.subscribe(broken)
    event = TurnStreamEvent(
        "preview-failure",
        TurnStreamEventKind.TURN_STARTED,
        TurnStartedPresentation("turn-failure", "client-failure"),
    )
    receipts = await generation.channel("feishu").publish_turn_event(event)
    assert receipts[0].status is DeliveryStatus.UNKNOWN
    with pytest.raises(RuntimeError, match="已因 UNKNOWN 终止"):
        await generation.channel("feishu").publish_turn_event(
            TurnStreamEvent(
                "preview-failure",
                TurnStreamEventKind.STREAM_DELTA,
                StreamDeltaPresentation("turn-failure", 1, "x", ""),
            )
        )
    await generation.stop()
    terminal = next(
        _diagnostic_fields(record)
        for record in caplog.records
        if _diagnostic_fields(record).get("event") == "plugin.operation.error"
        and _diagnostic_fields(record).get("operation") == "channel.turn_stream"
    )
    assert terminal["generation_id"] == "gen-1"
    assert terminal["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_c14d_callback_contract_mismatch_settles_unknown_before_raise() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    ports = adapter.presentation_ports
    assert ports.turn_stream is not None

    async def wrong_receipt(_event: TurnStreamEvent) -> PresentationReceipt:
        return PresentationReceipt("wrong-presentation", DeliveryStatus.DELIVERED)

    ports.turn_stream.subscribe(wrong_receipt)
    with pytest.raises(TypeError, match="identity 不匹配"):
        await generation.channel("feishu").publish_turn_event(
            TurnStreamEvent(
                "preview-mismatch",
                TurnStreamEventKind.TURN_STARTED,
                TurnStartedPresentation("turn-mismatch", "client-mismatch"),
            )
        )
    await generation.stop()


@pytest.mark.asyncio
async def test_c14d_callback_cancellation_waits_for_terminal_cleanup() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset(ChannelCapability),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    ports = adapter.presentation_ports
    assert ports.turn_stream is not None
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_event: TurnStreamEvent) -> PresentationReceipt:
        started.set()
        await release.wait()
        return PresentationReceipt("preview-cancel", DeliveryStatus.DELIVERED)

    ports.turn_stream.subscribe(slow)
    task = asyncio.create_task(
        generation.channel("feishu").publish_turn_event(
            TurnStreamEvent(
                "preview-cancel",
                TurnStreamEventKind.TURN_STARTED,
                TurnStartedPresentation("turn-cancel", "client-cancel"),
            )
        )
    )
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert generation.channel("feishu").in_flight == 0
    await generation.stop()


@pytest.mark.asyncio
async def test_c14d_ports_are_capability_gated_per_binding() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=PresentationAdapter,
        capabilities=frozenset({ChannelCapability.OUTBOUND, ChannelCapability.CONTROL}),
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    adapter = tuple(adapters.values())[0]
    ports = adapter.presentation_ports
    assert ports.control is not None
    assert ports.turn_stream is None
    assert adapter.context.control is ports.control
    assert adapter.context.turn_stream is None
    generation.open_admission()
    with pytest.raises(RuntimeError, match="exact snapshot lease"):
        await ports.control.interrupt(
            RawInbound(
                message_id="control-no-lease",
                message=ChannelInboundMessage(
                    channel="feishu",
                    sender="sender",
                    chat_id="chat",
                    content="/stop",
                    timestamp=datetime.now(timezone.utc),
                    metadata={},
                ),
            ),
            response_bodies=ControlResponseBodies("interrupted", "idle"),
        )
    await generation.stop()


async def _lease_for(snapshot: Any) -> _FakeSnapshotLease:
    return _FakeSnapshotLease(snapshot)


@pytest.mark.asyncio
async def test_exact_binding_lease_blocks_stop_until_snapshot_fork_closes() -> None:
    snapshot, factories, _ = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    source = _FakeSnapshotLease(snapshot)

    owner = host.acquire_binding(cast(Any, source), "feishu")
    assert owner.snapshot_id == snapshot.snapshot_id
    assert owner.generation_id == "gen-1"
    assert owner.channel_name == "feishu"
    assert owner.active
    stop = asyncio.create_task(generation.stop())
    await asyncio.sleep(0)
    assert not stop.done()

    await owner.aclose()
    assert not owner.active
    assert source.active
    assert len(source.forks) == 1 and not source.forks[0].active
    await stop


@pytest.mark.asyncio
async def test_exact_binding_lease_dispatches_one_outbound_envelope() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    source = _FakeSnapshotLease(snapshot)
    owner = host.acquire_binding(cast(Any, source), "feishu")
    envelope = OutboundEnvelope(
        logical_delivery_id="d1",
        delivery_id="d1",
        attempt_sequence=1,
        snapshot_id=snapshot.snapshot_id,
        generation_id="gen-1",
        binding_token=owner.binding_token,
        channel="feishu",
        recipient="u",
        body="hi",
        metadata={},
        commit_role=ChannelCommitRole.PASSIVE,
        thinking="thinking",
        reply_to="reply",
        session_message_id="message",
        control_turn_id="turn",
        execution_attempt_id="attempt",
        terminal_status=ChannelTerminalStatus.COMPLETED,
    )
    for adapter in adapters.values():
        adapter.release.set()

    receipt = await host.dispatch_outbound(envelope, owner)

    assert receipt == ChannelDeliveryReceipt(
        "d1",
        DeliveryStatus.DELIVERED,
        ("p1",),
    )
    assert tuple(adapters.values())[0].deliveries == ["d1"]
    request = tuple(adapters.values())[0].requests[0]
    assert request.commit_role is ChannelCommitRole.PASSIVE
    assert request.thinking == "thinking"
    assert request.reply_to == "reply"
    assert request.session_message_id == "message"
    assert request.control_turn_id == "turn"
    assert request.execution_attempt_id == "attempt"
    assert request.terminal_status is ChannelTerminalStatus.COMPLETED
    await owner.aclose()
    await generation.stop()


@pytest.mark.asyncio
async def test_exact_binding_lease_finishes_delivery_after_admission_closes() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    owner = host.acquire_binding(cast(Any, _FakeSnapshotLease(snapshot)), "feishu")
    envelope = OutboundEnvelope(
        logical_delivery_id="d1",
        delivery_id="d1",
        attempt_sequence=1,
        snapshot_id=snapshot.snapshot_id,
        generation_id="gen-1",
        binding_token=owner.binding_token,
        channel="feishu",
        recipient="u",
        body="hi",
        metadata={},
    )

    generation.close_admission()
    direct = generation.channel("feishu")
    with pytest.raises(RuntimeError, match="关闭"):
        await direct.deliver(
            ProviderDeliveryRequest(direct.binding_token, "new", "u", "hi")
        )
    delivery = asyncio.create_task(host.dispatch_outbound(envelope, owner))
    await asyncio.sleep(0)
    assert tuple(adapters.values())[0].deliveries == ["d1"]
    await owner.aclose()
    draining = asyncio.create_task(generation.drain())
    await asyncio.sleep(0)
    assert not draining.done()
    tuple(adapters.values())[0].release.set()
    receipt = await delivery
    await draining

    assert receipt.status is DeliveryStatus.DELIVERED
    assert tuple(adapters.values())[0].deliveries == ["d1"]
    await generation.stop()


@pytest.mark.asyncio
async def test_retained_delivery_rejects_forged_or_stopping_binding() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    source = _FakeSnapshotLease(snapshot)
    owner = host.acquire_binding(cast(Any, source), "feishu")
    envelope = OutboundEnvelope(
        logical_delivery_id="d1",
        delivery_id="d1",
        attempt_sequence=1,
        snapshot_id=snapshot.snapshot_id,
        generation_id="gen-1",
        binding_token=owner.binding_token,
        channel="feishu",
        recipient="u",
        body="hi",
        metadata={},
    )
    forged = ChannelBindingLease(
        host,
        (snapshot.snapshot_id, "feishu"),
        _FakeSnapshotLease(snapshot),
    )
    generation.close_admission()
    with pytest.raises(RuntimeError, match="Host 登记"):
        await host.dispatch_outbound(envelope, forged)
    assert tuple(adapters.values())[0].deliveries == []

    stopping = asyncio.create_task(generation.stop())
    for _ in range(20):
        if host._binding((snapshot.snapshot_id, "feishu")).stopping:
            break
        await asyncio.sleep(0)
    assert host._binding((snapshot.snapshot_id, "feishu")).stopping
    assert not stopping.done()
    with pytest.raises(RuntimeError, match="关闭"):
        await host.dispatch_outbound(envelope, owner)
    await owner.aclose()
    await stopping


@pytest.mark.asyncio
async def test_outbound_dispatch_rejects_foreign_host_binding() -> None:
    snapshot, factories, _ = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    owner = host.acquire_binding(cast(Any, _FakeSnapshotLease(snapshot)), "feishu")
    envelope = OutboundEnvelope(
        logical_delivery_id="d1",
        delivery_id="d1",
        attempt_sequence=1,
        snapshot_id=snapshot.snapshot_id,
        generation_id="gen-1",
        binding_token=owner.binding_token,
        channel="feishu",
        recipient="u",
        body="hi",
        metadata={},
    )

    with pytest.raises(RuntimeError, match="不属于当前 Host"):
        await _host().dispatch_outbound(envelope, owner)

    await owner.aclose()
    await generation.stop()


@pytest.mark.asyncio
async def test_formal_ingress_acquires_exact_binding_and_deduplicates() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        )
    )
    sources: list[_FakeSnapshotLease] = []

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        source = _FakeSnapshotLease(snapshot)
        sources.append(source)
        return source

    bus = MessageBus()
    host = _host(snapshot_lease_acquirer=acquire)
    host.bind_inbound_publisher(bus.publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    assert adapter.context.ingress is not None
    assert not hasattr(adapter.context, "recovery_ingress")
    assert adapter.runtime_ports.recovery_ingress is None
    raw = RawInbound(
        message_id="provider-message-1",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="user",
            chat_id="chat",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )

    assert await adapter.context.ingress.admit(raw) is True
    assert await adapter.context.ingress.admit(raw) is False
    assert len(sources) == 1 and not sources[0].active
    assert len(sources[0].forks) == 1 and sources[0].forks[0].active
    envelope = await bus.consume_inbound()
    assert envelope.message_id == raw.message_id  # type: ignore[union-attr]
    await bus.release_channel_inbound(envelope, InboundOwner.LANE)  # type: ignore[arg-type]
    assert not sources[0].forks[0].active

    await generation.stop()


@pytest.mark.asyncio
async def test_plugin_ingress_cannot_claim_mobile_durable_handoff() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset({ChannelCapability.INBOUND})
    )
    sources: list[_FakeSnapshotLease] = []

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        source = _FakeSnapshotLease(snapshot)
        sources.append(source)
        return source

    host = _host(snapshot_lease_acquirer=acquire)
    host.bind_inbound_publisher(MessageBus().publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    ingress = tuple(adapters.values())[0].context.ingress
    assert ingress is not None
    raw = RawInbound(
        message_id="forged-mobile-handoff",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="user",
            chat_id="chat",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={"mobile_v3_handoff": True},
        ),
    )

    with pytest.raises(RuntimeError, match="只属于 Core akashic"):
        await ingress.admit(raw)

    assert sources == []
    await generation.stop()


@pytest.mark.asyncio
async def test_external_mobile_adapter_never_receives_core_recovery_capability() -> None:
    module = _module(name="mobile")
    module.channel_names = ("mobile",)  # type: ignore[attr-defined]
    snapshot, factories, adapters = await _make_snapshot(
        module=module,
        capabilities=frozenset({ChannelCapability.INBOUND}),
    )
    generation = await _host().start_formal(snapshot, factories)
    adapter = tuple(adapters.values())[0]

    assert not hasattr(adapter.context, "recovery_ingress")
    assert adapter.runtime_ports.recovery_ingress is None

    await generation.stop()


@pytest.mark.asyncio
async def test_durable_recovery_replaces_retained_claim_without_weakening_duplicates(
    tmp_path: Any,
) -> None:
    root_token = object()
    adapters: dict[str, Adapter] = {}

    def factory(context: Any) -> Adapter:
        adapter = Adapter(context)
        adapters[context.binding_token] = adapter
        return adapter

    definition = CoreChannelDefinition(
        name="akashic",
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        ),
        factory=factory,
        inbound_identity=InboundIdentity.PROVIDER_MESSAGE_ID,
        source_revision="core-mobile-source-1",
        config_revision="core-mobile-config-1",
        generation_id="core-mobile-generation-1",
    )
    catalog = CommittedChannelCatalog(
        core_definitions=(definition,),
        root_instance_token=root_token,
    )
    snapshot = SimpleNamespace(
        snapshot_id="snapshot-core-mobile",
        state="committed",
        composition_root=SimpleNamespace(instance_token=root_token),
        channel_catalog=catalog,
        channel_registry=catalog.registry,
        channel_registry_identity=catalog.identity,
        generations={},
    )
    factories = {"akashic": ClientFactory()}
    manager = SessionManager(tmp_path / "workspace")
    session_key = "akashic:retained-recovery"
    manager.save(manager.get_or_create(session_key))
    bus = MessageBus()
    bus.bind_durable_inbound_store(manager.control_store)
    bus.bind_mobile_session_admission_owner(manager)

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        return _FakeSnapshotLease(snapshot)

    remember_calls = 0

    async def remember(_channel: str, _identity: str, _recipient: str) -> None:
        nonlocal remember_calls
        remember_calls += 1
        if remember_calls == 2:
            raise OSError("identity store unavailable")

    host = _host(
        snapshot_lease_acquirer=acquire,
        identity_resolver=lambda _channel, _identity: None,
        identity_rememberer=remember,
    )
    host.bind_inbound_publisher(bus.publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    adapter = tuple(adapters.values())[0]
    context = adapter.context
    assert context.ingress is not None
    assert not hasattr(context, "recovery_ingress")
    assert adapter.runtime_ports.recovery_ingress is not None
    bus.bind_mobile_channel_inbound_recoverer(
        adapter.runtime_ports.recovery_ingress.recover
    )
    raw = RawInbound(
        message_id="provider-retained-recovery",
        provider_identity="device:1",
        recipient="retained-recovery",
        message=ChannelInboundMessage(
            channel="akashic",
            sender="device:1",
            chat_id="retained-recovery",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_key_override": session_key,
                "client_message_id": "provider-retained-recovery",
                "mobile_v3_handoff": True,
                "mobile_handoff_id": "handoff-retained-recovery",
            },
        ),
    )
    assert await bus.reserve_mobile_channel_handoff(raw) is True
    assert await context.ingress.admit(raw) is True
    assert await context.ingress.admit(raw) is False
    first = await bus.consume_inbound()
    assert isinstance(first, InboundEnvelope)
    await bus.retain_mobile_channel_inbound(first, InboundOwner.LANE)

    with pytest.raises(OSError, match="identity store unavailable"):
        await bus.recover_durable_inbounds()
    assert await context.ingress.admit(raw) is False

    await bus.recover_durable_inbounds()
    assert await context.ingress.admit(raw) is False
    recovered = await bus.consume_inbound()
    assert isinstance(recovered, InboundEnvelope)
    recovered.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    await bus.complete_inbound(recovered)

    # 进程重启后 Host 没有旧 claim，durable row 仍能由 current binding 恢复。
    restart_session_key = "akashic:restart-no-claim"
    manager.save(manager.get_or_create(restart_session_key))
    restart_raw = RawInbound(
        message_id="provider-restart-no-claim",
        provider_identity="device:1",
        recipient="restart-no-claim",
        message=ChannelInboundMessage(
                channel="akashic",
            sender="device:1",
            chat_id="restart-no-claim",
            content="restart",
            timestamp=datetime.now(timezone.utc),
            metadata={
                "session_key_override": restart_session_key,
                "client_message_id": "provider-restart-no-claim",
                "mobile_v3_handoff": True,
                "mobile_handoff_id": "handoff-restart-no-claim",
            },
        ),
    )
    assert await bus.reserve_mobile_channel_handoff(restart_raw) is True
    await bus.defer_mobile_channel_handoff("handoff-restart-no-claim")
    await bus.recover_durable_inbounds()
    assert await context.ingress.admit(restart_raw) is False
    restart_recovered = await bus.consume_inbound()
    assert isinstance(restart_recovered, InboundEnvelope)
    restart_recovered.handoff(InboundOwner.LANE, InboundOwner.LOOP)
    await bus.complete_inbound(restart_recovered)

    assert manager.control_store.list_inbound_handoffs() == []
    assert remember_calls == 4
    await generation.stop()
    await bus.aclose()
    manager.close()


@pytest.mark.asyncio
async def test_outbound_only_binding_rejects_ingress_before_runtime_ports() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    acquired = 0

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        nonlocal acquired
        acquired += 1
        return _FakeSnapshotLease(snapshot)

    published: list[InboundEnvelope] = []

    async def publish(envelope: InboundEnvelope) -> None:
        published.append(envelope)

    host = _host(snapshot_lease_acquirer=acquire)
    host.bind_inbound_publisher(publish)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    raw = RawInbound(
        message_id="provider-message-outbound-only",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="user",
            chat_id="chat",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )

    assert tuple(adapters.values())[0].context.ingress is None

    assert acquired == 0
    assert published == []
    await generation.stop()


@pytest.mark.asyncio
async def test_formal_ingress_rejects_different_stable_snapshot_and_releases_claim() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        )
    )
    other_snapshot = SimpleNamespace(snapshot_id="other-snapshot", generations={})
    wrong = _FakeSnapshotLease(other_snapshot)
    right = _FakeSnapshotLease(snapshot)
    acquired = [wrong, right]

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        return acquired.pop(0)

    bus = MessageBus()
    host = _host(snapshot_lease_acquirer=acquire)
    host.bind_inbound_publisher(bus.publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    raw = RawInbound(
        message_id="provider-message-race",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="user",
            chat_id="chat",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )
    adapter = tuple(adapters.values())[0]

    with pytest.raises(RuntimeError, match="stable snapshot 不一致"):
        await adapter.context.ingress.admit(raw)

    assert not wrong.active
    assert await adapter.context.ingress.admit(raw) is True
    envelope = await bus.consume_inbound()
    assert isinstance(envelope, InboundEnvelope)
    await bus.release_channel_inbound(envelope, InboundOwner.LANE)
    await generation.stop()


@pytest.mark.asyncio
async def test_formal_ingress_scopes_dedupe_by_provider_identity_and_persists_mapping() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        )
    )
    mapping: dict[tuple[str, str], str] = {}

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        return _FakeSnapshotLease(snapshot)

    async def remember(channel: str, identity: str, recipient: str) -> None:
        mapping[(channel, identity)] = recipient

    host = _host(
        snapshot_lease_acquirer=acquire,
        identity_resolver=lambda channel, identity: mapping.get((channel, identity)),
        identity_rememberer=remember,
    )
    bus = MessageBus()
    host.bind_inbound_publisher(bus.publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    ingress = tuple(adapters.values())[0].context.ingress
    identity = tuple(adapters.values())[0].context.identity
    assert ingress is not None and identity is not None

    def raw(provider: str, recipient: str) -> RawInbound:
        return RawInbound(
            message_id="same-provider-message-id",
            provider_identity=provider,
            recipient=recipient,
            message=ChannelInboundMessage(
                channel="feishu",
                sender=provider,
                chat_id=recipient,
                content="hello",
                timestamp=datetime.now(timezone.utc),
                metadata={},
            ),
        )

    assert await ingress.admit(raw("open-a", "chat-a")) is True
    assert await ingress.admit(raw("open-b", "chat-b")) is True
    assert identity.resolve("open-a") == "chat-a"
    assert identity.resolve("open-b") == "chat-b"
    first = await bus.consume_inbound()
    second = await bus.consume_inbound()
    assert isinstance(first, InboundEnvelope)
    assert isinstance(second, InboundEnvelope)
    await bus.release_channel_inbound(first, InboundOwner.LANE)
    await bus.release_channel_inbound(second, InboundOwner.LANE)
    await generation.stop()


@pytest.mark.asyncio
async def test_identity_write_failure_releases_dedupe_claim_before_snapshot_acquire() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        )
    )
    acquire_calls = 0
    fail = True

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        nonlocal acquire_calls
        acquire_calls += 1
        return _FakeSnapshotLease(snapshot)

    async def remember(_channel: str, _identity: str, _recipient: str) -> None:
        nonlocal fail
        if fail:
            fail = False
            raise OSError("identity store unavailable")

    host = _host(
        snapshot_lease_acquirer=acquire,
        identity_resolver=lambda _channel, _identity: None,
        identity_rememberer=remember,
    )
    bus = MessageBus()
    host.bind_inbound_publisher(bus.publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    ingress = tuple(adapters.values())[0].context.ingress
    assert ingress is not None
    raw = RawInbound(
        message_id="identity-retry",
        provider_identity="open-id",
        recipient="chat-id",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="open-id",
            chat_id="chat-id",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )

    with pytest.raises(OSError, match="identity store unavailable"):
        await ingress.admit(raw)

    assert acquire_calls == 1
    assert await ingress.admit(raw) is True
    assert acquire_calls == 2
    envelope = await bus.consume_inbound()
    assert isinstance(envelope, InboundEnvelope)
    await bus.release_channel_inbound(envelope, InboundOwner.LANE)
    await generation.stop()


@pytest.mark.asyncio
async def test_publisher_failure_rolls_back_identity_receipt_and_binding() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        )
    )
    receipt = object()
    rolled_back: list[object] = []
    sources: list[_FakeSnapshotLease] = []

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        source = _FakeSnapshotLease(snapshot)
        sources.append(source)
        return source

    async def remember(
        _channel: str,
        _identity: str,
        _recipient: str,
    ) -> object:
        return receipt

    async def rollback(value: object) -> bool:
        rolled_back.append(value)
        return True

    async def publish(_envelope: InboundEnvelope) -> None:
        raise RuntimeError("publisher unavailable")

    host = _host(
        snapshot_lease_acquirer=acquire,
        identity_resolver=lambda _channel, _identity: None,
        identity_rememberer=remember,
        identity_rollbacker=rollback,
    )
    host.bind_inbound_publisher(publish)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    ingress = tuple(adapters.values())[0].context.ingress
    assert ingress is not None
    raw = RawInbound(
        message_id="publisher-failure",
        provider_identity="open-id",
        recipient="chat-id",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="open-id",
            chat_id="chat-id",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )

    with pytest.raises(RuntimeError, match="publisher unavailable"):
        await ingress.admit(raw)

    assert rolled_back == [receipt]
    assert generation.channel("feishu").in_flight == 0
    assert len(sources) == 1 and not sources[0].active
    assert len(sources[0].forks) == 1 and not sources[0].forks[0].active
    await generation.stop()


@pytest.mark.asyncio
async def test_identity_write_is_owned_by_binding_drain_during_publication() -> None:
    snapshot, factories, adapters = await _make_snapshot(
        capabilities=frozenset(
            {ChannelCapability.INBOUND, ChannelCapability.OUTBOUND}
        )
    )
    remember_started = asyncio.Event()
    remember_release = asyncio.Event()
    mapping: dict[str, str] = {}

    def acquire(snapshot_id: str) -> _FakeSnapshotLease:
        assert snapshot_id == snapshot.snapshot_id
        return _FakeSnapshotLease(snapshot)

    async def remember(_channel: str, identity: str, recipient: str) -> None:
        remember_started.set()
        await remember_release.wait()
        mapping[identity] = recipient

    host = _host(
        snapshot_lease_acquirer=acquire,
        identity_resolver=lambda _channel, identity: mapping.get(identity),
        identity_rememberer=remember,
    )
    bus = MessageBus()
    host.bind_inbound_publisher(bus.publish_channel_inbound)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    ingress = tuple(adapters.values())[0].context.ingress
    assert ingress is not None
    raw = RawInbound(
        message_id="publication-race",
        provider_identity="open-id",
        recipient="chat-id",
        message=ChannelInboundMessage(
            channel="feishu",
            sender="open-id",
            chat_id="chat-id",
            content="hello",
            timestamp=datetime.now(timezone.utc),
            metadata={},
        ),
    )

    admission = asyncio.create_task(ingress.admit(raw))
    await remember_started.wait()
    stop = asyncio.create_task(generation.stop())
    await asyncio.sleep(0)
    assert not stop.done()

    remember_release.set()
    assert await admission is True
    assert mapping == {"open-id": "chat-id"}
    envelope = await bus.consume_inbound()
    assert isinstance(envelope, InboundEnvelope)
    await bus.release_channel_inbound(envelope, InboundOwner.LANE)
    await stop


@pytest.mark.asyncio
async def test_binding_lease_cancel_waits_for_exact_snapshot_release() -> None:
    snapshot, factories, _ = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    release_gate = asyncio.Event()
    source = _FakeSnapshotLease(snapshot, release_gate=release_gate)
    owner = host.acquire_binding(cast(Any, source), "feishu")

    closing = asyncio.create_task(owner.aclose())
    await asyncio.sleep(0)
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert not owner.active
    assert len(source.forks) == 1 and not source.forks[0].active
    await generation.stop()


@pytest.mark.asyncio
async def test_wrong_binding_and_receipt_identity_fail_loud() -> None:
    snapshot, factories, _ = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    binding = generation.channel("feishu")
    binding.open_admission()
    with pytest.raises(RuntimeError, match="binding token"):
        await binding.deliver(ProviderDeliveryRequest("wrong", "d1", "u", "hi"))
    await generation.stop()

    snapshot, factories, adapters = await _make_snapshot(wrong_receipt=True)
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    binding = generation.channel("feishu")
    binding.open_admission()
    for adapter in adapters.values():
        adapter.release.set()
    with pytest.raises(RuntimeError, match="receipt identity"):
        await binding.deliver(ProviderDeliveryRequest(binding.binding_token, "d1", "u", "hi"))
    await generation.stop()


@pytest.mark.asyncio
async def test_journal_callback_happens_before_start_and_failure_keeps_count_zero() -> None:
    events: list[str] = []
    records: list[ChannelStartRecord] = []
    snapshot, factories, _ = await _make_snapshot(factory_events=events)

    async def before(record: ChannelStartRecord) -> None:
        records.append(record)
        events.append("journal")

    async def check(record: ChannelStartRecord) -> None:
        events.append("config-check")

    host = _host(on_before_start=before, config_revision_checker=check)
    generation = await host.start_formal(snapshot, factories)
    assert events == ["journal", "config-check", "factory"]
    assert records[0].source_revision == "source-1"
    assert records[0].config_revision == channel_config_revision(
        {"app_secret": CredentialRef(("app_secret",))}
    )
    assert records[0].raw_config_revision == "raw-config-1"
    assert len(records[0].descriptor_digest) == 64
    assert records[0].factory_export == "make_adapter"
    assert records[0].artifact_pointer == "/tmp/plugin"
    assert records[0].target == "formal"
    assert records[0].boot_owner == "plugin-manager"
    assert host.start_count(snapshot.snapshot_id, "feishu") == 1
    await generation.stop()

    async def fail_before(record: ChannelStartRecord) -> None:
        raise RuntimeError("journal failed")

    events = []
    snapshot, _, _ = await _make_snapshot(factory_events=events)
    host = _host(on_before_start=fail_before)
    with pytest.raises(RuntimeError, match="journal failed"):
        await host.start_formal(snapshot, {"feishu": ClientFactory()})
    assert host.start_count(snapshot.snapshot_id, "feishu") == 0
    assert events == []


def test_durable_callbacks_are_mandatory() -> None:
    with pytest.raises(TypeError):
        ChannelGenerationHost(
            on_before_start=None,  # type: ignore[arg-type]
            config_revision_checker=_noop_record,
        )
    with pytest.raises(TypeError):
        ChannelGenerationHost(
            on_before_start=_noop_record,
            config_revision_checker=None,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_identity_rollback_fence_conflict_is_fail_loud() -> None:
    async def remember(
        _channel: str,
        _identity: str,
        _recipient: str,
    ) -> object:
        return object()

    async def rollback(_receipt: object) -> bool:
        return False

    host = _host(
        identity_resolver=lambda _channel, _identity: None,
        identity_rememberer=remember,
        identity_rollbacker=rollback,
    )

    with pytest.raises(RuntimeError, match="rollback fence 已被并发状态取代"):
        await host._rollback_identity_write(object())


@pytest.mark.asyncio
async def test_config_revision_checker_failure_is_before_factory_and_start() -> None:
    events: list[str] = []
    snapshot, factories, _ = await _make_snapshot(factory_events=events)

    async def check(record: ChannelStartRecord) -> None:
        raise RuntimeError("config revision drift")

    host = _host(config_revision_checker=check)
    with pytest.raises(RuntimeError, match="config revision drift"):
        await host.start_formal(snapshot, factories)
    assert events == []
    assert factories["feishu"].closed == 1
    assert host.start_count(snapshot.snapshot_id, "feishu") == 0


@pytest.mark.asyncio
async def test_empty_registry_is_repeatable_noop_without_lock_or_fiber_owner() -> None:
    snapshot, _, _ = await _make_snapshot()
    root_token = snapshot.composition_root.instance_token
    empty_channels = PluginChannels(root_token)
    empty_registry = _freeze_plugin_channels(empty_channels, root_token)
    snapshot.channel_registry = empty_registry
    snapshot.channel_registry_identity = empty_registry.identity
    snapshot.generations = {}
    host = _host()
    first = await host.start_formal(snapshot, {})
    second = await host.start_formal(snapshot, {})
    assert first.snapshot_id == second.snapshot_id == snapshot.snapshot_id
    assert await first.stop() == ()
    assert await second.stop() == ()
    assert host._locks == {}
    assert not hasattr(host, "fiber")
    assert not hasattr(host, "context")


@pytest.mark.asyncio
async def test_partial_start_rolls_back_started_adapter_and_provider_factory() -> None:
    module = _module()
    module.channel_names = ("feishu", "qqbot")  # type: ignore[attr-defined]
    snapshot, failing_factories, adapters = await _make_snapshot(
        module=module,
        fail_after=2,
    )
    host = _host()
    with pytest.raises(RuntimeError, match="start failed"):
        await host.start_formal(snapshot, failing_factories)
    assert all(factory.closed == 1 for factory in failing_factories.values())
    assert len(adapters) == 2
    assert sum(adapter.stopped for adapter in adapters.values()) == 2
    assert host.failure(snapshot.snapshot_id) is None


@pytest.mark.asyncio
async def test_stop_failure_retains_tombstone_and_retry_cleans_exact_owner() -> None:
    snapshot, factories, adapters = await _make_snapshot(fail_stop=True)
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    with pytest.raises(RuntimeError, match="cleanup"):
        await generation.stop()
    tombstone = host.failure(snapshot.snapshot_id, "feishu")
    assert tombstone is not None
    assert tombstone.binding_token == generation.channel("feishu").binding_token
    assert tombstone.artifact_pointer == "/tmp/plugin"
    assert tombstone.factory_export == "make_adapter"
    assert tombstone.source_revision == "source-1"
    assert tombstone.config_revision == channel_config_revision(
        {"app_secret": CredentialRef(("app_secret",))}
    )
    assert tombstone.raw_config_revision == "raw-config-1"
    assert len(tombstone.descriptor_digest) == 64
    assert tombstone.target == "formal"
    assert tombstone.boot_owner == "plugin-manager"
    assert tombstone.adapter_stop_settled is True
    assert tombstone.adapter_stop_succeeded is False
    assert tombstone.factory_close_settled is True
    assert tombstone.factory_close_succeeded is True
    with pytest.raises(RuntimeError, match="未知"):
        await host.retry_generation_cleanup("wrong-binding-token")
    adapter = next(iter(adapters.values()))
    adapter.fail_stop = False
    await host.retry_generation_cleanup(tombstone.binding_token)
    assert adapter.stopped == 2
    assert factories["feishu"].closed == 1
    assert host.failure(snapshot.snapshot_id) is None


@pytest.mark.asyncio
async def test_incomplete_stop_receipt_is_diagnostic_error_and_retryable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="akashic.plugin.diagnostics")
    snapshot, factories, adapters = await _make_snapshot(
        adapter_cls=IncompleteStopAdapter
    )
    host = _host()
    generation = await host.start_formal(snapshot, factories)

    with pytest.raises(RuntimeError, match="cleanup"):
        await generation.stop()

    tombstone = host.failure(snapshot.snapshot_id, "feishu")
    assert tombstone is not None
    assert tombstone.adapter_stop_succeeded is False
    stop_terminals = [
        _diagnostic_fields(record)
        for record in caplog.records
        if _diagnostic_fields(record).get("operation") == "channel.stop"
        and _diagnostic_fields(record).get("event")
        in {"plugin.operation.done", "plugin.operation.error"}
    ]
    assert [item["event"] for item in stop_terminals] == [
        "plugin.operation.error"
    ]

    adapter = cast(IncompleteStopAdapter, next(iter(adapters.values())))
    adapter.resources_closed = True
    await host.retry_generation_cleanup(tombstone.binding_token)
    assert adapter.stopped == 2
    assert host.failure(snapshot.snapshot_id) is None


@pytest.mark.asyncio
async def test_retry_skips_successful_adapter_stop_when_factory_close_failed() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    factories["feishu"].fail_close = True
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    with pytest.raises(RuntimeError, match="cleanup"):
        await generation.stop()
    adapter = next(iter(adapters.values()))
    assert adapter.stopped == 1
    assert factories["feishu"].closed == 1
    tombstone = host.failure(snapshot.snapshot_id, "feishu")
    assert tombstone is not None
    assert tombstone.adapter_stop_succeeded is True
    assert tombstone.factory_close_settled is True
    assert tombstone.factory_close_succeeded is False
    factories["feishu"].fail_close = False
    await host.retry_generation_cleanup(tombstone.binding_token)
    assert adapter.stopped == 1
    assert factories["feishu"].closed == 2


@pytest.mark.asyncio
async def test_provider_cancel_and_failure_callback_cancel_retain_tombstone() -> None:
    snapshot, factories, _ = await _make_snapshot(cancel_stop=True)

    async def on_failure(record: Any) -> None:
        raise asyncio.CancelledError

    host = _host(on_failure=on_failure)
    generation = await host.start_formal(snapshot, factories)
    with pytest.raises(asyncio.CancelledError):
        await generation.stop()
    assert host.failure(snapshot.snapshot_id, "feishu") is not None


@pytest.mark.asyncio
async def test_failure_callback_error_is_not_logged_as_success() -> None:
    snapshot, factories, _ = await _make_snapshot(fail_stop=True)

    async def on_failure(record: Any) -> None:
        raise RuntimeError("journal unavailable")

    host = _host(on_failure=on_failure)
    generation = await host.start_formal(snapshot, factories)
    with pytest.raises(RuntimeError, match="journal unavailable"):
        await generation.stop()
    assert host.failure(snapshot.snapshot_id, "feishu") is not None


@pytest.mark.asyncio
async def test_factory_and_adapter_start_cancellation_keep_exact_tombstones() -> None:
    snapshot, factories, _ = await _make_snapshot(cancel_factory=True)
    host = _host()
    with pytest.raises(asyncio.CancelledError):
        await host.start_formal(snapshot, factories)
    factory_failure = host.failure(snapshot.snapshot_id, "feishu")
    assert factory_failure is not None
    assert factory_failure.binding_token
    assert factories["feishu"].closed == 1
    await host.retry_generation_cleanup(factory_failure.binding_token)
    assert host.failure(snapshot.snapshot_id) is None

    snapshot, factories, adapters = await _make_snapshot(cancel_start=True)
    host = _host()
    with pytest.raises(asyncio.CancelledError):
        await host.start_formal(snapshot, factories)
    adapter_failure = host.failure(snapshot.snapshot_id, "feishu")
    assert adapter_failure is not None
    assert adapter_failure.adapter is next(iter(adapters.values()))
    assert factories["feishu"].closed == 1
    await host.retry_generation_cleanup(adapter_failure.binding_token)
    assert host.failure(snapshot.snapshot_id) is None


@pytest.mark.asyncio
async def test_async_factory_and_noncallable_factory_are_rejected_before_start() -> None:
    snapshot, factories, _ = await _make_snapshot()

    async def async_factory(context: Any) -> Adapter:
        return Adapter(context)

    setattr(snapshot.generations["plugin.feishu"].instance.module, "make_adapter", async_factory)
    with pytest.raises(TypeError, match="async"):
        await _host().start_formal(snapshot, factories)
    assert factories["feishu"].closed == 1

    snapshot, factories, _ = await _make_snapshot()
    setattr(snapshot.generations["plugin.feishu"].instance.module, "make_adapter", None)
    with pytest.raises(TypeError, match="不可调用"):
        await _host().start_formal(snapshot, factories)
    assert factories["feishu"].closed == 1


@pytest.mark.asyncio
async def test_exact_root_and_factory_provenance_are_required() -> None:
    snapshot, factories, _ = await _make_snapshot()
    snapshot.composition_root = SimpleNamespace(instance_token=object())
    with pytest.raises(RuntimeError, match="exact composition Root"):
        await _host().start_formal(snapshot, factories)

    snapshot, factories, _ = await _make_snapshot()
    object.__setattr__(snapshot.channel_registry.factories[0], "config_revision", "drift")
    with pytest.raises(RuntimeError):
        await _host().start_formal(snapshot, factories)


@pytest.mark.asyncio
async def test_caller_cancellation_waits_for_cleanup() -> None:
    snapshot, factories, adapters = await _make_snapshot(block_stop=True)
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    stop_task = asyncio.create_task(generation.stop())
    adapter = next(iter(adapters.values()))
    await adapter.stop_started.wait()
    stop_task.cancel()
    stop_task.cancel()
    adapter.stop_release.set()
    with pytest.raises(asyncio.CancelledError):
        await stop_task
    assert factories["feishu"].closed == 1
    assert host.failure(snapshot.snapshot_id) is None


def test_attachment_ports_must_be_bound_as_a_pair() -> None:
    ref = _attachment_ref()
    import_port = _FakeAttachmentImportPort(ref)
    read_port = _FakeAttachmentReadPort(_FakeAttachmentReadLease(ref))
    with pytest.raises(TypeError, match="同时绑定"):
        _host(attachment_import=import_port)
    with pytest.raises(TypeError, match="同时绑定"):
        _host(attachment_read=read_port)


@pytest.mark.asyncio
async def test_formal_context_gets_per_binding_attachment_facades_and_none_without_ports() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    host = _host()
    generation = await host.start_formal(snapshot, factories)
    context = tuple(adapters.values())[0].context
    assert context.attachment_import is None
    assert context.attachment_read is None
    await generation.stop()

    snapshot, factories, adapters = await _make_snapshot()
    ref = _attachment_ref()
    import_port = _FakeAttachmentImportPort(ref)
    read_port = _FakeAttachmentReadPort(_FakeAttachmentReadLease(ref))
    host = _host(attachment_import=import_port, attachment_read=read_port)
    generation = await host.start_formal(snapshot, factories)
    context = tuple(adapters.values())[0].context
    assert context.attachment_import is not None
    assert context.attachment_read is not None
    assert context.attachment_import is not import_port
    assert context.attachment_read is not read_port
    generation.open_admission()
    imported = await context.attachment_import.import_bytes(
        b"hello",
        kind=AttachmentKind.FILE,
        filename="report.txt",
        media_type="text/plain",
    )
    assert imported == ref
    assert import_port.calls == 1
    await generation.stop()


@pytest.mark.asyncio
async def test_held_attachment_read_lease_blocks_generation_drain() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    ref = _attachment_ref()
    underlying = _FakeAttachmentReadLease(ref)
    import_port = _FakeAttachmentImportPort(ref)
    read_port = _FakeAttachmentReadPort(underlying)
    host = _host(attachment_import=import_port, attachment_read=read_port)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    context = tuple(adapters.values())[0].context
    binding = generation.channel("feishu")
    assert context.attachment_read is not None
    lease = await context.attachment_read.acquire(ref)
    assert binding.in_flight == 1
    assert await lease.read_bytes(max_bytes=5) == b"hello"

    stopping = asyncio.create_task(generation.stop())
    await asyncio.sleep(0)
    assert not stopping.done()
    await lease.aclose()
    assert binding.in_flight == 0
    await stopping
    assert underlying.close_calls == 1


@pytest.mark.asyncio
async def test_attachment_import_and_acquire_failure_or_cancel_release_in_flight() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    ref = _attachment_ref()
    import_port = _FakeAttachmentImportPort(ref)
    read_port = _FakeAttachmentReadPort(_FakeAttachmentReadLease(ref))
    host = _host(attachment_import=import_port, attachment_read=read_port)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    context = tuple(adapters.values())[0].context
    binding = generation.channel("feishu")
    assert context.attachment_import is not None
    assert context.attachment_read is not None

    import_port.fail = True
    with pytest.raises(OSError, match="import failed"):
        await context.attachment_import.import_bytes(
            b"hello",
            kind=AttachmentKind.FILE,
            filename="report.txt",
            media_type="text/plain",
        )
    assert binding.in_flight == 0

    read_port.fail = True
    with pytest.raises(OSError, match="acquire failed"):
        await context.attachment_read.acquire(ref)
    assert binding.in_flight == 0
    read_port.fail = False

    import_port.fail = False
    import_port.gate = asyncio.Event()
    import_task = asyncio.create_task(
        context.attachment_import.import_bytes(
            b"hello",
            kind=AttachmentKind.FILE,
            filename="report.txt",
            media_type="text/plain",
        )
    )
    await asyncio.sleep(0)
    assert binding.in_flight == 1
    import_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await import_task
    assert binding.in_flight == 0

    read_port.gate = asyncio.Event()
    acquire_task = asyncio.create_task(context.attachment_read.acquire(ref))
    await asyncio.sleep(0)
    assert binding.in_flight == 1
    acquire_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquire_task
    assert binding.in_flight == 0
    await generation.stop()


@pytest.mark.asyncio
async def test_closed_or_stale_binding_rejects_attachment_before_store_call() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    ref = _attachment_ref()
    import_port = _FakeAttachmentImportPort(ref)
    read_port = _FakeAttachmentReadPort(_FakeAttachmentReadLease(ref))
    host = _host(attachment_import=import_port, attachment_read=read_port)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    context = tuple(adapters.values())[0].context
    assert context.attachment_import is not None
    generation.close_admission()
    with pytest.raises(RuntimeError, match="关闭"):
        await context.attachment_import.import_bytes(
            b"hello",
            kind=AttachmentKind.FILE,
            filename="report.txt",
            media_type="text/plain",
        )
    assert import_port.calls == 0
    await generation.stop()
    with pytest.raises(KeyError):
        await context.attachment_import.import_bytes(
            b"hello",
            kind=AttachmentKind.FILE,
            filename="report.txt",
            media_type="text/plain",
        )
    assert import_port.calls == 0


@pytest.mark.asyncio
async def test_attachment_lease_close_is_critical_under_caller_cancellation() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    ref = _attachment_ref()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    underlying = _FakeAttachmentReadLease(
        ref,
        close_started=close_started,
        close_release=close_release,
    )
    import_port = _FakeAttachmentImportPort(ref)
    read_port = _FakeAttachmentReadPort(underlying)
    host = _host(attachment_import=import_port, attachment_read=read_port)
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    context = tuple(adapters.values())[0].context
    binding = generation.channel("feishu")
    assert context.attachment_read is not None
    lease = await context.attachment_read.acquire(ref)
    stopping = asyncio.create_task(generation.stop())
    await asyncio.sleep(0)
    assert not stopping.done()

    closing = asyncio.create_task(lease.aclose())
    await close_started.wait()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    assert binding.in_flight == 1

    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert binding.in_flight == 0
    await stopping
    assert underlying.close_calls == 1


@pytest.mark.asyncio
async def test_attachment_lease_concurrent_close_releases_host_once() -> None:
    snapshot, factories, adapters = await _make_snapshot()
    ref = _attachment_ref()
    underlying = _FakeAttachmentReadLease(ref)
    host = _host(
        attachment_import=_FakeAttachmentImportPort(ref),
        attachment_read=_FakeAttachmentReadPort(underlying),
    )
    generation = await host.start_formal(snapshot, factories)
    generation.open_admission()
    context = tuple(adapters.values())[0].context
    binding = generation.channel("feishu")
    assert context.attachment_read is not None
    lease = await context.attachment_read.acquire(ref)

    await asyncio.gather(lease.aclose(), lease.aclose())

    assert binding.in_flight == 0
    assert underlying.close_calls == 1
    await generation.stop()
