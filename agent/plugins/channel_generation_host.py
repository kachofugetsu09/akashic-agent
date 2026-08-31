"""Formal-only, generation-scoped host for v3 text channel adapters.

The host deliberately owns only live adapter bindings.  Publication, snapshot
leases and the current runtime remain Core/Manager responsibilities.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import nullcontext
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, ContextManager, Literal, cast

from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentReadLease,
    AttachmentRef,
    ChannelAdapter,
    ChannelAttachmentImportPort,
    ChannelAttachmentReadPort,
    ChannelCapability,
    ChannelCleanupFailure,
    ChannelControlPort,
    ChannelDeliveryReceipt,
    ChannelFactoryContext,
    ChannelPresentationPorts,
    ChannelReady,
    ChannelRegistrySnapshot,
    ChannelRuntimePorts,
    CommittedChannelCatalog,
    CoreChannelDefinition,
    CredentialRef,
    ControlReceipt,
    ControlResponseBodies,
    DeliveryStatus,
    InboundEnvelope,
    InboundIdentity,
    InboundOwner,
    OutboundEnvelope,
    ProviderClientFactory,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    PresentationReceipt,
    RawInbound,
    StreamSubscription,
    StopReceipt,
    TurnStreamCallback,
    TurnStreamEvent,
    TurnStreamEventKind,
    TurnStreamPort,
    TurnStartedPresentation,
    channel_config_revision,
)
from agent.plugins.composable import ComposablePlugin

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease

BeforeStartCallback = Callable[["ChannelStartRecord"], Awaitable[None]]
ConfigRevisionChecker = Callable[["ChannelStartRecord"], Awaitable[None]]
FailureCallback = Callable[["ChannelCleanupTombstone"], Awaitable[None] | None]
SnapshotLeaseAcquirer = Callable[[str], "RuntimeSnapshotLease"]
InboundPublisher = Callable[[InboundEnvelope], Awaitable[None]]
IdentityResolver = Callable[[str, str], str | None]
IdentityRememberer = Callable[
    [str, str, str],
    Coroutine[object, object, object | None],
]
IdentityRollbacker = Callable[[object], Coroutine[object, object, bool]]
ControlInterrupter = Callable[[RawInbound], Awaitable[object]]
ControlResponseDispatcher = Callable[..., Awaitable[ChannelDeliveryReceipt]]


class _PresentationContractFailure(TypeError):
    def __init__(self, message: str, receipt: PresentationReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class _ChannelStopReceiptFailure(RuntimeError):
    """Carry one valid but incomplete stop receipt through diagnostics."""

    def __init__(
        self,
        receipt: StopReceipt,
        failures: tuple[ChannelCleanupFailure, ...],
    ) -> None:
        super().__init__("channel adapter.stop 报告资源未完整关闭")
        self.receipt = receipt
        self.failures = failures


@dataclass(frozen=True, slots=True)
class ChannelStartRecord:
    """Durable-start identity written before an adapter can acquire resources."""

    snapshot_id: str
    catalog_identity: str
    plugin_id: str
    generation_id: str
    channel_name: str
    binding_token: str
    module_name: str
    artifact_pointer: str
    factory_export: str
    source_revision: str
    config_revision: str
    raw_config_revision: str
    descriptor_digest: str
    target: str
    boot_owner: str
    attempt: int = 1


@dataclass(frozen=True, slots=True)
class ChannelCleanupTombstone:
    """Retain every exact runtime owner until cleanup succeeds."""

    snapshot_id: str
    catalog_identity: str
    plugin_id: str
    generation_id: str
    channel_name: str
    module: ModuleType
    adapter: ChannelAdapter | None
    factory: Callable[[ChannelFactoryContext], ChannelAdapter] | None
    factory_context: ChannelFactoryContext | None
    provider_client_factory: ProviderClientFactory
    binding_token: str
    artifact_pointer: str
    factory_export: str
    source_revision: str
    config_revision: str
    raw_config_revision: str
    descriptor_digest: str
    target: str
    boot_owner: str
    adapter_stop_settled: bool
    adapter_stop_succeeded: bool
    factory_close_settled: bool
    factory_close_succeeded: bool
    resource: str
    error_type: str
    message: str
    action: str
    attempt_count: int = 1

    @property
    def error(self) -> str:
        """Return a stable human-readable cleanup error."""

        return self.message


@dataclass
class _ChannelBindingState:
    snapshot_id: str
    catalog_identity: str
    plugin_id: str
    generation_id: str
    channel_name: str
    capabilities: tuple[ChannelCapability, ...]
    inbound_identity: InboundIdentity | None
    module: ModuleType
    artifact_pointer: str
    factory: Callable[[ChannelFactoryContext], ChannelAdapter] | None
    adapter: ChannelAdapter | None
    provider_client_factory: ProviderClientFactory
    binding_token: str
    config: Mapping[str, object]
    credential_paths: tuple[str, ...]
    factory_context: ChannelFactoryContext | None
    factory_export: str
    source_revision: str
    config_revision: str
    raw_config_revision: str
    descriptor_digest: str
    target: str
    boot_owner: str
    start_attempt: int
    start_attempted: bool = False
    started: bool = False
    admission_open: bool = False
    stopping: bool = False
    stopped: bool = False
    in_flight: int = 0
    drain_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_receipt: StopReceipt | None = None
    ready: ChannelReady | None = None
    internal_cancellation: str | None = None
    runtime_attached: bool = False
    adapter_stop_settled: bool = False
    adapter_stop_succeeded: bool = False
    factory_close_settled: bool = False
    factory_close_succeeded: bool = False
    inbound_message_ids: deque[tuple[str, str]] = field(default_factory=deque)
    inbound_message_id_set: set[tuple[str, str]] = field(default_factory=set)
    control_port: _ChannelControl | None = None
    turn_stream_port: _ChannelTurnStream | None = None
    subscriptions: dict[int, _ChannelStreamSubscription] = field(default_factory=dict)
    control_message_ids: deque[tuple[str, str]] = field(default_factory=deque)
    control_message_id_set: set[tuple[str, str]] = field(default_factory=set)
    presentation_sequences: dict[str, int] = field(default_factory=dict)
    presentation_turn_ids: dict[str, str] = field(default_factory=dict)
    completed_presentations: set[str] = field(default_factory=set)
    failed_presentations: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.drain_event.set()


def _channel_entrypoint(
    state: _ChannelBindingState,
    operation: str,
) -> ContextManager[object]:
    """Record only external plugin adapters, not Core-owned channel bindings."""

    if state.plugin_id == "core":
        return nullcontext()
    return plugin_entrypoint(
        plugin_id=state.plugin_id,
        generation_id=state.generation_id,
        fiber=state.plugin_id,
        operation=operation,
        entrypoint=state.channel_name,
    )


class ChannelBinding:
    """Small facade for one exact channel binding owned by the Host."""

    def __init__(self, host: ChannelGenerationHost, key: tuple[str, str]) -> None:
        self._host = host
        self._key = key

    @property
    def snapshot_id(self) -> str:
        return self._host._binding(self._key).snapshot_id

    @property
    def generation_id(self) -> str:
        return self._host._binding(self._key).generation_id

    @property
    def plugin_id(self) -> str:
        return self._host._binding(self._key).plugin_id

    @property
    def channel_name(self) -> str:
        return self._host._binding(self._key).channel_name

    @property
    def binding_token(self) -> str:
        return self._host._binding(self._key).binding_token

    @property
    def admission_open(self) -> bool:
        return self._host._binding(self._key).admission_open

    @property
    def in_flight(self) -> int:
        return self._host._binding(self._key).in_flight

    @property
    def turn_stream(self) -> TurnStreamPort | None:
        return self._host._binding(self._key).turn_stream_port

    @property
    def stopped(self) -> bool:
        state = self._host._bindings.get(self._key)
        return state is None or state.stopped

    def open_admission(self) -> None:
        """Open this staged binding after publication has finalized."""

        self._host._open_admission(self._key)

    def close_admission(self) -> None:
        """Synchronously reject new deliveries while allowing in-flight work."""

        self._host._close_admission(self._key)

    async def drain(self) -> None:
        """Wait until every delivery accepted before close is terminal."""

        await self._host._drain(self._key)

    async def deliver(self, request: ProviderDeliveryRequest) -> ProviderDeliveryReceipt:
        """Deliver one text request through this exact binding."""

        return await self._host._deliver(self._key, request)

    async def publish_turn_event(
        self,
        event: TurnStreamEvent,
    ) -> tuple[PresentationReceipt, ...]:
        """Publish one typed turn event through this binding's subscriptions."""

        return await self._host.publish_turn_event(
            self.snapshot_id,
            self.channel_name,
            event,
        )

    async def stop(self) -> StopReceipt:
        """Close admission, drain and stop this binding."""

        return await self._host._stop_binding_critical(self._key)


class ChannelBindingLease:
    """Own one forked snapshot lease and one exact Host in-flight claim."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
        snapshot_lease: RuntimeSnapshotLease,
    ) -> None:
        self._host = host
        self._key = key
        self.snapshot_lease = snapshot_lease
        self._binding_released = False
        self._closed = False

    @property
    def snapshot_id(self) -> str:
        return self._key[0]

    @property
    def generation_id(self) -> str:
        return self._host._binding(self._key).generation_id

    @property
    def channel_name(self) -> str:
        return self._key[1]

    @property
    def binding_token(self) -> str:
        return self._host._binding(self._key).binding_token

    @property
    def turn_stream(self) -> TurnStreamPort | None:
        return self._host._binding(self._key).turn_stream_port

    @property
    def active(self) -> bool:
        return not self._closed

    async def deliver(self, envelope: OutboundEnvelope) -> ChannelDeliveryReceipt:
        """Deliver one envelope through this exact retained binding."""

        if self._closed:
            raise RuntimeError("Channel binding lease 已关闭")
        if not isinstance(envelope, OutboundEnvelope):
            raise TypeError("channel outbound 只接受 OutboundEnvelope")
        if (
            envelope.snapshot_id != self.snapshot_id
            or envelope.generation_id != self.generation_id
            or envelope.channel != self.channel_name
            or envelope.binding_token != self.binding_token
        ):
            raise RuntimeError("OutboundEnvelope 与 exact Channel binding 不一致")
        receipt = await self._host._deliver(
            self._key,
            ProviderDeliveryRequest(
                binding_token=self.binding_token,
                delivery_id=envelope.delivery_id,
                recipient=envelope.recipient,
                body=envelope.body,
                attachments=envelope.attachments,
                metadata=envelope.metadata,
                commit_role=envelope.commit_role,
                thinking=envelope.thinking,
                reply_to=envelope.reply_to,
                session_message_id=envelope.session_message_id,
                control_turn_id=envelope.control_turn_id,
                execution_attempt_id=envelope.execution_attempt_id,
                terminal_status=envelope.terminal_status,
            ),
            retained_binding=self,
        )
        return ChannelDeliveryReceipt(
            delivery_id=receipt.delivery_id,
            status=receipt.status,
            provider_ids=receipt.provider_ids,
            error=receipt.error,
        )

    async def publish_turn_event(
        self,
        event: TurnStreamEvent,
    ) -> tuple[PresentationReceipt, ...]:
        """Publish a preview while this exact old-turn binding lease remains active."""

        if self._closed:
            raise RuntimeError("Channel binding lease 已关闭")
        return await self._host.publish_turn_event(
            self.snapshot_id,
            self.channel_name,
            event,
            binding=self,
        )

    async def aclose(self) -> None:
        """Release both owners completely before propagating caller cancellation."""

        if self._closed:
            return
        task = asyncio.create_task(
            self._close(),
            name=f"channel_binding_lease_close:{self.snapshot_id}:{self.channel_name}",
        )
        await _await_task_after_cancellation(task)

    async def _close(self) -> None:
        if not self._binding_released:
            self._host._release_binding_lease(self)
            self._binding_released = True
        await self.snapshot_lease.release()
        self._closed = True


@dataclass(frozen=True, slots=True)
class _ChannelTurnBinding:
    lease: ChannelBindingLease
    owner_task: asyncio.Task[object] | None


_current_channel_binding: ContextVar[_ChannelTurnBinding | None] = ContextVar(
    "current_channel_binding",
    default=None,
)


def bind_channel_turn_binding(
    binding: object,
) -> Token[_ChannelTurnBinding | None]:
    """Bind the exact inbound Channel owner for one ConversationRuntime task."""

    active = getattr(binding, "active", None)
    if active is not True:
        raise RuntimeError("turn Channel binding 必须是当前 Host 的 active lease")
    return _current_channel_binding.set(
        _ChannelTurnBinding(
            cast(ChannelBindingLease, binding),
            asyncio.current_task(),
        )
    )


def reset_channel_turn_binding(
    token: Token[_ChannelTurnBinding | None],
) -> None:
    _current_channel_binding.reset(token)


def get_current_channel_turn_binding() -> ChannelBindingLease | None:
    binding = _current_channel_binding.get()
    if (
        binding is None
        or binding.owner_task is not asyncio.current_task()
        or not binding.lease.active
    ):
        return None
    return binding.lease


class _ChannelIngress:
    """Admit provider text into one exact formal binding."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
    ) -> None:
        self._host = host
        self._key = key

    async def admit(self, raw: RawInbound) -> bool:
        return await self._host._admit_inbound(self._key, raw)


class _ChannelRecoveryIngress:
    """Re-admit one Core-owned durable handoff without weakening provider dedupe."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
    ) -> None:
        self._host = host
        self._key = key

    async def recover(self, raw: RawInbound) -> bool:
        return await self._host._recover_inbound(self._key, raw)


class _ChannelIdentity:
    """Resolve recipients through the Core-owned durable identity index."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
    ) -> None:
        self._host = host
        self._key = key

    def resolve(self, provider_identity: str) -> str | None:
        return self._host._resolve_identity(self._key, provider_identity)


class _ChannelControl:
    """Expose one exact binding's deduplicated interrupt facade."""

    def __init__(self, host: ChannelGenerationHost, key: tuple[str, str]) -> None:
        self._host = host
        self._key = key

    async def interrupt(
        self,
        raw: RawInbound,
        *,
        response_bodies: ControlResponseBodies,
    ) -> ControlReceipt:
        """Claim, interrupt, and settle one provider control message."""

        if not isinstance(raw, RawInbound):
            raise TypeError("channel control 只接受 RawInbound")
        if not isinstance(response_bodies, ControlResponseBodies):
            raise TypeError("response_bodies 必须是 ControlResponseBodies")
        state = self._host._binding(self._key)
        if raw.message.channel != state.channel_name:
            raise RuntimeError("RawInbound channel 与 exact binding 不一致")
        if not state.admission_open or state.stopping or state.stopped:
            return ControlReceipt(False, "binding_closed")
        scope = raw.provider_identity or ""
        dedupe_key = (scope, raw.message_id)
        if dedupe_key in state.control_message_id_set:
            return ControlReceipt(False, "duplicate")
        state.control_message_id_set.add(dedupe_key)
        state.control_message_ids.append(dedupe_key)
        self._host._begin_presentation_operation(self._key, allow_closed=False)
        try:
            binding = await self._host._acquire_control_binding(self._key)
        except BaseException:
            state.control_message_id_set.discard(dedupe_key)
            try:
                state.control_message_ids.remove(dedupe_key)
            except ValueError:
                pass
            self._host._release_presentation_operation(self._key)
            raise
        try:
            task = asyncio.create_task(
                self._host._handle_control(
                    self._key,
                    raw,
                    response_bodies,
                    binding,
                ),
                name=f"channel-control:{state.channel_name}:{raw.message_id}",
            )
            return cast(ControlReceipt, await _await_task_after_cancellation(task))
        finally:
            try:
                await binding.aclose()
            finally:
                self._host._release_presentation_operation(self._key)
                while len(state.control_message_ids) > 500:
                    expired = state.control_message_ids.popleft()
                    state.control_message_id_set.discard(expired)


class _ChannelTurnStream:
    """Register callback subscriptions on one exact binding."""

    def __init__(self, host: ChannelGenerationHost, key: tuple[str, str]) -> None:
        self._host = host
        self._key = key

    def subscribe(self, callback: TurnStreamCallback) -> StreamSubscription:
        """Attach one async callback until it is explicitly closed."""

        if not _is_async_callback(callback):
            raise TypeError("turn stream callback 必须是 async callable")
        state = self._host._binding(self._key)
        if state.stopping or state.stopped or not state.start_attempted:
            raise RuntimeError("turn stream binding 已关闭或尚未 start")
        subscription = _ChannelStreamSubscription(self._host, self._key, callback)
        state.subscriptions[id(subscription)] = subscription
        return subscription


class _ChannelStreamSubscription:
    """Own accepted presentation callbacks and participate in exact drain."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
        callback: TurnStreamCallback,
    ) -> None:
        self._host = host
        self._key = key
        self._callback = callback
        self._admission_open = True
        self._closed = False
        self._running = 0
        self._quiescent = asyncio.Event()
        self._quiescent.set()

    def close_admission(self) -> None:
        """Synchronously reject new events while retaining accepted callbacks."""

        self._admission_open = False

    def _admit(self, *, allow_closed: bool = False) -> bool:
        if self._closed or (not self._admission_open and not allow_closed):
            return False
        state = self._host._binding(self._key)
        if state.stopped or (not allow_closed and (not state.admission_open or state.stopping)):
            return False
        self._running += 1
        self._quiescent.clear()
        try:
            self._host._begin_presentation_operation(
                self._key,
                allow_closed=allow_closed,
            )
        except BaseException:
            self._running -= 1
            if self._running == 0:
                self._quiescent.set()
            raise
        return True

    async def invoke(self, event: TurnStreamEvent) -> PresentationReceipt:
        """Invoke one already-admitted callback and settle its typed receipt."""

        try:
            return await self._invoke_plugin(event)
        except _PresentationContractFailure:
            raise
        except asyncio.CancelledError:
            receipt = self._host._unknown_presentation_receipt(
                event,
                "turn stream callback cancelled",
            )
            self._host._mark_presentation_failed(
                self._key,
                event.presentation_id,
            )
            return receipt
        except BaseException as error:
            receipt = self._host._unknown_presentation_receipt(event, str(error))
            self._host._mark_presentation_failed(
                self._key,
                event.presentation_id,
            )
            return receipt
        finally:
            self._running -= 1
            self._host._release_presentation_operation(self._key)
            if self._running == 0:
                self._quiescent.set()

    async def _invoke_plugin(self, event: TurnStreamEvent) -> PresentationReceipt:
        """Call and validate one plugin callback inside its exact boundary."""

        state = self._host._binding(self._key)
        with _channel_entrypoint(state, "channel.turn_stream"):
            result = self._callback(event)
            if not inspect.isawaitable(result):
                receipt = self._host._unknown_presentation_receipt(
                    event,
                    "turn stream callback 必须返回 awaitable",
                )
                self._host._mark_presentation_failed(
                    self._key,
                    event.presentation_id,
                )
                raise _PresentationContractFailure(
                    "turn stream callback 必须返回 awaitable",
                    receipt,
                )
            result = await result
            if not isinstance(result, PresentationReceipt):
                receipt = self._host._unknown_presentation_receipt(
                    event,
                    "turn stream callback 必须返回 PresentationReceipt",
                )
                self._host._mark_presentation_failed(
                    self._key,
                    event.presentation_id,
                )
                raise _PresentationContractFailure(
                    "turn stream callback 必须返回 PresentationReceipt",
                    receipt,
                )
            if result.presentation_id != event.presentation_id:
                receipt = self._host._unknown_presentation_receipt(
                    event,
                    "presentation receipt identity 不匹配",
                )
                self._host._mark_presentation_failed(
                    self._key,
                    event.presentation_id,
                )
                raise _PresentationContractFailure(
                    "presentation receipt identity 不匹配",
                    receipt,
                )
            if result.status is DeliveryStatus.UNKNOWN:
                self._host._mark_presentation_failed(
                    self._key,
                    event.presentation_id,
                )
            return result

    async def await_quiescence(self) -> None:
        await self._quiescent.wait()

    async def close(self) -> None:
        """Detach after admission is closed and every callback is terminal."""

        if self._closed:
            return
        self.close_admission()
        await self.await_quiescence()
        self._closed = True
        state = self._host._binding(self._key)
        state.subscriptions.pop(id(self), None)


class _ChannelAttachmentImport:
    """Expose the Core attachment importer only while this binding is admitted."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
        port: ChannelAttachmentImportPort,
    ) -> None:
        self._host = host
        self._key = key
        self._port = port

    async def import_bytes(
        self,
        data: bytes,
        *,
        kind: AttachmentKind,
        filename: str | None,
        media_type: str | None,
    ) -> AttachmentRef:
        """Import bytes while retaining this binding in the Host drain set."""

        self._host._begin_attachment_operation(self._key)
        try:
            result = self._port.import_bytes(
                data,
                kind=kind,
                filename=filename,
                media_type=media_type,
            )
            if not inspect.isawaitable(result):
                raise TypeError("attachment import 必须返回 awaitable")
            result = await result
            if not isinstance(result, AttachmentRef):
                raise TypeError("attachment import 必须返回 AttachmentRef")
            return result
        finally:
            self._host._release_attachment_operation(self._key)


class _ChannelAttachmentRead:
    """Expose Core read leases while charging the exact binding in-flight count."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
        port: ChannelAttachmentReadPort,
    ) -> None:
        self._host = host
        self._key = key
        self._port = port

    async def acquire(self, ref: AttachmentRef) -> AttachmentReadLease:
        """Acquire a binding-owned read lease or release the claim on failure."""

        if not isinstance(ref, AttachmentRef):
            raise TypeError("attachment read 只接受 AttachmentRef")
        self._host._begin_attachment_operation(self._key)
        try:
            result = self._port.acquire(ref)
            if not inspect.isawaitable(result):
                raise TypeError("attachment acquire 必须返回 awaitable")
            lease = await result
            _validate_attachment_read_lease(lease, ref)
            return _ChannelAttachmentReadLease(self._host, self._key, lease, ref)
        except BaseException:
            self._host._release_attachment_operation(self._key)
            raise


class _ChannelAttachmentReadLease:
    """Keep the Host claim until the underlying lease close has settled successfully."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        key: tuple[str, str],
        lease: AttachmentReadLease,
        ref: AttachmentRef,
    ) -> None:
        self._host = host
        self._key = key
        self._lease = lease
        self._ref = ref
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def ref(self) -> AttachmentRef:
        return self._ref

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        """Read through the retained Core lease until it is critically closed."""

        if self._closed:
            raise RuntimeError("attachment read lease 已关闭")
        result = self._lease.read_bytes(max_bytes=max_bytes)
        if not inspect.isawaitable(result):
            raise TypeError("attachment read_bytes 必须返回 awaitable")
        value = await result
        if not isinstance(value, bytes):
            raise TypeError("attachment read_bytes 必须返回 bytes")
        return value

    async def aclose(self) -> None:
        """Finish the underlying close before releasing Host drain ownership."""

        async with self._close_lock:
            if self._closed:
                return
            task = asyncio.create_task(
                _invoke_attachment_lease_close(self._lease),
                name=f"channel_attachment_lease_close:{self._key[0]}:{self._key[1]}",
            )
            try:
                await _await_task_after_cancellation(task)
            except asyncio.CancelledError:
                if _task_succeeded(task):
                    self._host._release_attachment_operation(self._key)
                    self._closed = True
                raise
            if task.cancelled():
                raise asyncio.CancelledError
            self._host._release_attachment_operation(self._key)
            self._closed = True


class ChannelGeneration:
    """A closed set of channel bindings staged for one committed snapshot."""

    def __init__(
        self,
        host: ChannelGenerationHost,
        snapshot_id: str,
        keys: tuple[tuple[str, str], ...],
    ) -> None:
        self._host = host
        self.snapshot_id = snapshot_id
        self._keys = keys

    def channel(self, channel_name: str) -> ChannelBinding:
        key = (self.snapshot_id, channel_name)
        if key not in self._host._bindings:
            raise KeyError(channel_name)
        return ChannelBinding(self._host, key)

    def open_admission(self) -> None:
        """Open all channels only after the caller has finalized publication."""

        for key in self._keys:
            self._host._open_admission(key)

    def close_admission(self) -> None:
        """Close all channels synchronously before draining them."""

        for key in self._keys:
            self._host._close_admission(key)

    async def drain(self) -> None:
        """Wait for all accepted deliveries in this generation."""

        await asyncio.gather(*(self._host._drain(key) for key in self._keys))

    async def stop(self) -> tuple[StopReceipt, ...]:
        """Stop all channels in reverse declaration order."""

        return await self._host.stop(self.snapshot_id)


class ChannelGenerationHost:
    """Materialize formal adapters without retaining a plugin Fiber or Context."""

    def __init__(
        self,
        *,
        on_before_start: BeforeStartCallback,
        config_revision_checker: ConfigRevisionChecker,
        on_failure: FailureCallback,
        snapshot_lease_acquirer: SnapshotLeaseAcquirer | None = None,
        identity_resolver: IdentityResolver | None = None,
        identity_rememberer: IdentityRememberer | None = None,
        identity_rollbacker: IdentityRollbacker | None = None,
        attachment_import: ChannelAttachmentImportPort | None = None,
        attachment_read: ChannelAttachmentReadPort | None = None,
        control_interrupter: ControlInterrupter | None = None,
        control_response_dispatcher: ControlResponseDispatcher | None = None,
    ) -> None:
        if not callable(on_before_start):
            raise TypeError("on_before_start 必须是 async callback")
        if not callable(config_revision_checker):
            raise TypeError("config_revision_checker 必须是 async callback")
        if not callable(on_failure):
            raise TypeError("on_failure 必须可调用")
        if snapshot_lease_acquirer is not None and not callable(
            snapshot_lease_acquirer
        ):
            raise TypeError("snapshot_lease_acquirer 必须可调用")
        if (identity_resolver is None) != (identity_rememberer is None):
            raise TypeError("identity resolver/rememberer 必须同时绑定")
        if identity_resolver is not None and not callable(identity_resolver):
            raise TypeError("identity_resolver 必须可调用")
        if identity_rememberer is not None and not callable(identity_rememberer):
            raise TypeError("identity_rememberer 必须可调用")
        if identity_rollbacker is not None and identity_rememberer is None:
            raise TypeError("identity rollbacker 需要 identity rememberer")
        if identity_rollbacker is not None and not callable(identity_rollbacker):
            raise TypeError("identity_rollbacker 必须可调用")
        if (attachment_import is None) != (attachment_read is None):
            raise TypeError("attachment import/read ports 必须同时绑定")
        if attachment_import is not None and not callable(
            getattr(attachment_import, "import_bytes", None)
        ):
            raise TypeError("attachment_import 必须提供 import_bytes(data, ...)")
        if attachment_read is not None and not callable(
            getattr(attachment_read, "acquire", None)
        ):
            raise TypeError("attachment_read 必须提供 acquire(ref)")
        if control_interrupter is not None and not callable(control_interrupter):
            raise TypeError("control_interrupter 必须可调用")
        if control_response_dispatcher is not None and not callable(
            control_response_dispatcher
        ):
            raise TypeError("control_response_dispatcher 必须可调用")
        self._on_before_start = on_before_start
        self._config_revision_checker = config_revision_checker
        self._on_failure = on_failure
        self._snapshot_lease_acquirer = snapshot_lease_acquirer
        self._inbound_publisher: InboundPublisher | None = None
        self._identity_resolver = identity_resolver
        self._identity_rememberer = identity_rememberer
        self._identity_rollbacker = identity_rollbacker
        self._attachment_import = attachment_import
        self._attachment_read = attachment_read
        self._control_interrupter = control_interrupter
        self._control_response_dispatcher = control_response_dispatcher
        self._bindings: dict[tuple[str, str], _ChannelBindingState] = {}
        self._binding_leases: set[ChannelBindingLease] = set()
        self._tombstones: dict[tuple[str, str], ChannelCleanupTombstone] = {}
        self._start_counts: dict[tuple[str, str], int] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def bind_inbound_publisher(self, publisher: InboundPublisher) -> None:
        """Bind the sole MessageBus inbound admission callback."""

        if not callable(publisher):
            raise TypeError("Channel inbound publisher 必须可调用")
        if self._inbound_publisher is not None:
            raise RuntimeError("Channel inbound publisher 已绑定")
        self._inbound_publisher = publisher

    def bind_control_interrupter(self, interrupter: ControlInterrupter) -> None:
        """Bind Core's typed interrupt effect owner exactly once."""

        if not callable(interrupter):
            raise TypeError("control interrupter 必须可调用")
        if self._control_interrupter is not None:
            raise RuntimeError("control interrupter 已绑定")
        self._control_interrupter = interrupter

    def bind_control_response_dispatcher(
        self,
        dispatcher: ControlResponseDispatcher,
    ) -> None:
        """Bind same-binding awaited control response dispatch exactly once."""

        if not callable(dispatcher):
            raise TypeError("control response dispatcher 必须可调用")
        if self._control_response_dispatcher is not None:
            raise RuntimeError("control response dispatcher 已绑定")
        self._control_response_dispatcher = dispatcher

    async def start_formal(
        self,
        snapshot: object,
        provider_client_factories: Mapping[str, ProviderClientFactory],
        *,
        boot_owner: str = "plugin-manager",
    ) -> ChannelGeneration:
        """Start one exact committed snapshot using only the formal target."""

        committed = _require_committed_snapshot(snapshot)
        _text(boot_owner, "boot_owner")
        catalog = getattr(committed, "channel_catalog", None)
        registry = (
            catalog.registry
            if isinstance(catalog, CommittedChannelCatalog)
            else committed.channel_registry
        )
        if registry is None:
            raise RuntimeError("committed snapshot 缺少 channel registry")
        snapshot_id = _text(committed.snapshot_id, "snapshot_id")
        if snapshot_id in self._locks or any(
            key[0] == snapshot_id for key in self._tombstones
        ):
            raise RuntimeError(f"channel generation 已存在: {snapshot_id}")
        if not isinstance(provider_client_factories, Mapping):
            raise TypeError("provider_client_factories 必须是 mapping")
        descriptors = tuple(registry.descriptors)
        expected_names = {descriptor.name for descriptor in descriptors}
        if set(provider_client_factories) != expected_names:
            raise RuntimeError("provider client factory 必须与 committed channel catalog 精确匹配")
        if not descriptors:
            return ChannelGeneration(self, snapshot_id, ())
        if len({id(value) for value in provider_client_factories.values()}) != len(
            provider_client_factories
        ):
            raise RuntimeError("一个 provider client factory 不能被多个 channel 共享")
        lock = asyncio.Lock()
        self._locks[snapshot_id] = lock
        started_keys: list[tuple[str, str]] = []
        try:
            for descriptor in descriptors:
                key = (snapshot_id, descriptor.name)
                state = await self._materialize_binding(
                    committed,
                    registry,
                    descriptor,
                    provider_client_factories[descriptor.name],
                    boot_owner=boot_owner,
                )
                self._bindings[key] = state
                started_keys.append(key)
                await self._start_binding(key)
            return ChannelGeneration(self, snapshot_id, tuple(started_keys))
        except BaseException as error:
            cleanup_task = asyncio.create_task(
                self._cleanup_keys(
                    snapshot_id,
                    tuple(started_keys)
                    + tuple(
                        key
                        for key in self._bindings
                        if key[0] == snapshot_id and key not in started_keys
                    ),
                    cause=error,
                ),
                name=f"channel_generation_cleanup:{snapshot_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError as cleanup_cancelled:
                if _task_succeeded(cleanup_task):
                    self._remove_generation(snapshot_id)
                if isinstance(error, asyncio.CancelledError):
                    raise error
                raise cleanup_cancelled from error
            except BaseException as cleanup_error:
                raise error from cleanup_error
            else:
                self._remove_generation(snapshot_id)
            raise error
        finally:
            if snapshot_id not in self._bindings and not any(
                key[0] == snapshot_id for key in self._tombstones
            ):
                self._locks.pop(snapshot_id, None)

    async def stop(self, snapshot_id: str) -> tuple[StopReceipt, ...]:
        """Stop a staged generation after closing admission and draining it."""

        keys = self._generation_keys(snapshot_id)
        if not keys:
            if any(key[0] == snapshot_id for key in self._tombstones):
                raise RuntimeError(f"channel generation cleanup 未完成: {snapshot_id}")
            return ()
        lock = self._locks.setdefault(snapshot_id, asyncio.Lock())
        async with lock:
            for key in keys:
                self._close_admission(key)
            cleanup_task = asyncio.create_task(
                self._stop_keys(keys),
                name=f"channel_generation_stop:{snapshot_id}",
            )
            try:
                receipts = await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if _task_succeeded(cleanup_task):
                    self._remove_generation(snapshot_id)
                raise
            except BaseException:
                raise
            if not any(key[0] == snapshot_id for key in self._tombstones):
                self._remove_generation(snapshot_id)
            return cast(tuple[StopReceipt, ...], receipts)

    def open_admission(self, snapshot_id: str) -> None:
        """Open every channel in one staged snapshot after publication."""

        generation = self.get(snapshot_id)
        if generation is None:
            raise KeyError(snapshot_id)
        generation.open_admission()

    def close_admission(self, snapshot_id: str) -> None:
        """Close every channel in one snapshot before a critical drain."""

        generation = self.get(snapshot_id)
        if generation is None:
            raise KeyError(snapshot_id)
        generation.close_admission()

    async def drain(self, snapshot_id: str) -> None:
        """Drain all in-flight provider deliveries for one snapshot."""

        generation = self.get(snapshot_id)
        if generation is None:
            return
        await generation.drain()

    async def retry_generation_cleanup(self, binding_token: str) -> None:
        """Retry one retained owner by exact binding token only."""

        await self._retry_binding_cleanup(binding_token)

    async def _retry_binding_cleanup(self, binding_token: str) -> None:
        """Retry one tombstone only when its exact binding token is supplied."""

        _text(binding_token, "binding_token")
        matches = tuple(
            (key, tombstone)
            for key, tombstone in self._tombstones.items()
            if tombstone.binding_token == binding_token
        )
        if len(matches) != 1:
            raise RuntimeError("channel cleanup binding token 未知或不唯一")
        (key, tombstone), = matches
        snapshot_id = key[0]
        lock = self._locks.setdefault(snapshot_id, asyncio.Lock())
        async with lock:
            cleanup_task = asyncio.create_task(
                self._retry_keys((key,)),
                name=f"channel_generation_retry:{snapshot_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if _task_succeeded(cleanup_task):
                    self._remove_generation(snapshot_id)
                raise
            if not any(key[0] == snapshot_id for key in self._tombstones):
                self._remove_generation(snapshot_id)

    def failure(
        self,
        snapshot_id: str,
        channel_name: str | None = None,
        *,
        binding_token: str | None = None,
    ) -> ChannelCleanupTombstone | tuple[ChannelCleanupTombstone, ...] | None:
        """Return the exact cleanup tombstone(s), if any."""

        if channel_name is not None:
            tombstone = self._tombstones.get((snapshot_id, channel_name))
            if binding_token is not None and (
                tombstone is None or tombstone.binding_token != binding_token
            ):
                return None
            return tombstone
        failures = tuple(
            tombstone
            for (owner, _), tombstone in self._tombstones.items()
            if owner == snapshot_id
            and (binding_token is None or tombstone.binding_token == binding_token)
        )
        return failures or None

    def start_count(self, snapshot_id: str, channel_name: str | None = None) -> int:
        """Return adapter.start invocation count; journal failure leaves it at zero."""

        if channel_name is not None:
            return self._start_counts.get((snapshot_id, channel_name), 0)
        return sum(
            count
            for (owner, _), count in self._start_counts.items()
            if owner == snapshot_id
        )

    def get(self, snapshot_id: str) -> ChannelGeneration | None:
        """Return a facade while at least one binding remains owned."""

        keys = self._generation_keys(snapshot_id)
        if not keys:
            return None
        return ChannelGeneration(self, snapshot_id, keys)

    def acquire_binding(
        self,
        snapshot_lease: RuntimeSnapshotLease,
        channel_name: str,
        *,
        _allow_claimed_after_close: bool = False,
    ) -> ChannelBindingLease:
        """Fork one exact stable lease and retain its live Channel binding."""

        snapshot = snapshot_lease.snapshot
        if not snapshot_lease.active:
            raise RuntimeError("RuntimeSnapshot lease 已关闭")
        snapshot_id = _text(snapshot.snapshot_id, "snapshot_id")
        key = (snapshot_id, _text(channel_name, "channel_name"))
        state = self._binding(key)
        catalog = getattr(snapshot, "channel_catalog", None)
        if state.plugin_id == "core":
            if not isinstance(catalog, CommittedChannelCatalog):
                raise RuntimeError("Core channel binding 缺少 committed catalog")
            definition = catalog.definition(state.channel_name)
            if definition is None or definition.generation_id != state.generation_id:
                raise RuntimeError("Core channel binding 与 catalog generation 不一致")
            registry = catalog.registry
        else:
            generation = snapshot.generations.get(state.plugin_id)
            if generation is None or generation.generation_id != state.generation_id:
                raise RuntimeError("Channel binding 与 RuntimeSnapshot generation 不一致")
            registry = snapshot.channel_registry
        if registry is None or not any(
            descriptor.name == state.channel_name
            and descriptor.owner == state.plugin_id
            for descriptor in registry.descriptors
        ):
            raise RuntimeError("Channel binding 不属于 exact RuntimeSnapshot catalog")
        if state.stopped or (
            not _allow_claimed_after_close
            and (not state.admission_open or state.stopping)
        ):
            raise RuntimeError("channel admission 已关闭")
        forked = snapshot_lease.fork()
        state.in_flight += 1
        state.drain_event.clear()
        binding = ChannelBindingLease(self, key, forked)
        self._binding_leases.add(binding)
        return binding

    async def dispatch_outbound(
        self,
        envelope: OutboundEnvelope,
        binding: object,
    ) -> ChannelDeliveryReceipt:
        """Dispatch through one lease created by this exact Host."""

        if not isinstance(binding, ChannelBindingLease) or binding._host is not self:
            raise RuntimeError("v3 Channel outbound binding 不属于当前 Host")
        return await binding.deliver(envelope)

    async def publish_turn_event(
        self,
        snapshot_id: str,
        channel_name: str,
        event: TurnStreamEvent,
        *,
        binding: ChannelBindingLease | None = None,
    ) -> tuple[PresentationReceipt, ...]:
        """Publish one typed event to callbacks attached to an exact binding."""

        if not isinstance(event, TurnStreamEvent):
            raise TypeError("turn stream 只接受 TurnStreamEvent")
        state = self._binding((_text(snapshot_id, "snapshot_id"), _text(channel_name, "channel_name")))
        if ChannelCapability.TURN_STREAM not in state.capabilities:
            raise RuntimeError("channel 未声明 turn stream capability")
        if binding is not None:
            if binding._host is not self or binding._key != (snapshot_id, channel_name):
                raise RuntimeError("turn stream binding lease 不属于 exact binding")
            if not binding.active:
                raise RuntimeError("turn stream binding lease 已关闭")
        elif not state.admission_open or state.stopping or state.stopped:
            raise RuntimeError("turn stream admission 已关闭")
        self._validate_presentation_event(state, event)
        tasks: list[asyncio.Task[PresentationReceipt]] = []
        for subscription in tuple(state.subscriptions.values()):
            if not subscription._admit(allow_closed=binding is not None):
                continue
            tasks.append(
                asyncio.create_task(
                    subscription.invoke(event),
                    name=(
                        f"channel-presentation:{state.channel_name}:"
                        f"{event.presentation_id}"
                    ),
                )
            )
        receipts: list[PresentationReceipt] = []
        cancelled = False
        contract_errors: list[_PresentationContractFailure] = []
        for task in tasks:
            try:
                receipts.append(
                    cast(PresentationReceipt, await _await_task_after_cancellation(task))
                )
            except asyncio.CancelledError:
                cancelled = True
            except _PresentationContractFailure as error:
                receipts.append(error.receipt)
                contract_errors.append(error)
        if cancelled:
            raise asyncio.CancelledError
        if contract_errors:
            raise contract_errors[0]
        return tuple(receipts)

    async def _acquire_control_binding(
        self,
        key: tuple[str, str],
    ) -> ChannelBindingLease:
        """Fork an exact snapshot lease for one control effect."""

        acquirer = self._snapshot_lease_acquirer
        if acquirer is None:
            raise RuntimeError("Channel control exact snapshot lease owner 未绑定")
        state = self._binding(key)
        source = acquirer(state.snapshot_id)
        binding: ChannelBindingLease | None = None
        try:
            try:
                if source.snapshot.snapshot_id != state.snapshot_id:
                    raise RuntimeError("Channel control 与当前 stable snapshot 不一致")
                binding = self.acquire_binding(
                    source,
                    state.channel_name,
                    _allow_claimed_after_close=True,
                )
            finally:
                release = asyncio.create_task(
                    source.release(),
                    name=f"channel-control-source-release:{state.channel_name}",
                )
                await _await_task_after_cancellation(release)
        except BaseException as error:
            if binding is not None:
                cleanup = asyncio.create_task(
                    binding.aclose(),
                    name=f"channel-control-binding-rollback:{state.channel_name}",
                )
                try:
                    await _await_task_after_cancellation(cleanup)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
            raise
        return cast(ChannelBindingLease, binding)

    async def _handle_control(
        self,
        key: tuple[str, str],
        raw: RawInbound,
        response_bodies: ControlResponseBodies,
        binding: ChannelBindingLease,
    ) -> ControlReceipt:
        interrupter = self._control_interrupter
        if interrupter is None:
            raise RuntimeError("Channel control interrupt owner 未绑定")
        result = interrupter(raw)
        if not inspect.isawaitable(result):
            raise TypeError("control interrupter 必须返回 awaitable")
        result = await result
        reason = _control_reason(result)
        accepted = reason == "interrupted"
        response = await self._dispatch_control_response(
            key,
            raw,
            response_bodies.interrupted if accepted else response_bodies.idle,
            binding,
        )
        return ControlReceipt(accepted, reason, response)

    async def _dispatch_control_response(
        self,
        key: tuple[str, str],
        raw: RawInbound,
        body: str,
        binding: ChannelBindingLease,
    ) -> ChannelDeliveryReceipt | None:
        dispatcher = self._control_response_dispatcher
        state = self._binding(key)
        delivery_id = _control_delivery_id(state.binding_token, raw.message_id)
        envelope = OutboundEnvelope(
            logical_delivery_id=delivery_id,
            delivery_id=delivery_id,
            attempt_sequence=1,
            snapshot_id=state.snapshot_id,
            generation_id=state.generation_id,
            binding_token=state.binding_token,
            channel=state.channel_name,
            recipient=raw.recipient or raw.message.chat_id,
            body=body,
            metadata={"control_message_id": raw.message_id},
        )
        try:
            if dispatcher is None:
                result = binding.deliver(envelope)
            else:
                result = _invoke_control_dispatcher(dispatcher, envelope, binding)
            if not inspect.isawaitable(result):
                raise TypeError("control response dispatcher 必须返回 awaitable")
            result = await result
            if not isinstance(result, ChannelDeliveryReceipt):
                raise TypeError("control response dispatcher 必须返回 ChannelDeliveryReceipt")
            if result.delivery_id != delivery_id:
                raise RuntimeError("control response receipt identity 不匹配")
            return result
        except asyncio.CancelledError:
            return ChannelDeliveryReceipt(
                delivery_id,
                DeliveryStatus.UNKNOWN,
                error="control response cancelled",
            )
        except Exception as error:
            return ChannelDeliveryReceipt(
                delivery_id,
                DeliveryStatus.UNKNOWN,
                error=str(error) or type(error).__name__,
            )

    def _validate_presentation_event(
        self,
        state: _ChannelBindingState,
        event: TurnStreamEvent,
    ) -> None:
        presentation_id = event.presentation_id
        if presentation_id in state.failed_presentations:
            raise RuntimeError(
                f"presentation 已因 UNKNOWN 终止，禁止继续 patch: {presentation_id}"
            )
        payload = event.payload
        turn_id = payload.turn_id
        previous_turn_id = state.presentation_turn_ids.get(presentation_id)
        if previous_turn_id is not None and previous_turn_id != turn_id:
            raise RuntimeError("presentation turn_id 不一致")
        state.presentation_turn_ids[presentation_id] = turn_id
        previous_sequence = state.presentation_sequences.get(presentation_id)
        if (
            presentation_id in state.completed_presentations
            and event.kind is not TurnStreamEventKind.TURN_OUTPUT_COMPLETED
        ):
            raise RuntimeError("turn.output.completed 后禁止继续 patch")
        if event.kind is TurnStreamEventKind.TURN_STARTED:
            if previous_sequence is not None:
                raise RuntimeError("turn.started 不能重复")
            if not isinstance(payload, TurnStartedPresentation):
                raise TypeError("turn.started payload 类型无效")
            state.presentation_sequences[presentation_id] = 0
            return
        sequence = getattr(payload, "sequence", None)
        if previous_sequence is None:
            raise RuntimeError("turn stream 必须先发送 turn.started")
        if not isinstance(sequence, int) or sequence <= previous_sequence:
            raise RuntimeError("turn stream sequence 必须单调递增")
        state.presentation_sequences[presentation_id] = sequence
        if event.kind is TurnStreamEventKind.TURN_OUTPUT_COMPLETED:
            state.completed_presentations.add(presentation_id)

    def _unknown_presentation_receipt(
        self,
        event: TurnStreamEvent,
        error: str,
    ) -> PresentationReceipt:
        return PresentationReceipt(
            presentation_id=event.presentation_id,
            status=DeliveryStatus.UNKNOWN,
            error=error or "turn stream callback failed",
        )

    def _mark_presentation_failed(
        self,
        key: tuple[str, str],
        presentation_id: str,
    ) -> None:
        state = self._binding(key)
        state.failed_presentations.add(presentation_id)

    def _begin_presentation_operation(
        self,
        key: tuple[str, str],
        *,
        allow_closed: bool = False,
    ) -> None:
        state = self._binding(key)
        if state.stopped or (not allow_closed and (state.stopping or not state.admission_open)):
            raise RuntimeError("presentation binding admission 已关闭")
        state.in_flight += 1
        state.drain_event.clear()

    def _release_presentation_operation(self, key: tuple[str, str]) -> None:
        self._release_in_flight(key)

    async def _recover_inbound(
        self,
        key: tuple[str, str],
        raw: RawInbound,
    ) -> bool:
        """Replace only a prior accepted claim for Core-owned durable recovery."""

        if not isinstance(raw, RawInbound):
            raise TypeError("Channel recovery 只接受 RawInbound")
        state = self._binding(key)
        if state.plugin_id != "core":
            raise RuntimeError("Channel recovery 只属于 Core durable inbound")
        if raw.message.metadata.get("mobile_v3_handoff") is not True:
            raise RuntimeError("Channel recovery 缺少 Mobile durable marker")
        if (
            ChannelCapability.INBOUND not in state.capabilities
            or state.inbound_identity is not InboundIdentity.PROVIDER_MESSAGE_ID
        ):
            raise RuntimeError("channel 未声明可用的 inbound capability")
        if raw.message.channel != state.channel_name:
            raise RuntimeError("RawInbound channel 与 exact binding 不一致")
        if not state.admission_open or state.stopping or state.stopped:
            raise RuntimeError("channel admission 已关闭")

        # 1. 进程内恢复复用旧 claim，不在任何 await 窗口释放 duplicate fence。
        dedupe_key = (raw.provider_identity or "", raw.message_id)
        if dedupe_key in state.inbound_message_id_set:
            if state.inbound_message_ids.count(dedupe_key) != 1:
                raise RuntimeError("Channel inbound dedupe index 不一致")
            return await self._admit_inbound(
                key,
                raw,
                _retained_claim=dedupe_key,
            )

        # 2. 进程重启时无内存 claim，由 current binding 新建正常 claim。
        return await self._admit_inbound(key, raw)

    async def _admit_inbound(
        self,
        key: tuple[str, str],
        raw: RawInbound,
        *,
        _retained_claim: tuple[str, str] | None = None,
    ) -> bool:
        """Acquire, enqueue, and retain one deduplicated exact inbound lease."""

        if not isinstance(raw, RawInbound):
            raise TypeError("Channel ingress 只接受 RawInbound")
        state = self._binding(key)
        if (
            ChannelCapability.INBOUND not in state.capabilities
            or state.inbound_identity is not InboundIdentity.PROVIDER_MESSAGE_ID
        ):
            raise RuntimeError("channel 未声明可用的 inbound capability")
        if raw.message.channel != state.channel_name:
            raise RuntimeError("RawInbound channel 与 exact binding 不一致")
        if not state.admission_open or state.stopping or state.stopped:
            raise RuntimeError("channel admission 已关闭")
        if raw.message.metadata.get("mobile_v3_handoff") is True and not (
            state.plugin_id == "core" and state.channel_name == "akashic"
        ):
            raise RuntimeError("Mobile durable handoff 只属于 Core akashic binding")
        provider_scope = raw.provider_identity or ""
        dedupe_key = (provider_scope, raw.message_id)
        retained_claim = _retained_claim is not None
        if retained_claim and (
            _retained_claim != dedupe_key
            or dedupe_key not in state.inbound_message_id_set
            or state.inbound_message_ids.count(dedupe_key) != 1
        ):
            raise RuntimeError("Channel retained recovery claim 不一致")
        if not retained_claim and dedupe_key in state.inbound_message_id_set:
            return False
        acquirer = self._snapshot_lease_acquirer
        publisher = self._inbound_publisher
        if acquirer is None or publisher is None:
            raise RuntimeError("Channel ingress runtime ports 未绑定")

        # 1. Claim before any await so concurrent duplicate callbacks serialize.
        if not retained_claim:
            state.inbound_message_id_set.add(dedupe_key)
            state.inbound_message_ids.append(dedupe_key)
        self._begin_presentation_operation(key, allow_closed=False)
        accepted = False
        binding: ChannelBindingLease | None = None
        envelope: InboundEnvelope | None = None
        identity_receipt: object | None = None
        try:
            source = acquirer(state.snapshot_id)
            try:
                if source.snapshot.snapshot_id != key[0]:
                    raise RuntimeError("Channel ingress 与当前 stable snapshot 不一致")
                binding = self.acquire_binding(
                    source,
                    state.channel_name,
                    _allow_claimed_after_close=True,
                )
            finally:
                release = asyncio.create_task(
                    source.release(),
                    name=f"channel-ingress-source-release:{state.channel_name}",
                )
                await _await_task_after_cancellation(release)
            if raw.provider_identity is not None:
                rememberer = self._identity_rememberer
                if rememberer is None or raw.recipient is None:
                    raise RuntimeError("Channel identity runtime port 未绑定")
                identity_task = asyncio.create_task(
                    rememberer(
                        state.channel_name,
                        raw.provider_identity,
                        raw.recipient,
                    ),
                    name=f"channel-identity-remember:{state.channel_name}",
                )
                try:
                    identity_receipt = await _await_task_after_cancellation(
                        identity_task
                    )
                    if identity_task.cancelled():
                        raise asyncio.CancelledError
                except BaseException:
                    if identity_task.done() and not identity_task.cancelled():
                        identity_receipt = identity_task.result()
                    raise
            envelope = InboundEnvelope(
                message_id=raw.message_id,
                snapshot_id=binding.snapshot_id,
                generation_id=binding.generation_id,
                binding_token=binding.binding_token,
                message=raw.message,
                lease=binding,
            )
            await publisher(envelope)
            if envelope.owner in (InboundOwner.INGRESS, InboundOwner.CLOSED):
                raise RuntimeError("Channel inbound publisher 未接管 envelope")
            accepted = True
            while len(state.inbound_message_ids) > 500:
                expired = state.inbound_message_ids.popleft()
                state.inbound_message_id_set.remove(expired)
            return True
        except BaseException as error:
            cleanup_errors: list[BaseException] = []
            accepted_elsewhere = envelope is not None and envelope.owner not in (
                InboundOwner.INGRESS,
                InboundOwner.CLOSED,
            )
            if accepted_elsewhere:
                accepted = True
            else:
                if envelope is not None and envelope.owner is InboundOwner.INGRESS:
                    close_task = asyncio.create_task(
                        envelope.close(InboundOwner.INGRESS),
                        name=f"channel-ingress-close:{state.channel_name}",
                    )
                    try:
                        await _settle_cleanup_task(close_task)
                    except BaseException as close_error:
                        cleanup_errors.append(close_error)
                if identity_receipt is not None:
                    rollback_task = asyncio.create_task(
                        self._rollback_identity_write(identity_receipt),
                        name=f"channel-identity-rollback:{state.channel_name}",
                    )
                    try:
                        await _settle_cleanup_task(rollback_task)
                    except BaseException as rollback_error:
                        cleanup_errors.append(rollback_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "Channel inbound acceptance 回滚失败",
                    [error, *cleanup_errors],
                ) from error
            raise
        finally:
            try:
                if not accepted:
                    if not retained_claim:
                        state.inbound_message_id_set.discard(dedupe_key)
                        try:
                            state.inbound_message_ids.remove(dedupe_key)
                        except ValueError:
                            pass
                    if binding is not None and binding.active:
                        await binding.aclose()
            finally:
                self._release_presentation_operation(key)

    async def _rollback_identity_write(self, receipt: object) -> None:
        """精确撤销失败 acceptance 写入且显式暴露 rollback fence 冲突。"""

        rollbacker = self._identity_rollbacker
        if rollbacker is None:
            raise RuntimeError("Channel identity rollback runtime port 未绑定")
        if await rollbacker(receipt) is not True:
            raise RuntimeError("Channel identity rollback fence 已被并发状态取代")

    def _resolve_identity(
        self,
        key: tuple[str, str],
        provider_identity: str,
    ) -> str | None:
        """Resolve only while the exact binding still accepts work."""

        state = self._binding(key)
        if not state.admission_open or state.stopping or state.stopped:
            raise RuntimeError("channel admission 已关闭")
        resolver = self._identity_resolver
        if resolver is None:
            raise RuntimeError("Channel identity runtime port 未绑定")
        return resolver(state.channel_name, provider_identity)

    async def _materialize_binding(
        self,
        snapshot: Any,
        registry: ChannelRegistrySnapshot,
        descriptor: Any,
        provider_client_factory: ProviderClientFactory,
        *,
        boot_owner: str,
    ) -> _ChannelBindingState:
        catalog = getattr(snapshot, "channel_catalog", None)
        core_definition: CoreChannelDefinition | None = None
        if descriptor.owner == "core":
            if not isinstance(catalog, CommittedChannelCatalog):
                raise RuntimeError("Core channel 缺少 committed catalog")
            core_definition = catalog.definition(descriptor.name)
            if core_definition is None:
                raise RuntimeError(f"Core channel definition 缺失: {descriptor.name}")
            module = ModuleType(f"akashic_core_channel_{descriptor.name}")
            provenance = core_definition.provenance
            config = core_definition.config
            generation_id = core_definition.generation_id
            factory: Callable[[ChannelFactoryContext], ChannelAdapter] | None = (
                core_definition.factory
            )
            artifact_pointer = "core"
            source_revision = core_definition.source_revision
            raw_config_revision = core_definition.config_revision
        else:
            generation = snapshot.generations.get(descriptor.owner)
            if generation is None:
                raise RuntimeError(f"channel owner generation 缺失: {descriptor.owner}")
            if not isinstance(generation.instance, ComposablePlugin):
                raise RuntimeError(f"channel owner 不是 ComposablePlugin: {descriptor.owner}")
            plugin = generation.instance
            module = plugin.module
            if not isinstance(module, ModuleType):
                raise RuntimeError(f"channel owner module 无效: {descriptor.owner}")
            provenance = _find_provenance(
                registry,
                descriptor.owner,
                generation.generation_id,
                descriptor.name,
            )
            if (
                provenance.source_revision != generation.source_revision
                or provenance.config_revision
                != channel_config_revision(generation.config_projection)
                or provenance.factory_export != descriptor.factory_export
            ):
                raise RuntimeError(f"channel factory provenance drift: {descriptor.name}")
            config = generation.config_projection
            if not isinstance(config, Mapping):
                raise RuntimeError(
                    f"channel generation config projection 无效: {descriptor.owner}"
                )
            generation_id = generation.generation_id
            factory = None
            artifact_pointer = str(generation.plugin_dir)
            source_revision = generation.source_revision
            raw_config_revision = generation.config_revision
        if provenance.factory_export != descriptor.factory_export:
            raise RuntimeError(f"channel factory provenance drift: {descriptor.name}")
        binding_token = uuid.uuid4().hex
        descriptor_digest = _descriptor_digest(descriptor)
        _validate_provider_factory(provider_client_factory, descriptor.name)
        return _ChannelBindingState(
            snapshot_id=snapshot.snapshot_id,
            catalog_identity=registry.identity,
            plugin_id=descriptor.owner,
            generation_id=generation_id,
            channel_name=descriptor.name,
            capabilities=descriptor.capabilities,
            inbound_identity=descriptor.inbound_identity,
            module=module,
            artifact_pointer=artifact_pointer,
            factory=factory,
            adapter=None,
            provider_client_factory=provider_client_factory,
            binding_token=binding_token,
            config=config,
            credential_paths=descriptor.credential_paths,
            factory_context=None,
            factory_export=descriptor.factory_export,
            source_revision=source_revision,
            config_revision=provenance.config_revision,
            raw_config_revision=raw_config_revision,
            descriptor_digest=descriptor_digest,
            target="formal",
            boot_owner=boot_owner,
            start_attempt=1,
        )

    async def _start_binding(self, key: tuple[str, str]) -> None:
        state = self._binding(key)
        record = ChannelStartRecord(
            snapshot_id=state.snapshot_id,
            catalog_identity=state.catalog_identity,
            plugin_id=state.plugin_id,
            generation_id=state.generation_id,
            channel_name=state.channel_name,
            binding_token=state.binding_token,
            module_name=state.module.__name__,
            artifact_pointer=state.artifact_pointer,
            factory_export=state.factory_export,
            source_revision=state.source_revision,
            config_revision=state.config_revision,
            raw_config_revision=state.raw_config_revision,
            descriptor_digest=state.descriptor_digest,
            target=state.target,
            boot_owner=state.boot_owner,
            attempt=state.start_attempt,
        )
        await _require_awaitable(self._on_before_start(record), "on_before_start")
        await _require_awaitable(
            self._config_revision_checker(record),
            "config_revision_checker",
        )
        factory = state.factory
        if factory is None:
            factory = _resolve_sync_factory(
                state.module,
                state.factory_export,
            )
            state.factory = factory
        credentials = _resolve_credentials(state.config, state.credential_paths)
        state.control_port = (
            _ChannelControl(self, key)
            if ChannelCapability.CONTROL in state.capabilities
            else None
        )
        state.turn_stream_port = (
            _ChannelTurnStream(self, key)
            if ChannelCapability.TURN_STREAM in state.capabilities
            else None
        )
        state.factory_context = ChannelFactoryContext(
            snapshot_id=state.snapshot_id,
            generation_id=state.generation_id,
            binding_token=state.binding_token,
            config=state.config,
            credentials=credentials,
            provider_client_factory=state.provider_client_factory,
            ingress=(
                _ChannelIngress(self, key)
                if ChannelCapability.INBOUND in state.capabilities
                else None
            ),
            identity=(
                _ChannelIdentity(self, key)
                if ChannelCapability.INBOUND in state.capabilities
                and self._identity_resolver is not None
                else None
            ),
            attachment_import=(
                _ChannelAttachmentImport(self, key, self._attachment_import)
                if self._attachment_import is not None
                else None
            ),
            attachment_read=(
                _ChannelAttachmentRead(self, key, self._attachment_read)
                if self._attachment_read is not None
                else None
            ),
            control=state.control_port,
            turn_stream=state.turn_stream_port,
        )
        try:
            with _channel_entrypoint(state, "channel.factory"):
                adapter = factory(state.factory_context)
                if inspect.isawaitable(adapter):
                    _close_awaitable(adapter)
                    raise TypeError(
                        f"channel factory 不得是 async: {state.channel_name}"
                    )
                _validate_adapter(adapter, state.channel_name)
        except asyncio.CancelledError:
            state.internal_cancellation = "factory-start"
            raise
        state.adapter = cast(ChannelAdapter, adapter)
        self._start_counts[key] = self._start_counts.get(key, 0) + 1
        state.start_attempted = True
        if ChannelCapability.INBOUND in state.capabilities:
            attach_runtime = getattr(adapter, "attach_runtime", None)
            open_admission = getattr(adapter, "open_admission", None)
            close_admission = getattr(adapter, "close_admission", None)
            if not all(
                callable(item)
                for item in (attach_runtime, open_admission, close_admission)
            ):
                raise TypeError(
                    f"inbound channel adapter 缺少 runtime lifecycle: {state.channel_name}"
                )
            context = state.factory_context
            if context is None:
                raise RuntimeError("channel factory context 尚未保存")
            with _channel_entrypoint(state, "channel.attach_runtime"):
                attached = attach_runtime(
                    ChannelRuntimePorts(
                        snapshot_id=context.snapshot_id,
                        generation_id=context.generation_id,
                        binding_token=context.binding_token,
                        ingress=context.ingress,
                        identity=context.identity,
                        attachment_import=context.attachment_import,
                        recovery_ingress=(
                            _ChannelRecoveryIngress(self, key)
                            if state.plugin_id == "core"
                            else None
                        ),
                    )
                )
                if inspect.isawaitable(attached):
                    _close_awaitable(attached)
                    raise TypeError("channel adapter.attach_runtime 必须同步返回")
            state.runtime_attached = True
        if (
            ChannelCapability.CONTROL in state.capabilities
            or ChannelCapability.TURN_STREAM in state.capabilities
        ):
            attach_presentation = getattr(adapter, "attach_presentation", None)
            if not callable(attach_presentation):
                raise TypeError(
                    f"channel adapter 缺少 attach_presentation: {state.channel_name}"
                )
            with _channel_entrypoint(state, "channel.attach_presentation"):
                attached = attach_presentation(
                    ChannelPresentationPorts(
                        control=state.control_port,
                        turn_stream=state.turn_stream_port,
                    )
                )
                if inspect.isawaitable(attached):
                    _close_awaitable(attached)
                    raise TypeError(
                        "channel adapter.attach_presentation 必须同步返回"
                    )
        try:
            with _channel_entrypoint(state, "channel.start"):
                result = await _invoke_async(
                    cast(ChannelAdapter, state.adapter),
                    "start",
                )
                if not isinstance(result, ChannelReady):
                    raise TypeError(
                        f"channel adapter.start 返回值无效: {state.channel_name}"
                    )
                if result.binding_token != state.binding_token:
                    raise RuntimeError(
                        f"channel adapter binding token 不匹配: {state.channel_name}"
                    )
                if result.admission_open:
                    raise RuntimeError(
                        f"channel adapter 必须以 closed 状态启动: {state.channel_name}"
                    )
        except asyncio.CancelledError:
            state.internal_cancellation = "adapter-start"
            raise
        state.ready = result
        state.started = True

    async def _deliver(
        self,
        key: tuple[str, str],
        request: ProviderDeliveryRequest,
        *,
        retained_binding: ChannelBindingLease | None = None,
    ) -> ProviderDeliveryReceipt:
        """Deliver through a live binding or an exact lease admitted before close."""

        state = self._binding(key)
        if not isinstance(request, ProviderDeliveryRequest):
            raise TypeError("channel deliver 只接受 ProviderDeliveryRequest")
        if request.binding_token != state.binding_token:
            raise RuntimeError("channel delivery binding token 不匹配")
        if retained_binding is not None:
            if (
                retained_binding not in self._binding_leases
                or retained_binding._host is not self
                or retained_binding._key != key
                or not retained_binding.active
            ):
                raise RuntimeError("Channel binding lease 未由当前 Host 登记")
        if state.stopping or state.stopped:
            raise RuntimeError("channel admission 已关闭")
        if retained_binding is None and not state.admission_open:
            raise RuntimeError("channel admission 已关闭")
        state.in_flight += 1
        state.drain_event.clear()
        try:
            with _channel_entrypoint(state, "channel.deliver"):
                result = await _invoke_async(
                    cast(ChannelAdapter, state.adapter),
                    "deliver",
                    request,
                )
                if not isinstance(result, ProviderDeliveryReceipt):
                    raise TypeError(
                        f"channel deliver receipt 类型无效: {state.channel_name}"
                    )
                if result.delivery_id != request.delivery_id:
                    raise RuntimeError("channel delivery receipt identity 不匹配")
            return result
        finally:
            state.in_flight -= 1
            if state.in_flight == 0:
                state.drain_event.set()

    def _open_admission(self, key: tuple[str, str]) -> None:
        state = self._binding(key)
        if state.stopped or state.stopping:
            raise RuntimeError("不能打开已停止的 channel binding")
        if not state.started:
            raise RuntimeError("channel binding 尚未 start")
        state.admission_open = True
        if state.runtime_attached and state.adapter is not None:
            open_admission = getattr(state.adapter, "open_admission")
            try:
                with _channel_entrypoint(state, "channel.open_admission"):
                    open_admission()
            except BaseException:
                state.admission_open = False
                raise

    def _close_admission(self, key: tuple[str, str]) -> None:
        state = self._binding(key)
        was_open = state.admission_open
        state.admission_open = False
        if not was_open:
            return
        try:
            if state.runtime_attached and state.adapter is not None:
                with _channel_entrypoint(state, "channel.close_admission"):
                    getattr(state.adapter, "close_admission")()
        finally:
            state.admission_open = False

    async def _drain(self, key: tuple[str, str]) -> None:
        state = self._binding(key)
        await state.drain_event.wait()

    def _begin_attachment_operation(self, key: tuple[str, str]) -> None:
        """Admit one attachment import/acquire before its first await."""

        state = self._binding(key)
        if self._attachment_import is None or self._attachment_read is None:
            raise RuntimeError("Channel attachment runtime ports 未绑定")
        if not state.admission_open or state.stopping or state.stopped:
            raise RuntimeError("channel admission 已关闭")
        state.in_flight += 1
        state.drain_event.clear()

    def _release_attachment_operation(self, key: tuple[str, str]) -> None:
        """Release one attachment operation only after its owner is settled."""

        self._release_in_flight(key)

    def _release_binding_lease(self, binding: ChannelBindingLease) -> None:
        if binding not in self._binding_leases:
            raise RuntimeError("Channel binding lease 未由当前 Host 登记")
        self._binding_leases.remove(binding)
        self._release_in_flight(binding._key)

    def _release_in_flight(self, key: tuple[str, str]) -> None:
        state = self._binding(key)
        if state.in_flight <= 0:
            raise RuntimeError("channel binding lease 计数下溢")
        state.in_flight -= 1
        if state.in_flight == 0:
            state.drain_event.set()

    async def _stop_binding(self, key: tuple[str, str]) -> StopReceipt:
        state = self._binding(key)
        self._close_admission(key)
        if state.stopped and state.stop_receipt is not None:
            return state.stop_receipt
        state.stopping = True
        for subscription in tuple(state.subscriptions.values()):
            subscription.close_admission()
        await state.drain_event.wait()
        try:
            subscriptions = tuple(state.subscriptions.values())
            for subscription in subscriptions:
                await subscription.close()
            failures: list[ChannelCleanupFailure] = []
            receipt = state.stop_receipt
            if not state.adapter_stop_succeeded:
                state.adapter_stop_settled = False
                if state.start_attempted and state.adapter is not None:
                    try:
                        with _channel_entrypoint(state, "channel.stop"):
                            result = await _invoke_async(state.adapter, "stop")
                            if not isinstance(result, StopReceipt):
                                raise TypeError("channel adapter.stop 返回值无效")
                            if result.binding_token != state.binding_token:
                                raise RuntimeError(
                                    "channel stop receipt binding token 不匹配"
                                )
                            if any(
                                failure.binding_token != state.binding_token
                                or failure.plugin_id != state.plugin_id
                                or failure.generation_id != state.generation_id
                                for failure in result.failures
                            ):
                                raise RuntimeError(
                                    "channel stop receipt failure owner 不匹配"
                                )
                            receipt_failures = list(result.failures)
                            if not result.resources_closed:
                                receipt_failures.append(
                                    _cleanup_failure(
                                        state,
                                        "adapter",
                                        "adapter resources_closed=false",
                                    )
                                )
                            if receipt_failures:
                                raise _ChannelStopReceiptFailure(
                                    result,
                                    tuple(receipt_failures),
                                )
                        receipt = result
                        state.stop_receipt = result
                        state.adapter_stop_succeeded = True
                    except _ChannelStopReceiptFailure as error:
                        receipt = error.receipt
                        state.stop_receipt = error.receipt
                        failures.extend(error.failures)
                    except BaseException as error:
                        failures.append(
                            _cleanup_failure(
                                state,
                                "adapter",
                                str(error),
                                error,
                            )
                        )
                    finally:
                        state.adapter_stop_settled = True
                else:
                    state.adapter_stop_succeeded = True
                    state.adapter_stop_settled = True
            if not state.factory_close_succeeded:
                state.factory_close_settled = False
                try:
                    await _close_provider_factory(state.provider_client_factory)
                    state.factory_close_succeeded = True
                except BaseException as error:
                    failures.append(
                        _cleanup_failure(
                            state,
                            "provider-client-factory",
                            str(error),
                            error,
                        )
                    )
                finally:
                    state.factory_close_settled = True
            if state.internal_cancellation is not None and not failures:
                failures.append(
                    _cleanup_failure(
                        state,
                        state.internal_cancellation,
                        f"{state.internal_cancellation} cancelled",
                    )
                )
            if failures:
                error = RuntimeError("channel cleanup failed: " + "; ".join(item.message for item in failures))
                await self._retain_tombstone(key, state, failures[0], error)
                raise error
            if receipt is None:
                receipt = StopReceipt(binding_token=state.binding_token, resources_closed=True)
            state.stop_receipt = receipt
            state.stopped = True
            state.stopping = False
            self._tombstones.pop(key, None)
            return receipt
        except asyncio.CancelledError:
            state.stopping = True
            raise

    async def _stop_binding_critical(self, key: tuple[str, str]) -> StopReceipt:
        """Finish one binding cleanup before restoring caller cancellation."""

        cleanup_task = asyncio.create_task(
            self._stop_binding(key),
            name=f"channel_binding_stop:{key[0]}:{key[1]}",
        )
        return cast(StopReceipt, await _await_task_after_cancellation(cleanup_task))

    async def _stop_keys(self, keys: tuple[tuple[str, str], ...]) -> tuple[StopReceipt, ...]:
        receipts: list[StopReceipt] = []
        failures: list[BaseException] = []
        for key in reversed(keys):
            try:
                receipts.append(await self._stop_binding(key))
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError(
                "channel generation cleanup failed: "
                + "; ".join(str(error) or type(error).__name__ for error in failures)
            ) from failures[0]
        return tuple(receipts)

    async def _cleanup_keys(
        self,
        generation_id: str,
        keys: tuple[tuple[str, str], ...],
        *,
        cause: BaseException,
    ) -> None:
        for key in reversed(keys):
            state = self._bindings.get(key)
            if state is None:
                continue
            self._close_admission(key)
        await asyncio.gather(*(self._drain(key) for key in keys if key in self._bindings))
        await self._stop_keys(keys)

    async def _retry_keys(self, keys: tuple[tuple[str, str], ...]) -> None:
        failures: list[BaseException] = []
        for key in reversed(keys):
            state = self._bindings.get(key)
            tombstone = self._tombstones.get(key)
            if state is None or tombstone is None:
                continue
            if tombstone.binding_token != state.binding_token:
                failures.append(RuntimeError("channel cleanup exact binding token drift"))
                continue
            try:
                state.internal_cancellation = None
                await self._stop_binding(key)
                self._tombstones.pop(key, None)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                failures.append(error)
        if failures:
            raise RuntimeError("channel generation cleanup retry failed") from failures[0]

    async def _retain_tombstone(
        self,
        key: tuple[str, str],
        state: _ChannelBindingState,
        failure: ChannelCleanupFailure,
        error: BaseException,
    ) -> None:
        previous = self._tombstones.get(key)
        tombstone = ChannelCleanupTombstone(
            snapshot_id=state.snapshot_id,
            catalog_identity=state.catalog_identity,
            plugin_id=state.plugin_id,
            generation_id=state.generation_id,
            channel_name=state.channel_name,
            module=state.module,
            adapter=state.adapter,
            factory=state.factory,
            factory_context=state.factory_context,
            provider_client_factory=state.provider_client_factory,
            binding_token=state.binding_token,
            artifact_pointer=state.artifact_pointer,
            factory_export=state.factory_export,
            source_revision=state.source_revision,
            config_revision=state.config_revision,
            raw_config_revision=state.raw_config_revision,
            descriptor_digest=state.descriptor_digest,
            target=state.target,
            boot_owner=state.boot_owner,
            adapter_stop_settled=state.adapter_stop_settled,
            adapter_stop_succeeded=state.adapter_stop_succeeded,
            factory_close_settled=state.factory_close_settled,
            factory_close_succeeded=state.factory_close_succeeded,
            resource=failure.resource,
            error_type=type(error).__name__,
            message=str(error),
            action="retry_generation_cleanup",
            attempt_count=1 if previous is None else previous.attempt_count + 1,
        )
        self._tombstones[key] = tombstone
        result = self._on_failure(tombstone)
        if inspect.isawaitable(result):
            await result

    def _binding(self, key: tuple[str, str]) -> _ChannelBindingState:
        state = self._bindings.get(key)
        if state is None:
            tombstone = self._tombstones.get(key)
            if tombstone is not None:
                raise RuntimeError(f"channel binding cleanup 未完成: {key[1]}")
            raise KeyError(key[1])
        return state

    def _generation_keys(self, generation_id: str) -> tuple[tuple[str, str], ...]:
        return tuple(key for key in self._bindings if key[0] == generation_id)

    def _remove_generation(self, generation_id: str) -> None:
        states = [state for key, state in self._bindings.items() if key[0] == generation_id]
        if any(not state.stopped for state in states):
            return
        for key in tuple(self._bindings):
            if key[0] == generation_id:
                self._bindings.pop(key, None)
        if not any(key[0] == generation_id for key in self._tombstones):
            self._locks.pop(generation_id, None)


def _require_committed_snapshot(snapshot: object) -> Any:
    if not hasattr(snapshot, "snapshot_id") or not hasattr(snapshot, "channel_registry"):
        raise TypeError("ChannelGenerationHost 只接受 RuntimeSnapshot")
    if getattr(snapshot, "state", None) != "committed":
        raise RuntimeError("ChannelGenerationHost 只接受 committed RuntimeSnapshot")
    root = getattr(snapshot, "composition_root", None)
    registry = getattr(snapshot, "channel_registry", None)
    catalog = getattr(snapshot, "channel_catalog", None)
    if catalog is not None:
        if not isinstance(catalog, CommittedChannelCatalog):
            raise TypeError("channel_catalog 类型无效")
        if root is not None and catalog.root_instance_token is not root.instance_token:
            raise RuntimeError("committed channel catalog 不属于 exact composition Root")
        registry = catalog.registry
    if root is None or registry is None:
        raise RuntimeError("committed snapshot 必须带 exact composition Root/channel registry")
    if registry.root_instance_token is not root.instance_token:
        raise RuntimeError("channel registry 不属于 exact composition Root")
    if catalog is None and getattr(snapshot, "channel_registry_identity", registry.identity) != registry.identity:
        raise RuntimeError("channel registry identity drift")
    if not isinstance(registry, ChannelRegistrySnapshot):
        raise TypeError("channel_registry 类型无效")
    return snapshot


def _find_provenance(
    registry: ChannelRegistrySnapshot,
    owner: str,
    generation_id: str,
    channel_name: str,
) -> Any:
    matches = tuple(
        item
        for item in registry.factories
        if item.plugin_id == owner
        and item.generation_id == generation_id
        and item.channel_name == channel_name
    )
    if len(matches) != 1:
        raise RuntimeError(f"channel factory provenance 缺失或重复: {channel_name}")
    return matches[0]


def _descriptor_digest(descriptor: Any) -> str:
    """Hash the complete immutable descriptor identity for durable ownership."""

    payload = {
        "owner": descriptor.owner,
        "name": descriptor.name,
        "capabilities": [item.value for item in descriptor.capabilities],
        "factory_export": descriptor.factory_export,
        "inbound_identity": (
            None if descriptor.inbound_identity is None else descriptor.inbound_identity.value
        ),
        "credential_paths": list(descriptor.credential_paths),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _resolve_sync_factory(module: ModuleType, export: str) -> Callable[[ChannelFactoryContext], ChannelAdapter]:
    value: object = module
    for segment in export.replace(":", ".").split("."):
        value = getattr(value, segment)
    if not callable(value):
        raise TypeError(f"channel factory export 不可调用: {export}")
    if inspect.iscoroutinefunction(value):
        raise TypeError(f"channel factory 不得是 async: {export}")
    try:
        inspect.signature(value).bind(cast(object, None))
    except (TypeError, ValueError) as error:
        raise TypeError(f"channel factory ABI 必须是 factory(context): {export}") from error
    return cast(Callable[[ChannelFactoryContext], ChannelAdapter], value)


def _resolve_credentials(
    config: Mapping[str, object],
    paths: tuple[str, ...],
) -> Mapping[str, CredentialRef]:
    result: dict[str, CredentialRef] = {}
    for path in paths:
        current: object = config
        found = True
        for segment in path.split("."):
            if not isinstance(current, Mapping) or segment not in current:
                found = False
                break
            current = current[segment]
        if not found:
            continue
        if not isinstance(current, CredentialRef):
            raise RuntimeError(f"channel credential path 未被 redacted: {path}")
        result[path] = current
    if paths and not result:
        raise RuntimeError("channel credential paths 均未出现在正式配置投影")
    return result


def _validate_adapter(adapter: object, channel_name: str) -> None:
    if any(not callable(getattr(adapter, name, None)) for name in ("start", "deliver", "stop")):
        raise TypeError(f"channel adapter ABI 无效: {channel_name}")


def _validate_provider_factory(factory: object, channel_name: str) -> None:
    if any(not callable(getattr(factory, name, None)) for name in ("create", "aclose")):
        raise TypeError(f"provider client factory ABI 无效: {channel_name}")


async def _invoke_async(adapter: object, method_name: str, *args: object) -> object:
    result = getattr(adapter, method_name)(*args)
    if not inspect.isawaitable(result):
        raise TypeError(f"channel adapter.{method_name} 必须返回 awaitable")
    return await result


async def _close_provider_factory(factory: ProviderClientFactory) -> None:
    result = factory.aclose()
    if not inspect.isawaitable(result):
        raise TypeError("provider client factory.aclose 必须返回 awaitable")
    await result


def _validate_attachment_read_lease(
    lease: object,
    ref: AttachmentRef,
) -> None:
    """Validate the store lease before transferring its drain ownership."""

    if not callable(getattr(lease, "read_bytes", None)):
        raise TypeError("attachment read lease 必须提供 read_bytes(max_bytes=...)")
    if not callable(getattr(lease, "aclose", None)):
        raise TypeError("attachment read lease 必须提供 aclose()")
    if getattr(lease, "ref", None) != ref:
        raise RuntimeError("attachment read lease ref 不匹配")


async def _invoke_attachment_lease_close(lease: AttachmentReadLease) -> None:
    """Invoke a store lease close and preserve its cancellation/error result."""

    result = lease.aclose()
    if not inspect.isawaitable(result):
        raise TypeError("attachment read lease aclose 必须返回 awaitable")
    await result


async def _require_awaitable(result: object, name: str) -> None:
    if not inspect.isawaitable(result):
        raise TypeError(f"{name} 必须是 async callback")
    await cast(Awaitable[None], result)


def _cleanup_failure(
    state: _ChannelBindingState,
    resource: str,
    message: str,
    error: BaseException | None = None,
) -> ChannelCleanupFailure:
    normalized_message = message or (type(error).__name__ if error is not None else "cleanup failed")
    return ChannelCleanupFailure(
        stage="channel-stop",
        plugin_id=state.plugin_id,
        generation_id=state.generation_id,
        binding_token=state.binding_token,
        resource=resource,
        error_type=type(error).__name__ if error is not None else "RuntimeError",
        message=normalized_message,
        retry_action="retry_generation_cleanup",
    )


def _close_awaitable(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} 必须是非空字符串")
    return value


async def _await_task_after_cancellation(task: asyncio.Task[Any]) -> Any:
    """Finish critical cleanup before restoring caller cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            continue
    try:
        result = task.result()
    except asyncio.CancelledError:
        result = None
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _settle_cleanup_task(task: asyncio.Task[Any]) -> Any:
    """Settle one acceptance rollback task before preserving the original failure."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _task_succeeded(task: asyncio.Task[Any]) -> bool:
    if not task.done():
        return False
    try:
        return task.exception() is None
    except asyncio.CancelledError:
        return False


def _control_reason(
    value: object,
) -> Literal["interrupted", "idle"]:
    if value is True or value == "interrupted":
        return "interrupted"
    if value is False or value == "idle":
        return "idle"
    raise TypeError("control interrupter 必须返回 interrupted/idle 或 bool")


def _control_delivery_id(binding_token: str, message_id: str) -> str:
    payload = f"control\x00{binding_token}\x00{message_id}".encode("utf-8")
    return "control:" + hashlib.sha256(payload).hexdigest()


def _invoke_control_dispatcher(
    dispatcher: ControlResponseDispatcher,
    envelope: OutboundEnvelope,
    binding: ChannelBindingLease,
) -> object:
    """Dispatch one control response through its exact retained binding."""

    return dispatcher(envelope, binding)


def _is_async_callback(callback: object) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


__all__ = [
    "ChannelBinding",
    "ChannelBindingLease",
    "ChannelCleanupTombstone",
    "ChannelGeneration",
    "ChannelGenerationHost",
    "ChannelStartRecord",
]
