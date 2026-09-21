"""Channel provider：贡献 Context 拥有连接，原 binding 拥有外部效果回执。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ContextManager, Literal, Protocol, cast

from agent.plugin_composition.context import Context, RuntimeScope
from agent.plugin_composition.model import CompositionError, FiberState, ServiceKey
from agent.plugin_composition.requests import RequestContext
from agent.plugin_composition.diagnostics import plugin_entrypoint
from agent.plugin_composition.channels import (
    CHANNELS,
    CHANNEL_INPUT,
    ChannelDefinition,
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
    ChannelRuntimePorts,
    ControlReceipt,
    ControlResponseBodies,
    DeliveryStatus,
    InboundEnvelope,
    InboundIdentity,
    OutboundEnvelope,
    ProviderDeliveryReceipt,
    ProviderDeliveryRequest,
    PresentationReceipt,
    RawInbound,
    DURABLE_HANDOFF_ID,
    DURABLE_INBOUND_MARKER,
    DURABLE_PROVIDER_MESSAGE_ID,
    StreamSubscription,
    StopReceipt,
    TurnStreamCallback,
    TurnStreamEvent,
    TurnStreamEventKind,
    TurnStreamPort,
    TurnStartedPresentation,
)

from agent.plugin_composition.admission import SOURCE_ADMISSION
from agent.plugin_composition.channel_io import (
    InputCustody, INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
)
from agent.plugin_composition.runtime_lifecycle import RUNTIME_STARTING, RuntimeStarting
from agent.plugins.snapshot import get_current_runtime_lease

if TYPE_CHECKING:
    from agent.plugins.snapshot import RuntimeSnapshotLease

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


@dataclass
class _ChannelBindingState:
    snapshot_id: str
    plugin_id: str
    generation_id: str
    channel_name: str
    capabilities: tuple[ChannelCapability, ...]
    inbound_identity: InboundIdentity | None
    factory: Callable[[ChannelFactoryContext], ChannelAdapter]
    adapter: ChannelAdapter | None
    binding_token: str
    config: Mapping[str, object]
    factory_context: ChannelFactoryContext | None
    plugin_context: Context | None = None
    activation_token: object | None = None
    listeners: set[asyncio.Task[object]] = field(default_factory=set)
    start_task: asyncio.Task[object] | None = None
    stop_task: asyncio.Task[StopReceipt] | None = None
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
    inbound_message_ids: deque[tuple[str, str]] = field(default_factory=deque)
    inbound_message_id_set: set[tuple[str, str]] = field(default_factory=set)
    durable_reservations: dict[str, "_DurableReservation"] = field(default_factory=dict)
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


@dataclass(frozen=True, slots=True)
class _DurableReservation:
    """Keep the exact channel-owned authority for one durable handoff."""

    handoff_id: str
    channel: str
    session_key: str
    provider_message_id: str


def _channel_entrypoint(
    state: _ChannelBindingState,
    operation: str,
) -> ContextManager[object]:
    """记录真实贡献插件的调用边界。"""

    return plugin_entrypoint(
        plugin_id=state.plugin_id,
        generation_id=state.generation_id,
        fiber=state.plugin_id,
        operation=operation,
        entrypoint=state.channel_name,
    )


class ChannelBindingLease:
    """Own one forked snapshot lease and one exact Host in-flight claim."""

    def __init__(
        self,
        host: PluginChannels,
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


class _ChannelIngress:
    """Admit provider text into one exact formal binding."""

    def __init__(
        self,
        host: PluginChannels,
        key: tuple[str, str],
    ) -> None:
        self._host = host
        self._key = key

    async def admit(self, raw: RawInbound) -> bool:
        return await self._host._admit_inbound(self._key, raw)


class _ChannelDurableInbound:
    """Expose only the durable handoff operations declared by one binding."""

    def __init__(
        self,
        host: PluginChannels,
        key: tuple[str, str],
    ) -> None:
        self._host = host
        self._key = key

    def _state(self, *, allow_closed: bool = False) -> _ChannelBindingState:
        state = self._host._binding(self._key)
        if (
            ChannelCapability.DURABLE_INBOUND not in state.capabilities
            or ChannelCapability.INBOUND not in state.capabilities
            or state.inbound_identity is not InboundIdentity.PROVIDER_MESSAGE_ID
        ):
            raise RuntimeError("channel 未声明 durable inbound capability")
        if state.stopped or (not allow_closed and (state.stopping or not state.admission_open)):
            raise RuntimeError("channel durable inbound admission 已关闭")
        return state

    def _custody(self, *, allow_closed: bool = False) -> InputCustody:
        self._state(allow_closed=allow_closed)
        custody = self._host._input_custody
        if custody is None:
            raise RuntimeError("Channel input custody runtime port 未绑定")
        return custody

    @staticmethod
    def _reservation_metadata(raw: RawInbound) -> _DurableReservation:
        metadata = raw.message.metadata
        handoff_id = metadata.get(DURABLE_HANDOFF_ID)
        provider_message_id = metadata.get(DURABLE_PROVIDER_MESSAGE_ID)
        session_key = metadata.get("session_key_override")
        if (
            metadata.get(DURABLE_INBOUND_MARKER) is not True
            or not isinstance(handoff_id, str)
            or not handoff_id
            or provider_message_id != raw.message_id
            or not isinstance(session_key, str)
            or not session_key.strip()
        ):
            raise RuntimeError("durable inbound reservation identity 无效")
        return _DurableReservation(
            handoff_id=handoff_id,
            channel=raw.message.channel,
            session_key=session_key.strip(),
            provider_message_id=raw.message_id,
        )

    def _remembered_reservation(
        self,
        state: _ChannelBindingState,
        *,
        handoff_id: str | None = None,
        session_key: str | None = None,
        provider_message_id: str | None = None,
    ) -> _DurableReservation | None:
        if handoff_id is not None:
            reservation = state.durable_reservations.get(handoff_id)
            if reservation is None or self._host._durable_reservation_owners.get(
                reservation.handoff_id
            ) != self._key:
                return None
            return reservation
        for reservation in state.durable_reservations.values():
            if (
                reservation.session_key == session_key
                and reservation.provider_message_id == provider_message_id
                and self._host._durable_reservation_owners.get(
                    reservation.handoff_id
                ) == self._key
            ):
                return reservation
        return None

    async def reserve(self, raw: RawInbound) -> bool:
        state = self._state()
        reservation = self._reservation_metadata(raw)
        if reservation.channel != state.channel_name:
            raise RuntimeError("RawInbound channel 与 exact binding 不一致")
        owner = self._host._durable_reservation_owners.get(reservation.handoff_id)
        if owner is not None and owner != self._key:
            raise RuntimeError("durable inbound reservation 已由另一 binding 持有")
        reserve_task = asyncio.create_task(
            self._custody().reserve_durable_inbound(raw),
            name=f"channel-durable-reserve:{reservation.handoff_id}",
        )
        accepted, cancelled = await _await_reservation_after_cancellation(
            reserve_task
        )
        if accepted:
            self._host._remember_durable_reservation(self._key, reservation)
        if cancelled:
            # The Bus task has settled and the exact binding is recorded before
            # restoring cancellation, so stop/recovery can still own this row.
            raise asyncio.CancelledError
        return accepted

    async def defer(self, handoff_id: str) -> None:
        state = self._state(allow_closed=True)
        reservation = self._remembered_reservation(state, handoff_id=handoff_id)
        if reservation is None:
            raise RuntimeError("durable inbound defer 不属于 exact binding reservation")
        released = await self._custody(allow_closed=True).defer_durable_inbound(handoff_id)
        if not released:
            raise RuntimeError("durable inbound reservation 仍有执行 owner")
        self._host._forget_durable_reservation(self._key, handoff_id)

    async def settle_rejected(
        self,
        *,
        session_key: str,
        provider_message_id: str,
    ) -> None:
        state = self._state(allow_closed=True)
        reservation = self._remembered_reservation(
            state,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )
        if reservation is None:
            raise RuntimeError("durable inbound settle 不属于 exact binding reservation")
        await self._custody(allow_closed=True).settle_rejected_inbound(
            channel=state.channel_name,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )
        self._host._forget_durable_reservation(self._key, reservation.handoff_id)

    def has_pending(
        self,
        *,
        session_key: str,
        provider_message_id: str,
    ) -> bool:
        state = self._host._binding(self._key)
        reservation = self._remembered_reservation(
            state,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )
        self._state(allow_closed=reservation is not None)
        return self._custody(allow_closed=reservation is not None).has_pending_durable_inbound(
            channel=state.channel_name,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )

    def pending_attachment_refs(
        self,
        *,
        session_key: str,
        provider_message_id: str,
    ) -> tuple[AttachmentRef, ...] | None:
        state = self._host._binding(self._key)
        reservation = self._remembered_reservation(
            state,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )
        self._state(allow_closed=reservation is not None)
        return self._custody(allow_closed=reservation is not None).pending_durable_attachment_refs(
            channel=state.channel_name,
            session_key=session_key,
            provider_message_id=provider_message_id,
        )

    async def recover(self, raw: RawInbound) -> bool:
        state = self._state()
        if raw.message.channel != state.channel_name:
            raise RuntimeError("RawInbound channel 与 exact binding 不一致")
        _ = self._reservation_metadata(raw)
        return await self._host._recover_inbound(self._key, raw)


class _ChannelIdentity:
    """Resolve recipients through the Core-owned durable identity index."""

    def __init__(
        self,
        host: PluginChannels,
        key: tuple[str, str],
    ) -> None:
        self._host = host
        self._key = key

    def resolve(self, provider_identity: str) -> str | None:
        return self._host._resolve_identity(self._key, provider_identity)


class _ChannelControl:
    """Expose one exact binding's deduplicated interrupt facade."""

    def __init__(self, host: PluginChannels, key: tuple[str, str]) -> None:
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

    def __init__(self, host: PluginChannels, key: tuple[str, str]) -> None:
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
        host: PluginChannels,
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
            receipt = self._host._failed_presentation_receipt(
                event,
                "turn stream callback cancelled",
            )
            self._host._mark_presentation_failed(
                self._key,
                event.presentation_id,
            )
            return receipt
        except BaseException as error:
            receipt = self._host._failed_presentation_receipt(event, str(error))
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
                receipt = self._host._failed_presentation_receipt(
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
                receipt = self._host._failed_presentation_receipt(
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
                receipt = self._host._failed_presentation_receipt(
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
            if result.status is DeliveryStatus.FAILED:
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
        host: PluginChannels,
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
        host: PluginChannels,
        key: tuple[str, str],
        port: ChannelAttachmentReadPort,
    ) -> None:
        self._host = host
        self._key = key
        self._port = port

    def resolve_refs(self, artifact_ids: tuple[str, ...]) -> tuple[AttachmentRef, ...]:
        """Resolve opaque artifact ids through the exact read owner."""

        resolver = getattr(self._port, "resolve_refs", None)
        if not callable(resolver):
            raise TypeError("attachment read owner 缺少 resolve_refs(ids)")
        result = resolver(artifact_ids)
        if not isinstance(result, tuple) or any(
            not isinstance(ref, AttachmentRef) for ref in result
        ):
            raise TypeError("attachment resolver 必须返回 AttachmentRef tuple")
        if tuple(ref.artifact_id for ref in result) != artifact_ids:
            raise RuntimeError("attachment resolver 未保留请求顺序")
        return result

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
        host: PluginChannels,
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

    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes:
        """Read one bounded chunk without releasing the binding claim."""

        if self._closed:
            raise RuntimeError("attachment read lease 已关闭")
        result = self._lease.read_chunk(offset=offset, max_bytes=max_bytes)
        if not inspect.isawaitable(result):
            raise TypeError("attachment read_chunk 必须返回 awaitable")
        value = await result
        if not isinstance(value, bytes):
            raise TypeError("attachment read_chunk 必须返回 bytes")
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


class PluginChannels:
    """一个 Root 的普通 Channel provider；不跨 Root 复用连接。"""

    def __init__(self, ctx: Context) -> None:
        self._context = ctx
        self._root_token = ctx.root_instance_token
        self._admission = ctx.require(SOURCE_ADMISSION)
        self._input_custody = ctx.require(INPUT_CUSTODY)
        identity = ctx.require(CHANNEL_IDENTITY)
        self._identity_resolver = identity.resolve
        self._identity_rememberer = identity.remember
        self._identity_rollbacker = identity.rollback
        self._attachment_import = ctx.require(CHANNEL_ATTACHMENT_IMPORT)
        self._attachment_read = ctx.require(CHANNEL_ATTACHMENT_READ)
        self._snapshot_lease_acquirer = self._admission.lease
        self._recovery_snapshot_lease_acquirer = self._admission.lease
        self._boot_id = self._admission.boot_id
        self._bindings: dict[tuple[str, str], _ChannelBindingState] = {}
        self._declarations: dict[str, ChannelDefinition] = {}
        self._durable_reservation_owners: dict[str, tuple[str, str]] = {}
        self._binding_leases: set[ChannelBindingLease] = set()
        self._startup_snapshot_leases: dict[str, RuntimeSnapshotLease] = {}
        self._sealed = False
        self._opened = asyncio.Event()

    def seal(self) -> None:
        """封存本 Root 贡献；不向 Core 复制目录。"""
        self._sealed = True

    async def register(self, ctx: Context, definition: ChannelDefinition) -> None:
        """连接、factory 和停止回执由贡献 Context 持有。"""
        if ctx.root_instance_token is not self._root_token or ctx.require(CHANNELS) is not self:
            raise CompositionError("CHANNEL_SERVICE_ROOT_MISMATCH", "Channel provider 不属于当前 Root")
        if self._sealed:
            raise CompositionError("PLUGIN_CHANNELS_FROZEN", "Channel 贡献已封存")
        if not isinstance(definition, ChannelDefinition):
            raise TypeError("Channel 贡献必须是 ChannelDefinition")
        if definition.name in self._declarations:
            raise CompositionError("DUPLICATE_PLUGIN_CHANNEL", definition.name)
        if ChannelCapability.INBOUND in definition.capabilities:
            ctx.require(CHANNEL_INPUT)
        def declare() -> Callable[[], None]:
            self._declarations[definition.name] = definition

            def remove() -> None:
                del self._declarations[definition.name]

            return remove

        await ctx.effect(declare, label=f"channel-definition:{definition.name}")
        key: tuple[str, str] | None = None

        def close() -> None:
            if key is not None:
                self._close_admission(key)

        def open() -> None:
            if key is None:
                raise RuntimeError("Channel 必须 closed 启动 ready 后才能开放")
            self._open_admission(key)
            self._opened.set()

        await self._admission.watch(ctx, close=close, open=open)

        async def stop() -> None:
            nonlocal key
            if key is not None:
                await self._stop_binding_critical(key)
                del self._bindings[key]
                key = None

        await ctx.effect(lambda: stop, label=f"channel:{definition.name}")

        async def start(_event: RuntimeStarting) -> None:
            nonlocal key
            self._admission.require_starting(ctx)
            if key is not None:
                raise RuntimeError("同一 Channel Context 不允许重新启动旧连接")
            lease = get_current_runtime_lease()
            assert lease is not None
            key = (lease.snapshot.snapshot_id, definition.name)
            self._bindings[key] = _ChannelBindingState(
                snapshot_id=key[0], plugin_id=ctx.runtime.plugin_id,
                generation_id=ctx.runtime.generation_id, channel_name=definition.name,
                capabilities=tuple(definition.capabilities), inbound_identity=definition.inbound_identity,
                factory=definition.factory, adapter=None, binding_token=uuid.uuid4().hex,
                config=definition.config, factory_context=None, plugin_context=ctx,
                activation_token=ctx.fiber.activation_token,
            )
            self._startup_snapshot_leases[key[0]] = lease
            try:
                await self._start_binding(key)
            finally:
                self._startup_snapshot_leases.pop(key[0], None)

        await ctx.on(RUNTIME_STARTING, start)

    async def start_recovery(self) -> None:
        """提交开放后才恢复 pending 输入，任务归当前 provider Scope。"""
        if not any(ChannelCapability.DURABLE_INBOUND in item.capabilities for item in self._declarations.values()):
            return
        # _opened 是一次性事件：每次启动都等本次提交的开放，不能复用上一代已置位的事件。
        self._opened = asyncio.Event()

        async def recover() -> None:
            await self._opened.wait()
            await self._input_custody.recover_durable_inbounds()

        await self._context.spawn(recover(), name="channel-pending-inputs")

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
        root = snapshot.composition_root
        context = state.plugin_context
        if (root is None or root.instance_token is not self._root_token
                or root.context.require(CHANNELS) is not self
                or context is None or root.context_owner(context) != state.plugin_id
                or context.fiber.activation_token is not state.activation_token):
            raise RuntimeError("Channel binding 不属于 exact Root/provider/贡献 Context")
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

    async def _spawn_owned[T](
        self, key: tuple[str, str], coroutine: Coroutine[Any, Any, T], *, name: str,
    ) -> asyncio.Task[T]:
        """启动期登记监听任务；请求或遗留 factory 引用不能再创建任务。"""

        state = self._bindings.get(key)
        if state is None or state.plugin_context is None or state.start_task is not asyncio.current_task():
            coroutine.close()
            raise RuntimeError("channel spawn_owned 只允许 adapter.start 当前任务调用")
        # binding 的关闭 Effect 已在 factory 前登记，先让 adapter 正常停止监听。
        # 不能再登记更晚的 task Effect，否则 Scope 会先取消监听再调用 adapter.stop。
        async def run() -> T:
            with _channel_entrypoint(state, "channel.listener"):
                return await coroutine

        task = asyncio.create_task(run(), name=name)
        state.listeners.add(task)

        def finished(result: asyncio.Task[T]) -> None:
            if not result.cancelled() and result.exception() is not None:
                state.plugin_context.report_incident("CHANNEL_LISTENER_FAILED", str(result.exception()))

        task.add_done_callback(finished)
        return task

    @asynccontextmanager
    async def _open_request_scope(self, key: tuple[str, str]) -> AsyncIterator[RequestContext]:
        """让外部请求占有精确 binding，关闭接纳后排空再释放插件。"""

        # 1. 首个 await 前占位，防止关闭接纳与取得 snapshot 之间漏过排空。
        state = self._binding(key)
        context = state.plugin_context
        if context is None:
            raise RuntimeError("channel 没有插件声明 Context")
        # ``channel.start`` runs before public admission opens.  A channel
        # may still need one exact scope during startup to validate and bind
        # its declared providers.  Only the task currently starting this
        # binding receives that closed-admission exception; later requests
        # continue to require an open binding.
        state = self._binding(key)
        allow_start_scope = (
            state.start_task is asyncio.current_task()
            and not state.started
            and not state.stopping
        )
        self._begin_presentation_operation(key, allow_closed=allow_start_scope)
        binding: ChannelBindingLease | None = None
        try:
            binding = await self._acquire_control_binding(key)
            root = binding.snapshot_lease.snapshot.composition_root
            if (
                root is None
                or root.context_owner(context) != state.plugin_id
                or context.fiber.state is not FiberState.ACTIVE
                or context.fiber.activation_token is not state.activation_token
            ):
                raise RuntimeError("channel 请求 Context 不属于当前 activation")
            # 2. 原 Context 的 get 允许 Root 查询；请求只解析声明 Fiber 的依赖。
            allowed = frozenset(context._declared_dependencies())
            runtime = context.runtime
            active = True

            def resolve(key: ServiceKey[object]) -> object:
                from agent.plugins.snapshot import get_current_runtime_lease

                current = get_current_runtime_lease()
                if not active or current is not scope_lease:
                    raise CompositionError("REQUEST_SCOPE_MISSING", "插件请求作用域已关闭")
                if context.fiber.activation_token is not state.activation_token:
                    raise CompositionError("REQUEST_SCOPE_MISSING", "请求声明 activation 已失效")
                if key not in allowed:
                    raise CompositionError("SERVICE_UNDECLARED", f"请求未声明能力: {key.name}")
                return context.require(key)

            request = RequestContext(
                plugin_id=runtime.plugin_id,
                plugin_dir=runtime.plugin_dir,
                data_root=runtime.data_dir,
                validation=self._admission.validation,
                _workspace_roots=tuple((name, runtime.workspace_root(name)) for name in runtime.workspace_roots),
                _workspace_files=tuple((name, runtime.workspace_file(name)) for name in runtime.workspace_files),
                _resolve=resolve,
            )
            scope_lease = binding.snapshot_lease.fork()
            try:
                async with RuntimeScope(scope_lease):
                    yield request
            finally:
                active = False
        finally:
            # 3. 取消也必须等 lease 清理完成，才能解除 Host 的排空占位。
            try:
                if binding is not None:
                    cleanup = asyncio.create_task(binding.aclose(), name="channel-request-release")
                    await _await_task_after_cancellation(cleanup)
            finally:
                self._release_presentation_operation(key)

    async def _acquire_control_binding(
        self,
        key: tuple[str, str],
    ) -> ChannelBindingLease:
        """Fork an exact snapshot lease for one control effect."""

        state = self._binding(key)
        startup_lease = self._startup_snapshot_leases.get(state.snapshot_id)
        if startup_lease is not None and state.start_task is asyncio.current_task():
            source = startup_lease.fork()
        else:
            acquirer = self._snapshot_lease_acquirer
            if acquirer is None:
                raise RuntimeError("Channel control exact snapshot lease owner 未绑定")
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
        interrupter = self._declarations[self._binding(key).channel_name].interrupt
        if interrupter is None:
            raise RuntimeError("Channel control interrupt owner 未绑定")
        async with RuntimeScope(binding.snapshot_lease.fork()):
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
            result = binding.deliver(envelope)
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
                DeliveryStatus.FAILED,
                error="control response cancelled",
            )
        except Exception as error:
            return ChannelDeliveryReceipt(
                delivery_id,
                DeliveryStatus.FAILED,
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
                f"presentation 已因发送失败终止，禁止继续 patch: {presentation_id}"
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

    def _failed_presentation_receipt(
        self,
        event: TurnStreamEvent,
        error: str,
    ) -> PresentationReceipt:
        return PresentationReceipt(
            presentation_id=event.presentation_id,
            status=DeliveryStatus.FAILED,
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

    async def recover_inbound(self, raw: RawInbound) -> bool:
        """Route a persisted handoff to the one current exact channel binding."""

        if not isinstance(raw, RawInbound):
            raise TypeError("durable recovery 只接受 RawInbound")
        candidates = tuple(
            key
            for key, state in self._bindings.items()
            if (
                state.channel_name == raw.message.channel
                and ChannelCapability.INBOUND in state.capabilities
                and ChannelCapability.DURABLE_INBOUND in state.capabilities
                and state.inbound_identity is InboundIdentity.PROVIDER_MESSAGE_ID
                and state.admission_open
                and not state.stopping
                and not state.stopped
            )
        )
        if not candidates:
            # A different channel may be restored by a later generation.  The
            # Bus leaves this row pending when the current catalog has no exact
            # owner; malformed identity/session failures still fail in
            # _recover_inbound after an owner is selected.
            return False
        if len(candidates) != 1:
            raise RuntimeError(
                f"durable inbound channel binding 不唯一: {raw.message.channel}"
            )
        return await self._recover_inbound(
            candidates[0],
            raw,
            _use_recovery_snapshot_lease=True,
        )

    async def _recover_inbound(
        self,
        key: tuple[str, str],
        raw: RawInbound,
        *,
        _use_recovery_snapshot_lease: bool = False,
    ) -> bool:
        """Replace only a prior accepted claim for one durable recovery."""

        if not isinstance(raw, RawInbound):
            raise TypeError("Channel recovery 只接受 RawInbound")
        state = self._binding(key)
        if (
            ChannelCapability.DURABLE_INBOUND not in state.capabilities
            or ChannelCapability.INBOUND not in state.capabilities
            or state.inbound_identity is not InboundIdentity.PROVIDER_MESSAGE_ID
        ):
            raise RuntimeError("channel 未声明 durable inbound capability")
        if raw.message.channel != state.channel_name:
            raise RuntimeError("RawInbound channel 与 exact binding 不一致")
        _ = _ChannelDurableInbound._reservation_metadata(raw)
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
                _use_recovery_snapshot_lease=_use_recovery_snapshot_lease,
            )

        # 2. 进程重启时无内存 claim，由 current binding 新建正常 claim。
        return await self._admit_inbound(
            key,
            raw,
            _use_recovery_snapshot_lease=_use_recovery_snapshot_lease,
        )

    async def _admit_inbound(
        self,
        key: tuple[str, str],
        raw: RawInbound,
        *,
        _retained_claim: tuple[str, str] | None = None,
        _use_recovery_snapshot_lease: bool = False,
    ) -> bool:
        """在 exact Root 接纳 Input，再完成传输收束；没有回复队列。"""

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
        durable_marker = raw.message.metadata.get(DURABLE_INBOUND_MARKER) is True
        if durable_marker and ChannelCapability.DURABLE_INBOUND not in state.capabilities:
            raise RuntimeError("durable handoff 只属于声明 durable capability 的 binding")
        session_key = f"{state.channel_name}:{raw.message.chat_id}"
        if "session_key_override" in raw.message.metadata:
            override = raw.message.metadata["session_key_override"]
            if not (
                ChannelCapability.DURABLE_INBOUND in state.capabilities
                and durable_marker
                and isinstance(override, str) and override.strip()
            ):
                raise RuntimeError("Session override 只属于已验证的 durable handoff")
            session_key = override.strip()
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
        acquirer = (
            self._recovery_snapshot_lease_acquirer
            if _use_recovery_snapshot_lease
            else self._snapshot_lease_acquirer
        )
        custody = self._input_custody
        if acquirer is None or custody is None:
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
                session_key=session_key,
                snapshot_id=binding.snapshot_id,
                generation_id=binding.generation_id,
                binding_token=binding.binding_token,
                message=raw.message,
                lease=binding,
            )
            await custody.prepare_channel_input(envelope)

            async def commit_input() -> None:
                nonlocal accepted
                # 显式传入原 binding 的 lease；热更新不能让输入落到后来发布的 Root。
                assert binding is not None and envelope is not None
                lease = binding.snapshot_lease.fork()
                async with RuntimeScope(lease):
                    root = lease.snapshot.composition_root
                    if root is None:
                        raise RuntimeError("Channel input 缺少 composition Root")
                    accept = root.context.require(CHANNEL_INPUT)
                    _ = await accept(session_key, raw.message_id, raw.message)
                    accepted = True
                    # 此后失败只能保留 cleanup/recovery，不能回滚身份或去重记录。
                    await custody.complete_channel_input(envelope)
                    handoff_id = raw.message.metadata.get(DURABLE_HANDOFF_ID)
                    if isinstance(handoff_id, str):
                        self._forget_durable_reservation(key, handoff_id)

            commit = asyncio.create_task(commit_input(), name=f"channel-input:{raw.message_id}")
            await _await_task_after_cancellation(commit)
            while len(state.inbound_message_ids) > 500:
                expired = state.inbound_message_ids.popleft()
                state.inbound_message_id_set.remove(expired)
            return True
        except BaseException as error:
            cleanup_errors: list[BaseException] = []
            if not accepted:
                if envelope is not None:
                    close_task = asyncio.create_task(
                        custody.retain_channel_input(envelope),
                        name=f"channel-input-retain:{state.channel_name}",
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

    async def _start_binding(self, key: tuple[str, str]) -> None:
        state = self._binding(key)
        factory = state.factory
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
            boot_id=self._boot_id,
            binding_token=state.binding_token,
            config=state.config,
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
            data_root=None if state.plugin_context is None else state.plugin_context.data_root,
            open_scope=None if state.plugin_context is None else lambda: self._open_request_scope(key),
            spawn_owned=None if state.plugin_context is None else lambda coroutine, *, name: self._spawn_owned(key, coroutine, name=name),
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
                        durable_inbound=(
                            _ChannelDurableInbound(self, key)
                            if ChannelCapability.DURABLE_INBOUND in state.capabilities
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
        state.start_task = asyncio.current_task()
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
        finally:
            state.start_task = None
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
        for subscription in tuple(state.subscriptions.values()):
            subscription.close_admission()
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
            # The adapter still owns its accepted tasks until stop() returns.
            # Those tasks may need the exact port to defer or reject a handoff
            # in their finally blocks.  Transfer only the reservations left
            # after a successful adapter stop; a failed stop keeps the old
            # binding as the cleanup owner and therefore cannot be reclaimed.
            if state.adapter_stop_succeeded:
                for listener in state.listeners:
                    if not listener.done():
                        listener.cancel()
                if state.listeners:
                    await asyncio.gather(*state.listeners, return_exceptions=True)
                    state.listeners.clear()
                failures.extend(await self._defer_durable_reservations(key, state))
            if failures:
                error = RuntimeError("channel cleanup failed: " + "; ".join(item.message for item in failures))
                assert state.plugin_context is not None
                state.plugin_context.report_incident("CHANNEL_STOP_FAILED", str(error))
                raise error
            if receipt is None:
                receipt = StopReceipt(binding_token=state.binding_token, resources_closed=True)
            state.stop_receipt = receipt
            state.stopped = True
            state.stopping = False
            return receipt
        except asyncio.CancelledError:
            state.stopping = True
            raise

    async def _stop_binding_critical(self, key: tuple[str, str]) -> StopReceipt:
        """Finish one binding cleanup before restoring caller cancellation."""

        state = self._binding(key)
        if state.stop_task is None or state.stop_task.done():
            state.stop_task = asyncio.create_task(
                self._stop_binding(key), name=f"channel-binding-stop:{key[0]}:{key[1]}",
            )
        return cast(StopReceipt, await _await_task_after_cancellation(state.stop_task))

    def _binding(self, key: tuple[str, str]) -> _ChannelBindingState:
        state = self._bindings.get(key)
        if state is None:
            raise KeyError(key[1])
        return state

    def _remember_durable_reservation(
        self,
        key: tuple[str, str],
        reservation: _DurableReservation,
    ) -> None:
        owner = self._durable_reservation_owners.get(reservation.handoff_id)
        if owner is not None and owner != key:
            raise RuntimeError("durable inbound reservation 已由另一 binding 持有")
        self._durable_reservation_owners[reservation.handoff_id] = key
        self._binding(key).durable_reservations[reservation.handoff_id] = reservation

    def _forget_durable_reservation(
        self,
        key: tuple[str, str],
        handoff_id: str,
    ) -> None:
        owner = self._durable_reservation_owners.get(handoff_id)
        if owner == key:
            self._durable_reservation_owners.pop(handoff_id, None)
        self._binding(key).durable_reservations.pop(handoff_id, None)

    async def _defer_durable_reservations(
        self,
        key: tuple[str, str],
        state: _ChannelBindingState,
    ) -> tuple[ChannelCleanupFailure, ...]:
        """转交 stop 前仍未接纳的 reservation，并清除旧 binding 权限。"""

        custody = self._input_custody
        if custody is None:
            return ()
        failures: list[ChannelCleanupFailure] = []
        for handoff_id in tuple(state.durable_reservations):
            try:
                released = await custody.defer_durable_inbound(handoff_id)
                if not released:
                    failures.append(
                        _cleanup_failure(
                            state,
                            "durable-inbound-reservation",
                            f"handoff {handoff_id} 仍有执行 owner",
                        )
                    )
                    continue
                self._forget_durable_reservation(key, handoff_id)
            except BaseException as error:
                failures.append(
                    _cleanup_failure(
                        state,
                        "durable-inbound-reservation",
                        str(error),
                        error,
                    )
                )
        return tuple(failures)


def _validate_attachment_read_lease(lease: object, ref: AttachmentRef) -> None:
    """附件边界确认回执及释放方法后才交接排空占位。"""
    if not callable(getattr(lease, "read_bytes", None)):
        raise TypeError("attachment read lease 必须提供 read_bytes(max_bytes=...)")
    if not callable(getattr(lease, "read_chunk", None)):
        raise TypeError(
            "attachment read lease 必须提供 read_chunk(offset=..., max_bytes=...)"
        )
    if not callable(getattr(lease, "aclose", None)):
        raise TypeError("attachment read lease 必须提供 aclose()")
    if getattr(lease, "ref", None) != ref:
        raise RuntimeError("attachment read lease ref 不匹配")


async def _invoke_attachment_lease_close(lease: AttachmentReadLease) -> None:
    result = lease.aclose()
    if not inspect.isawaitable(result):
        raise TypeError("attachment read lease aclose 必须返回 awaitable")
    await result


def _validate_adapter(adapter: object, channel_name: str) -> None:
    if any(not callable(getattr(adapter, name, None)) for name in ("start", "deliver", "stop")):
        raise TypeError(f"channel adapter ABI 无效: {channel_name}")


async def _invoke_async(adapter: object, method_name: str, *args: object) -> object:
    result = getattr(adapter, method_name)(*args)
    if not inspect.isawaitable(result):
        raise TypeError(f"channel adapter.{method_name} 必须返回 awaitable")
    return await result


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
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


async def _await_reservation_after_cancellation(
    task: asyncio.Task[bool],
) -> tuple[bool, bool]:
    """Finish Bus reserve and report outer cancellation after owner handoff."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    return task.result(), cancelled


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


def _is_async_callback(callback: object) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )
