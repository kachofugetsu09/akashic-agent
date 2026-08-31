from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from agent.control.errors import (
    ControlAdmissionError,
    RuntimeClosedError,
    ThreadBusyError,
)
from agent.control.models import (
    TurnItemKind,
    TurnRecord,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from agent.control.runtime import ConversationRuntime, TurnHandle
from agent.looping.core import AgentLoop
from agent.plugin_composition.channels import (
    ChannelCommitRole,
    DeliveryStatus as ChannelDeliveryStatus,
    AttachmentRef,
    InboundEnvelope,
    InboundOwner,
    OutboundEnvelope,
    AttachmentReadLease,
    ChannelDeliveryReceipt,
    ChannelTerminalStatus,
)
from bus.events import (
    ChannelMessage,
    InboundMessage,
    OutboundMessage,
    TurnTerminalStatus,
)
from bus.queue import MessageBus
from bus.events import channel_message_from_outbound
from bootstrap.channel_attachment_import import import_channel_attachments
from core.common.diagnostic_log import turn_milestone

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from infra.channels.artifacts import ChannelAttachmentArtifactStore

T = TypeVar("T")

ChannelDeliveryDispatcher = Callable[
    [ChannelMessage, bool],
    Awaitable[ChannelDeliveryReceipt],
]


class _ModelAttachmentLease(AttachmentReadLease, Protocol):
    @property
    def model_path(self) -> str: ...


async def _complete_critical(awaitable: Awaitable[T]) -> T:
    """Finish attachment ownership cleanup before restoring caller cancellation."""

    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    result = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return result


_TERMINAL_LANE_RETRY_DELAY = 1.0


def _has_mobile_handoff(value: object) -> bool:
    """Identify the Mobile durable owner without coupling it to channel name."""

    if isinstance(value, InboundEnvelope):
        return value.metadata.get("mobile_v3_handoff") is True
    return isinstance(value, InboundMessage) and value.handoff_id is not None


class _TerminalHandoffRetainedError(RuntimeError):
    """终态尚未 durable delivery，handoff 仍由当前 lane 持有。"""


class PassiveMessageWorker:
    """把渠道入站消息转换为 ConversationRuntime turn。"""

    def __init__(
        self,
        bus: MessageBus,
        runtime: ConversationRuntime,
        legacy_loop: AgentLoop,
        *,
        attachment_store: ChannelAttachmentArtifactStore | None = None,
        channel_dispatcher: ChannelDeliveryDispatcher | None = None,
    ) -> None:
        self._bus = bus
        self._runtime = runtime
        self._legacy_loop = legacy_loop
        self._attachment_store = attachment_store
        self._channel_dispatcher = channel_dispatcher
        self._running = False
        self._lane_queues: dict[
            str,
            asyncio.Queue[InboundMessage | InboundEnvelope | object],
        ] = {}
        self._lane_tasks: dict[str, asyncio.Task[None]] = {}
        self._result_tasks: set[asyncio.Task[None]] = set()
        self._channel_result_tasks: dict[asyncio.Task[None], TurnHandle] = {}

    def bind_channel_dispatcher(
        self,
        dispatcher: ChannelDeliveryDispatcher,
    ) -> None:
        """Bind the committed Channel dispatcher used by legacy ingress projection."""

        if not callable(dispatcher):
            raise TypeError("passive Channel dispatcher 必须可调用")
        if self._channel_dispatcher is not None:
            raise RuntimeError("passive Channel dispatcher 已绑定")
        self._channel_dispatcher = dispatcher

    async def run(self) -> None:
        self._running = True
        try:
            # 1. 仅重放有界 durable handoff 页，lane 准入由 MessageBus 统一持有。
            await self._bus.recover_durable_inbounds()
            while self._running:
                try:
                    item = await asyncio.wait_for(
                        self._bus.consume_inbound(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                self._enqueue(item)
        finally:
            self._running = False
            for task in tuple(self._lane_tasks.values()):
                task.cancel()
            if self._lane_tasks:
                await asyncio.gather(
                    *tuple(self._lane_tasks.values()),
                    return_exceptions=True,
                )
            await self._drain_channel_lane_queues()
            self._lane_tasks.clear()
            self._lane_queues.clear()
            await self._interrupt_channel_results()
            for task in tuple(self._result_tasks - self._channel_result_tasks.keys()):
                task.cancel()
            if self._result_tasks:
                await asyncio.gather(*tuple(self._result_tasks), return_exceptions=True)
            self._result_tasks.clear()
            self._channel_result_tasks.clear()

    def _enqueue(self, item: object) -> None:
        key = cast(Any, item).session_key
        queue = self._lane_queues.setdefault(key, asyncio.Queue())
        queue.put_nowait(item)
        task = self._lane_tasks.get(key)
        if task is None or task.done():
            self._lane_tasks[key] = asyncio.create_task(
                self._run_lane(key, queue),
                name=f"passive-lane:{key}",
            )

    async def _run_lane(
        self,
        key: str,
        queue: asyncio.Queue[InboundMessage | InboundEnvelope | object],
    ) -> None:
        """串行执行单 thread 队列，并隔离单条消息失败。"""

        while True:
            item = await queue.get()
            while True:
                try:
                    if isinstance(item, InboundEnvelope):
                        result_task = await self._admit_channel_envelope(item)
                        await asyncio.shield(result_task)
                    elif isinstance(item, InboundMessage):
                        result_task = await self._admit_message(item)
                        if result_task is not None:
                            await result_task
                    else:
                        await self._legacy_loop._run_inbound_turn(cast(Any, item))
                    break
                except asyncio.CancelledError:
                    if (
                        isinstance(item, InboundEnvelope)
                        and item.owner is InboundOwner.LANE
                    ):
                        if _has_mobile_handoff(item):
                            await self._bus.retain_mobile_channel_inbound(
                                item,
                                InboundOwner.LANE,
                            )
                        else:
                            await self._bus.release_channel_inbound(
                                item,
                                InboundOwner.LANE,
                            )
                    raise
                except _TerminalHandoffRetainedError as error:
                    # 终态已持久化，只重投同一权威 terminal；同 session 后续消息
                    # 保持排队，绝不重跑 Provider，也不把 accepted owner 变成孤儿。
                    logger.error(
                        "passive terminal retained; retrying thread=%s delay=%.1fs error=%s",
                        key,
                        _TERMINAL_LANE_RETRY_DELAY,
                        error,
                    )
                    await asyncio.sleep(_TERMINAL_LANE_RETRY_DELAY)
                except Exception:
                    logger.exception("passive lane message failed thread=%s", key)
                    if (
                        isinstance(item, InboundEnvelope)
                        and item.owner is not InboundOwner.CLOSED
                    ):
                        if _has_mobile_handoff(item):
                            if not self._bus.mobile_inbound_cleanup_pending(item):
                                await self._bus.retain_mobile_channel_inbound(
                                    item,
                                    item.owner,
                                )
                        else:
                            await self._bus.release_channel_inbound(item, item.owner)
                    break
            if queue.empty():
                task = asyncio.current_task()
                if self._lane_tasks.get(key) is task:
                    self._lane_tasks.pop(key)
                    self._lane_queues.pop(key)
                return

    async def _run_message(self, item: InboundMessage) -> None:
        """兼容直接调用：等待由本条消息新建的 turn 完成。"""

        result_task = await self._admit_message(item)
        if result_task is not None:
            await result_task

    async def _admit_channel_envelope(
        self,
        envelope: InboundEnvelope,
    ) -> asyncio.Task[None]:
        """Transfer one exact Channel lease into a ConversationRuntime turn."""

        # 1. Lane owns the accepted envelope until the runtime handle exists.
        envelope.handoff(InboundOwner.LANE, InboundOwner.LOOP)
        transferred = False
        session_admission_id: str | None = None
        attachment_leases: tuple[_ModelAttachmentLease, ...] = ()
        try:
            message = envelope.message
            if _has_mobile_handoff(envelope):
                session_admission_id = self._bus.mobile_session_admission_id(envelope)
                store = self._legacy_loop.session_manager.control_store
                matched = store.find_turn_by_client_message_id(
                    envelope.session_key,
                    envelope.message_id,
                )
                if matched is not None:
                    if not matched.status.is_terminal:
                        raise RuntimeError(
                            f"mobile turn 非终态未恢复: "
                            f"{matched.id}/{matched.status.value}"
                        )
                    task = asyncio.create_task(
                        self._settle_channel_envelope(
                            envelope,
                            TurnResult.from_record(matched),
                            (),
                            session_admission_id,
                        ),
                        name=f"channel-passive-recovery:{matched.id}",
                    )
                    self._result_tasks.add(task)
                    task.add_done_callback(self._result_tasks.discard)
                    transferred = True
                    return task
            else:
                _, session_admission_id = (
                    self._legacy_loop.session_manager.admit_existing(
                        envelope.session_key
                    )
                )
            attachment_leases = await self._acquire_attachment_refs(message.attachments)
            request = TurnRequest(
                envelope.session_key,
                message.content,
                {
                    "channel": message.channel,
                    "chatId": message.chat_id,
                    "sender": message.sender,
                    "media": [],
                    "inputTimestamp": message.timestamp.isoformat(),
                    "inboundMetadata": {
                        **dict(message.metadata),
                        "client_message_id": envelope.message_id,
                        **(
                            {
                                "attachment_ids": [
                                    ref.artifact_id for ref in message.attachments
                                ]
                            }
                            if message.attachments
                            else {}
                        ),
                    },
                    "channelMessageId": envelope.message_id,
                    "channelSnapshotId": envelope.snapshot_id,
                    "channelGenerationId": envelope.generation_id,
                    "channelBindingToken": envelope.binding_token,
                },
            )
            retry_source = message.metadata.get("retry_of_client_message_id")
            if retry_source is not None and (
                not isinstance(retry_source, str) or not retry_source
            ):
                raise ValueError("retry_of_client_message_id 必须是非空字符串")
            fresh_after_failure = _has_mobile_handoff(envelope) and retry_source is None

            # 2. Capacity waits retain the exact old binding; they never reacquire current.
            while True:
                try:
                    handle = await self._runtime.start_turn(
                        request,
                        runtime_snapshot_lease=cast(
                            Any,
                            envelope.lease.snapshot_lease,
                        ),
                        channel_binding_lease=cast(Any, envelope.lease),
                        live_media=tuple(
                            lease.model_path for lease in attachment_leases
                        ),
                        retry_source_client_message_id=cast(
                            str | None,
                            retry_source,
                        ),
                        fresh_interaction_after_failure=fresh_after_failure,
                    )
                    break
                except ThreadBusyError:
                    await self._runtime.wait_thread_available(envelope.session_key)
                except ControlAdmissionError:
                    if self._runtime.admission_request_never_fits(
                        request,
                        fresh_interaction_after_failure=fresh_after_failure,
                        retry_source_client_message_id=cast(str | None, retry_source),
                    ):
                        handle = await self._runtime.reject_never_fit_turn(
                            request,
                            fresh_interaction_after_failure=fresh_after_failure,
                            retry_source_client_message_id=cast(
                                str | None, retry_source
                            ),
                        )
                        break
                    await self._runtime.wait_capacity_available(
                        request,
                        fresh_interaction_after_failure=fresh_after_failure,
                        retry_source_client_message_id=cast(str | None, retry_source),
                    )
                except RuntimeClosedError:
                    await self._runtime.wait_until_accepting_turns()

            # 3. The result task owns delivery and terminal lease release.
            task = asyncio.create_task(
                self._finish_channel_envelope(
                    envelope,
                    handle,
                    attachment_leases,
                    session_admission_id,
                ),
                name=f"channel-passive-result:{handle.id}",
            )
            self._result_tasks.add(task)
            self._channel_result_tasks[task] = handle

            def forget(completed: asyncio.Task[None]) -> None:
                self._result_tasks.discard(completed)
                self._channel_result_tasks.pop(completed, None)

            task.add_done_callback(forget)
            transferred = True
            return task
        finally:
            if not transferred:
                try:
                    await self._close_attachment_leases(attachment_leases)
                finally:
                    try:
                        if (
                            session_admission_id is not None
                            and not _has_mobile_handoff(envelope)
                        ):
                            self._legacy_loop.session_manager.release_admission(
                                session_admission_id
                            )
                    finally:
                        if _has_mobile_handoff(envelope):
                            await self._bus.retain_mobile_channel_inbound(
                                envelope,
                                InboundOwner.LOOP,
                            )
                        else:
                            await self._bus.release_channel_inbound(
                                envelope, InboundOwner.LOOP
                            )

    async def _finish_channel_envelope(
        self,
        envelope: InboundEnvelope,
        handle: TurnHandle,
        attachment_leases: tuple[_ModelAttachmentLease, ...],
        session_admission_id: str,
    ) -> None:
        """Await the turn and settle one exact non-retryable provider delivery."""

        result = await handle.result()
        await self._settle_channel_envelope(
            envelope,
            result,
            attachment_leases,
            session_admission_id,
        )

    async def _settle_channel_envelope(
        self,
        envelope: InboundEnvelope,
        result: TurnResult,
        attachment_leases: tuple[_ModelAttachmentLease, ...],
        session_admission_id: str,
    ) -> None:
        """Deliver one authoritative terminal and settle its exact inbound owner."""

        terminal_durable = not _has_mobile_handoff(envelope)
        try:
            legacy_view = InboundMessage(
                channel=envelope.channel,
                sender=envelope.sender,
                chat_id=envelope.chat_id,
                content=envelope.content,
                timestamp=envelope.timestamp,
                metadata=dict(envelope.metadata),
            )
            terminal = self._terminal_outbound(legacy_view, result)
            terminal_message = channel_message_from_outbound(terminal)
            attachment_store = self._attachment_store
            if terminal_message.attachments and terminal_message.attachment_refs:
                raise RuntimeError("Channel terminal 不得同时携带 path 与 opaque refs")
            if terminal_message.attachments and attachment_store is None:
                raise RuntimeError("PassiveWorker Channel attachment store 尚未绑定")
            attachment_refs = terminal_message.attachment_refs
            if terminal_message.attachments:
                attachment_refs = await import_channel_attachments(
                    cast("ChannelAttachmentArtifactStore", attachment_store),
                    terminal_message.attachments,
                )
            delivery_id = result.id
            receipt = await self._bus.publish_channel_outbound_awaited(
                OutboundEnvelope(
                    logical_delivery_id=delivery_id,
                    delivery_id=delivery_id,
                    attempt_sequence=1,
                    snapshot_id=envelope.snapshot_id,
                    generation_id=envelope.generation_id,
                    binding_token=envelope.binding_token,
                    channel=envelope.channel,
                    recipient=envelope.chat_id,
                    body=terminal_message.content,
                    metadata=cast(Any, terminal_message.metadata),
                    attachments=attachment_refs,
                    commit_role=ChannelCommitRole.PASSIVE,
                    thinking=terminal_message.thinking,
                    reply_to=terminal_message.reply_to,
                    session_message_id=terminal_message.session_message_id,
                    control_turn_id=terminal_message.control_turn_id,
                    execution_attempt_id=terminal_message.execution_attempt_id,
                    terminal_status=(
                        ChannelTerminalStatus(terminal_message.terminal_status.value)
                        if terminal_message.terminal_status is not None
                        else None
                    ),
                ),
                envelope.lease,
            )
            if receipt.status is not ChannelDeliveryStatus.DELIVERED:
                logger.error(
                    "v3 channel terminal settled channel=%s delivery_id=%s status=%s error=%s",
                    envelope.channel,
                    receipt.delivery_id,
                    receipt.status.value,
                    receipt.error,
                )
            else:
                terminal_durable = True
        finally:
            try:
                await self._close_attachment_leases(attachment_leases)
            finally:
                try:
                    if terminal_durable:
                        await self._bus.complete_inbound(envelope)
                    elif _has_mobile_handoff(envelope):
                        await self._bus.retain_mobile_channel_inbound(
                            envelope,
                            InboundOwner.LOOP,
                        )
                    else:
                        await self._bus.release_channel_inbound(
                            envelope,
                            InboundOwner.LOOP,
                        )
                finally:
                    if not _has_mobile_handoff(envelope):
                        self._legacy_loop.session_manager.release_admission(
                            session_admission_id
                        )

    async def _drain_channel_lane_queues(self) -> None:
        """Close every v3 envelope still owned by a stopped lane."""

        for queue in self._lane_queues.values():
            retained: list[object] = []
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if isinstance(item, InboundEnvelope):
                    if _has_mobile_handoff(item):
                        await self._bus.retain_mobile_channel_inbound(
                            item,
                            InboundOwner.LANE,
                        )
                    else:
                        await self._bus.release_channel_inbound(
                            item,
                            InboundOwner.LANE,
                        )
                else:
                    retained.append(item)
            for item in retained:
                queue.put_nowait(item)

    async def _interrupt_channel_results(self) -> None:
        """Make every running v3 turn terminal before releasing its exact lease."""

        owned = tuple(self._channel_result_tasks.items())
        failures: list[Exception] = []
        for task, handle in owned:
            if task.done():
                continue
            try:
                await handle.interrupt()
            except Exception as error:
                failures.append(error)
        if owned:
            results = await asyncio.gather(
                *(task for task, _ in owned),
                return_exceptions=True,
            )
            failures.extend(
                result for result in results if isinstance(result, Exception)
            )
        if failures:
            raise ExceptionGroup("v3 Channel turn cleanup 失败", failures)

    async def _admit_message(
        self,
        item: InboundMessage,
    ) -> asyncio.Task[None] | None:
        """快速准入渠道消息，并把唯一终态发送职责交给新 turn owner。"""

        # 阶段1：mobile durable handoff 以权威 turn 为 owner，恢复绝不直接删 row。
        if _has_mobile_handoff(item) and item.handoff_id is not None:
            matched = self._matched_mobile_turn(item)
            if matched is not None:
                if matched.status.is_terminal:
                    await self._redeliver_terminal(item, matched)
                    return None
                raise RuntimeError(
                    f"mobile turn 非终态未恢复: {matched.id}/{matched.status.value}"
                )
        # 阶段2：mobile 消息需要 session admission；channel 已给出则本轮复用。
        if _has_mobile_handoff(item) and item.session_admission_id is None:
            _, item.session_admission_id = (
                self._legacy_loop.session_manager.admit_existing(item.session_key)
            )
        transferred = False
        attachment_leases: tuple[_ModelAttachmentLease, ...] = ()
        try:
            attachment_leases = await self._acquire_mobile_attachment_ids(item.metadata)
            # 阶段3：渠道信息只作为 executor 所需的受控 metadata，不改变 thread identity。
            request = TurnRequest(
                item.session_key,
                item.content,
                {
                    "channel": item.channel,
                    "chatId": item.chat_id,
                    "sender": item.sender,
                    "media": [] if attachment_leases else list(item.media),
                    "inputTimestamp": item.timestamp.isoformat(),
                    "inboundMetadata": dict(item.metadata),
                },
            )
            retry_source = item.metadata.get("retry_of_client_message_id")
            if retry_source is not None and (
                not isinstance(retry_source, str) or not retry_source
            ):
                raise ValueError("retry_of_client_message_id 必须是非空字符串")
            fresh_after_failure = _has_mobile_handoff(item) and retry_source is None
            while True:
                try:
                    handle = await self._runtime.start_turn(
                        request,
                        live_media=tuple(
                            lease.model_path for lease in attachment_leases
                        ),
                        retry_source_client_message_id=cast(
                            str | None,
                            retry_source,
                        ),
                        fresh_interaction_after_failure=fresh_after_failure,
                    )
                    break
                except ThreadBusyError:
                    await self._runtime.wait_thread_available(item.session_key)
                    continue
                except ControlAdmissionError:
                    if self._runtime.admission_request_never_fits(
                        request,
                        fresh_interaction_after_failure=fresh_after_failure,
                        retry_source_client_message_id=cast(str | None, retry_source),
                    ):
                        handle = await self._runtime.reject_never_fit_turn(
                            request,
                            fresh_interaction_after_failure=fresh_after_failure,
                            retry_source_client_message_id=cast(
                                str | None, retry_source
                            ),
                        )
                        break
                    self._release_admission_once(item)
                    await self._runtime.wait_capacity_available(
                        request,
                        fresh_interaction_after_failure=fresh_after_failure,
                        retry_source_client_message_id=cast(str | None, retry_source),
                    )
                    if _has_mobile_handoff(item) and item.session_admission_id is None:
                        _, item.session_admission_id = (
                            self._legacy_loop.session_manager.admit_existing(
                                item.session_key
                            )
                        )
                    continue
                except RuntimeClosedError:
                    # Restart 取消会在同一进程恢复 admission：当前 coroutine 继续
                    # 持有 accepted owner，等待栅栏后原地重试。完整 shutdown 会
                    # 取消 worker；durable row 留给下一进程恢复。
                    self._release_admission_once(item)
                    await self._runtime.wait_until_accepting_turns()
                    if _has_mobile_handoff(item) and item.session_admission_id is None:
                        _, item.session_admission_id = (
                            self._legacy_loop.session_manager.admit_existing(
                                item.session_key
                            )
                        )
                    continue
            task = asyncio.create_task(
                self._finish_message(item, handle, attachment_leases),
                name=f"passive-result:{handle.id}",
            )
            self._result_tasks.add(task)
            task.add_done_callback(self._result_tasks.discard)
            transferred = True
            return task
        finally:
            # 阶段4：turn owner 建立前只释放 admission，durable handoff 保留供恢复。
            if not transferred:
                try:
                    await self._close_attachment_leases(attachment_leases)
                finally:
                    self._release_admission_once(item)

    async def _finish_message(
        self,
        item: InboundMessage,
        handle: TurnHandle,
        attachment_leases: tuple[_ModelAttachmentLease, ...] = (),
    ) -> None:
        """等待新 turn 的唯一终态；mobile 只在 durable delivered 后完成 inbound。

        finally 兜底本轮 session admission：取消、receipt False 与异常路径都
        只释放一次；成功路径 _complete_message 已清空 identity，finally 不重复。
        """

        try:
            result = await handle.result()
            outbound = self._terminal_outbound(item, result)
            if _has_mobile_handoff(item) and item.handoff_id is not None:
                # 阶段1：terminal 实际送达收据是 handoff 删除的唯一授权。
                await self._commit_mobile_terminal(
                    item,
                    outbound,
                    turn_id=result.id,
                    client_message_id=self._verified_client_message_id(result),
                    terminal_status=result.status.value,
                    mode="live",
                )
            else:
                if not await self._deliver_terminal(outbound):
                    raise RuntimeError(
                        f"Channel terminal delivery 未完成: {item.channel}/{result.id}"
                    )
        finally:
            try:
                await self._close_attachment_leases(attachment_leases)
            finally:
                self._release_admission_once(item)

    async def _acquire_mobile_attachment_ids(
        self,
        metadata: dict[str, object],
    ) -> tuple[_ModelAttachmentLease, ...]:
        raw = metadata.get("attachment_ids")
        if raw is None:
            return ()
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item for item in raw
        ):
            raise ValueError("mobile attachment_ids 必须是非空字符串数组")
        store = self._attachment_store
        if store is None:
            raise RuntimeError("PassiveWorker attachment store 未绑定")
        refs = store.resolve_refs(tuple(cast(list[str], raw)))
        return await self._acquire_attachment_refs(refs)

    async def _acquire_attachment_refs(
        self,
        refs: tuple[AttachmentRef, ...],
    ) -> tuple[_ModelAttachmentLease, ...]:
        if not refs:
            return ()
        store = self._attachment_store
        if store is None:
            raise RuntimeError("PassiveWorker attachment store 未绑定")
        leases: list[_ModelAttachmentLease] = []
        try:
            for ref in refs:
                lease = await store.acquire(ref)
                leases.append(cast(_ModelAttachmentLease, lease))
        except BaseException:
            await self._close_attachment_leases(tuple(leases))
            raise
        return tuple(leases)

    async def _close_attachment_leases(
        self,
        leases: tuple[_ModelAttachmentLease, ...],
    ) -> None:
        if not leases:
            return

        async def close_all() -> None:
            results = await asyncio.gather(
                *(lease.aclose() for lease in reversed(leases)),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, Exception)]
            if failures:
                raise ExceptionGroup("attachment read lease cleanup 失败", failures)

        await _complete_critical(close_all())

    async def _commit_mobile_terminal(
        self,
        item: InboundMessage,
        outbound: OutboundMessage,
        *,
        turn_id: str,
        client_message_id: str,
        terminal_status: str,
        mode: str,
    ) -> None:
        """送达 Mobile 终态并完成 handoff，以唯一闭合 span 报告结果。"""

        started = time.monotonic()
        self._observe_terminal_milestone(
            "tl:worker.terminal.start",
            session_id=item.session_key,
            turn_id=turn_id,
            client_message_id=client_message_id,
            outcome=terminal_status,
            counts=f"mode={mode}",
        )
        try:
            # 1. 终态必须先提交到 Mobile durable inbox；失败时 handoff 保留。
            if not await self._deliver_terminal(outbound):
                raise _TerminalHandoffRetainedError(
                    f"mobile terminal delivery failed turn={turn_id} handoff retained"
                )
            # 2. 只有 durable inbox 与 handoff DELETE 都成功，span 才能记 done。
            await self._complete_message(item)
        except asyncio.CancelledError:
            self._observe_terminal_milestone(
                "tl:worker.terminal.cancelled",
                session_id=item.session_key,
                turn_id=turn_id,
                client_message_id=client_message_id,
                duration_ms=(time.monotonic() - started) * 1000,
                outcome="cancelled",
                counts=f"mode={mode}",
                level=logging.WARNING,
            )
            raise
        except Exception as error:
            self._observe_terminal_milestone(
                "tl:worker.terminal.error",
                session_id=item.session_key,
                turn_id=turn_id,
                client_message_id=client_message_id,
                duration_ms=(time.monotonic() - started) * 1000,
                outcome="error",
                counts=f"mode={mode} error_type={type(error).__name__}",
                level=logging.ERROR,
            )
            raise
        self._observe_terminal_milestone(
            "tl:worker.terminal.done",
            session_id=item.session_key,
            turn_id=turn_id,
            client_message_id=client_message_id,
            duration_ms=(time.monotonic() - started) * 1000,
            outcome="delivered",
            counts=f"mode={mode}",
        )

    async def _deliver_terminal(self, outbound: OutboundMessage) -> bool:
        """Project one legacy ingress terminal through the committed Channel dispatcher."""

        dispatcher = self._channel_dispatcher
        if dispatcher is None:
            raise RuntimeError("Passive terminal exact Channel dispatcher 未绑定")
        message = channel_message_from_outbound(outbound)
        receipt = await dispatcher(message, True)
        if not isinstance(receipt, ChannelDeliveryReceipt):
            raise TypeError(
                "Passive terminal Channel dispatcher 必须返回 ChannelDeliveryReceipt"
            )
        return receipt.status is ChannelDeliveryStatus.DELIVERED

    def _observe_terminal_milestone(
        self,
        event: str,
        *,
        session_id: str,
        turn_id: str,
        client_message_id: str,
        outcome: str = "",
        duration_ms: float | None = None,
        counts: str = "",
        level: int = logging.INFO,
    ) -> None:
        """打一条 worker terminal 观测里程碑；观测自身异常绝不覆盖业务异常。"""

        try:
            turn_milestone(
                logger,
                event,
                session_id=session_id,
                turn_id=turn_id,
                client_message_id=client_message_id,
                duration_ms=duration_ms,
                outcome=outcome,
                counts=counts,
                level=level,
            )
        except Exception as error:
            logger.error(
                "passive worker 观测失败（业务不中断）: event=%s turn=%s error=%s",
                event,
                turn_id,
                error,
            )

    def _verified_client_message_id(self, result: TurnResult) -> str:
        """从唯一 userMessage item 的已验证 metadata 取 client_message_id。

        阶段1：扫描全部 userMessage item 的 data.metadata.client_message_id；
        阶段2：多个不同非空值即内部身份冲突，fail-loud，绝不静默选中；
        阶段3：唯一非空值返回，缺失返回空串由调用方按 channel 语义处理。
        """

        retry_client_message_id = result.metadata.get("retryClientMessageId")
        if retry_client_message_id is not None:
            if (
                not isinstance(retry_client_message_id, str)
                or not retry_client_message_id
            ):
                raise ValueError("retryClientMessageId 必须是非空字符串")
            return retry_client_message_id
        values: set[str] = set()
        for entry in result.items:
            if entry.kind is not TurnItemKind.USER_MESSAGE:
                continue
            data = entry.data
            raw_metadata = data.get("metadata")
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"turn userMessage metadata 无效: {entry.id}")
            metadata = cast(dict[str, object], raw_metadata)
            raw = metadata.get("client_message_id")
            if raw is not None and not isinstance(raw, str):
                raise ValueError(
                    f"turn userMessage client_message_id 必须是字符串: {entry.id}"
                )
            if isinstance(raw, str) and raw:
                values.add(raw)
        if len(values) > 1:
            raise RuntimeError(
                f"同一 turn 存在多个 userMessage client_message_id: "
                f"{sorted(values)}"
            )
        return next(iter(values), "")

    async def _redeliver_terminal(
        self,
        item: InboundMessage,
        record: TurnRecord,
    ) -> None:
        """恢复路径：用权威 turn 终态投影重投递，durable delivered 后才 ACK。"""

        result = TurnResult.from_record(record)
        outbound = self._terminal_outbound(item, result)
        await self._commit_mobile_terminal(
            item,
            outbound,
            turn_id=record.id,
            client_message_id=self._verified_client_message_id(result),
            terminal_status=record.status.value,
            mode="recovery",
        )

    def _matched_mobile_turn(self, item: InboundMessage) -> TurnRecord | None:
        """以 turns.items_json 的 client_message_id 唯一匹配为恢复 owner。"""

        client_message_id = item.metadata.get("client_message_id")
        if not isinstance(client_message_id, str) or not client_message_id:
            return None
        return self._legacy_loop.session_manager.control_store.find_turn_by_client_message_id(
            item.session_key,
            client_message_id,
        )

    def _terminal_outbound(
        self,
        item: InboundMessage,
        result: TurnResult,
    ) -> OutboundMessage:
        """把权威 turn 终态投影为出站消息；正常与恢复分支共用同一投影。

        completed/failed/recovered 一律从已验证 userMessage item 贯通
        client_message_id 到 outbound metadata；缺失（mobile）或 userMessage
        多值冲突 fail-fast。assistant metadata 只是冗余投影，不拥有该身份。
        """

        verified_cmid = self._verified_client_message_id(result)
        if result.status is TurnStatus.COMPLETED:
            assistant = next(
                entry
                for entry in reversed(result.items)
                if entry.kind.value == "assistantMessage"
            )
            data = assistant.data
            metadata = dict(cast(dict[str, Any], data.get("metadata", {})))
            attachment_refs = self._terminal_attachment_refs(data)
            if verified_cmid:
                metadata["client_message_id"] = verified_cmid
            elif _has_mobile_handoff(item) and item.handoff_id is not None:
                # durable handoff 链必须在 channel/gateway 贯通身份，缺失即 fail-fast。
                raise RuntimeError(
                    f"mobile completed turn 缺少已验证 client_message_id: "
                    f"{result.id}"
                )
            return OutboundMessage(
                channel=item.channel,
                chat_id=item.chat_id,
                content=result.final_response or "",
                thinking=cast(str | None, data.get("thinking")),
                reply_to=cast(str | None, data.get("replyTo")),
                media=list(cast(list[str], data.get("media", []))),
                attachment_refs=attachment_refs,
                metadata=metadata,
                control_turn_id=result.interaction_id,
                execution_attempt_id=result.id,
                session_message_id=cast(str | None, data.get("sessionMessageId")),
                terminal_status=TurnTerminalStatus.COMPLETED,
            )
        metadata: dict[str, Any] = {}
        if verified_cmid:
            metadata["client_message_id"] = verified_cmid
        elif _has_mobile_handoff(item) and item.handoff_id is not None:
            raise RuntimeError(
                f"mobile terminal 缺少已验证 client_message_id: {result.id}"
            )
        terminal_status = TurnTerminalStatus(result.status.value)
        if result.status in (TurnStatus.INTERRUPTED, TurnStatus.CANCELLED):
            return OutboundMessage(
                channel=item.channel,
                chat_id=item.chat_id,
                content="本轮已中断。",
                control_turn_id=result.interaction_id,
                execution_attempt_id=result.id,
                metadata=metadata,
                terminal_status=terminal_status,
            )
        if result.status is not TurnStatus.FAILED:
            raise ValueError(f"不支持的 terminal 状态: {result.status.value}")
        content = (
            result.error.message
            if result.error is not None
            else "处理消息时出错，请稍后再试。"
        )
        if result.error is not None:
            metadata["retryable"] = result.error.retryable
        return OutboundMessage(
            channel=item.channel,
            chat_id=item.chat_id,
            content=content,
            control_turn_id=result.interaction_id,
            execution_attempt_id=result.id,
            metadata=metadata,
            terminal_status=terminal_status,
        )

    def _terminal_attachment_refs(
        self,
        assistant_data: dict[str, Any],
    ) -> tuple[AttachmentRef, ...]:
        """把 durable control-turn attachment identity 还原为终态投递引用。"""

        # 1. Control turns persist opaque identities, never live artifact paths.
        raw_ids = assistant_data.get("attachmentIds")
        if raw_ids is None:
            return ()
        if not isinstance(raw_ids, list) or not all(
            isinstance(item, str) and item for item in raw_ids
        ):
            raise ValueError("assistant attachmentIds 必须是非空字符串数组")
        artifact_ids = tuple(cast(list[str], raw_ids))
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("assistant attachmentIds 不得重复")

        # 2. The Core artifact owner must reconstruct the exact immutable refs.
        store = self._attachment_store
        if store is None:
            raise RuntimeError("PassiveWorker attachment store 尚未绑定")
        refs = store.resolve_refs(artifact_ids)
        if tuple(ref.artifact_id for ref in refs) != artifact_ids:
            raise RuntimeError("assistant attachmentIds 无法解析为 exact artifacts")
        return refs

    async def _complete_message(self, item: InboundMessage) -> None:
        """完成 durable inbound 确认，并恰一次释放 mobile session admission。"""

        try:
            await self._bus.complete_inbound(item)
        finally:
            self._release_admission_once(item)

    def _release_admission_once(self, item: InboundMessage) -> None:
        """恰一次释放本轮取得的 mobile session admission，不留重复释放身份。"""

        admission_id = item.session_admission_id
        if admission_id is None:
            return
        item.session_admission_id = None
        self._legacy_loop.session_manager.release_admission(admission_id)

    def stop(self) -> None:
        self._running = False
