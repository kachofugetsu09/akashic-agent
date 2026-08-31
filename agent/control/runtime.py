from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable, cast

from agent.control.errors import (
    ControlAdmissionError,
    ControlExecutionError,
    RuntimeClosedError,
    SlowConsumerError,
    ThreadBusyError,
    TurnAdmissionUncertainError,
    TurnNotFoundError,
)
from agent.control.events import TurnEvent
from agent.control.ids import new_item_id, new_turn_id
from agent.control.models import (
    TurnError,
    TurnItem,
    TurnItemKind,
    TurnRecord,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from agent.control.turn_scope import (
    TurnExecutionScope,
    bind_turn_scope,
    reset_turn_scope,
)
from agent.turn_effects import PostCommitEffect, set_post_commit_effect
from agent.control.replay_format import (
    METADATA_ATTEMPT_REPLAY,
    METADATA_PRIOR_TOOL_CHAIN,
    replay_messages,
)
from agent.control.ports import (
    ControlExecutionResult,
    TurnExecutor,
    InputLock,
    TurnUserInput,
)
from agent.restart import RestartCoordinator
from core.common.diagnostic_log import turn_milestone
from session.store import SessionStore
from agent.looping.interrupt import InterruptResult

if TYPE_CHECKING:
    from agent.plugins.channel_generation_host import ChannelBindingLease
    from agent.plugins.snapshot import RuntimeSnapshotLease

logger = logging.getLogger(__name__)
_STREAM_END = object()
StreamValue = TurnEvent | BaseException | object

DEFAULT_CONTROL_MAX_ACTIVE_TURNS = 16
DEFAULT_CONTROL_MAX_ACTIVE_BYTES = 32 * 1024 * 1024
DEFAULT_CONTROL_MAX_RUNTIME_OBJECTS = 16
DEFAULT_REPLAY_EVENTS_PER_TURN = 256
DEFAULT_REPLAY_BYTES_PER_TURN = 4 * 1024 * 1024
DEFAULT_REPLAY_BYTES_GLOBAL = 32 * 1024 * 1024
DEFAULT_TERMINAL_REPLAY_TTL_SECONDS = 5 * 60
_INTERACTION_ID = "interactionId"
_ATTEMPT_ORDINAL = "attemptOrdinal"
_CONTINUED_FROM_TURN_ID = "continuedFromTurnId"
_PRIOR_INPUT_COUNT = "priorInputCount"
_INTERACTION_REJECTED = "interactionRejected"
_FRESH_INTERACTION = "freshInteraction"
_SUPERSEDES_INTERACTION_ID = "supersedesInteractionId"
_RETRY_SOURCE_CLIENT_MESSAGE_ID = "retrySourceClientMessageId"
_RETRY_CLIENT_MESSAGE_ID = "retryClientMessageId"


@dataclass(frozen=True)
class _PreparedTurnRequest:
    effective_request: TurnRequest
    prior_inputs: list[TurnUserInput]
    replay_attempts: list[TurnRecord]
    retrying: bool


def _validate_turn_request_metadata(request: TurnRequest) -> None:
    """拒绝调用方伪造由 Control Runtime 独占的 turn metadata。"""

    reserved = (
        _INTERACTION_REJECTED,
        _FRESH_INTERACTION,
        _SUPERSEDES_INTERACTION_ID,
        _RETRY_SOURCE_CLIENT_MESSAGE_ID,
        _RETRY_CLIENT_MESSAGE_ID,
    )
    forged = next((field for field in reserved if field in request.metadata), None)
    if forged is not None:
        raise ValueError(f"turn metadata 的 {forged} 为 Runtime 保留字段")


def _encoded_turn_bytes(request: TurnRequest) -> int:
    """计算控制面 turn 请求的 UTF-8 编码字节数。"""

    return len(
        json.dumps(
            request.to_dict(), ensure_ascii=False, sort_keys=True, default=str
        ).encode()
    )


def _build_effective_turn_request(
    request: TurnRequest,
    *,
    turn_id: str,
    previous_attempts: list[TurnRecord],
    prior_inputs: list[TurnUserInput],
) -> TurnRequest:
    """计算 start_turn 实际计费与执行的 effective request（含续接元数据）。

    容量等待、永久超限判断与 start_turn 必须使用同一个 effective request
    投影，否则等待方看到的字节数会与真实保留容量漂移。
    """

    interaction_id = (
        ConversationRuntime._interaction_id(previous_attempts[-1])
        if previous_attempts
        else turn_id
    )
    metadata = {
        **request.metadata,
        _INTERACTION_ID: interaction_id,
        _ATTEMPT_ORDINAL: len(previous_attempts),
        _PRIOR_INPUT_COUNT: len(prior_inputs),
    }
    if previous_attempts:
        metadata[_CONTINUED_FROM_TURN_ID] = previous_attempts[-1].id
    return TurnRequest(request.thread_id, request.input, metadata)


def _persist_execution_scope(
    request: TurnRequest,
    scope: TurnExecutionScope | None,
) -> TurnRequest:
    """Project durable scope facts before the Runtime locks and stores the input."""

    if scope is None:
        return request
    raw_inbound = request.metadata.get("inboundMetadata", {})
    if not isinstance(raw_inbound, dict) or not all(
        isinstance(key, str) for key in raw_inbound
    ):
        raise ValueError("control inboundMetadata 必须是字符串键对象")
    inbound: dict[str, object] = dict(cast(dict[str, object], raw_inbound))
    if scope.disabled_prompt_sections:
        raw_disabled = inbound.get("disabled_prompt_sections", [])
        if not isinstance(raw_disabled, list) or not all(
            isinstance(section, str) and section for section in raw_disabled
        ):
            raise ValueError(
                "control inboundMetadata disabled_prompt_sections 必须是非空字符串数组"
            )
        disabled = cast(list[str], raw_disabled)
        inbound["disabled_prompt_sections"] = sorted(
            {*disabled, *scope.disabled_prompt_sections}
        )
    if scope.post_commit_effect is PostCommitEffect.SUPPRESS:
        set_post_commit_effect(inbound, PostCommitEffect.SUPPRESS)
    return TurnRequest(
        request.thread_id,
        request.input,
        {**request.metadata, "inboundMetadata": inbound},
    )


def _control_client_message_id(metadata: dict[str, object]) -> str:
    """读取 start_turn 已验证的 inboundMetadata 客户端消息标识。

    start_turn 边界已对 inboundMetadata 做完整结构校验（字符串键对象、
    client_message_id 非字符串即 fail-loud），这里只按已建立不变量读取，
    不再重复同一结构校验；普通非 mobile 入口没有该字段时返回 "missing"。
    """

    inbound_metadata = metadata.get("inboundMetadata")
    if not isinstance(inbound_metadata, dict):
        return "missing"
    value = inbound_metadata.get("client_message_id", "")
    if not isinstance(value, str) or not value:
        return "missing"
    return value


def _encoded_event_bytes(event: TurnEvent) -> int:
    """计算 replay event 的 UTF-8 编码字节数。"""

    return len(
        json.dumps(
            event.to_notification(), ensure_ascii=False, sort_keys=True, default=str
        ).encode()
    )


def _merge_turn_items(base: list[TurnItem], updates: list[TurnItem]) -> list[TurnItem]:
    """按 item identity 保留顺序并应用最新 checkpoint。"""

    merged = list(base)
    positions = {item.id: index for index, item in enumerate(merged)}
    if len(positions) != len(merged):
        raise RuntimeError("turn items 包含重复 identity")
    for item in updates:
        index = positions.get(item.id)
        if index is None:
            positions[item.id] = len(merged)
            merged.append(item)
        else:
            merged[index] = item
    return merged


class TurnHandle:
    """持有一个 turn 的结果、事件流和精确中断入口。"""

    def __init__(
        self, runtime: ConversationRuntime, thread_id: str, turn_id: str
    ) -> None:
        self._runtime = runtime
        self.thread_id = thread_id
        self.id = turn_id

    def record(self) -> dict[str, object]:
        return self._runtime.read_turn(self.thread_id, self.id).to_dict()

    async def result(self) -> TurnResult:
        return await self._runtime.wait_result(self.thread_id, self.id)

    def events(self, *, after_event: int | None = None) -> AsyncIterator[TurnEvent]:
        return self._runtime.subscribe(self.thread_id, self.id, after_event=after_event)

    async def interrupt(self) -> TurnRecord:
        return await self._runtime.interrupt_turn(self.thread_id, self.id)


class _RuntimeInputLock(InputLock):
    """把 reasoner 的最终 lock 交回 runtime 唯一 owner。"""

    def __init__(self, runtime: ConversationRuntime, turn_id: str) -> None:
        self._runtime = runtime
        self._turn_id = turn_id

    async def lock(self) -> None:
        await self._runtime._lock_turn_input(self._turn_id)

    def used_inputs(self) -> tuple[TurnUserInput, ...]:
        return self._runtime._used_turn_inputs(self._turn_id)


class ConversationRuntime:
    """拥有 turn 排队、执行、中断、事件和持久状态。"""

    def __init__(
        self,
        store: SessionStore,
        executor: TurnExecutor,
        *,
        subscriber_queue_size: int = 256,
        restart_coordinator: RestartCoordinator | None = None,
        max_active_turns: int = DEFAULT_CONTROL_MAX_ACTIVE_TURNS,
        max_active_bytes: int = DEFAULT_CONTROL_MAX_ACTIVE_BYTES,
        max_live_runtime_objects: int = DEFAULT_CONTROL_MAX_RUNTIME_OBJECTS,
        replay_events_per_turn: int = DEFAULT_REPLAY_EVENTS_PER_TURN,
        replay_bytes_per_turn: int = DEFAULT_REPLAY_BYTES_PER_TURN,
        replay_bytes_global: int = DEFAULT_REPLAY_BYTES_GLOBAL,
        terminal_replay_ttl_seconds: float = DEFAULT_TERMINAL_REPLAY_TTL_SECONDS,
        turn_terminal: (
            Callable[[str, TurnStatus, dict[str, object], tuple[object, ...]], None]
            | None
        ) = None,
    ) -> None:
        if subscriber_queue_size < 2:
            raise ValueError("subscriber_queue_size 必须至少为 2")
        for name, value in (
            ("max_active_turns", max_active_turns),
            ("max_active_bytes", max_active_bytes),
            ("max_live_runtime_objects", max_live_runtime_objects),
            ("replay_events_per_turn", replay_events_per_turn),
            ("replay_bytes_per_turn", replay_bytes_per_turn),
            ("replay_bytes_global", replay_bytes_global),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if terminal_replay_ttl_seconds <= 0:
            raise ValueError("terminal_replay_ttl_seconds 必须为正数")
        self._store = store
        self._executor = executor
        self._subscriber_queue_size = subscriber_queue_size
        self._control_admission_lock = asyncio.Lock()
        self._admission_capacity_event = asyncio.Event()
        self._max_active_turns = max_active_turns
        self._max_active_bytes = max_active_bytes
        self._max_live_runtime_objects = max_live_runtime_objects
        self._active_turn_bytes: dict[str, int] = {}
        self._active_admission_bytes = 0
        self._live_runtime_objects = 0
        self._replay_events_per_turn = replay_events_per_turn
        self._replay_bytes_per_turn = replay_bytes_per_turn
        self._replay_bytes_global = replay_bytes_global
        self._terminal_replay_ttl_seconds = terminal_replay_ttl_seconds
        self._turn_terminal = turn_terminal
        self._active_by_thread: dict[str, str] = {}
        self._consumed_inputs: dict[str, list[TurnUserInput]] = {}
        self._locked_turn_inputs: set[str] = set()
        self._turn_input_sources: dict[str, _RuntimeInputLock] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._results: dict[str, asyncio.Future[TurnResult]] = {}
        self._history: dict[str, list[TurnEvent]] = {}
        self._history_sequences: dict[str, list[int]] = {}
        self._next_event_sequence: dict[str, int] = {}
        self._history_truncated: set[str] = set()
        self._replay_order: OrderedDict[tuple[str, int], tuple[TurnEvent, int]] = (
            OrderedDict()
        )
        self._history_byte_totals: dict[str, int] = {}
        self._replay_bytes = 0
        self._terminal_replay_expiry: dict[str, float] = {}
        self._replay_reaper_task: asyncio.Task[None] | None = None
        self._replay_reaper_wakeup = asyncio.Event()
        self._replay_reaper_error: BaseException | None = None
        self._subscribers: dict[str, set[asyncio.Queue[StreamValue]]] = {}
        self._interrupt_requested: set[str] = set()
        self._thread_idle: dict[str, asyncio.Event] = {}
        self._closed = False
        self._accepting_turns = True
        self._restart_owner_turn_id: str | None = None
        self._restart_coordinator = restart_coordinator
        recovered = self._store.recover_in_progress_turns()
        if recovered:
            logger.warning(
                "Recovered %d stale control turns as terminal",
                len(recovered),
            )

    async def start_turn(
        self,
        request: TurnRequest,
        *,
        runtime_snapshot_lease: RuntimeSnapshotLease | None = None,
        channel_binding_lease: ChannelBindingLease | None = None,
        live_media: tuple[str, ...] = (),
        execution_scope: TurnExecutionScope | None = None,
        fresh_interaction: bool = False,
        fresh_interaction_after_failure: bool = False,
        retry_source_client_message_id: str | None = None,
    ) -> TurnHandle:
        """拒绝 active thread，并仅把本次进程可用的 media 交给 executor。"""

        # 1. 在唯一 owner 处检查 thread 与控制面容量；拒绝不写 SessionStore。
        _validate_turn_request_metadata(request)
        async with self._control_admission_lock:
            self._raise_replay_reaper_failure()
            if self._closed or not self._accepting_turns:
                raise RuntimeClosedError("conversation runtime is shutting down")
            if request.thread_id in self._active_by_thread:
                raise ThreadBusyError(f"thread 已有 active turn: {request.thread_id}")
            turn_id = new_turn_id()
            prepared = self._prepare_turn_request(
                request,
                turn_id=turn_id,
                fresh_interaction=fresh_interaction,
                fresh_interaction_after_failure=fresh_interaction_after_failure,
                retry_source_client_message_id=retry_source_client_message_id,
                execution_scope=execution_scope,
            )
            effective_request = prepared.effective_request
            prior_inputs = prepared.prior_inputs
            replay_attempts = prepared.replay_attempts
            retrying = prepared.retrying
            attempt_replay = replay_messages(
                replay_attempts,
                tool_group_from_item=ConversationRuntime._tool_group_from_item,
            )
            prior_tool_chain = self._attempt_tool_chain(replay_attempts)
            request_bytes = _encoded_turn_bytes(effective_request)
            admission_token = self._reserve_admission(request_bytes)

            # 2. 先持久化 queued handle；失败时只回滚本轮 admission token。
            try:
                initial_input = (
                    None
                    if retrying
                    else self._build_turn_user_input(
                        effective_request,
                        ordinal=len(prior_inputs),
                    )
                )
                user_item = (
                    None
                    if initial_input is None
                    else self._user_input_item(initial_input)
                )
                record = self._store.create_turn(
                    TurnRecord(
                        id=turn_id,
                        thread_id=request.thread_id,
                        status=TurnStatus.QUEUED,
                        input=effective_request.input,
                        metadata=dict(effective_request.metadata),
                        items=[] if user_item is None else [user_item],
                        usage=None,
                        error=None,
                        created_at=datetime.now(UTC),
                    )
                )
            except BaseException:
                self._release_admission(admission_token)
                raise
            self._commit_admission_token(admission_token, turn_id, request_bytes)
            self._active_by_thread[request.thread_id] = turn_id
            self._thread_idle[request.thread_id] = asyncio.Event()
            self._consumed_inputs[turn_id] = (
                list(prior_inputs)
                if initial_input is None
                else [*prior_inputs, initial_input]
            )
            source = _RuntimeInputLock(self, turn_id)
            self._turn_input_sources[turn_id] = source
            loop = asyncio.get_running_loop()
            self._results[turn_id] = loop.create_future()
            self._history[turn_id] = []
            self._history_sequences[turn_id] = []
            self._history_byte_totals[turn_id] = 0
            self._next_event_sequence[turn_id] = 0
            self._subscribers[turn_id] = set()
            handle = TurnHandle(self, request.thread_id, turn_id)
        try:
            self._publish(
                TurnEvent.create(
                    "turn/queued", request.thread_id, turn_id, turn=record.to_dict()
                )
            )
            if user_item is not None:
                self._publish_user_item(request.thread_id, turn_id, user_item)
            execution_request = (
                TurnRequest(
                    effective_request.thread_id,
                    effective_request.input,
                    {**effective_request.metadata, "media": list(live_media)},
                )
                if live_media
                else effective_request
            )
            task = asyncio.create_task(
                self._run(
                    execution_request,
                    turn_id,
                    attempt_replay=attempt_replay,
                    prior_tool_chain=prior_tool_chain,
                    runtime_snapshot_lease=runtime_snapshot_lease,
                    channel_binding_lease=channel_binding_lease,
                    execution_scope=execution_scope,
                ),
                name=f"conversation-turn:{turn_id}",
            )
        except BaseException as error:
            try:
                self._fail_queued_start(request.thread_id, turn_id, error)
            except BaseException as cleanup_error:
                raise TurnAdmissionUncertainError(
                    turn_id,
                    "turn 已持久化，且 start_turn cleanup 未完成",
                ) from cleanup_error
            raise TurnAdmissionUncertainError(
                turn_id,
                "turn 已持久化，但 start_turn 未返回 handle",
            ) from error
        self._tasks[turn_id] = task
        return handle

    def _fail_queued_start(
        self,
        thread_id: str,
        turn_id: str,
        error: BaseException,
    ) -> None:
        """Terminalize a persisted Turn whose execution task was not published."""

        terminal: TurnRecord | None = None
        try:
            current = self._store.read_turn(turn_id)
            if current is not None and current.status is TurnStatus.QUEUED:
                current = self._store.transition_turn(
                    turn_id,
                    expected_status=TurnStatus.QUEUED,
                    status=TurnStatus.IN_PROGRESS,
                    thread_id=thread_id,
                )
                terminal = self._store.transition_turn(
                    turn_id,
                    expected_status=current.status,
                    status=TurnStatus.FAILED,
                    thread_id=thread_id,
                    error=TurnError(
                        type=type(error).__name__,
                        message=str(error),
                        retryable=False,
                    ),
                )
            future = self._results.get(turn_id)
            if terminal is not None and future is not None and not future.done():
                future.set_result(TurnResult.from_record(terminal))
                self._finish_streams(turn_id)
        finally:
            self._release_turn_ownership(thread_id, turn_id)

    async def reject_never_fit_turn(
        self,
        request: TurnRequest,
        *,
        fresh_interaction: bool = False,
        fresh_interaction_after_failure: bool = False,
        retry_source_client_message_id: str | None = None,
    ) -> TurnHandle:
        """把永久超过单请求容量的输入持久化为可观察 failed turn。"""

        # 1. 在准入 owner 内复核关闭、thread 与永久超限不变量。
        _validate_turn_request_metadata(request)
        async with self._control_admission_lock:
            self._raise_replay_reaper_failure()
            if self._closed or not self._accepting_turns:
                raise RuntimeClosedError("conversation runtime is shutting down")
            if request.thread_id in self._active_by_thread:
                raise ThreadBusyError(f"thread 已有 active turn: {request.thread_id}")
            turn_id = new_turn_id()
            prepared = self._prepare_turn_request(
                request,
                turn_id=turn_id,
                fresh_interaction=fresh_interaction,
                fresh_interaction_after_failure=fresh_interaction_after_failure,
                retry_source_client_message_id=retry_source_client_message_id,
            )
            effective_request = prepared.effective_request
            effective_request = TurnRequest(
                effective_request.thread_id,
                effective_request.input,
                {**effective_request.metadata, _INTERACTION_REJECTED: True},
            )
            request_bytes = _encoded_turn_bytes(effective_request)
            if request_bytes <= self._max_active_bytes:
                raise RuntimeError("reject_never_fit_turn 仅接受永久超限请求")

            # 2. Runtime 持久化 queued → in_progress → failed；channel 不伪造终态。
            initial_input = (
                None
                if prepared.retrying
                else self._build_turn_user_input(
                    effective_request,
                    ordinal=len(prepared.prior_inputs),
                )
            )
            user_item = (
                None if initial_input is None else self._user_input_item(initial_input)
            )
            queued = self._store.create_turn(
                TurnRecord(
                    id=turn_id,
                    thread_id=request.thread_id,
                    status=TurnStatus.QUEUED,
                    input=effective_request.input,
                    metadata=dict(effective_request.metadata),
                    items=[] if user_item is None else [user_item],
                    usage=None,
                    error=None,
                    created_at=datetime.now(UTC),
                )
            )
            loop = asyncio.get_running_loop()
            future: asyncio.Future[TurnResult] = loop.create_future()
            self._results[turn_id] = future
            self._history[turn_id] = []
            self._history_sequences[turn_id] = []
            self._history_byte_totals[turn_id] = 0
            self._next_event_sequence[turn_id] = 0
            self._subscribers[turn_id] = set()
            self._publish(
                TurnEvent.create(
                    "turn/queued",
                    request.thread_id,
                    turn_id,
                    turn=queued.to_dict(),
                )
            )
            if user_item is not None:
                self._publish_user_item(request.thread_id, turn_id, user_item)
            started = self._store.transition_turn(
                turn_id,
                expected_status=TurnStatus.QUEUED,
                status=TurnStatus.IN_PROGRESS,
                thread_id=request.thread_id,
            )
            self._publish(
                TurnEvent.create(
                    "turn/started",
                    request.thread_id,
                    turn_id,
                    turn=started.to_dict(),
                )
            )
            terminal = self._store.transition_turn(
                turn_id,
                expected_status=TurnStatus.IN_PROGRESS,
                status=TurnStatus.FAILED,
                thread_id=request.thread_id,
                error=TurnError(
                    type="resource-exhausted",
                    message="消息超过单条容量上限，无法受理。",
                    retryable=False,
                ),
            )
            self._publish(
                TurnEvent.create(
                    "turn/completed",
                    request.thread_id,
                    turn_id,
                    turn=terminal.to_dict(),
                )
            )
            result = TurnResult.from_record(terminal)
            future.set_result(result)
            self._finish_streams(turn_id)

        # 3. 终态诊断与回调使用同一个持久 turn/client identity。
        client_message_id = _control_client_message_id(effective_request.metadata)
        turn_milestone(
            logger,
            "tl:turn.terminal",
            session_id=request.thread_id,
            turn_id=turn_id,
            client_message_id=client_message_id,
            counts="status=failed rejection=never_fit",
            outcome="failed",
        )
        if self._restart_coordinator is not None:
            self._restart_coordinator.mark_turn_terminal(turn_id, "failed")
        if self._turn_terminal is not None:
            self._turn_terminal(
                turn_id,
                TurnStatus.FAILED,
                {**effective_request.metadata, "turnId": turn_id},
                tuple(terminal.items),
            )
        return TurnHandle(self, request.thread_id, turn_id)

    def _open_interaction_attempts(
        self,
        thread_id: str,
        *,
        include_fresh: bool = False,
    ) -> list[TurnRecord]:
        """从最新未完成 attempt 沿显式前驱恢复 logical interaction。"""

        # 1. completed 与永久拒绝都关闭当前 logical interaction；普通失败和
        #    中断仍允许下一 attempt 沿显式前驱续接。
        latest_page = self._store.list_turns(thread_id, limit=1)
        if not latest_page or latest_page[0].status is TurnStatus.COMPLETED:
            return []
        fresh = latest_page[0].metadata.get(_FRESH_INTERACTION)
        if fresh is not None and not isinstance(fresh, bool):
            raise ValueError("freshInteraction 必须是布尔值")
        superseded = latest_page[0].metadata.get(_SUPERSEDES_INTERACTION_ID)
        if superseded is not None and (
            not isinstance(superseded, str) or not superseded
        ):
            raise ValueError("supersedesInteractionId 必须是非空字符串")
        if fresh is True and not include_fresh:
            return []
        rejected = latest_page[0].metadata.get(_INTERACTION_REJECTED)
        if rejected is not None and not isinstance(rejected, bool):
            raise ValueError("interactionRejected 必须是布尔值")
        if rejected is True:
            return []

        # 2. 新数据沿 continuedFromTurnId 精确回溯；旧数据作为单 attempt 兼容。
        attempts = [latest_page[0]]
        seen = {latest_page[0].id}
        while previous_id := attempts[-1].metadata.get(_CONTINUED_FROM_TURN_ID):
            if not isinstance(previous_id, str) or not previous_id:
                raise ValueError("continuedFromTurnId 必须是非空字符串")
            if previous_id in seen:
                raise RuntimeError(f"control attempt 前驱成环: {previous_id}")
            previous = self._store.read_turn(previous_id)
            if previous is None or previous.thread_id != thread_id:
                raise RuntimeError(
                    f"control attempt 前驱不存在或 thread 漂移: {previous_id}"
                )
            if previous.status is TurnStatus.COMPLETED:
                raise RuntimeError(
                    f"completed turn 不得成为未完成 interaction 前驱: {previous_id}"
                )
            attempts.append(previous)
            seen.add(previous_id)
        attempts.reverse()
        interaction_id = self._interaction_id(attempts[-1])
        if any(self._interaction_id(item) != interaction_id for item in attempts):
            raise RuntimeError(
                f"control attempt interaction identity 漂移: {interaction_id}"
            )
        return attempts

    @staticmethod
    def _interaction_id(record: TurnRecord) -> str:
        raw = record.metadata.get(_INTERACTION_ID, record.id)
        if not isinstance(raw, str) or not raw:
            raise ValueError("interactionId 必须是非空字符串")
        return raw

    @staticmethod
    def _attempt_user_inputs(attempts: list[TurnRecord]) -> list[TurnUserInput]:
        """按 logical ordinal 恢复此前 attempt 的所有用户输入。"""

        inputs: list[TurnUserInput] = []
        for attempt in attempts:
            for item in attempt.items:
                if item.kind is not TurnItemKind.USER_MESSAGE:
                    continue
                data = item.data
                ordinal = data.get("ordinal")
                content = data.get("content")
                media = data.get("media", [])
                metadata = data.get("metadata", {})
                timestamp = data.get("timestamp")
                if (
                    not isinstance(ordinal, int)
                    or isinstance(ordinal, bool)
                    or ordinal != len(inputs)
                ):
                    raise ValueError(
                        f"logical interaction user ordinal 不连续: {attempt.id}/{ordinal}"
                    )
                if not isinstance(content, str):
                    raise ValueError(f"turn user content 必须是字符串: {item.id}")
                if not isinstance(media, list) or not all(
                    isinstance(value, str) for value in media
                ):
                    raise ValueError(f"turn user media 必须是字符串数组: {item.id}")
                if not isinstance(metadata, dict) or not all(
                    isinstance(key, str) for key in metadata
                ):
                    raise ValueError(
                        f"turn user metadata 必须是字符串键对象: {item.id}"
                    )
                if not isinstance(timestamp, str):
                    raise ValueError(f"turn user timestamp 必须是字符串: {item.id}")
                parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    raise ValueError(f"turn user timestamp 必须包含时区: {item.id}")
                inputs.append(
                    TurnUserInput(
                        item_id=item.id,
                        ordinal=ordinal,
                        content=content,
                        media=tuple(media),
                        metadata=dict(cast(dict[str, object], metadata)),
                        timestamp=parsed.astimezone(UTC),
                    )
                )
        return inputs

    @staticmethod
    def _request_client_message_id(request: TurnRequest) -> str:
        """Read the current transport identity from the validated inbound metadata."""

        inbound = request.metadata.get("inboundMetadata", {})
        if not isinstance(inbound, dict):
            raise ValueError("control inboundMetadata 必须是对象")
        value = inbound.get("client_message_id")
        if not isinstance(value, str) or not value:
            raise ValueError("retry turn 缺少当前 client_message_id")
        return value

    @staticmethod
    def _prepare_retry(
        request: TurnRequest,
        attempts: list[TurnRecord],
        prior_inputs: list[TurnUserInput],
        source_client_message_id: str,
    ) -> tuple[list[TurnUserInput], list[TurnRecord], TurnRequest]:
        """Reuse the latest logical input while starting a new execution attempt."""

        if not attempts or not prior_inputs:
            raise ValueError("没有可重试的 logical turn")
        latest = attempts[-1]
        if (
            latest.status is not TurnStatus.FAILED
            or latest.error is None
            or latest.error.retryable is not True
        ):
            raise ValueError("只有最新的可重试 failed attempt 可以重试")

        source_index = next(
            (
                index
                for index in range(len(attempts) - 1, -1, -1)
                if any(
                    item.kind is TurnItemKind.USER_MESSAGE
                    for item in attempts[index].items
                )
            ),
            None,
        )
        if source_index is None:
            raise RuntimeError("failed interaction 缺少用户输入 owner")
        logical_input_id = prior_inputs[-1].metadata.get("client_message_id")
        if logical_input_id != source_client_message_id:
            raise ValueError("retry source 不是当前 failed attempt 的用户输入")

        inbound = request.metadata.get("inboundMetadata", {})
        if not isinstance(inbound, dict) or not all(
            isinstance(key, str) for key in inbound
        ):
            raise ValueError("control inboundMetadata 必须是字符串键对象")
        current_client_message_id = inbound.get("client_message_id")
        if (
            not isinstance(current_client_message_id, str)
            or not current_client_message_id
        ):
            raise ValueError("retry turn 缺少当前 client_message_id")

        source = prior_inputs[-1]
        effective_request = TurnRequest(
            request.thread_id,
            source.content,
            dict(request.metadata),
        )
        return prior_inputs, attempts[:source_index], effective_request

    @staticmethod
    def _tool_group_from_item(item: TurnItem) -> dict[str, Any] | None:
        """把一个已闭合 tool item 转换为标准 replay group。"""

        if item.kind is not TurnItemKind.TOOL_CALL:
            return None
        data = item.data
        status = data.get("status")
        result = data.get("resultPreview")
        if status in {"in_progress", "interrupted", "cancelled"}:
            return None
        call_id = data.get("callId")
        name = data.get("name")
        arguments = data.get("arguments", {})
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(f"completed tool callId 无效: {item.id}")
        if not isinstance(name, str) or not name:
            raise ValueError(f"completed tool name 无效: {item.id}")
        if not isinstance(arguments, dict):
            raise ValueError(f"completed tool arguments 无效: {item.id}")
        if not isinstance(result, str):
            raise ValueError(f"completed tool resultPreview 无效: {item.id}")
        return {
            "text": "",
            "calls": [
                {
                    "call_id": call_id,
                    "name": name,
                    "status": status,
                    "arguments": dict(arguments),
                    "final_arguments": dict(arguments),
                    "result": result,
                }
            ],
        }

    @staticmethod
    def _attempt_tool_chain(attempts: list[TurnRecord]) -> list[dict[str, Any]]:
        """把完成的工具 item 投影为正常 session replay 使用的 tool_chain。"""

        chain: list[dict[str, Any]] = []
        for attempt in attempts:
            for item in attempt.items:
                group = ConversationRuntime._tool_group_from_item(item)
                if group is not None:
                    chain.append(group)
        return chain

    def _build_turn_user_input(
        self,
        request: TurnRequest,
        *,
        ordinal: int,
    ) -> TurnUserInput:
        raw_timestamp = request.metadata.get("inputTimestamp")
        if isinstance(raw_timestamp, str):
            timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("control inputTimestamp 必须包含时区")
            timestamp = timestamp.astimezone(UTC)
        else:
            timestamp = datetime.now(UTC)
        media = request.metadata.get("media", [])
        if not isinstance(media, list) or not all(
            isinstance(item, str) for item in media
        ):
            raise ValueError("control metadata media 必须是字符串数组")
        inbound_metadata = request.metadata.get("inboundMetadata", {})
        if not isinstance(inbound_metadata, dict) or not all(
            isinstance(key, str) for key in inbound_metadata
        ):
            raise ValueError("control inboundMetadata 必须是字符串键对象")
        client_message_id = inbound_metadata.get("client_message_id", "")
        if not isinstance(client_message_id, str):
            raise ValueError("control inboundMetadata client_message_id 必须是字符串")
        return TurnUserInput(
            item_id=new_item_id(),
            ordinal=ordinal,
            content=request.input,
            media=tuple(media),
            metadata=dict(cast(dict[str, object], inbound_metadata)),
            timestamp=timestamp,
        )

    @staticmethod
    def _user_input_item(user_input: TurnUserInput) -> TurnItem:
        return TurnItem(
            TurnItemKind.USER_MESSAGE,
            user_input.item_id,
            {
                "content": user_input.content,
                "ordinal": user_input.ordinal,
                "media": list(user_input.media),
                "metadata": dict(user_input.metadata),
                "timestamp": user_input.timestamp.isoformat(),
            },
        )

    def _publish_user_item(self, thread_id: str, turn_id: str, item: TurnItem) -> None:
        self._publish(
            TurnEvent.create("item/started", thread_id, turn_id, item=item.to_dict())
        )
        self._publish(
            TurnEvent.create("item/completed", thread_id, turn_id, item=item.to_dict())
        )

    async def _lock_turn_input(self, turn_id: str) -> None:
        """在 admission lock 下原子封口，active attempt 不存在输入队列。"""

        async with self._control_admission_lock:
            if turn_id not in self._consumed_inputs:
                raise RuntimeError(f"turn input source 已释放: {turn_id}")
            self._locked_turn_inputs.add(turn_id)

    def _used_turn_inputs(self, turn_id: str) -> tuple[TurnUserInput, ...]:
        consumed = self._consumed_inputs.get(turn_id)
        if consumed is None:
            raise RuntimeError(f"turn input source 已释放: {turn_id}")
        return tuple(consumed)

    def _reserve_admission(self, request_bytes: int) -> str:
        """在控制准入锁内保留一个 queued/running turn 的容量 token。"""

        active_turns = len(self._active_turn_bytes)
        if not self._admission_can_fit(request_bytes):
            raise ControlAdmissionError(
                "resource-exhausted: control admission capacity busy "
                f"(turns={active_turns}/{self._max_active_turns}, "
                f"bytes={self._active_admission_bytes + request_bytes}/{self._max_active_bytes}, "
                f"runtime_objects={self._live_runtime_objects + 1}/{self._max_live_runtime_objects})"
            )
        # The turn id is not available until after the persistent record is built.
        token = f"__reserved__:{time.monotonic_ns()}:{active_turns}"
        self._active_turn_bytes[token] = request_bytes
        self._active_admission_bytes += request_bytes
        self._live_runtime_objects += 1
        return token

    def _commit_admission_token(
        self, token: str, turn_id: str, request_bytes: int
    ) -> None:
        if token not in self._active_turn_bytes:
            raise RuntimeError(f"control admission token missing for turn: {turn_id}")
        if self._active_turn_bytes.pop(token) != request_bytes:
            raise RuntimeError(f"control admission token bytes mismatch: {turn_id}")
        self._active_turn_bytes[turn_id] = request_bytes

    def _release_admission(self, turn_id: str) -> None:
        stored = self._active_turn_bytes.pop(turn_id, None)
        if stored is None:
            return
        self._active_admission_bytes -= stored
        self._live_runtime_objects -= 1
        self._admission_capacity_event.set()

    def _admission_can_fit(self, request_bytes: int) -> bool:
        """判断当前全局容量加上 request_bytes 后是否可再容纳一个新 turn。

        保留与等待共用同一个条件：字节数必须包含本次请求，否则
        active=900/max=1000/request=200 这类边界会被误判为可立即通过，
        导致调用方忙轮询 start_turn。
        """

        return (
            len(self._active_turn_bytes) < self._max_active_turns
            and self._active_admission_bytes + request_bytes <= self._max_active_bytes
            and self._live_runtime_objects < self._max_live_runtime_objects
        )

    def _prepare_turn_request(
        self,
        request: TurnRequest,
        *,
        turn_id: str,
        fresh_interaction: bool = False,
        fresh_interaction_after_failure: bool = False,
        retry_source_client_message_id: str | None = None,
        execution_scope: TurnExecutionScope | None = None,
    ) -> _PreparedTurnRequest:
        """Build the one effective request used by admission and execution."""

        recoverable_attempts = self._open_interaction_attempts(
            request.thread_id,
            include_fresh=(
                retry_source_client_message_id is not None
                or fresh_interaction_after_failure
            ),
        )
        if (
            fresh_interaction_after_failure
            and recoverable_attempts
            and recoverable_attempts[-1].status is TurnStatus.FAILED
        ):
            fresh_interaction = True
        if fresh_interaction and retry_source_client_message_id is not None:
            raise ValueError("fresh interaction 不能同时重试既有输入")
        previous_attempts = [] if fresh_interaction else recoverable_attempts
        prior_inputs = self._attempt_user_inputs(previous_attempts)
        retrying = retry_source_client_message_id is not None
        replay_attempts = previous_attempts
        effective_source = request
        if retrying:
            prior_inputs, replay_attempts, effective_source = self._prepare_retry(
                request,
                previous_attempts,
                prior_inputs,
                retry_source_client_message_id,
            )
        effective_request = _build_effective_turn_request(
            effective_source,
            turn_id=turn_id,
            previous_attempts=previous_attempts,
            prior_inputs=prior_inputs,
        )
        if retrying:
            effective_request = TurnRequest(
                effective_request.thread_id,
                effective_request.input,
                {
                    **effective_request.metadata,
                    _PRIOR_INPUT_COUNT: len(prior_inputs) - 1,
                    _RETRY_SOURCE_CLIENT_MESSAGE_ID: retry_source_client_message_id,
                    _RETRY_CLIENT_MESSAGE_ID: self._request_client_message_id(request),
                },
            )
        if fresh_interaction:
            metadata: dict[str, Any] = {
                **effective_request.metadata,
                _FRESH_INTERACTION: True,
            }
            if recoverable_attempts:
                metadata[_SUPERSEDES_INTERACTION_ID] = self._interaction_id(
                    recoverable_attempts[-1]
                )
            effective_request = TurnRequest(
                effective_request.thread_id,
                effective_request.input,
                metadata,
            )
        effective_request = _persist_execution_scope(effective_request, execution_scope)
        return _PreparedTurnRequest(
            effective_request=effective_request,
            prior_inputs=prior_inputs,
            replay_attempts=replay_attempts,
            retrying=retrying,
        )

    def _effective_request_bytes(
        self,
        request: TurnRequest,
        *,
        fresh_interaction: bool = False,
        fresh_interaction_after_failure: bool = False,
        retry_source_client_message_id: str | None = None,
    ) -> int:
        """计算 start_turn 会实际计费的 effective request 字节数。"""

        prepared = self._prepare_turn_request(
            request,
            turn_id=new_turn_id(),
            fresh_interaction=fresh_interaction,
            fresh_interaction_after_failure=fresh_interaction_after_failure,
            retry_source_client_message_id=retry_source_client_message_id,
        )
        return _encoded_turn_bytes(prepared.effective_request)

    def admission_request_never_fits(
        self,
        request: TurnRequest,
        *,
        fresh_interaction: bool = False,
        fresh_interaction_after_failure: bool = False,
        retry_source_client_message_id: str | None = None,
    ) -> bool:
        """单个请求永久超过最大字节容量时返回 True，等待容量无意义。"""

        return (
            self._effective_request_bytes(
                request,
                fresh_interaction=fresh_interaction,
                fresh_interaction_after_failure=fresh_interaction_after_failure,
                retry_source_client_message_id=retry_source_client_message_id,
            )
            > self._max_active_bytes
        )

    async def wait_capacity_available(
        self,
        request: TurnRequest,
        *,
        fresh_interaction: bool = False,
        fresh_interaction_after_failure: bool = False,
        retry_source_client_message_id: str | None = None,
    ) -> None:
        """等待控制面全局容量或关闭状态变化，由调用方重新尝试 start_turn。

        阶段1：先按单请求永久超出容量边界直接返回，避免无意义的无限等待；
        阶段2：clear 后检查再等待：任何在 clear 之后发生的容量释放都会
        重新 set 事件并唤醒，绝不丢失 wakeup；
        阶段3：关闭或排空也会唤醒，随后 start_turn 暴露 RuntimeClosedError。
        """

        request_bytes = self._effective_request_bytes(
            request,
            fresh_interaction=fresh_interaction,
            fresh_interaction_after_failure=fresh_interaction_after_failure,
            retry_source_client_message_id=retry_source_client_message_id,
        )
        while not self._closed and self._accepting_turns:
            self._admission_capacity_event.clear()
            if request_bytes > self._max_active_bytes or self._admission_can_fit(
                request_bytes
            ):
                return
            await self._admission_capacity_event.wait()

    async def wait_until_accepting_turns(self) -> None:
        """等待 restart 取消恢复准入；完整 shutdown 立即暴露关闭。"""

        # clear 后复查，避免 resume 恰好发生在 clear 与 wait 之间而丢 wakeup。
        while not self._closed and not self._accepting_turns:
            self._admission_capacity_event.clear()
            if self._closed or self._accepting_turns:
                break
            await self._admission_capacity_event.wait()
        if self._closed:
            raise RuntimeClosedError("conversation runtime is shutting down")

    def _release_turn_ownership(
        self,
        thread_id: str,
        turn_id: str,
        *,
        require_admission: bool = False,
    ) -> None:
        """释放 turn 在 runtime 的全部所有权状态；中断与正常收束共用。"""

        _ = self._active_by_thread.pop(thread_id, None)
        idle = self._thread_idle.pop(thread_id, None)
        if idle is not None:
            idle.set()
        _ = self._tasks.pop(turn_id, None)
        if require_admission and turn_id not in self._active_turn_bytes:
            raise RuntimeError(f"queued turn admission missing: {turn_id}")
        self._interrupt_requested.discard(turn_id)
        self._consumed_inputs.pop(turn_id, None)
        self._locked_turn_inputs.discard(turn_id)
        self._turn_input_sources.pop(turn_id, None)
        self._release_admission(turn_id)

    async def _run(
        self,
        request: TurnRequest,
        turn_id: str,
        *,
        attempt_replay: list[dict[str, Any]],
        prior_tool_chain: list[dict[str, Any]],
        runtime_snapshot_lease: RuntimeSnapshotLease | None,
        channel_binding_lease: ChannelBindingLease | None,
        execution_scope: TurnExecutionScope | None,
    ) -> None:
        """执行已按 thread 和容量准入的 turn，并保证只写一个终态。"""

        terminal: TurnRecord | None = None
        fatal_error: BaseException | None = None
        terminal_client_message_id = _control_client_message_id(request.metadata)
        observed_items: dict[str, TurnItem] = {}
        open_item_ids: dict[str, None] = {}

        def close_observed_items(status: TurnStatus) -> list[TurnItem]:
            """闭合并返回本轮已经实时发布的全部 item。"""

            for item_id in tuple(open_item_ids):
                item = observed_items[item_id]
                closed = TurnItem(
                    item.kind,
                    item.id,
                    {**item.data, "status": status.value},
                )
                observed_items[item_id] = closed
                open_item_ids.pop(item_id)
                self._publish(
                    TurnEvent.create(
                        "item/completed",
                        request.thread_id,
                        turn_id,
                        item=closed.to_dict(),
                    )
                )
            return list(observed_items.values())

        try:
            # 1. 不同 thread 可并发；同 thread 已由 start_turn 的唯一 owner 拒绝。
            record = self._store.transition_turn(
                turn_id,
                expected_status=TurnStatus.QUEUED,
                status=TurnStatus.IN_PROGRESS,
                thread_id=request.thread_id,
            )
            self._publish(
                TurnEvent.create(
                    "turn/started", request.thread_id, turn_id, turn=record.to_dict()
                )
            )

            # 2. 核心执行不依赖 transport；成功结果进入正式 assistant item。
            execution_request = TurnRequest(
                request.thread_id,
                request.input,
                {
                    **request.metadata,
                    "turnId": turn_id,
                    "_controlTurnInputSource": self._turn_input_sources[turn_id],
                    METADATA_ATTEMPT_REPLAY: attempt_replay,
                    METADATA_PRIOR_TOOL_CHAIN: prior_tool_chain,
                },
            )
            live_item_ids: set[str] = set()

            def publish_item(method: str, item: TurnItem) -> None:
                live_item_ids.add(item.id)
                if method == "item/started":
                    if item.id in observed_items:
                        raise ValueError(f"item 重复 started: {item.id}")
                    observed_items[item.id] = item
                    open_item_ids[item.id] = None
                    self._store.append_active_turn_item(
                        turn_id,
                        thread_id=request.thread_id,
                        item=item,
                    )
                elif method == "item/completed":
                    if item.id not in open_item_ids:
                        raise ValueError(f"item 未 started 即 completed: {item.id}")
                    observed_items[item.id] = item
                    open_item_ids.pop(item.id)
                    self._store.replace_active_turn_item(
                        turn_id,
                        thread_id=request.thread_id,
                        item=item,
                    )
                else:
                    raise ValueError(f"未知 control item event: {method}")
                self._publish(
                    TurnEvent.create(
                        method,
                        request.thread_id,
                        turn_id,
                        item=item.to_dict(),
                    )
                )

            execution_request.metadata["_controlItemEvent"] = publish_item
            snapshot_token = None
            channel_token = None
            scope_token = None
            if runtime_snapshot_lease is not None:
                if not runtime_snapshot_lease.active:
                    raise RuntimeError("turn exact RuntimeSnapshot lease 已关闭")
                from agent.plugins.snapshot import bind_runtime_snapshot

                snapshot_token = bind_runtime_snapshot(runtime_snapshot_lease)
            if channel_binding_lease is not None:
                from agent.plugins.channel_generation_host import (
                    bind_channel_turn_binding,
                )

                channel_token = bind_channel_turn_binding(channel_binding_lease)
            if execution_scope is not None:
                scope_token = bind_turn_scope(execution_scope)
            try:
                execution = await self._executor(execution_request)
            finally:
                if scope_token is not None:
                    reset_turn_scope(scope_token)
                if channel_token is not None:
                    from agent.plugins.channel_generation_host import (
                        reset_channel_turn_binding,
                    )

                    reset_channel_turn_binding(channel_token)
                if snapshot_token is not None:
                    from agent.plugins.snapshot import reset_runtime_snapshot

                    reset_runtime_snapshot(snapshot_token)
            await self._turn_input_sources[turn_id].lock()
            if open_item_ids:
                raise RuntimeError(
                    f"executor 返回时仍有未闭合 item: {sorted(open_item_ids)}"
                )
            for item in execution.items:
                if item.id in live_item_ids:
                    continue
                self._publish(
                    TurnEvent.create(
                        "item/started",
                        request.thread_id,
                        turn_id,
                        item=item.to_dict(),
                    )
                )
                self._publish(
                    TurnEvent.create(
                        "item/completed",
                        request.thread_id,
                        turn_id,
                        item=item.to_dict(),
                    )
                )
            assistant_item = TurnItem(
                TurnItemKind.ASSISTANT_MESSAGE,
                new_item_id(),
                {"content": execution.response, **execution.assistant_data},
            )
            self._publish(
                TurnEvent.create(
                    "item/started",
                    request.thread_id,
                    turn_id,
                    item=assistant_item.to_dict(),
                )
            )
            deltas = execution.deltas or [execution.response]
            for sequence, delta in enumerate(deltas):
                self._publish(
                    TurnEvent.create(
                        "item/assistantMessage/delta",
                        request.thread_id,
                        turn_id,
                        itemId=assistant_item.id,
                        delta=delta,
                        sequence=sequence,
                    )
                )
                # 2a. 让订阅者消费事后回放，避免突发填满有界队列。
                await asyncio.sleep(0)
            self._publish(
                TurnEvent.create(
                    "item/completed",
                    request.thread_id,
                    turn_id,
                    item=assistant_item.to_dict(),
                )
            )
            current = self._store.read_turn(turn_id)
            if current is None:
                raise TurnNotFoundError(f"turn 不存在: {turn_id}")
            items = _merge_turn_items(
                current.items,
                [*execution.items, assistant_item],
            )
            terminal = self._store.transition_turn(
                turn_id,
                expected_status=TurnStatus.IN_PROGRESS,
                status=TurnStatus.COMPLETED,
                thread_id=request.thread_id,
                items=items,
                final_response=execution.response,
                usage=execution.usage,
            )
        except asyncio.CancelledError:
            current = self._store.read_turn(turn_id)
            if current is not None and current.status.is_terminal:
                terminal = current
            elif current is not None:
                status = (
                    TurnStatus.INTERRUPTED
                    if current.status is TurnStatus.IN_PROGRESS
                    and turn_id in self._interrupt_requested
                    else TurnStatus.CANCELLED
                )
                items = _merge_turn_items(
                    current.items,
                    close_observed_items(status),
                )
                terminal = self._store.transition_turn(
                    turn_id,
                    expected_status=current.status,
                    status=status,
                    thread_id=request.thread_id,
                    items=items,
                )
        except Exception as exc:
            logger.exception(
                "conversation turn failed thread=%s turn=%s", request.thread_id, turn_id
            )
            current = self._store.read_turn(turn_id)
            if current is not None and current.status.is_terminal:
                terminal = current
            elif current is not None and current.status is TurnStatus.IN_PROGRESS:
                items = _merge_turn_items(
                    current.items,
                    close_observed_items(TurnStatus.FAILED),
                )
                terminal = self._store.transition_turn(
                    turn_id,
                    expected_status=current.status,
                    status=TurnStatus.FAILED,
                    thread_id=request.thread_id,
                    items=items,
                    error=TurnError(
                        type=(
                            exc.error_type
                            if isinstance(exc, ControlExecutionError)
                            else type(exc).__name__
                        ),
                        message=str(exc),
                        retryable=bool(getattr(exc, "retryable", False)),
                    ),
                )
            else:
                fatal_error = exc
        finally:
            # 3. terminal 是唯一结束通知；结果 future 与 active owner 一起收束。
            future = self._results[turn_id]
            if terminal is not None:
                # 3a. 时间链：SessionDB 权威 turn 终态（owner；展示投影不得自持终态）
                turn_milestone(
                    logger,
                    "tl:turn.terminal",
                    session_id=request.thread_id,
                    turn_id=turn_id,
                    client_message_id=terminal_client_message_id,
                    counts=f"status={terminal.status.value}",
                    outcome=terminal.status.value,
                )
                if self._restart_coordinator is not None:
                    self._restart_coordinator.mark_turn_terminal(
                        turn_id,
                        terminal.status.value,
                    )
                event = TurnEvent.create(
                    "turn/completed",
                    request.thread_id,
                    turn_id,
                    turn=terminal.to_dict(),
                )
                self._publish(event)
                if not future.done():
                    future.set_result(TurnResult.from_record(terminal))
                self._finish_streams(turn_id)
            else:
                error = fatal_error or RuntimeError(f"turn 未建立终态: {turn_id}")
                if not future.done():
                    future.set_exception(error)
                self._fail_streams(turn_id, error)
            self._release_turn_ownership(request.thread_id, turn_id)
            if terminal is not None and self._turn_terminal is not None:
                self._turn_terminal(
                    turn_id,
                    terminal.status,
                    {**request.metadata, "turnId": turn_id},
                    tuple(terminal.items),
                )

    def _publish(self, event: TurnEvent) -> None:
        self._raise_replay_reaper_failure()
        self._gc_terminal_replay()
        history = self._history[event.turn_id]
        sequences = self._history_sequences[event.turn_id]
        sequence = self._next_event_sequence[event.turn_id]
        self._next_event_sequence[event.turn_id] = sequence + 1
        event_bytes = _encoded_event_bytes(event)
        history.append(event)
        sequences.append(sequence)
        self._replay_order[(event.turn_id, sequence)] = (event, event_bytes)
        self._replay_bytes += event_bytes
        self._history_byte_totals[event.turn_id] += event_bytes
        self._trim_turn_replay(event.turn_id)
        self._trim_global_replay()
        for queue in tuple(self._subscribers[event.turn_id]):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                while not queue.empty():
                    _ = queue.get_nowait()
                queue.put_nowait(
                    SlowConsumerError(f"turn event consumer too slow: {event.turn_id}")
                )
                self._subscribers[event.turn_id].discard(queue)

    def _trim_turn_replay(self, turn_id: str) -> None:
        history = self._history[turn_id]
        sequences = self._history_sequences[turn_id]
        while history and (
            len(history) > self._replay_events_per_turn
            or self._history_bytes(turn_id) > self._replay_bytes_per_turn
        ):
            self._remove_oldest_replay_event(turn_id)
            if len(history) != len(sequences):
                raise RuntimeError(f"control replay index corrupted: {turn_id}")
        if len(history) < self._next_event_sequence[turn_id]:
            self._history_truncated.add(turn_id)

    def _trim_global_replay(self) -> None:
        while self._replay_bytes > self._replay_bytes_global and self._replay_order:
            (turn_id, sequence), (event, event_bytes) = self._replay_order.popitem(
                last=False
            )
            self._remove_replay_event(
                turn_id, sequence, event, event_bytes=event_bytes, global_removed=True
            )
            self._history_truncated.add(turn_id)

    def _history_bytes(self, turn_id: str) -> int:
        return self._history_byte_totals[turn_id]

    def _remove_oldest_replay_event(self, turn_id: str) -> None:
        history = self._history[turn_id]
        sequences = self._history_sequences[turn_id]
        if not history or len(history) != len(sequences):
            raise RuntimeError(f"control replay index corrupted: {turn_id}")
        self._remove_replay_event(turn_id, sequences[0], history[0])

    def _remove_replay_event(
        self,
        turn_id: str,
        sequence: int,
        event: TurnEvent,
        *,
        event_bytes: int | None = None,
        global_removed: bool = False,
    ) -> None:
        history = self._history.get(turn_id)
        sequences = self._history_sequences.get(turn_id)
        if history is None or sequences is None:
            raise RuntimeError(f"control replay index corrupted: {turn_id}")
        if len(history) != len(sequences):
            raise RuntimeError(f"control replay index corrupted: {turn_id}")
        if len(set(sequences)) != len(sequences):
            raise RuntimeError(f"control replay index corrupted: {turn_id}")

        target_index: int | None = None
        for index, (item_sequence, item) in enumerate(zip(sequences, history)):
            key = (turn_id, item_sequence)
            entry = self._replay_order.get(key)
            if global_removed and item_sequence == sequence:
                if item is not event:
                    raise RuntimeError(f"control replay index corrupted: {turn_id}")
                if target_index is not None:
                    raise RuntimeError(f"control replay index corrupted: {turn_id}")
                target_index = index
                continue
            if entry is None or entry[0] is not item:
                raise RuntimeError(f"control replay index corrupted: {turn_id}")
            if item_sequence == sequence:
                if item is not event:
                    raise RuntimeError(f"control replay index corrupted: {turn_id}")
                target_index = index

        if target_index is None:
            raise RuntimeError(f"control replay index corrupted: {turn_id}")
        key = (turn_id, sequence)
        entry = self._replay_order.get(key)
        if global_removed:
            if entry is not None:
                raise RuntimeError(f"control replay index corrupted: {turn_id}")
            if event_bytes is None:
                raise RuntimeError(f"control replay index corrupted: {turn_id}")
            size = event_bytes
        else:
            if entry is None or entry[0] is not event:
                raise RuntimeError(f"control replay index corrupted: {turn_id}")
            if event_bytes is not None and entry[1] != event_bytes:
                raise RuntimeError(f"control replay index corrupted: {turn_id}")
            size = entry[1]
        turn_total = self._history_byte_totals.get(turn_id)
        if (
            turn_total is None
            or size < 0
            or turn_total < size
            or self._replay_bytes < size
        ):
            raise RuntimeError(f"control replay index corrupted: {turn_id}")

        history.pop(target_index)
        sequences.pop(target_index)
        if not global_removed:
            self._replay_order.pop(key)
        self._replay_bytes -= size
        self._history_byte_totals[turn_id] = turn_total - size

    def _gc_terminal_replay(self) -> None:
        now = time.monotonic()
        for turn_id, expiry in tuple(self._terminal_replay_expiry.items()):
            if expiry > now:
                continue
            history = self._history.get(turn_id)
            if history is None:
                self._terminal_replay_expiry.pop(turn_id, None)
                continue
            while history:
                self._remove_oldest_replay_event(turn_id)
            if not history:
                self._history.pop(turn_id, None)
                self._history_sequences.pop(turn_id, None)
                self._history_byte_totals.pop(turn_id, None)
                self._history_truncated.discard(turn_id)
                self._next_event_sequence.pop(turn_id, None)
                self._subscribers.pop(turn_id, None)
                self._results.pop(turn_id, None)
                self._terminal_replay_expiry.pop(turn_id, None)

    def _finish_streams(self, turn_id: str) -> None:
        self._raise_replay_reaper_failure()
        self._terminal_replay_expiry[turn_id] = (
            time.monotonic() + self._terminal_replay_ttl_seconds
        )
        self._ensure_replay_reaper()
        self._replay_reaper_wakeup.set()
        for queue in tuple(self._subscribers[turn_id]):
            try:
                queue.put_nowait(_STREAM_END)
            except asyncio.QueueFull:
                while not queue.empty():
                    _ = queue.get_nowait()
                queue.put_nowait(
                    SlowConsumerError(f"turn event consumer too slow: {turn_id}")
                )

    def _fail_streams(self, turn_id: str, error: BaseException) -> None:
        for queue in tuple(self._subscribers[turn_id]):
            while not queue.empty():
                _ = queue.get_nowait()
            queue.put_nowait(error)

    async def subscribe(
        self,
        thread_id: str,
        turn_id: str,
        *,
        after_event: int | None = None,
    ) -> AsyncIterator[TurnEvent]:
        """订阅 live stream，并在 replay 被截断或过期时发出权威快照。"""

        if after_event is not None and (
            not isinstance(after_event, int)
            or isinstance(after_event, bool)
            or after_event < -1
        ):
            raise ValueError("after_event 必须是大于等于 -1 的整数")
        self._raise_replay_reaper_failure()
        self._gc_terminal_replay()
        record = self.read_turn(thread_id, turn_id)
        history = self._history.get(turn_id)
        sequences = self._history_sequences.get(turn_id, [])
        replay_expired = record.status.is_terminal and history is None
        if history is None and not replay_expired:
            raise TurnNotFoundError(f"turn 不在当前 runtime: {thread_id}/{turn_id}")

        replay_events: list[TurnEvent] = [] if history is None else list(history)
        replay_sequences = [] if history is None else list(sequences)
        replay_truncated = False
        if replay_expired:
            replay_events = [self._replay_notice("replay/expired", record)]
        elif record.status.is_terminal and self._terminal_replay_expired(turn_id):
            replay_expired = True
            replay_events = [self._replay_notice("replay/expired", record)]
        else:
            if after_event is not None:
                replay_truncated = bool(
                    replay_sequences and after_event < replay_sequences[0] - 1
                )
                replay_events = [
                    event
                    for sequence, event in zip(replay_sequences, replay_events)
                    if sequence > after_event
                ]
            elif turn_id in self._history_truncated:
                replay_truncated = True
            if replay_truncated:
                replay_events.insert(0, self._replay_notice("replay/truncated", record))

        queue: asyncio.Queue[StreamValue] = asyncio.Queue(
            max(self._subscriber_queue_size, len(replay_events) + 1)
        )
        for event in replay_events:
            queue.put_nowait(event)
        if record.status.is_terminal:
            queue.put_nowait(_STREAM_END)
        else:
            self._subscribers[turn_id].add(queue)
        try:
            while True:
                value = await queue.get()
                if value is _STREAM_END:
                    return
                if isinstance(value, BaseException):
                    raise value
                yield cast(TurnEvent, value)
        finally:
            self._subscribers.get(turn_id, set()).discard(queue)

    def _terminal_replay_expired(self, turn_id: str) -> bool:
        expiry = self._terminal_replay_expiry.get(turn_id)
        return expiry is not None and expiry <= time.monotonic()

    def _raise_replay_reaper_failure(self) -> None:
        if self._replay_reaper_error is not None:
            raise RuntimeError(
                "control replay reaper failed"
            ) from self._replay_reaper_error

    def _ensure_replay_reaper(self) -> None:
        self._raise_replay_reaper_failure()
        if self._replay_reaper_task is None:
            loop = asyncio.get_running_loop()
            self._replay_reaper_task = loop.create_task(
                self._replay_reaper(),
                name="conversation-replay-reaper",
            )
            self._replay_reaper_task.add_done_callback(self._observe_replay_reaper_task)

    def _observe_replay_reaper_task(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None and self._replay_reaper_error is None:
            self._replay_reaper_error = error

    async def _replay_reaper(self) -> None:
        """按最早 terminal expiry 回收 replay，并把内部损坏升级为 runtime fatal。"""

        try:
            while not self._closed:
                # 1. 先清除旧唤醒，再读取当前最早 expiry，避免错过更早的新终态。
                self._replay_reaper_wakeup.clear()
                expiry = min(self._terminal_replay_expiry.values(), default=None)
                if expiry is None:
                    await self._replay_reaper_wakeup.wait()
                    continue

                delay = expiry - time.monotonic()
                if delay > 0:
                    try:
                        await asyncio.wait_for(
                            self._replay_reaper_wakeup.wait(), timeout=delay
                        )
                    except asyncio.TimeoutError:
                        pass
                    continue

                # 2. 到期清理只触碰运行时 replay 投影；SessionStore 保持权威终态。
                self._gc_terminal_replay()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._replay_reaper_error = exc
            self._accepting_turns = False
            self._admission_capacity_event.set()
            logger.critical(
                "event=runtime_fatal owner=control.replay_reaper reason=terminal_replay_cleanup_failed",
                exc_info=True,
            )
            raise

    @staticmethod
    def _replay_notice(method: str, record: TurnRecord) -> TurnEvent:
        status = (
            "replay_truncated" if method.endswith("truncated") else "replay_expired"
        )
        return TurnEvent.create(
            method,
            record.thread_id,
            record.id,
            error=status,
            replay_status=status,
            snapshot=record.to_dict(),
        )

    def read_turn(self, thread_id: str, turn_id: str) -> TurnRecord:
        record = self._store.read_turn(turn_id)
        if record is None or record.thread_id != thread_id:
            raise TurnNotFoundError(f"turn 不存在: {thread_id}/{turn_id}")
        return record

    def admission_snapshot(self) -> dict[str, int]:
        """返回 queued/running turn 的控制准入计数，不含历史 thread。"""

        return {
            "turns": len(self._active_turn_bytes),
            "bytes": self._active_admission_bytes,
            "runtime_objects": self._live_runtime_objects,
            "max_turns": self._max_active_turns,
            "max_bytes": self._max_active_bytes,
            "max_runtime_objects": self._max_live_runtime_objects,
        }

    @property
    def active_turn_count(self) -> int:
        return len(self._active_turn_bytes)

    @property
    def active_admission_bytes(self) -> int:
        return self._active_admission_bytes

    @property
    def live_runtime_objects(self) -> int:
        return self._live_runtime_objects

    @property
    def replay_bytes(self) -> int:
        return self._replay_bytes

    def is_thread_active(self, thread_id: str) -> bool:
        return thread_id in self._active_by_thread

    async def quiesce_and_drain(self) -> None:
        """停止接收新 turn，并等待已经持久化的 turn 自然结束。"""
        self._accepting_turns = False
        self._admission_capacity_event.set()
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks)

    def quiesce_for_restart(self, caller_turn_id: str) -> None:
        """仅在 caller 是唯一 turn 时冻结新的 turn 准入。"""

        # 1. caller 必须是当前 runtime 唯一已经持久化的 turn。
        active_turns = set(self._tasks)
        if caller_turn_id not in active_turns:
            raise RuntimeClosedError(
                f"restart caller turn 不在当前 runtime: {caller_turn_id}"
            )
        others = active_turns - {caller_turn_id}
        if others:
            raise RuntimeClosedError(
                f"仍有其他 turn 等待或执行，拒绝重启: {sorted(others)}"
            )
        if not self._accepting_turns:
            if self._restart_owner_turn_id == caller_turn_id:
                return
            raise RuntimeClosedError("conversation runtime 已在排空")

        # 2. 不获取全局 admission，避免 caller 在工具执行中自锁。
        self._accepting_turns = False
        self._restart_owner_turn_id = caller_turn_id
        self._admission_capacity_event.set()

    def resume_after_restart_cancel(self, caller_turn_id: str) -> None:
        """只允许原 restart owner 在提交前恢复准入。"""

        if self._restart_owner_turn_id != caller_turn_id:
            raise RuntimeError(f"restart admission owner 不匹配: {caller_turn_id}")
        self._restart_owner_turn_id = None
        if not self._closed:
            self._accepting_turns = True
        self._admission_capacity_event.set()

    async def wait_thread_available(self, thread_id: str) -> None:
        """等待当前 thread owner 释放，不获取新的 owner。"""

        while event := self._thread_idle.get(thread_id):
            await event.wait()

    async def wait_result(self, thread_id: str, turn_id: str) -> TurnResult:
        record = self.read_turn(thread_id, turn_id)
        if record.status.is_terminal:
            return TurnResult.from_record(record)
        future = self._results.get(turn_id)
        if future is None:
            raise TurnNotFoundError(f"turn 不在当前 runtime: {thread_id}/{turn_id}")
        return await asyncio.shield(future)

    async def interrupt_turn(self, thread_id: str, turn_id: str) -> TurnRecord:
        # 1. 与普通输入 admission/final lock 共用栅栏，先锁定再取消执行。
        async with self._control_admission_lock:
            record = self.read_turn(thread_id, turn_id)
            if record.status.is_terminal:
                return record
            if self._active_by_thread.get(thread_id) != turn_id:
                raise TurnNotFoundError(f"active turn 不匹配: {thread_id}/{turn_id}")
            already_locked = turn_id in self._locked_turn_inputs
            task = self._tasks[turn_id]
            future = self._results[turn_id]
            if not already_locked:
                self._locked_turn_inputs.add(turn_id)
                if record.status is TurnStatus.IN_PROGRESS:
                    self._interrupt_requested.add(turn_id)
                task.cancel()

        if already_locked:
            await asyncio.shield(future)
            return self.read_turn(thread_id, turn_id)

        if record.status is TurnStatus.QUEUED:
            # 2. 先让已启动 task 自行收束；启动前取消则由 owner 补交 cancelled。
            _ = await asyncio.gather(task, return_exceptions=True)
            if future.done():
                return self.read_turn(thread_id, turn_id)
            terminal = self._store.transition_turn(
                turn_id,
                expected_status=TurnStatus.QUEUED,
                status=TurnStatus.CANCELLED,
                thread_id=thread_id,
            )
            self._publish(
                TurnEvent.create(
                    "turn/completed", thread_id, turn_id, turn=terminal.to_dict()
                )
            )
            future.set_result(TurnResult.from_record(terminal))
            self._finish_streams(turn_id)
            self._release_turn_ownership(thread_id, turn_id, require_admission=True)
            return terminal

        # 3. in-progress task 自己在取消处理器中提交 interrupted。
        await asyncio.shield(future)
        return self.read_turn(thread_id, turn_id)

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult:
        """为现有 channel 命令提供 session 定位的同步 adapter。"""
        turn_id = self._active_by_thread.get(session_key)
        if turn_id is None:
            return InterruptResult("idle", session_key, "当前没有正在执行的任务。")
        _ = asyncio.create_task(
            self.interrupt_turn(session_key, turn_id),
            name=f"channel-interrupt:{turn_id}",
        )
        return InterruptResult(
            "interrupted",
            session_key,
            "本轮已中断。你可以继续补充要求，我会接着这件事处理。",
        )

    async def shutdown(self) -> None:
        if self._closed:
            self._raise_replay_reaper_failure()
            return
        self._closed = True
        self._accepting_turns = False
        self._admission_capacity_event.set()
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        reaper = self._replay_reaper_task
        if reaper is not None:
            reaper.cancel()
            result = await asyncio.gather(reaper, return_exceptions=True)
            if (
                result
                and isinstance(result[0], BaseException)
                and not isinstance(result[0], asyncio.CancelledError)
            ):
                self._replay_reaper_error = result[0]
        self._raise_replay_reaper_failure()
