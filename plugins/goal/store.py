"""目标状态的 Ledger owner 记录：编码、CAS 事务与续跑追加。"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.content.contract import CONTENT
from plugins.ledger.contract import (
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    ContentPart,
    Input,
    MessageReader,
    MessageWriter,
    OwnerState,
    OwnerStore,
    OwnerTransaction,
)
from plugins.ledger.contract import MessageConflict

from .state import (
    Goal,
    GoalStatus,
    UsageCoverage,
    mark_continuation,
    normalize_objective,
    normalize_token_budget,
    now_utc,
    settle_round,
)

import hashlib

GOAL_PREFIX = "goal:"
OPERATION_PREFIX = "goalop:"
MAX_OPERATIONS = 1024
_OPERATION_KINDS = frozenset({"create", "update", "pause", "resume", "clear", "budget", "command"})
_GOAL_FIELDS = frozenset({
    "version", "session_id", "goal_id", "objective", "status", "token_budget", "tokens_used",
    "continuations", "consecutive_empty", "settled_through_seq",
    "last_settled_message_id", "continuation_id", "usage_coverage",
    "created_at", "updated_at", "last_reason",
})
_OPERATION_FIELDS = frozenset({
    "version", "kind", "digest", "response", "goal", "created_at",
})


@dataclass(frozen=True, slots=True)
class Operation:
    kind: str
    digest: str
    response: str
    goal: Goal | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RoundSettlement:
    activity: bool
    tokens: int
    coverage: UsageCoverage
    settled_through_seq: int
    last_settled_message_id: str | None
    reason: str | None
    append: bool
    pause_after: bool = False
    preempted: bool = False


class GoalStale(ValueError):
    """prepare 时捕获的目标快照已被替换或清除。"""


def digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def goal_key(session_id: str) -> str:
    if not session_id:
        raise ValueError("目标记录需要 Session")
    return GOAL_PREFIX + session_id


def operation_key(kind: str, session_id: str, operation_id: str) -> str:
    if kind not in _OPERATION_KINDS:
        raise ValueError(f"未知的目标操作 kind: {kind}")
    if not session_id or not operation_id:
        raise ValueError("目标操作需要 Session 与 operation_id")
    return f"{OPERATION_PREFIX}{kind}:{session_id}:{operation_id}"


def _datetime(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"goal owner {field} 必须是 ISO 时间")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"goal owner {field} 不是合法 ISO 时间") from error
    if parsed.utcoffset() is None:
        raise ValueError(f"goal owner {field} 必须带时区")
    return parsed


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"goal owner {field} 无效")
    return value


def decode_goal(session_id: str, raw: object) -> Goal:
    if not isinstance(raw, Mapping):
        raise ValueError("goal owner 目标字段无效")
    value = cast(Mapping[str, Any], raw)
    if set(value) != _GOAL_FIELDS:
        raise ValueError("goal owner 目标字段无效")
    if value["version"] != 1:
        raise ValueError("goal owner 目标版本不支持")
    if value["session_id"] != session_id:
        raise ValueError("goal owner 目标 Session 不一致")
    goal_id = value["goal_id"]
    objective = value["objective"]
    status = value["status"]
    token_budget = value["token_budget"]
    tokens_used = value["tokens_used"]
    continuations = value["continuations"]
    consecutive_empty = value["consecutive_empty"]
    settled_through_seq = value["settled_through_seq"]
    coverage = value["usage_coverage"]
    if not isinstance(goal_id, str) or not goal_id:
        raise ValueError("goal owner goal_id 无效")
    if not isinstance(objective, str) or normalize_objective(objective) != objective:
        raise ValueError("goal owner objective 无效")
    if not isinstance(status, str):
        raise ValueError("goal owner status 无效")
    try:
        parsed_status = GoalStatus(status)
    except ValueError as error:
        raise ValueError(f"goal owner 未知状态: {status}") from error
    if token_budget is not None and (
        type(token_budget) is not int or normalize_token_budget(token_budget) != token_budget
    ):
        raise ValueError("goal owner token_budget 无效")
    for name, number in (
        ("tokens_used", tokens_used),
        ("continuations", continuations),
        ("consecutive_empty", consecutive_empty),
    ):
        if type(number) is not int or number < 0:
            raise ValueError(f"goal owner {name} 必须是非负整数")
    if type(settled_through_seq) is not int or settled_through_seq < -1:
        raise ValueError("goal owner settled_through_seq 无效")
    if coverage not in ("exact", "partial", "unavailable"):
        raise ValueError("goal owner usage_coverage 无效")
    return Goal(
        session_id=session_id,
        goal_id=goal_id,
        objective=objective,
        status=parsed_status,
        token_budget=token_budget,
        tokens_used=tokens_used,
        continuations=continuations,
        consecutive_empty=consecutive_empty,
        settled_through_seq=settled_through_seq,
        last_settled_message_id=_optional_text(value["last_settled_message_id"], "last_settled_message_id"),
        continuation_id=_optional_text(value["continuation_id"], "continuation_id"),
        usage_coverage=cast(UsageCoverage, coverage),
        created_at=_datetime(value["created_at"], "created_at"),
        updated_at=_datetime(value["updated_at"], "updated_at"),
        last_reason=_optional_text(value["last_reason"], "last_reason"),
    )


def encode_goal(goal: Goal) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "session_id": goal.session_id,
        "goal_id": goal.goal_id,
        "objective": goal.objective,
        "status": goal.status.value,
        "token_budget": goal.token_budget,
        "tokens_used": goal.tokens_used,
        "continuations": goal.continuations,
        "consecutive_empty": goal.consecutive_empty,
        "settled_through_seq": goal.settled_through_seq,
        "last_settled_message_id": goal.last_settled_message_id,
        "continuation_id": goal.continuation_id,
        "usage_coverage": goal.usage_coverage,
        "created_at": goal.created_at.isoformat(),
        "updated_at": goal.updated_at.isoformat(),
        "last_reason": goal.last_reason,
    }
    _ = decode_goal(goal.session_id, value)
    return value


def decode_operation(raw: object) -> Operation:
    if not isinstance(raw, Mapping):
        raise ValueError("goal owner 操作字段无效")
    value = cast(Mapping[str, Any], raw)
    if set(value) != _OPERATION_FIELDS:
        raise ValueError("goal owner 操作字段无效")
    if value["version"] != 1:
        raise ValueError("goal owner 操作版本不支持")
    kind = value["kind"]
    digest = value["digest"]
    response = value["response"]
    goal = value["goal"]
    if kind not in _OPERATION_KINDS:
        raise ValueError("goal owner 操作 kind 无效")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ValueError("goal owner 操作 digest 无效")
    if not isinstance(response, str) or not response:
        raise ValueError("goal owner 操作 response 无效")
    decoded_goal = None
    if goal is not None:
        if not isinstance(goal, Mapping) or not isinstance(goal.get("session_id"), str):
            raise ValueError("goal owner 操作目标无效")
        decoded_goal = decode_goal(cast(str, goal["session_id"]), goal)
    return Operation(
        kind=kind,
        digest=digest,
        response=response,
        goal=decoded_goal,
        created_at=_datetime(value["created_at"], "created_at"),
    )


def encode_operation(operation: Operation) -> dict[str, object]:
    value: dict[str, object] = {
        "version": 1,
        "kind": operation.kind,
        "digest": operation.digest,
        "response": operation.response,
        "goal": None if operation.goal is None else encode_goal(operation.goal),
        "created_at": operation.created_at.isoformat(),
    }
    _ = decode_operation(value)
    return value


class GoalStore:
    """目标 owner 的窄门面；所有写路径都走 OWNER_STATE 的同一事务。"""

    def __init__(
        self, ctx: Context, *, state: OwnerState | None = None,
        catalog: Any | None = None, writers: Any | None = None,
        content: Any | None = None,
    ) -> None:
        self._ctx = ctx
        self._state = state if state is not None else ctx.require(OWNER_STATE)
        self._catalog = catalog if catalog is not None else ctx.require(MESSAGE_CATALOG)
        self._writers = writers if writers is not None else ctx.require(MESSAGE_WRITERS)
        self._content = content if content is not None else ctx.require(CONTENT)

    def _open(self) -> OwnerStore:
        return self._state.open(self._ctx)

    def _access(self, session_id: str) -> tuple[OwnerStore, MessageReader, MessageWriter]:
        reader = self._catalog.reader(session_id)
        writer = self._writers.bind(
            self._ctx, author="goal", source="conversation",
            body_types=(Input,), content={"text": self._content.check_text},
        )(session_id)
        store = self._open()
        store.check_access(reader, writer)
        return store, reader, writer

    async def get(self, session_id: str) -> Goal | None:
        store = self._open()
        record = await run_file_io(lambda: store.read(goal_key(session_id)))
        return None if record is None else decode_goal(session_id, record.value)

    async def get_with_version(self, session_id: str) -> tuple[Goal, int] | None:
        store = self._open()
        record = await run_file_io(lambda: store.read(goal_key(session_id)))
        if record is None:
            return None
        return decode_goal(session_id, record.value), record.version

    async def version(self, session_id: str) -> int:
        store = self._open()
        record = await run_file_io(lambda: store.read(goal_key(session_id)))
        if record is None:
            raise MessageConflict("目标记录不存在")
        return record.version

    async def active_sessions(self) -> frozenset[str]:
        store = self._open()
        records = await run_file_io(store.list)
        sessions: set[str] = set()
        for key, record in records:
            if not key.startswith(GOAL_PREFIX):
                continue
            session_id = key[len(GOAL_PREFIX):]
            goal = decode_goal(session_id, record.value)
            if goal.active:
                sessions.add(session_id)
        return frozenset(sessions)

    async def runtime_sessions(self) -> frozenset[str]:
        """返回需要 runtime 处理的 Session：active，或还有未结算的目标轮次。"""
        store = self._open()
        records = await run_file_io(store.list)
        sessions: set[str] = set()
        for key, record in records:
            if not key.startswith(GOAL_PREFIX):
                continue
            session_id = key[len(GOAL_PREFIX):]
            goal = decode_goal(session_id, record.value)
            if goal.active or goal.continuation_id is not None:
                sessions.add(session_id)
        return frozenset(sessions)

    async def conversation_head(self, session_id: str) -> int:
        reader = self._catalog.reader(session_id)
        return await run_file_io(lambda: reader.head(source="conversation"))

    async def operation(
        self, kind: str, session_id: str, operation_id: str,
    ) -> Operation | None:
        store = self._open()
        record = await run_file_io(
            lambda: store.read(operation_key(kind, session_id, operation_id))
        )
        return None if record is None else decode_operation(record.value)

    async def apply(
        self,
        session_id: str,
        operation_id: str,
        kind: str,
        request_digest: str,
        action: Callable[[Goal | None], tuple[Goal | None, str]],
        *,
        expected_head: int | None = None,
        expected_goal: tuple[str, int] | None = None,
    ) -> tuple[Goal | None, str]:
        store, reader, _writer = self._access(session_id)
        key = operation_key(kind, session_id, operation_id)

        def transact(
            transaction: OwnerTransaction, stale: tuple[tuple[str, int], ...],
        ) -> tuple[Goal | None, str]:
            previous = transaction.read(key)
            if previous is not None:
                operation = decode_operation(previous.value)
                if (operation.kind, operation.digest) != (kind, request_digest):
                    raise MessageConflict("目标操作 key 已用于不同请求")
                return operation.goal, operation.response
            if expected_head is not None and transaction.source_changed(
                reader, "conversation", expected_head,
            ):
                raise MessageConflict("目标操作前对话来源已有新消息")
            current = transaction.read(goal_key(session_id))
            current_goal = None if current is None else decode_goal(session_id, current.value)
            if expected_goal is not None:
                if current is None or current_goal is None:
                    raise GoalStale("目标已不存在；请重新读取目标")
                if (current_goal.goal_id, current.version) != expected_goal:
                    raise GoalStale("目标已变化；请重新读取目标")
            goal, response = action(current_goal)
            if goal is None:
                if current is not None:
                    transaction.delete(goal_key(session_id), expected_version=current.version)
            else:
                if goal.session_id != session_id:
                    raise ValueError("目标操作返回了其他 Session 的目标")
                _ = transaction.save(
                    goal_key(session_id), encode_goal(goal),
                    expected_version=None if current is None else current.version,
                )
            operation = Operation(
                kind=kind, digest=request_digest, response=response,
                goal=goal, created_at=now_utc(),
            )
            _ = transaction.save(
                key, encode_operation(operation),
                expected_version=None if previous is None else previous.version,
            )
            # 回执保留上限与本次写入同事务；并发改写让整个事务重试，
            # 不能吞掉 MessageConflict（事务已经标记失败）。
            for stale_key, stale_version in stale:
                transaction.delete(stale_key, expected_version=stale_version)
            return goal, response

        last_error: Exception | None = None
        for _attempt in range(3):
            stale = self._stale_operations(store, reserve=1)
            try:
                return await store.transact_async(
                    lambda transaction, stale=stale: transact(transaction, stale),
                )
            except MessageConflict as error:
                existing = store.read(key)
                if existing is not None:
                    operation = decode_operation(existing.value)
                    if (operation.kind, operation.digest) != (kind, request_digest):
                        raise
                last_error = error
                continue
        raise MessageConflict(f"目标操作事务持续冲突：{last_error}")

    async def settle(
        self,
        session_id: str,
        *,
        expected_goal_id: str,
        expected_version: int,
        expected_head: int,
        settlement: RoundSettlement,
        next_message_id: str | Callable[[Goal], str] | None,
        next_text: str | Callable[[Goal], str] | None,
        allow_inactive: bool = False,
    ) -> Goal:
        store, reader, writer = self._access(session_id)

        def transact(transaction: OwnerTransaction) -> Goal:
            current = transaction.read(goal_key(session_id))
            if current is None or current.version != expected_version:
                raise MessageConflict("目标记录已变化")
            goal = decode_goal(session_id, current.value)
            if goal.goal_id != expected_goal_id:
                raise MessageConflict("目标已被替换")
            if not goal.active and not allow_inactive:
                raise MessageConflict("目标不是 active")
            if transaction.source_changed(reader, "conversation", expected_head):
                raise MessageConflict("对话来源在结算前已有新消息")
            updated = settle_round(
                goal,
                activity=settlement.activity,
                tokens=settlement.tokens,
                coverage=settlement.coverage,
                settled_through_seq=settlement.settled_through_seq,
                last_settled_message_id=settlement.last_settled_message_id,
                reason=settlement.reason,
                preempted=settlement.preempted,
                allow_inactive=allow_inactive,
            )
            if settlement.pause_after and updated.active:
                from .state import pause
                updated = pause(updated, settlement.reason or "用户停止目标轮次")
            if settlement.append and updated.active:
                if next_message_id is None or next_text is None:
                    raise ValueError("续跑输入缺少消息身份或正文")
                message_id = (
                    next_message_id(updated)
                    if callable(next_message_id) else next_message_id
                )
                text = next_text(updated) if callable(next_text) else next_text
                updated = mark_continuation(updated, message_id)
                _ = transaction.append(
                    writer, message_id,
                    Input((ContentPart("text", text),)),
                    expected_source_head=expected_head,
                )
            _ = transaction.save(
                goal_key(session_id), encode_goal(updated),
                expected_version=current.version,
            )
            return updated

        return await store.transact_async(transact)

    async def start(
        self,
        session_id: str,
        *,
        expected_goal_id: str,
        expected_version: int,
        expected_head: int,
        message_id: str,
        text: str,
    ) -> Goal:
        store, reader, writer = self._access(session_id)

        def transact(transaction: OwnerTransaction) -> Goal:
            current = transaction.read(goal_key(session_id))
            if current is None or current.version != expected_version:
                raise MessageConflict("目标记录已变化")
            goal = decode_goal(session_id, current.value)
            if goal.goal_id != expected_goal_id:
                raise MessageConflict("目标已被替换")
            if not goal.active:
                raise MessageConflict("目标不是 active")
            if transaction.source_changed(reader, "conversation", expected_head):
                raise MessageConflict("对话来源在续跑前已有新消息")
            updated = mark_continuation(goal, message_id)
            _ = transaction.append(
                writer, message_id,
                Input((ContentPart("text", text),)),
                expected_source_head=expected_head,
            )
            _ = transaction.save(
                goal_key(session_id), encode_goal(updated),
                expected_version=current.version,
            )
            return updated

        return await store.transact_async(transact)

    def _stale_operations(
        self, store: OwnerStore, *, reserve: int = 0,
    ) -> tuple[tuple[str, int], ...]:
        records = [
            (key, record) for key, record in store.list()
            if key.startswith(OPERATION_PREFIX)
        ]
        keep = MAX_OPERATIONS - reserve
        if len(records) <= keep:
            return ()
        decoded: list[tuple[datetime, str, int]] = []
        for key, record in records:
            try:
                decoded.append((decode_operation(record.value).created_at, key, record.version))
            except ValueError:
                continue
        decoded.sort(key=lambda item: item[0])
        return tuple(
            (key, version)
            for _created, key, version in decoded[: max(0, len(decoded) - keep)]
        )
