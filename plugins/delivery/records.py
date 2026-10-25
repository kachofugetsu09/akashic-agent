from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Annotated, Literal, Protocol, Self, cast, runtime_checkable

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from plugins.ledger.contract import MessageConflict, MessageReader, MessageWriter, OwnerRecord, OwnerStore, OwnerTransaction
from plugins.ledger.contract import Body, Message, Output
from plugins.ledger.contract import json_value
from core.common.file_io import run_file_io

from .api import Receipt, Sink, Text
from .history import Confirmation, time_key


class Selection(BaseModel):
    """首次策略选择的完整集合；显式追加目的地不改写这个事实。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    session_id: Text
    recovery_owner: Text
    passive: bool
    sinks: tuple[Text, ...]


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    version: Literal[1] = 1
    sink: Sink
    phase: Literal["prepared", "started", "delivered", "rejected", "failed"]
    receipt: Receipt | None = None
    confirmed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def check_receipt(self) -> Self:
        if self.phase in {"prepared", "started"}:
            if self.receipt is not None:
                raise ValueError("未结算发送不能包含终态回执")
        elif self.receipt is None or self.receipt.status != self.phase:
            raise ValueError("发送阶段与回执不一致")
        if self.confirmed_at is not None and self.phase != "delivered":
            raise ValueError("只有已送达消息可以包含确认时间")
        return self


def read_delivery(value: Mapping[str, object]) -> Delivery:
    """只在持久读取边界解释旧失败状态，不改写原存储。"""
    data = dict(value)
    if data.get("phase") == "unknown":
        receipt = data.get("receipt")
        if not isinstance(receipt, Mapping) or receipt.get("status") != "unknown":
            raise ValueError("旧发送阶段与回执不一致")
        data = {**data, "phase": "failed", "receipt": {**receipt, "status": "failed"}}
    return Delivery.model_validate_json(json.dumps(json_value(data)))


class Cursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    through_seq: Annotated[int, Field(ge=-1)]


def delivery_key(message_id: str, sink: str) -> str:
    return "delivery:" + json.dumps([message_id, sink], ensure_ascii=False, separators=(",", ":"))


# 未终结发送的索引与发送记录同事务维护；恢复只读这个小区间，不随历史增长。
_OPEN_PREFIX = "open:"
_OPEN_STOP = "open;"
_OPEN_INDEX = "open-index"
_TERMINAL = frozenset({"delivered", "rejected", "failed"})
# 不可选路消息只推进进程内游标；领先耐久值满这么多条才写回，重启重读量因此有界。
_UNROUTABLE_PERSIST_GAP = 100


def _open_key(message_id: str, sink: str) -> str:
    return _OPEN_PREFIX + json.dumps([message_id, sink], ensure_ascii=False, separators=(",", ":"))


def _identity(key: str, prefix: str) -> tuple[str, str]:
    identities: object = json.loads(key[len(prefix):])
    if not isinstance(identities, list):
        raise ValueError("发送记录身份损坏")
    values = cast(list[object], identities)
    if len(values) != 2 or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("发送记录身份损坏")
    message_id, sink = cast(list[str], identities)
    return message_id, sink


class DeliveryRecords:
    """选路、prepared 与消费进度同事务提交；正文只按 Message 引用读取。"""

    def __init__(self, state: OwnerStore, recovery_owner: str):
        self._state = state
        self.recovery_owner = recovery_owner
        # cursor 只由本 owner 的消费事务单调推进；缓存只可能滞后，滞后由消费路径的
        # 已消费分支兼容（读回既有选择，不重复发送）。
        self._cursor_cache: dict[str, int] = {}
        # 已知的耐久 cursor；本进程首次越过不可选路消息的会话也写回一次，
        # 否则从不选路的会话每次启动都要从头重读全部历史。
        self._durable_cursor: dict[str, int] = {}
        self._unroutable_seen: set[str] = set()

    def selection(self, message_id: str) -> Selection | None:
        row = self._state.read("selection:" + message_id)
        if row is None:
            return None
        if row.version != 0:
            raise ValueError("首次发送选择不能被改写")
        return Selection.model_validate_json(json.dumps(json_value(row.value)))

    def _cursor_key(self, session_id: str) -> str:
        return "cursor:" + json.dumps([self.recovery_owner, session_id], separators=(",", ":"))

    def check_owner(self, message_id: str) -> Selection:
        selection = self.selection(message_id)
        if selection is None:
            raise ValueError("发送缺少首次选择")
        if selection.recovery_owner != self.recovery_owner:
            raise PermissionError("发送属于另一个恢复 owner")
        return selection

    def cursor(self, session_id: str) -> int:
        cached = self._cursor_cache.get(session_id)
        if cached is not None:
            return cached
        row = self._state.read(self._cursor_key(session_id))
        through = -1 if row is None else Cursor.model_validate(dict(row.value)).through_seq
        self._cursor_cache[session_id] = through
        self._durable_cursor[session_id] = through
        return through

    def read(self, message_id: str, sink: str) -> tuple[OwnerRecord, Delivery]:
        _ = self.check_owner(message_id)
        row = self._state.read(delivery_key(message_id, sink))
        if row is None:
            raise ValueError("发送尚未 prepared")
        delivery = read_delivery(row.value)
        if delivery.sink.name != sink:
            raise ValueError("发送目的地与记录身份不一致")
        return row, delivery

    def _prepare_message(
        self,
        reader: MessageReader,
        message: Message,
        sinks: tuple[Sink, ...],
        *,
        passive: bool = False,
    ) -> Selection:
        """显式发送固定全体目的地，不替自动消费者跳过其他消息。"""
        self._check_message(reader, message, sinks)
        def commit(tx: OwnerTransaction) -> Selection:
            existing = self.selection(message.message_id)
            if existing is not None:
                return self.check_owner(message.message_id)
            return self._prepare(tx, message, sinks, passive=passive)
        return self._state.transact(commit)


    async def prepare_async(self, reader: MessageReader, message: Message,
                            sinks: tuple[Sink | Mapping[str, object], ...], *, passive: bool = False) -> Selection:
        fixed = _normalize_sinks(sinks)
        return await run_file_io(lambda: self._prepare_message(reader, message, fixed, passive=passive))

    async def publish_async(self, writer: MessageWriter, message_id: str, body: Body,
                            sinks: tuple[Sink | Mapping[str, object], ...], *, passive: bool = False) -> tuple[Message, Selection]:
        """原 scope 准备正文，SQL 事务共同提交消息与首次目的地。"""
        fixed = _normalize_sinks(sinks)
        if len({sink.name for sink in fixed}) != len(fixed):
            raise ValueError("一次选路不能重复同一目的地")
        prepared = await writer.prepare_async(message_id, body)

        def commit(tx: OwnerTransaction) -> tuple[Message, Selection]:
            message = tx.append_prepared(prepared)
            existing = self.selection(message_id)
            selected = self.check_owner(message_id) if existing is not None else self._prepare(tx, message, fixed, passive=passive)
            return message, selected
        return await self._state.transact_async(commit)

    async def consume_async(self, reader: MessageReader, message: Message,
                            sinks: tuple[Sink | Mapping[str, object], ...] | None, *, passive: bool = False) -> Selection | None:
        fixed = None if sinks is None else _normalize_sinks(sinks)
        return await run_file_io(lambda: self._consume_message(reader, message, fixed, passive=passive))

    async def consume_batch_async(self, reader: MessageReader,
                                  items: tuple[tuple[Message, tuple[Sink | Mapping[str, object], ...] | None], ...],
                                  *, passive: bool = False) -> tuple[Selection | None, ...]:
        """同一批消息的消费与 cursor 推进一次事务提交；顺序与冲突语义与逐条 consume 一致。"""
        fixed = tuple((message, None if sinks is None else _normalize_sinks(sinks)) for message, sinks in items)
        return await run_file_io(lambda: self._consume_batch(reader, fixed, passive=passive))

    async def add_async(self, message_id: str, sink: Sink | Mapping[str, object]) -> None:
        fixed = _normalize_sink(sink)
        await run_file_io(lambda: self._add_destination(message_id, fixed))

    def _check_message(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...]) -> None:
        self._state.check_access(reader)
        if reader.session_id != message.session_id or reader.get(message.message_id) != message:
            raise ValueError("发送必须引用本 Session 已提交的原消息")
        if len({sink.name for sink in sinks}) != len(sinks):
            raise ValueError("一次选路不能重复同一目的地")

    def _consume_message(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...] | None,
                *, passive: bool = False) -> Selection | None:
        """按序消费；None 表示不拥有选路权，空集合是明确的零目标选择。"""
        return self._consume_batch(reader, ((message, sinks),), passive=passive)[0]

    def _consume_batch(self, reader: MessageReader,
                       items: tuple[tuple[Message, tuple[Sink, ...] | None], ...],
                       *, passive: bool = False) -> tuple[Selection | None, ...]:
        # 1. 不可能选路的消息只推进游标，按 _advance_unroutable 的条件写回，不进本批事务。
        if self._advance_unroutable(reader, items):
            return (None,) * len(items)
        for message, sinks in items:
            self._check_message(reader, message, () if sinks is None else sinks)

        committed_through: list[int] = []

        def commit(tx: OwnerTransaction) -> tuple[Selection | None, ...]:
            # cursor 写回只由本事务内的持久值决定：内存缓存只服务读路径，
            # 不得参与决定写回什么（单调性由 expected_version 冲突重试保证）。
            key = self._cursor_key(reader.session_id)
            persisted = tx.read(key)
            persisted_through = (-1 if persisted is None else
                                 Cursor.model_validate(dict(persisted.value)).through_seq)
            # 进程内游标只会越过不可选路消息，可作为本事务的起点。
            through = max(persisted_through, self._cursor_cache.get(reader.session_id, -1))
            selections: list[Selection | None] = []
            for message, sinks in items:
                if message.seq <= through:
                    existing = self.selection(message.message_id)
                    if existing is None and sinks is not None:
                        raise ValueError("已消费消息没有本次要求的发送选择")
                    selections.append(existing)
                    continue
                following = reader.read(after_seq=through, limit=1)
                if not following or following[0] != message:
                    raise MessageConflict("发送消费不能跳过消息")
                selection = (self.selection(message.message_id) if sinks is None else
                             self._prepare(tx, message, sinks, passive=passive))
                through = message.seq
                selections.append(selection)
            # 所属目的地已 prepared 后才推进 cursor；其他来源保留自己的选路权。
            if through > persisted_through:
                _ = tx.save(key, {"through_seq": through}, expected_version=None if persisted is None else persisted.version)
                committed_through.append(through)
            return tuple(selections)

        selections = self._state.transact(commit)
        if committed_through:
            self._cursor_cache[reader.session_id] = committed_through[-1]
            self._durable_cursor[reader.session_id] = committed_through[-1]
        return selections

    # 只有 complete Output 可能被选路；其余消息在不拥有选路权时无需持久化消费事实。
    def _advance_unroutable(self, reader: MessageReader,
                            items: tuple[tuple[Message, tuple[Sink, ...] | None], ...]) -> bool:
        if not items or any(sinks is not None or (isinstance(message.body, Output) and message.body.finish == "complete")
                            for message, sinks in items):
            return False
        through = self.cursor(reader.session_id)
        if items[0][0].seq <= through:
            return False
        following = reader.read(after_seq=through, limit=1)
        if not following or following[0].message_id != items[0][0].message_id:
            raise MessageConflict("发送消费不能跳过消息")
        session_id = reader.session_id
        through = items[-1][0].seq
        self._cursor_cache[session_id] = through
        if (session_id not in self._unroutable_seen
                or through - self._durable_cursor.get(session_id, -1) >= _UNROUTABLE_PERSIST_GAP):
            self._unroutable_seen.add(session_id)
            self._persist_cursor(session_id, through)
        return True

    # 只写回已越过的不可选路前缀；与消费事务一样按耐久值单调推进。
    def _persist_cursor(self, session_id: str, through: int) -> None:
        key = self._cursor_key(session_id)

        def commit(tx: OwnerTransaction) -> int:
            persisted = tx.read(key)
            persisted_through = (-1 if persisted is None else
                                 Cursor.model_validate(dict(persisted.value)).through_seq)
            if through <= persisted_through:
                return persisted_through
            _ = tx.save(key, {"through_seq": through}, expected_version=None if persisted is None else persisted.version)
            return through

        self._durable_cursor[session_id] = self._state.transact(commit)

    def _prepare(self, tx: OwnerTransaction, message: Message, sinks: tuple[Sink, ...], *, passive: bool = False) -> Selection:
        """同一消息的首次选路不可变；重复策略计算只采用原集合。"""
        existing = self.selection(message.message_id)
        if existing is not None:
            if existing.session_id != message.session_id:
                raise ValueError("发送选择属于另一个 Session")
            return existing
        selection = Selection(session_id=message.session_id, recovery_owner=self.recovery_owner, passive=passive, sinks=tuple(sink.name for sink in sinks))
        _ = tx.save("selection:" + message.message_id, selection.model_dump(mode="json"), expected_version=None)
        for sink in sinks:
            self._add(tx, message.message_id, sink)
        return selection

    def _add_destination(self, message_id: str, sink: Sink) -> None:
        """显式新增一个目的地；相同发送键不能改绑地址或旧 generation。"""
        def commit(tx: OwnerTransaction) -> None:
            _ = self.check_owner(message_id)
            self._add(tx, message_id, sink)
        self._state.transact(commit)

    def _add(self, tx: OwnerTransaction, message_id: str, sink: Sink) -> None:
        key = delivery_key(message_id, sink.name)
        previous = tx.read(key)
        if previous is not None:
            _, existing = self.read(message_id, sink.name)
            if existing.sink != sink:
                raise MessageConflict("原发送不能更换目的地或 binding")
            return
        value = Delivery(sink=sink, phase="prepared")
        _ = tx.save(key, value.model_dump(mode="json"), expected_version=None)
        self._sync_open(tx, message_id, sink.name, value.phase)

    # 索引项只表达"该发送尚未终结"；重试回到 prepared 时重新加入。
    def _sync_open(self, tx: OwnerTransaction, message_id: str, sink: str, phase: str) -> None:
        key = _open_key(message_id, sink)
        row = tx.read(key)
        if phase in _TERMINAL:
            if row is not None:
                tx.delete(key, expected_version=row.version)
        elif row is None:
            _ = tx.save(key, {}, expected_version=None)


    async def save_async(self, message_id: str, previous: OwnerRecord, delivery: Delivery, *,
                         on_commit: Callable[[OwnerRecord], None] | None = None) -> OwnerRecord:
        return await self._state.transact_async(
            lambda tx: self._save(tx, message_id, previous, delivery), on_commit=on_commit,
        )

    async def cancel_prepared_async(self, message_id: str, sink: str, receipt: Receipt, *,
                                    on_commit: Callable[[bool], None]) -> bool:
        """同事务判定并撤回 prepared，不能把已 started 的真实效果改成 rejected。"""
        def commit(tx: OwnerTransaction) -> bool:
            previous, delivery = self.read(message_id, sink)
            if delivery.phase != "prepared":
                return False
            self._save(tx, message_id, previous, Delivery(sink=delivery.sink, phase="rejected", receipt=receipt))
            return True
        return await self._state.transact_async(commit, on_commit=on_commit)

    def _save(self, tx: OwnerTransaction, message_id: str, previous: OwnerRecord, delivery: Delivery) -> OwnerRecord:
        """真实回执与首个送达时间索引同事务提交；旧回执不补造历史时间。"""
        selection = self.check_owner(message_id)
        old = read_delivery(previous.value)
        if old.phase == "delivered" and delivery != old:
            raise MessageConflict("已送达回执不能改写")
        if delivery.phase == "delivered" and old.phase != "delivered":
            delivery = delivery.model_copy(update={"confirmed_at": datetime.now(timezone.utc)})

        row = tx.save(delivery_key(message_id, delivery.sink.name),
                      cast(Mapping[str, object], delivery.model_dump(mode="json")),
                      expected_version=previous.version)
        self._sync_open(tx, message_id, delivery.sink.name, delivery.phase)
        key = "confirmed-message:" + message_id
        if delivery.confirmed_at is not None and tx.read(key) is None:
            confirmation = Confirmation(message_id=message_id, session_id=selection.session_id,
                                        confirmed_at=delivery.confirmed_at)
            value = confirmation.model_dump(mode="json")
            _ = tx.save(key, value, expected_version=None)
            _ = tx.save(time_key(delivery.confirmed_at) + message_id, value, expected_version=None)
        return row

    def pending(self) -> tuple[tuple[str, str], ...]:
        """恢复只枚举已固定的效果；不因新策略新增目的地。"""
        # 1. 升级后首次恢复按全部发送记录补齐索引，此后只读索引区间。
        if self._state.read(_OPEN_INDEX) is None:
            self._state.transact(self._backfill_open)
        keys: list[str] = []
        stop = _OPEN_STOP
        while page := self._state.scan(start=_OPEN_PREFIX, stop=stop, limit=1000):
            keys.extend(key for key, _row in page)
            stop = page[-1][0]
        # 2. 索引只缩小候选；归属与阶段仍以原记录为准。
        pending: list[tuple[str, str]] = []
        for key in sorted(keys):
            message_id, sink = _identity(key, _OPEN_PREFIX)
            selection = self.selection(message_id)
            if selection is None:
                raise ValueError("发送缺少首次选择")
            if selection.recovery_owner != self.recovery_owner:
                continue
            _, delivery = self.read(message_id, sink)
            if delivery.phase not in _TERMINAL:
                pending.append((message_id, sink))
        return tuple(pending)

    # 在写事务内按全部发送记录重建索引，并以标记记录宣告索引完整。
    def _backfill_open(self, tx: OwnerTransaction) -> None:
        if tx.read(_OPEN_INDEX) is not None:
            return
        for key, row in self._state.list():
            if key.startswith("delivery:"):
                message_id, sink = _identity(key, "delivery:")
                self._sync_open(tx, message_id, sink, read_delivery(row.value).phase)
        _ = tx.save(_OPEN_INDEX, {"version": 1}, expected_version=None)


@runtime_checkable
class _SinkValue(Protocol):
    """允许归档 generation 的同形目的地跨 owner 边界进入校验。"""

    name: str
    binding_id: str
    address: str


def _normalize_sinks(sinks: tuple[Sink | Mapping[str, object] | _SinkValue, ...]) -> tuple[Sink, ...]:
    """在 Delivery owner 边界把消费者结构重新校验成内部 Sink。"""
    return tuple(_normalize_sink(sink) for sink in sinks)


def _normalize_sink(value: Sink | Mapping[str, object] | _SinkValue) -> Sink:
    """接纳 Mapping 或跨 generation 同形值，再由 owner 重新校验。"""
    if isinstance(value, Sink):
        return value
    if isinstance(value, Mapping):
        return Sink.model_validate(dict(value))
    if isinstance(value, _SinkValue):
        return Sink.model_validate({
            "name": value.name,
            "binding_id": value.binding_id,
            "address": value.address,
        })
    raise TypeError("发送目的地必须是结构映射")
