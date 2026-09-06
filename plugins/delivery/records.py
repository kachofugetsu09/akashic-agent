from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Annotated, Literal, Self, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from session.log import MessageConflict, MessageReader, MessageWriter, OwnerRecord, OwnerStore, OwnerTransaction
from session.message import Body, Message
from session.message_codec import json_value

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
    phase: Literal["prepared", "started", "delivered", "rejected", "unknown"]
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


class Cursor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    through_seq: Annotated[int, Field(ge=-1)]


def delivery_key(message_id: str, sink: str) -> str:
    return "delivery:" + json.dumps([message_id, sink], ensure_ascii=False, separators=(",", ":"))


class DeliveryRecords:
    """选路、prepared 与消费进度同事务提交；正文只按 Message 引用读取。"""

    def __init__(self, state: OwnerStore, recovery_owner: str):
        self._state = state
        self.recovery_owner = recovery_owner

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
        row = self._state.read(self._cursor_key(session_id))
        return -1 if row is None else Cursor.model_validate(dict(row.value)).through_seq

    def read(self, message_id: str, sink: str) -> tuple[OwnerRecord, Delivery]:
        _ = self.check_owner(message_id)
        row = self._state.read(delivery_key(message_id, sink))
        if row is None:
            raise ValueError("发送尚未 prepared")
        delivery = Delivery.model_validate_json(json.dumps(json_value(row.value)))
        if delivery.sink.name != sink:
            raise ValueError("发送目的地与记录身份不一致")
        return row, delivery

    def prepare(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...], *, passive: bool = False) -> Selection:
        """显式发送固定全体目的地，不替自动消费者跳过其他消息。"""
        sinks = tuple(Sink.model_validate(sink.model_dump()) for sink in sinks)
        self._check_message(reader, message, sinks)
        def commit(tx: OwnerTransaction) -> Selection:
            existing = self.selection(message.message_id)
            if existing is not None:
                return self.check_owner(message.message_id)
            return self._prepare(tx, message, sinks, passive=passive)
        return self._state.transact(commit)

    def publish(self, writer: MessageWriter, message_id: str, body: Body,
                sinks: tuple[Sink, ...], *, passive: bool = False) -> tuple[Message, Selection]:
        """已授权 writer 的新消息与首次选路同事务，崩溃不会留下可被抢选的通知。"""
        sinks = tuple(Sink.model_validate(sink.model_dump()) for sink in sinks)
        if len({sink.name for sink in sinks}) != len(sinks):
            raise ValueError("一次选路不能重复同一目的地")
        def commit(tx: OwnerTransaction) -> tuple[Message, Selection]:
            message = tx.append(writer, message_id, body)
            existing = self.selection(message_id)
            selected = self.check_owner(message_id) if existing is not None else self._prepare(tx, message, sinks, passive=passive)
            return message, selected
        return self._state.transact(commit)

    def _check_message(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...]) -> None:
        self._state.check_access(reader)
        if reader.session_id != message.session_id or reader.get(message.message_id) != message:
            raise ValueError("发送必须引用本 Session 已提交的原消息")
        if len({sink.name for sink in sinks}) != len(sinks):
            raise ValueError("一次选路不能重复同一目的地")

    def consume(self, reader: MessageReader, message: Message, sinks: tuple[Sink, ...] | None,
                *, passive: bool = False) -> Selection | None:
        """按序消费；None 表示不拥有选路权，空集合是明确的零目标选择。"""
        if sinks is not None:
            sinks = tuple(Sink.model_validate(sink.model_dump()) for sink in sinks)
        self._check_message(reader, message, () if sinks is None else sinks)

        def commit(tx: OwnerTransaction) -> Selection | None:
            cursor = self.cursor(reader.session_id)
            if message.seq <= cursor:
                existing = self.selection(message.message_id)
                if existing is None and sinks is not None:
                    raise ValueError("已消费消息没有本次要求的发送选择")
                return existing
            following = reader.read(after_seq=cursor, limit=1)
            if not following or following[0] != message:
                raise MessageConflict("发送消费不能跳过消息")
            selection = (self.selection(message.message_id) if sinks is None else
                         self._prepare(tx, message, sinks, passive=passive))
            # 所属目的地已 prepared 后才推进 cursor；其他来源保留自己的选路权。
            key = self._cursor_key(reader.session_id)
            previous = tx.read(key)
            _ = tx.save(key, {"through_seq": message.seq}, expected_version=None if previous is None else previous.version)
            return selection

        return self._state.transact(commit)

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

    def add(self, message_id: str, sink: Sink) -> None:
        """显式新增一个目的地；相同发送键不能改绑地址或旧 generation。"""
        sink = Sink.model_validate(sink.model_dump())
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

    def save(self, message_id: str, previous: OwnerRecord, delivery: Delivery) -> OwnerRecord:
        """真实回执与首个送达时间索引同事务提交；旧回执不补造历史时间。"""
        selection = self.check_owner(message_id)
        old = Delivery.model_validate_json(json.dumps(json_value(previous.value)))
        if old.phase == "delivered" and delivery != old:
            raise MessageConflict("已送达回执不能改写")
        if delivery.phase == "delivered" and old.phase != "delivered":
            delivery = delivery.model_copy(update={"confirmed_at": datetime.now(timezone.utc)})

        def commit(tx: OwnerTransaction) -> OwnerRecord:
            row = tx.save(delivery_key(message_id, delivery.sink.name),
                          cast(Mapping[str, object], delivery.model_dump(mode="json")),
                          expected_version=previous.version)
            key = "confirmed-message:" + message_id
            if delivery.confirmed_at is not None and tx.read(key) is None:
                confirmation = Confirmation(message_id=message_id, session_id=selection.session_id,
                                            confirmed_at=delivery.confirmed_at)
                value = confirmation.model_dump(mode="json")
                _ = tx.save(key, value, expected_version=None)
                _ = tx.save(time_key(delivery.confirmed_at) + message_id, value, expected_version=None)
            return row

        return self._state.transact(commit)

    def pending(self) -> tuple[tuple[str, str], ...]:
        """恢复只枚举已固定的效果；不因新策略新增目的地。"""
        pending: list[tuple[str, str]] = []
        for key, _row in self._state.list():
            if not key.startswith("delivery:"):
                continue
            identities: object = json.loads(key[len("delivery:"):])
            if not isinstance(identities, list):
                raise ValueError("发送记录身份损坏")
            values = cast(list[object], identities)
            if len(values) != 2 or any(not isinstance(value, str) or not value for value in values):
                raise ValueError("发送记录身份损坏")
            message_id, sink = cast(list[str], identities)
            selection = self.selection(message_id)
            if selection is None:
                raise ValueError("发送缺少首次选择")
            if selection.recovery_owner != self.recovery_owner:
                continue
            _, delivery = self.read(message_id, sink)
            if delivery.phase not in {"delivered", "rejected"}:
                pending.append((message_id, sink))
        return tuple(pending)
