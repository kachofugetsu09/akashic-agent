from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol, cast

from agent.plugin_composition.messages import MessageReader, MessageWriter, OwnerStore
from agent.plugin_contracts import CallRef, ContentPart, Control, Message, Output, ToolCall, ToolResult
from agent.plugin_contracts.tool_api import (
    Authorize,
    BoundTool,
    CallSource,
    Denied,
    InvalidArguments,
    OpenTool,
    Outcome,
    Result,
    display_name,
    durable_call_key,
    MessageReplyPort,
    result_message_id,
)

@dataclass(frozen=True, slots=True)
class MessageReply(MessageReplyPort):
    """已获授的调用结果写入位置；独立程序调用不需要它。"""

    message_id: str
    call_ref: CallRef
    reader: MessageReader
    writer: MessageWriter
    check_start: Callable[[], None]

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, str) or not self.message_id:
            raise ValueError("工具结果 message_id 不能为空")
        if not isinstance(self.call_ref, CallRef):
            raise TypeError("工具结果必须引用 CallRef")

    def request(self) -> ToolCall:
        """直接读取已提交请求，调用者不能另外指定 binding、参数或执行 key。"""
        if self.reader.session_id != self.writer.session_id:
            raise ValueError("结果 reader 与 writer 的 Session 不一致")
        message = self.reader.get(self.call_ref.message_id)
        if message is None or not isinstance(message.body, Output):
            raise ValueError("工具调用消息缺失")
        if self.call_ref.part_index >= len(message.body.parts):
            raise ValueError("工具调用位置不存在")
        call = message.body.parts[self.call_ref.part_index]
        if not isinstance(call, ToolCall):
            raise ValueError("调用引用不指向 ToolCall")
        return call

    def source(self) -> CallSource:
        """只取实际请求的日志前缀，后来接纳的输入不能改变参数准备。"""
        message = self.reader.get(self.call_ref.message_id)
        if message is None:
            raise ValueError("工具调用消息缺失")
        return CallSource(self.call_ref, self.reader.snapshot(through_seq=message.seq))

    def check(self, state: OwnerStore) -> None:
        state.check_access(self.reader, self.writer)
        self.writer.check(ToolResult(self.call_ref, "error", ()))

    def abandoned(self) -> bool:
        """放弃由同来源的持久前缀决定，普通取消不代表放弃。"""
        call = self.reader.get(self.call_ref.message_id)
        if call is None:
            raise ValueError("工具调用消息缺失")
        return any(message.source == call.source and isinstance(message.body, Control)
                   and message.body.action == "abandon" and message.body.through_seq >= call.seq
                   for message in self.reader.snapshot(after_seq=call.seq))

    def read(self, pointer: object) -> Result:
        """按持久指针读取正文，不在工具回执中保留第二份结果。"""
        if not isinstance(pointer, Mapping):
            raise ValueError("工具结果指针损坏")
        value = cast(Mapping[str, object], pointer)
        if (
            set(value) != {"message_id", "seq"}
            or value["message_id"] != self.message_id
        ):
            raise ValueError("工具结果指针不匹配")
        seq = value["seq"]
        if type(seq) is not int or seq < 0:
            raise ValueError("工具结果序号无效")
        messages = self.reader.read(after_seq=seq - 1, through_seq=seq, limit=1)
        if len(messages) != 1 or messages[0].message_id != self.message_id:
            raise ValueError("工具结果消息缺失")
        body = messages[0].body
        if not isinstance(body, ToolResult) or body.call_ref != self.call_ref:
            raise ValueError("工具结果不属于原调用")
        return Result(body.outcome, body.parts)


