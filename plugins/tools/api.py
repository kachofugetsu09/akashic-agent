from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from session.log import MessageReader, MessageWriter, OwnerStore
from session.message import CallRef, ContentPart, Message, Output, ToolCall, ToolResult


Outcome = Literal["success", "denied", "error", "unknown"]


@dataclass(frozen=True, slots=True)
class Result:
    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "unknown"}:
            raise ValueError("工具结果状态无效")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("工具结果必须是内容块")
        object.__setattr__(self, "parts", parts)


@dataclass(frozen=True, slots=True)
class CallSource:
    """实际调用的不可变消息前缀；不携带 reader 或任何写入能力。"""

    call_ref: CallRef
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class MessageReply:
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
        self.writer.check(ToolResult(self.call_ref, "unknown", ()))

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


def display_name(metadata: Mapping[str, object]) -> str:
    """从原 binding 读取工具名称，不打开工具或暴露其恢复配置。"""
    tool = metadata.get("tool")
    if not isinstance(tool, Mapping):
        raise ValueError("工具 binding 描述无效")
    name = cast(Mapping[str, object], tool).get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("工具 binding 缺少工具名")
    return name


class InvalidArguments(ValueError):
    """工具明确拒绝请求参数；可以返回错误结果供调用者修正。"""


class Denied(Exception):
    """授权 owner 明确拒绝当前最终参数；没有发生本次调用。"""


class BoundTool(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object]: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result: ...

    async def query(self, key: str) -> Result | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


OpenTool = Callable[[str], AbstractAsyncContextManager[BoundTool]]
Authorize = Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]
