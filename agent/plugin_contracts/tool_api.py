"""工具 ABI 的公开结构合同。

工具插件与消费者共享一套调用 ABI：结果值、调用来源、拒绝/拒绝理由、以及
`BoundTool` 协议。这些只引用消息词汇表，不持有存储或 I/O，因此由合同层拥有，
插件实现（`plugins/tools/...`）按结构满足。

`MessageReply` 需要会话读写端口，属于存储 seam，仍留在 `plugins/tools/api.py`。
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping, Protocol, cast, runtime_checkable

from agent.plugin_contracts.message import CallRef, ContentPart, Message, ToolCall

if TYPE_CHECKING:
    # 仅类型标注用；合同层运行时不依赖组合内核的模型词汇。
    from agent.plugin_composition.models import ModelToolCall

Outcome = Literal["success", "denied", "error", "interrupted"]


def result_message_id(call_ref: CallRef) -> str:
    """调用结果的默认消息身份；恢复消费者与普通程序共用。"""
    return f"tool-result:{call_ref.message_id}:{call_ref.part_index}"


def durable_call_key(call_ref: CallRef) -> str:
    """返回一次已提交 ToolCall 的稳定 effect key。"""
    if not isinstance(call_ref, CallRef):
        raise TypeError("工具调用引用无效")
    return "message:" + json.dumps(
        [call_ref.message_id, call_ref.part_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class Result:
    """一次工具调用的结算结果。"""

    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
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
    """一次已绑定工具的执行接口。"""

    @property
    def idempotent(self) -> bool:
        """同一 binding 与参数是否可安全重放。"""
        ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object]:
        """规范化最终参数；不产生外部效果。"""
        ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        """执行调用并返回不可变结果。"""
        ...

    async def query(self, key: str) -> Result | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


OpenTool = Callable[[str], AbstractAsyncContextManager[BoundTool]]
Authorize = Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object]]]


@runtime_checkable
class MessageReplyPort(Protocol):
    """已获授的调用结果写入位置；实现由 `plugins/tools/api.py` 提供。"""

    def request(self) -> ToolCall:
        """读取已提交请求。"""
        ...

    def source(self) -> CallSource:
        """返回调用的不可变消息前缀。"""
        ...

    def check(self, state: object) -> None:
        """核对调用仍属于当前 owner 状态。"""
        ...

    def abandoned(self) -> bool:
        """调用是否已被放弃。"""
        ...

    def read(self, pointer: object) -> Result:
        """读取已有结果。"""
        ...


class InvalidToolCall(ValueError):
    """模型调用不符合当前展示协议；可反馈模型纠正，不代表工具效果。"""


class ToolPresentation(Protocol):
    """定义一次程序固定的 schema、wire 解码和系统提示词。"""

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...

    @property
    def system_prompt(self) -> str: ...

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]]: ...

    def configuration(self, name: str) -> Mapping[str, object] | None: ...


class NativePresentation:
    """把固定 binding 描述直接展示给模型。"""

    def __init__(self, descriptions: Mapping[str, Mapping[str, object]]):
        self._descriptions = dict(descriptions)

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(tool_schema(self._descriptions[name]) for name in self._descriptions)

    @property
    def system_prompt(self) -> str:
        return ""

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]]:
        if call.name not in self._descriptions:
            raise InvalidToolCall(f"工具不属于获授 view: {call.name}；请使用当前工具目录。")
        return call.name, cast(Mapping[str, object], call.arguments)

    def configuration(self, name: str) -> Mapping[str, object] | None:
        return None


def tool_schema(description: Mapping[str, object]) -> Mapping[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": description["name"],
            "description": description["description"],
            "parameters": description["parameters"],
        },
    }
