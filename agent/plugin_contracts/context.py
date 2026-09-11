"""上下文装配能力的公开结构合同。

预置上下文（system prompt + reminders + summary + references）与摘要 reducer
是 Core 与多个业务插件共用的结构边界，因此值模型与 Protocol 由合同层拥有，
实现留在 `plugins/context`。

`ContextModel` 需要引用模型请求词汇；本模块只在类型检查期引用
`agent.plugin_composition.models`，运行时不依赖它。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.content import Reference
from agent.plugin_contracts.message import (
    CallRef,
    ContentPart,
    ContentReferences,
    Control,
    Message,
    Output,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from agent.plugin_composition.models import BoundChatModel, ModelRequest


def settled_prefixes(messages: tuple[Message, ...]) -> tuple[int, ...]:
    """返回工具已结算或被明确放弃的前缀长度；不伪造任何工具结果。"""
    pending: dict[CallRef, Message] = {}
    ends: list[int] = []
    for index, message in enumerate(messages):
        body = message.body
        if isinstance(body, Output):
            pending.update((CallRef(message.message_id, pos), message) for pos, part in enumerate(body.parts)
                           if isinstance(part, ToolCall))
        elif isinstance(body, ToolResult):
            _ = pending.pop(body.call_ref, None)
        elif isinstance(body, Control) and body.action == "abandon":
            pending = {ref: call for ref, call in pending.items()
                       if call.source != message.source or call.seq > body.through_seq}
        if not pending:
            ends.append(index + 1)
    return tuple(ends)


def check_summary(part: ContentPart) -> ContentReferences:
    """摘要使用记录只引用已固定的来源；内容 writer 不接受内联摘要。"""
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("context.summary 必须是对象")
    value = cast(Mapping[str, object], value)
    if (set(value) != {"reference"} or not isinstance(value["reference"], str) or not value["reference"]):
        raise ValueError("context.summary 必须包含唯一的摘要 binding 引用")
    return ContentReferences(binding_ids=(value["reference"],))


def summary_range(snapshot: tuple[Message, ...], source_message_ids: tuple[str, ...]) -> range:
    """按真实身份定位摘要的连续区间；窗口外旧消息不冒充摘要来源。"""
    identities = tuple(message.message_id for message in snapshot)
    if not source_message_ids or source_message_ids[0] not in identities:
        raise ValueError("摘要来源缺少实际消息")
    start = identities.index(source_message_ids[0])
    end = start + len(source_message_ids)
    if identities[start:end] != source_message_ids:
        raise ValueError("摘要来源不等于实际连续消息范围")
    return range(start, end)


@dataclass(frozen=True, slots=True)
class Summary:
    """摘要 owner 已持久发布的内容与精确覆盖范围。"""

    reference: str
    source_message_ids: tuple[str, ...]
    content: str

    def __post_init__(self) -> None:
        ids = tuple(self.source_message_ids)
        if not self.reference or not isinstance(self.reference, str):
            raise ValueError("摘要必须有持久来源引用")
        if not ids or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("摘要必须声明覆盖的消息")
        if len(set(ids)) != len(ids):
            raise ValueError("摘要消息引用不能重复")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("摘要正文不能为空")
        object.__setattr__(self, "source_message_ids", ids)


@dataclass(frozen=True, slots=True)
class Reminder:
    """一次请求中的具名文本；priority 只决定显示顺序，不授予权限。"""

    name: str
    text: str
    priority: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("提醒必须有稳定的局部名称")
        if not isinstance(self.text, str):
            raise TypeError("提醒正文必须是字符串")
        if type(self.priority) is not int:
            raise TypeError("提醒 priority 必须是整数")


@dataclass(frozen=True, slots=True)
class Materials:
    """权限已由组合确定的 Prompt，以及保持低信任的检索材料。"""

    system_prompt: str
    reminders: tuple[Reminder, ...] = ()
    summary: Summary | None = None
    references: tuple[Reference, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("system Prompt 必须是字符串")
        parts = tuple(self.reminders)
        if any(not isinstance(part, Reminder) for part in parts):
            raise TypeError("提醒必须是已校验的文本块")
        if self.summary is not None and not isinstance(self.summary, Summary):
            raise TypeError("摘要必须来自已发布的 Summary")
        references = tuple(self.references)
        if any(not isinstance(ref, Reference) for ref in references):
            raise TypeError("引用必须是材料 owner 已取得的 Reference")
        object.__setattr__(self, "reminders", parts)
        object.__setattr__(self, "references", references)


class ContextModel(Protocol):
    """Model 的只读请求投影；这里没有 complete 或工具执行权。"""

    @property
    def context_window(self) -> int | None: ...

    @property
    def max_tool_schemas(self) -> int | None: ...

    def render(self, messages: tuple[Message, ...], *, after_seq: int,
               summary_reference: str | None = None, fresh: bool = False) -> ModelRequest:
        """接收完整事实；after_seq 是摘要覆盖末尾，-1 表示没有覆盖。

        fresh 明确从选定近期窗口开始新请求，不接续旧 opaque 状态。
        summary_reference 明确要求从这份摘要开始新请求；只有同一摘要下的
        后续成功响应才接续 opaque state。只给 after_seq 不授权丢弃 replay。
        """
        ...

    def estimate(self, request: ModelRequest) -> int: ...


class SummaryReducer(Protocol):
    """摘要 owner 先持久发布再返回；None 表示保留已有摘要，不做缩减。"""

    async def __call__(
        self, snapshot: tuple[Message, ...], materials: Materials,
        request: ModelRequest, model: BoundChatModel, projection: ContextModel,
        *, source: str, force: bool,
    ) -> Summary | None: ...


class ContextOverflow(ValueError):
    def __init__(self, estimated_tokens: int, output_tokens: int, capacity: int,
                 *, request: ModelRequest):
        self.estimated_tokens = estimated_tokens
        self.output_tokens = output_tokens
        self.capacity = capacity
        self.request = request
        super().__init__(
            f"请求需要约 {estimated_tokens}+{output_tokens} tokens，容量 {capacity}"
        )



@runtime_checkable
class MaterialViewPort(Protocol):
    """固定本次请求贡献者后的材料收集 view。"""

    def close(self) -> None:
        """关闭 view，之后不再接受 prepare。"""
        ...

    async def prepare(
        self,
        snapshot: tuple[Message, ...],
        source: str,
        *,
        caller: object | None = None,
        reminders: tuple[Reminder, ...] = (),
    ) -> Materials:
        """按固定贡献者收集预置上下文。"""
        ...


@runtime_checkable
class ContextMaterialsPort(Protocol):
    """预置上下文材料注册表对消费者可见的方法子集。"""

    async def register(
        self,
        ctx: object,
        *,
        name: str,
        prepare: object,
        priority: int = 0,
        prompt: bool = False,
        reduce: object | None = None,
    ) -> object:
        """登记一个材料来源；同名称只有一个真实 owner。"""
        ...

    def bind(
        self, *, exclude: frozenset[str] = frozenset()
    ) -> AbstractAsyncContextManager[MaterialViewPort]:
        """固定本次请求的材料来源。"""
        ...


@runtime_checkable
class ContextBuilderPort(Protocol):
    """把预置上下文组装为 ModelRequest 的纯组装器。"""

    def build(
        self,
        snapshot: Sequence[Message],
        *,
        materials: Materials,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
        window_start: str | None = None,
    ) -> ModelRequest:
        """纯函数式组装；容量不足明确报错。"""
        ...


MATERIALS = ServiceKey[ContextMaterialsPort]("context.materials.v1")
CONTEXT = ServiceKey[ContextBuilderPort]("context.v1")
