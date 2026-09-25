"""工具的公共合同；注册身份、允许目录和展示各自独立于提供方实现。"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.context import Context
from agent.plugin_composition.effect import Effect
from agent.plugin_composition.messages import MessageReader
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import ToolCall as ModelToolCall
from agent.plugin_composition.tasks import ExternalRootPermit, Task
from agent.plugin_contracts import (
    CallRef,
    ContentPart,
    ContentReferences,
    Message,
    ToolCall,
)

Outcome = Literal["success", "denied", "error", "interrupted"]


def durable_call_key(call_ref: CallRef) -> str:
    """Return the stable effect key already used by a submitted ToolCall."""
    if not isinstance(call_ref, CallRef):
        raise TypeError("工具调用引用无效")
    return "message:" + json.dumps(
        [call_ref.message_id, call_ref.part_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@runtime_checkable
class ResultLike(Protocol):
    """provider 返回的结构结果；tools owner 不依赖 provider 的类身份。"""

    @property
    def outcome(self) -> Outcome: ...
    @property
    def parts(self) -> tuple[ContentPart, ...]: ...


@dataclass(frozen=True, slots=True)
class CallSource:
    """实际调用的不可变消息前缀；不携带 reader 或任何写入能力。"""

    call_ref: CallRef
    messages: tuple[Message, ...]

    @property
    def effect_key(self) -> str:
        return durable_call_key(self.call_ref)


class ProviderBoundTool(Protocol):
    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ResultLike: ...

    async def query(self, key: str) -> ResultLike | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


@dataclass(frozen=True, slots=True)
class ToolRef:
    """引用当前 composition Root 中的一次真实工具注册。"""

    name: str
    description: Mapping[str, object]


def tool_key(name: str) -> ServiceKey[ToolRef]:
    """声明对单个已注册工具的依赖，随工具 owner 激活与释放。"""
    return ServiceKey[ToolRef](f"tools.ref.{name}.v1")


@dataclass(frozen=True, slots=True)
class ToolView:
    """消费者获授的一组真实工具引用。"""

    refs: tuple[ToolRef, ...]

    def __post_init__(self) -> None:
        refs = tuple(self.refs)
        names = tuple(ref.name for ref in refs)
        if len(set(names)) != len(names):
            raise ValueError("工具 view 不能包含重复名称")
        object.__setattr__(self, "refs", refs)

    def select(self, name: str) -> ToolRef:
        for ref in self.refs:
            if ref.name == name:
                return ref
        raise PermissionError(f"工具不属于获授 view: {name}")

    def without(self, names: frozenset[str]) -> ToolView:
        return ToolView(tuple(ref for ref in self.refs if ref.name not in names))

    @classmethod
    def combine(cls, *views: ToolView) -> ToolView:
        return cls(tuple(ref for view in views for ref in view.refs))


class ToolPresentation(Protocol):
    """定义一次程序固定的 schema、wire 解码和系统提示词。"""

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...

    @property
    def system_prompt(self) -> str: ...

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str: ...

    def configuration(self, name: str) -> Mapping[str, object] | None: ...


@dataclass(frozen=True, slots=True)
class Result:
    outcome: Outcome
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"success", "denied", "error", "interrupted"}:
            raise ValueError("工具结果状态无效")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("工具结果必须是内容块")
        object.__setattr__(self, "parts", parts)


class BoundTool(Protocol):
    """tools owner 暴露给执行器的已归一化工具 facade。"""

    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result: ...

    async def query(self, key: str) -> Result | None:
        """查询原调用；None 只表示无法确定，不能解释为没有效果。"""
        ...


class ToolCatalog(Protocol):
    """注册表拥有真实引用；消费者只选择本次允许的目录。"""

    async def declare_group(
        self, ctx: Context, *, always_on: bool = False, description: str = "未声明用途"
    ) -> Effect: ...
    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: Callable[
            [Mapping[str, object]], AbstractAsyncContextManager[ProviderBoundTool]
        ],
        capture: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        public: bool = True,
        idempotent: bool = False,
        risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
        search_hint: str | None = None,
    ) -> ToolRef: ...
    async def register_prepare(
        self,
        ctx: Context,
        *,
        tool: ToolRef,
        name: str,
        prepare: Callable[[Mapping[str, object]], Awaitable[Mapping[str, object]]],
    ) -> Effect: ...
    async def register_authorize(
        self,
        ctx: Context,
        *,
        tool: ToolRef,
        name: str,
        authorize: Callable[[Mapping[str, object]], Awaitable[str | None]],
    ) -> Effect: ...
    def view(self, *refs: ToolRef) -> ToolView: ...
    def group_description(self, ref: ToolRef) -> str: ...
    def group_always_on(self, ref: ToolRef) -> bool: ...
    def bind(
        self,
        ref: ToolRef,
        bindings: Bindings,
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> str: ...
    async def bind_scoped(
        self,
        ref: ToolRef,
        bindings: Bindings,
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> str: ...
    def bind_saved(
        self,
        metadata: Mapping[str, object],
        bindings: Bindings,
        *,
        configuration: Mapping[str, object],
    ) -> str: ...
    def open(
        self, metadata: Mapping[str, object]
    ) -> AbstractAsyncContextManager[BoundTool]: ...
    async def authorize(
        self, metadata: Mapping[str, object], arguments: Mapping[str, object]
    ) -> str | None: ...
    async def drain_calls(self, calls: tuple[CallRef, ...]) -> None: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")
ALL_TOOLS = ServiceKey[Callable[[], ToolView]]("tools.all.v1")
TOOL_DISPLAY_NAME = ServiceKey[Callable[[str], str]]("tools.display-name.v1")
TOOL_SEARCH_PRESENTATION = ServiceKey[
    Callable[[ToolView], tuple[ToolView, ToolPresentation]]
]("tool-search.presentation.v2")


class DecodedCall(Protocol):
    @property
    def binding_id(self) -> str | None: ...
    @property
    def arguments(self) -> Mapping[str, object]: ...
    @property
    def rejection(self) -> Mapping[str, object] | None: ...


class ToolMenu(Protocol):
    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def names(self) -> frozenset[str]: ...
    @property
    def system_prompt(self) -> str: ...
    def name(self, binding_id: str) -> str: ...
    def decode(self, call: ModelToolCall) -> DecodedCall: ...
    def check_call(self, call: ToolCall) -> None: ...
    async def execute(self, call: CallRef) -> Result: ...
    async def settle_abandoned(self, call: CallRef) -> Result: ...


class ToolProgram(Protocol):
    async def create_menu(
        self,
        reader: MessageReader,
        source: str,
        *,
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
        check_start: Callable[[], None],
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        view: ToolView | None = None,
        fixed_bindings: Mapping[str, str] | None = None,
        limit: int | None = None,
        presentation: ToolPresentation | None = None,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolMenu: ...


class ToolCleanup(Protocol):
    """程序消费者提供本次工具 owner 的真实收尾边界。"""

    def __call__(
        self,
        reader: MessageReader,
        source: str,
        from_seq: int,
        *,
        task: Task,
        drain: Callable[[tuple[CallRef, ...]], Awaitable[None]],
    ) -> AbstractAsyncContextManager[None]: ...


class BindSavedTool(Protocol):
    async def __call__(
        self,
        bindings: Bindings,
        binding_id: str,
        *,
        configuration: Mapping[str, object],
    ) -> str: ...


TOOL_PROGRAM = ServiceKey[ToolProgram]("tools.program.v1")
TOOL_CLEANUP = ServiceKey[ToolCleanup]("tools.cleanup.v1")
TOOL_BIND_SAVED = ServiceKey[BindSavedTool]("tools.bind-saved.v1")
