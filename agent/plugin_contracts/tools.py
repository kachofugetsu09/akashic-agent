"""工具能力的公开结构合同。

`tools.v1` / `tools.all.v1` / `tools.display-name.v1` 是「工具注册表」的公开
名字：Core 与插件都要声明它们，但都不应 import `plugins/tools` 的实现模块。
合同层因此拥有 key、`ToolRef`/`ToolView` 值模型，以及描述注册表的 Protocol。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager

    from agent.plugin_composition.bindings import Bindings
    from agent.plugin_composition.context import Context
    from agent.plugin_composition.effect import Effect
    from agent.plugin_contracts.tool_api import (
        Authorize,
        BoundTool,
        MessageReplyPort,
        Result,
    )
    from agent.plugin_contracts.restart import ExternalRootPermit

Prepare = Callable[[Mapping[str, object]], Awaitable[Mapping[str, object]]]
BindingAuthorize = Callable[[Mapping[str, object]], Awaitable[None]]
OpenTarget = Callable[[Mapping[str, object]], "AbstractAsyncContextManager[BoundTool]"]
Capture = Callable[[Mapping[str, object]], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class ToolRef:
    """引用当前 composition Root 中的一次真实工具注册。"""

    name: str
    description: Mapping[str, object]


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
        """按名字取得引用；不属于本 view 时 fail-loud。"""
        for ref in self.refs:
            if ref.name == name:
                return ref
        raise PermissionError(f"工具不属于获授 view: {name}")

    def without(self, names: frozenset[str]) -> ToolView:
        """排除若干名字后生成新 view。"""
        return ToolView(tuple(ref for ref in self.refs if ref.name not in names))

    @classmethod
    def combine(cls, *views: ToolView) -> ToolView:
        """按顺序合并多个 view。"""
        return cls(tuple(ref for view in views for ref in view.refs))


@runtime_checkable
class ToolExecutionPort(Protocol):
    """一次已授权工具调用的执行入口；实现由 `plugins/tools/execution.py` 提供。"""

    async def execute_call(self, reply: MessageReplyPort) -> Result:
        """按已提交调用执行并把结果写回 reply。"""
        ...

    async def deny_call(self, reply: MessageReplyPort, reason: str) -> Result:
        """按授权结果拒绝调用并写回 reply。"""
        ...


@runtime_checkable
class ToolCatalogPort(Protocol):
    """工具注册表对 Core 与插件可见的方法子集；实现由 plugins/tools 提供。"""

    async def declare_group(
        self, ctx: Context, *, always_on: bool = False, description: str = "未声明用途"
    ) -> Effect:
        """声明唯一组级展示事实。"""
        ...

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: OpenTarget,
        capture: Capture | None = None,
        public: bool = True,
        idempotent: bool = False,
        risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
        search_hint: str | None = None,
    ) -> ToolRef:
        """登记一个工具并返回其引用。"""
        ...

    def view(self, *refs: ToolRef) -> ToolView:
        """构造消费者 view。"""
        ...

    def bind(
        self,
        ref: ToolRef,
        bindings: Bindings,
        *,
        configuration: Mapping[str, object] | None = None,
    ) -> str:
        """从真实注册 Context 固定闭包。"""
        ...

    async def bind_saved(
        self,
        bindings: Bindings,
        binding_id: str,
        *,
        configuration: Mapping[str, object],
    ) -> str:
        """从真实原 binding 派生新配置。"""
        ...

    def execution(
        self,
        authorize: Authorize,
        *,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolExecutionPort:
        """返回该注册表的执行入口。"""
        ...


TOOLS = ServiceKey[ToolCatalogPort]("tools.v1")
ALL_TOOLS = ServiceKey[Callable[[], ToolView]]("tools.all.v1")
TOOL_DISPLAY_NAME = ServiceKey[Callable[[str], str]]("tools.display-name.v1")
