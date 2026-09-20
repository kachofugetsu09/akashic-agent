"""tool_search 与 tools owner 之间的最小本地边界。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.models import ToolCall
from agent.plugin_contracts import ContentPart


ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResultValue:
    """tool_search 的本地结果；tools owner 在执行边界重新校验。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


class CallSource(Protocol):
    """工具只接收已提交调用的可选来源，不持有任何写入能力。"""


class BoundTool(Protocol):
    """tools owner 打开的真实工具；搜索插件只消费其最小方法集。"""

    @property
    def idempotent(self) -> bool: ...

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str: ...

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> ToolResultValue: ...

    async def query(self, key: str) -> ToolResultValue | None: ...


class ToolRef(Protocol):
    """注册表返回的不透明真实引用；tool_search 不构造它。"""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> Mapping[str, object]: ...


class ToolView(Protocol):
    """catalog.view 返回的真实 view；tool_search 只转交它。"""

    @property
    def refs(self) -> tuple[ToolRef, ...]: ...


class ToolCatalog(Protocol):
    """tools owner 提供的注册与只读展示入口。"""

    async def declare_group(
        self,
        ctx: Context,
        *,
        always_on: bool = False,
        description: str = "未声明用途",
    ) -> Effect: ...

    async def register(
        self,
        ctx: Context,
        *,
        name: str,
        description: str,
        parameters: Mapping[str, object],
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[object]],
        capture: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
        public: bool = True,
        idempotent: bool = False,
        risk: Literal["read-only", "read-write", "external-side-effect"] = "read-write",
        search_hint: str | None = None,
    ) -> ToolRef: ...

    def view(self, *refs: object) -> ToolView: ...

    def group_description(self, ref: ToolRef) -> str: ...

    def group_always_on(self, ref: ToolRef) -> bool: ...


class ToolPresentation(Protocol):
    """一次展示的固定 schema、wire 解码和系统提示词。"""

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...

    @property
    def system_prompt(self) -> str: ...

    def decode(self, call: ToolCall) -> tuple[str, Mapping[str, object]] | str: ...

    def configuration(self, name: str) -> Mapping[str, object] | None: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")


__all__ = [
    "BoundTool",
    "CallSource",
    "ToolCatalog",
    "ToolPresentation",
    "ToolRef",
    "ToolResultValue",
    "ToolView",
    "TOOLS",
]
