"""standard_web 与 tools owner 之间的最小本地边界。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_contracts import ContentPart


ToolOutcome = Literal["success", "denied", "error", "interrupted"]


@dataclass(frozen=True, slots=True)
class ToolResultValue:
    """standard_web 的本地结果；tools owner 在执行边界重新校验。"""

    outcome: ToolOutcome
    parts: tuple[ContentPart, ...]


class CallSource(Protocol):
    """Web provider 不读取来源，只保留窄的可选调用参数。"""


class ToolRef(Protocol):
    """注册表返回的不透明真实引用；provider 不构造它。"""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> Mapping[str, object]: ...


class ToolView(Protocol):
    """catalog.view 返回的真实 view；provider 只转交它。"""

    @property
    def refs(self) -> tuple[ToolRef, ...]: ...


class ToolCatalog(Protocol):
    """tools owner 提供的注册入口；注册表和执行器仍由 tools 拥有。"""

    async def declare_group(
        self, ctx: Context, *, always_on: bool = False,
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
    ) -> object: ...

    def view(self, *refs: object) -> ToolView: ...


TOOLS = ServiceKey[ToolCatalog]("tools.v1")
