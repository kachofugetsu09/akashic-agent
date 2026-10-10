"""模型工具呈现、菜单与程序的公共输入合同。"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol
from agent.plugin_composition.messages import MessageReader
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.models import ToolCall as ModelToolCall
from agent.plugin_composition.tasks import ExternalRootPermit
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences, ToolCall
from agent.plugin_contracts.tools import (
    CommitAfter,
    DecodedCall,
    Result,
    StartCheck,
    ToolView,
)


class ToolPresentation(Protocol):
    """定义一次程序固定的 schema、wire 解码和系统提示词。"""

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...

    @property
    def system_prompt(self) -> str: ...

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str: ...

    def configuration(self, name: str) -> Mapping[str, object] | None: ...


TOOL_LOADING_PRESENTATION = ServiceKey[
    Callable[[ToolView], tuple[ToolView, ToolPresentation]]
]("tools.loading.presentation.v1")


class ToolMenu(Protocol):
    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...
    @property
    def names(self) -> frozenset[str]: ...
    @property
    def system_prompt(self) -> str: ...
    def name(self, binding_id: str) -> str: ...
    def parallel(self, binding_id: str) -> bool: ...
    def decode(self, call: ModelToolCall) -> DecodedCall: ...
    def check_call(self, call: ToolCall) -> None: ...
    async def execute(self, call: CallRef, *, commit_after: CommitAfter | None = None) -> Result: ...
    async def settle_abandoned(self, call: CallRef) -> Result: ...


class OrderedToolProgram(Protocol):
    async def create_menu(
        self,
        reader: MessageReader,
        source: str,
        *,
        content: Mapping[str, Callable[[ContentPart], ContentReferences]],
        check_start: StartCheck,
        authorize: Callable[
            [str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]
        ],
        view: ToolView | None = None,
        fixed_bindings: Mapping[str, str] | None = None,
        limit: int | None = None,
        presentation: ToolPresentation | None = None,
        child_permit: Callable[[], ExternalRootPermit] | None = None,
    ) -> ToolMenu: ...


TOOL_PROGRAM_V2 = ServiceKey[OrderedToolProgram]("tools.program.v2")
