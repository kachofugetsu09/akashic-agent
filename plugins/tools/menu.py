from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.models import ToolCall as ModelToolCall
from agent.plugin_contracts import CallRef, ToolCall

from .execution import MessageReply, Result, ToolExecution
from .plugin import TOOLS, ToolRef, ToolView
from agent.plugin_contracts.tools import ToolCatalogPort as ToolCatalog, ToolExecutionPort
from agent.plugin_contracts.tool_api import (  # noqa: F401  (再导出)
    InvalidToolCall,
    NativePresentation,
    ToolPresentation,
    tool_schema,
)


class ToolMenu:
    """把获授引用或归档 binding 固定成同一种模型菜单。"""

    def __init__(
        self,
        catalog: ToolCatalog,
        bindings: Bindings,
        execution: ToolExecutionPort,
        reply: Callable[[CallRef], MessageReply],
        *,
        view: ToolView | None = None,
        limit: int | None = None,
        fixed_bindings: Mapping[str, str] | None = None,
        presentation: ToolPresentation | None = None,
    ):
        if (view is None) == (fixed_bindings is None):
            raise ValueError("工具菜单必须且只能取得 current view 或固定 binding")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("工具菜单容量必须为正整数或 None")
        self._bindings = bindings
        self._execution = execution
        self._reply = reply

        if fixed_bindings is not None:
            self._bound = dict(fixed_bindings)
            descriptions = self._descriptions()
            self._presentation = presentation or NativePresentation(descriptions)
        else:
            assert view is not None
            current = {ref.name: ref for ref in view.refs}
            descriptions = {name: ref.description for name, ref in current.items()}
            self._presentation = presentation or NativePresentation(descriptions)
            self._bound = {
                name: catalog.bind(
                    ref,
                    bindings,
                    configuration=self._presentation.configuration(name),
                )
                for name, ref in current.items()
            }
        if limit is not None and len(self._presentation.schemas) > limit:
            raise ValueError(
                "模型工具容量不足以容纳固定展示: "
                f"required={len(self._presentation.schemas)} limit={limit}"
            )

    def _descriptions(self) -> dict[str, Mapping[str, object]]:
        descriptions: dict[str, Mapping[str, object]] = {}
        for name, identity in self._bound.items():
            if not isinstance(identity, str) or not identity:
                raise ValueError("固定工具缺少 binding ID")
            description = cast(
                Mapping[str, object], self._bindings.describe(identity, TOOLS)["tool"]
            )
            if description["name"] != name:
                raise ValueError("固定工具名称与 binding 不一致")
            descriptions[name] = description
        return descriptions

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]:
        return self._presentation.schemas

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._bound)

    @property
    def system_prompt(self) -> str:
        return self._presentation.system_prompt

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]]:
        name, arguments = self._presentation.decode(call)
        identity = self._bound.get(name)
        if identity is None:
            raise PermissionError(f"展示层返回了未获授工具: {name}")
        return identity, arguments

    def bind(self, name: str) -> str:
        """让归档旧 ReAct 只解析已经固定在本菜单中的 binding。"""
        identity = self._bound.get(name)
        if identity is None:
            raise PermissionError(f"模型未获授工具: {name}")
        return identity

    def name(self, binding_id: str) -> str:
        description = cast(
            Mapping[str, object], self._bindings.describe(binding_id, TOOLS)["tool"]
        )
        return cast(str, description["name"])

    def check_call(self, call: ToolCall) -> None:
        if call.binding_id not in self._bound.values():
            raise PermissionError("工具请求不属于本次获授 view")

    async def execute(self, call: CallRef) -> Result:
        reply = self._reply(call)
        try:
            return await self._execution.execute_call(reply)
        finally:
            reply.writer.expire()
