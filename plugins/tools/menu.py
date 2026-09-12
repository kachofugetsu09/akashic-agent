from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.models import ToolCall as ModelToolCall
from agent.plugin_contracts import CallRef, ToolCall

from .execution import MessageReply, Result, ToolExecution
from .plugin import TOOLS, ToolCatalog, ToolView


@dataclass(frozen=True, slots=True)
class ToolCallDecode:
    """工具 owner 对一次 wire 调用的结构化解码结果。"""

    binding_id: str | None
    arguments: Mapping[str, object]
    rejection: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arguments, Mapping):
            raise TypeError("工具解码参数必须是对象")
        accepted = self.binding_id is not None
        if accepted == (self.rejection is not None):
            raise ValueError("工具解码结果必须且只能是成功或拒绝")
        if accepted and (not isinstance(self.binding_id, str) or not self.binding_id):
            raise ValueError("工具解码结果缺少 binding ID")
        if self.rejection is not None and not isinstance(self.rejection, Mapping):
            raise TypeError("工具拒绝反馈必须是对象")

    @property
    def accepted(self) -> bool:
        """判断模型调用是否已经解码到获授 binding。"""
        return self.binding_id is not None



class ToolPresentation(Protocol):
    """定义一次程序固定的 schema、wire 解码和系统提示词。"""

    @property
    def schemas(self) -> tuple[Mapping[str, Any], ...]: ...

    @property
    def system_prompt(self) -> str: ...

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str: ...

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

    def decode(self, call: ModelToolCall) -> tuple[str, Mapping[str, object]] | str:
        if call.name not in self._descriptions:
            return f"工具不属于获授 view: {call.name}；请使用当前工具目录。"
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


class ToolMenu:
    """把获授引用或归档 binding 固定成同一种模型菜单。"""

    def __init__(
        self,
        catalog: ToolCatalog,
        bindings: Bindings,
        execution: ToolExecution,
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

    def decode(self, call: ModelToolCall) -> ToolCallDecode:
        """把 wire 调用变成真实 binding 或模型可修正的拒绝反馈。"""
        decoded = self._presentation.decode(call)
        if isinstance(decoded, str):
            return ToolCallDecode(None, {}, {
                "name": call.name, "arguments": call.arguments, "error": decoded,
            })
        name, arguments = decoded
        identity = self._bound.get(name)
        if identity is None:
            raise PermissionError(f"展示层返回了未获授工具: {name}")
        return ToolCallDecode(identity, arguments)

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
