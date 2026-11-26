"""Self-CLI Operation Bus 插件入口：提供 OPERATIONS_V1 与唯一模型工具 self。"""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart, json_value
from plugins.tools.contract import TOOLS, CallSource, Result

from .contract import OPERATIONS_V1, OperationResult
from .registry import OperationRegistry

api_version = 3
name = "operations"
version = "1.0.0"
desc = "Self-CLI Operation Bus 与正交扩展注册中心"


class SelfCliInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    argv: list[str] = Field(
        description='命令行参数数组，例如 ["help"]、["describe", "wd"] 或 ["wd", "current"]。',
    )


class SelfCliTool:
    # 混合读写单工具保守声明非幂等；具体写操作由各自 request_id 回执恢复。
    idempotent = False

    def __init__(self, registry: OperationRegistry) -> None:
        self._registry = registry

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object] | str:
        try:
            parsed = SelfCliInput.model_validate(json_value(arguments))
        except ValidationError as err:
            return str(err)
        session_id = source.messages[-1].session_id if source and source.messages else None
        return {"argv": parsed.argv, "session_id": session_id}

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        argv = [str(item) for item in list(arguments.get("argv", []))]  # type: ignore[arg-type]
        session_raw = arguments.get("session_id")
        session_id = session_raw if isinstance(session_raw, str) else None
        res = await self._registry.invoke(
            argv, actor="agent", session_id=session_id, request_id=key
        )
        return self._to_tool_result(res)

    async def query(self, key: str) -> Result | None:
        res = self._registry.receipt(key)
        return None if res is None else self._to_tool_result(res)

    def _to_tool_result(self, res: OperationResult) -> Result:
        text = res.summary
        if res.data and not text.lstrip().startswith("{"):
            text = f"{text}\n{json.dumps(res.data, ensure_ascii=False)}"
        parts = (ContentPart("text", text), *res.parts)
        return Result("success" if res.ok else "error", parts)


async def apply(ctx: Context) -> None:
    """发布 OPERATIONS_V1，并在存在 TOOLS 时暴露唯一模型工具 `self`。"""
    registry = OperationRegistry()
    _ = await ctx.provide(OPERATIONS_V1, registry)

    async def register_tool(child: Context) -> None:
        tools = child.require(TOOLS)
        _ = await tools.declare_group(child, always_on=True, description=desc)

        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[SelfCliTool]:
            yield SelfCliTool(registry)

        _ = await tools.register(
            child,
            name="self",
            description="通过统一内省式 CLI 调用 Akashic 自身能力；传 [\"help\"] 查看当前已注册命令与正交扩展选项。",
            parameters=SelfCliInput.model_json_schema(),
            open=open_tool,
            idempotent=False,
            icon="terminal",
        )

    _ = await ctx.inject((TOOLS,), register_tool, name="self-cli-tool")
