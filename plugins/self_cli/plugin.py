"""唯一模型工具 self：把 argv 交给 OPERATIONS_V1；与 operations 分开，免得 operations 依赖工具目录而成环。"""
from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import cast
from contextlib import asynccontextmanager
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart, RpcMethod, json_value
from plugins.operations.contract import OPERATIONS_V1, OperationResult, Operations
from plugins.tools.contract import TOOLS, CallSource, Result

api_version = 3
name = "self_cli"
version = "1.0.0"
desc = "Self-CLI 模型工具 self：通过统一内省式 CLI 调用 Akashic 自身能力"
# 终端里的 `akashic self ...` 经 Gateway 调同一张操作表，操作者是用户本人。
INVOKE_METHOD = "self/invoke"


class SelfCliInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    argv: list[str] = Field(
        description='命令行参数数组，例如 ["help"]、["describe", "wd"] 或 ["wd", "current"]。',
    )


class SelfCliTool:
    # 混合读写单工具保守声明非幂等；具体写操作由各自 request_id 回执恢复。
    idempotent = False

    def __init__(self, registry: Operations) -> None:
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
    """工具目录与 operation 总线都在时暴露唯一模型工具 `self`。"""
    async def register_tool(child: Context) -> None:
        tools = child.require(TOOLS)
        _ = await tools.declare_group(child, always_on=True, description=desc)

        @asynccontextmanager
        async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[SelfCliTool]:
            yield SelfCliTool(child.require(OPERATIONS_V1))

        _ = await tools.register(
            child,
            name="self",
            description="通过统一内省式 CLI 调用 Akashic 自身能力；传 [\"help\"] 查看当前已注册命令与正交扩展选项。",
            parameters=SelfCliInput.model_json_schema(),
            open=open_tool,
            idempotent=False,
            icon="terminal",
        )

    _ = await ctx.inject((TOOLS, OPERATIONS_V1), register_tool, name="self-cli-tool")
    _ = await ctx.inject((OPERATIONS_V1,), _register_shell, name="self-cli-shell")


class SelfInvokeParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    argv: list[str] = Field(max_length=256)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    session_id: str | None = Field(default=None, min_length=1, max_length=512)


# 终端入口与模型工具共用一张操作表；差别只有 actor，user_only 操作因此只在这里可用。
async def _register_shell(child: Context) -> None:
    registry = child.require(OPERATIONS_V1)

    async def invoke(params: BaseModel) -> dict[str, object]:
        call = cast(SelfInvokeParams, params)
        res = await registry.invoke(call.argv, actor="user", session_id=call.session_id, request_id=call.request_id)
        return {
            "ok": res.ok, "summary": res.summary, "data": json_value(dict(res.data)),
            "error_code": res.error_code, "parts": [json_value({"kind": p.kind, "value": p.value}) for p in res.parts],
        }

    _ = await child.provide(RpcMethod.key(INVOKE_METHOD), RpcMethod(SelfInvokeParams, child.entrypoint(invoke)))
