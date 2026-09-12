from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts import ContentPart


@dataclass(frozen=True)
class Result:
    outcome: str
    parts: tuple[ContentPart, ...]


class Tools(Protocol):
    async def declare_group(self, ctx: Context, *, description: str) -> object: ...

    async def register(
        self, ctx: Context, *, name: str, description: str,
        parameters: Mapping[str, object],
        open: Callable[[Mapping[str, object]], AbstractAsyncContextManager[object]],
        idempotent: bool, risk: str,
    ) -> Mapping[str, object]: ...


TOOLS = ServiceKey[Tools]("tools.v1")

api_version = 3
name = "memory_recall"
version = "1.0.0"
inject = (TOOLS,)

MEMORY_RECALL = ServiceKey[object]("memory.recall.v1")


async def recall_fixture(
    context: object,
    arguments: Mapping[str, object],
) -> str:
    _ = context
    query = arguments.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    return json.dumps(
        {
            "items": [
                {
                    "text": "用户不喜欢只有基准提升、没有实质新能力的模型更新。",
                    "score": 0.94,
                }
            ]
        },
        ensure_ascii=False,
    )


class _RecallTool:
    """通过 tools.v1 暴露 fixture 的真实查询能力。"""

    idempotent = True

    async def prepare(
        self,
        arguments: Mapping[str, object],
        source: object | None = None,
    ) -> Mapping[str, object] | str:
        del source
        query = arguments.get("query")
        if not isinstance(query, str) or not query:
            return "query must be a non-empty string"
        return dict(arguments)

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        del key
        return Result(
            "success",
            (ContentPart("text", await recall_fixture(None, arguments)),),
        )

    async def query(self, key: str) -> None:
        del key
        return None


async def apply(ctx: Context, config: object) -> None:
    _ = config
    _ = await ctx.provide(MEMORY_RECALL, object())
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, description="Fixture memory recall")

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]):
        yield _RecallTool()

    await catalog.register(
        ctx,
        name="recall_fixture",
        description="Recall one user preference relevant to a candidate.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        open=open_tool,
        idempotent=True,
        risk="read-only",
    )
