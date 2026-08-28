from __future__ import annotations

import json
from collections.abc import Mapping

from agent.plugin_composition import (
    TOOL_CATALOG,
    Context,
    PluginToolDefinition,
    ServiceKey,
)

api_version = 3
name = "memory_recall"
version = "1.0.0"
inject = (TOOL_CATALOG,)

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


async def apply(ctx: Context, config: object) -> None:
    _ = config
    _ = await ctx.provide(MEMORY_RECALL, object())
    await ctx.require(TOOL_CATALOG).register(
        ctx,
        PluginToolDefinition(
            name="recall_fixture",
            description="Recall one user preference relevant to a candidate.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            handler_export="recall_fixture",
            risk="read-only",
        ),
        recall_fixture,
        provided_for=MEMORY_RECALL,
    )
