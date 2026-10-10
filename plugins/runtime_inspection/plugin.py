from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from agent.plugin_composition import Context
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_contracts.inspection import DOCUMENTS
from .rpc import rpc_methods

from .inspection import (
    SKILL_INSPECTION,
    RuntimeInspectionProvider,
)

api_version = 3
name = "runtime_inspection"
version = "1.0.0"
desc = "汇总 owner 文档与可选任务、技能的只读运行时检查投影"
inject = ()

_T = TypeVar("_T")


async def _bind_optional(
    ctx: Context,
    provider: RuntimeInspectionProvider,
    key: ServiceKey[_T],
    name: str,
    bind: Callable[[_T], None],
    unbind: Callable[[_T], None],
) -> None:
    """Mount one optional business provider without blocking the Root."""

    async def apply(child: Context) -> None:
        service = child.require(key)

        async def setup() -> object:
            bind(service)

            async def cleanup() -> None:
                unbind(service)

            return cleanup

        _ = await child.effect(setup, label=f"runtime-inspection:{name}")

    _ = await ctx.inject((key,), apply, name=f"runtime-inspection-{name}")


async def apply(ctx: Context) -> None:
    """发布文档目录，并在依赖存在时组合任务与技能只读 provider。"""

    provider = RuntimeInspectionProvider(ctx)
    _ = await ctx.provide(DOCUMENTS, provider)
    for name, operation in rpc_methods(provider).items():
        _ = await ctx.provide(rpc_method_key(name), operation)
    await _bind_optional(
        ctx,
        provider,
        SKILL_INSPECTION,
        "skills",
        provider.bind_skills,
        provider.unbind_skills,
    )
