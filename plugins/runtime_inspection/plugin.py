from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from agent.plugin_composition import Context
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.rpc import rpc_method_key
from .rpc import rpc_methods

from .inspection import (
    SCHEDULER_INSPECTION,
    SKILL_INSPECTION,
    RuntimeInspectionProvider,
)

api_version = 3
name = "runtime_inspection"
version = "1.0.0"
desc = "提供固定文档与可选任务、技能的只读运行时检查投影"
workspace_files = ("memory/MEMORY.md", "memory/SELF.md", "memory/VEDA.md")
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


async def apply(ctx: Context, config: object) -> None:
    """发布文档 owner，并在依赖存在时组合任务与技能只读 provider。"""

    provider = RuntimeInspectionProvider(
        {
            "memory": ctx.workspace_file("memory/MEMORY.md"),
            "self": ctx.workspace_file("memory/SELF.md"),
            "veda": ctx.workspace_file("memory/VEDA.md"),
        }
    )
    for name, operation in rpc_methods(provider).items():
        _ = await ctx.provide(rpc_method_key(name), operation)
    await _bind_optional(
        ctx,
        provider,
        SCHEDULER_INSPECTION,
        "scheduler",
        provider.bind_scheduler,
        provider.unbind_scheduler,
    )
    await _bind_optional(
        ctx,
        provider,
        SKILL_INSPECTION,
        "skills",
        provider.bind_skills,
        provider.unbind_skills,
    )
