from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from agent.plugin_composition import Context
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.gateway.contract import RpcMethod
from plugins.runtime_inspection.contract import DOCUMENTS
from .rpc import rpc_methods
from plugins.operations.contract import OPERATIONS_V1, OptionSpec, RpcMirror, register_rpc_mirrors

# 运行时自省对终端/SDK 与助手一视同仁：同一处理函数。
MIRRORS = (
    RpcMirror("inspection/capabilities.get", ("inspection", "capabilities"), "读取当前组合的能力快照（插件、工具、命令、技能）"),
    RpcMirror("inspection/documents.list", ("inspection", "documents"), "列出运行时文档"),
    RpcMirror("inspection/documents.get", ("inspection", "document"), "读取一篇运行时文档",
              (OptionSpec("document_id", "文档 ID", required=True),)),
    RpcMirror("inspection/jobs.list", ("inspection", "jobs"), "列出定时任务"),
    RpcMirror("inspection/jobs.get", ("inspection", "job"), "读取一个定时任务", (OptionSpec("job_id", "任务 ID", required=True),)),
    RpcMirror("inspection/skills.list", ("inspection", "skills"), "列出技能与本地技能来源的探测状态"),
)

from plugins.skills.contract import SKILL_INSPECTION
from .inspection import (
    RuntimeInspectionProvider,
)

api_version = 3
name = "runtime_inspection"
version = "1.1.0"
desc = "汇总 owner 文档与可选任务、技能的只读运行时检查投影"
inject = (RUNTIME_CATALOG,)

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
    methods = rpc_methods(ctx, provider)
    for name, operation in methods.items():
        _ = await ctx.provide(RpcMethod.key(name), operation)

    async def register(child: Context) -> None:
        await register_rpc_mirrors(child, child.require(OPERATIONS_V1), methods=methods, mirrors=MIRRORS)
    _ = await ctx.inject((OPERATIONS_V1,), register, name="operations")
    await _bind_optional(
        ctx,
        provider,
        SKILL_INSPECTION,
        "skills",
        provider.bind_skills,
        provider.unbind_skills,
    )
