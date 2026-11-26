"""Self-CLI Operation Bus 插件入口：提供 OPERATIONS_V1；模型工具 self 由 self_cli 插件挂到工具目录。"""
from __future__ import annotations

from agent.plugin_composition import Context

from .contract import OPERATIONS_V1
from .registry import OperationRegistry

api_version = 3
name = "operations"
version = "1.0.0"
desc = "Self-CLI Operation Bus 与正交扩展注册中心"


# operations 只依赖 ledger 契约，处在插件依赖图底层，任何业务插件都能软挂载 operation 而不成环。
async def apply(ctx: Context) -> None:
    """发布 OPERATIONS_V1。"""
    _ = await ctx.provide(OPERATIONS_V1, OperationRegistry())
