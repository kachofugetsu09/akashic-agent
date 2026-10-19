from agent.plugin_composition import Context
from plugins.context.contract import MATERIALS_V4
from plugins.host_execution.contract import PROCESSES

from ._tool_boundary import TOOLS
from .shell import register_shell

api_version = 3
name = "shell"
version = "1.0.0"
desc = "执行命令并管理其进程"
inject = (TOOLS, PROCESSES, MATERIALS_V4)


# 工作目录与后台作业通知都在调用时借用：缺席时退回配置目录、不提供后台运行。
async def apply(ctx: Context) -> None:
    _ = await ctx.require(TOOLS).declare_group(ctx, always_on=True, description=desc)
    _ = await register_shell(ctx)
