from agent.plugin_composition import Context, PROCESSES, ServiceKey
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from plugins.standard_tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from plugins.context.materials import MATERIALS
from plugins.tools.plugin import TOOLS, ToolView

from .files import register_file
from .shell import register_shell
from .skills import register_skills

api_version = 3
name = "standard_tools"
version = "1.0.0"
desc = "提供文件、命令与技能读取工具"
inject = (TOOLS, PROCESSES, ARTIFACT_IMPORT, MATERIALS)

STANDARD_TOOLS = ServiceKey[ToolView]("standard-tools.tools.v1")


async def apply(ctx: Context, config: object) -> None:
    """注册既有工具的普通入口；安装和归档装配不访问文件、进程或网络。"""
    catalog = ctx.require(TOOLS)
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    refs = []
    for backend in (ReadFileTool, ListDirTool, WriteFileTool, EditFileTool):
        refs.append(
            await register_file(
                ctx,
                backend,
                allowed_dir=(
                    ctx.runtime.workspace
                    if backend in (ReadFileTool, ListDirTool)
                    else None
                ),
            )
        )
    refs.extend(await register_shell(ctx))
    refs.append(await register_skills(ctx))
    _ = await ctx.provide(STANDARD_TOOLS, catalog.view(*refs))
