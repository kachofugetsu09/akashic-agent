from agent.plugin_composition import PROCESSES, Context
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.plugin_composition.assets import INSTALLED_ASSETS
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.tasks import TASKS
from agent.plugin_composition.messages import OWNER_STATE, SESSION_ADMISSION
from agent.plugin_contracts.directories import WORKING_DIRECTORY

from ._materials_boundary import MATERIALS
from ._tool_boundary import TOOLS
from .files import register_file
from .filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from .shell import register_shell
from .skills import register_skills
from .working_directory import WorkingDirectories

api_version = 3
name = "standard_tools"
version = "1.0.0"
desc = "提供文件、命令与技能读取工具"
inject = (BINDINGS, TASKS, TOOLS, PROCESSES, ARTIFACT_IMPORT, MATERIALS, INSTALLED_ASSETS,
          OWNER_STATE, SESSION_ADMISSION)



async def apply(ctx: Context) -> None:
    """注册既有工具的普通入口；安装和归档装配不访问文件、进程或网络。"""
    catalog = ctx.require(TOOLS)
    directories = WorkingDirectories(ctx.require(OWNER_STATE).open(ctx))
    _ = await ctx.provide(WORKING_DIRECTORY, directories)
    _ = await ctx.require(SESSION_ADMISSION).register_initializer(
        ctx, name="working-directory", initialize=directories.initialize,
    )
    _ = await catalog.declare_group(ctx, always_on=True, description=desc)
    for backend in (ReadFileTool, ListDirTool, WriteFileTool, EditFileTool):
        await register_file(
            ctx,
            backend,
            allowed_dir=(
                ctx.runtime.workspace
                if backend in (ReadFileTool, ListDirTool)
                else None
            ),
        )
    await register_shell(ctx)
    await register_skills(ctx)
