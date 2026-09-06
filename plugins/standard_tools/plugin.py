from agent.plugin_composition import Context, PROCESSES
from agent.plugin_composition.artifacts import ARTIFACT_IMPORT
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from plugins.tools.plugin import TOOLS

from .files import register_file
from .shell import register_shell
from .web import register_web

api_version = 3
name = "standard_tools"
version = "1.0.0"
desc = "文件、命令与 Web 工具，通过普通工具合同授权和记录结果"
inject = (TOOLS, PROCESSES, ARTIFACT_IMPORT)


async def apply(ctx: Context, config: object) -> None:
    """注册既有工具的普通入口；安装和归档装配不访问文件、进程或网络。"""
    for backend in (ReadFileTool, ListDirTool, WriteFileTool, EditFileTool):
        await register_file(ctx, backend, allowed_dir=ctx.runtime.workspace if backend in (ReadFileTool, ListDirTool) else None)
    await register_shell(ctx)
    await register_web(ctx)
