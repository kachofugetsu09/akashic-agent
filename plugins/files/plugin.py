from agent.plugin_composition import Context
from plugins.host_execution.contract import FILES
from plugins.ledger.contract import ARTIFACT_IMPORT

from ._tool_boundary import TOOLS
from .files import register_file
from .filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool

api_version = 3
name = "files"
version = "1.0.0"
desc = "读取、列出、写入和编辑文件"
inject = (TOOLS, FILES, ARTIFACT_IMPORT)


# 读与列目录默认限于 workspace；会话工作目录在调用时借用，缺席时按配置目录访问。
async def apply(ctx: Context) -> None:
    _ = await ctx.require(TOOLS).declare_group(ctx, always_on=True, description=desc)
    for backend in (ReadFileTool, ListDirTool, WriteFileTool, EditFileTool):
        _ = await register_file(
            ctx, backend, allowed_dir=ctx.runtime.workspace if backend in (ReadFileTool, ListDirTool) else None,
        )
