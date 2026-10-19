"""图片理解：主模型不能直接看图时，经 vision 角色读取图片并返回文本。"""
from agent.plugin_composition import Context
from plugins.host_execution.contract import FILES
from plugins.models.contract import CHAT_MODELS
from plugins.standard_tools.contract import WORKING_DIRECTORY

from ._tool_boundary import TOOLS
from .vision import register_vision

api_version = 3
name = "vision"
version = "1.0.0"
desc = "通过已配置的 vision 角色查看图片"
inject = (TOOLS, FILES, CHAT_MODELS, WORKING_DIRECTORY)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(TOOLS).declare_group(ctx, always_on=True, description=desc)
    await register_vision(ctx, ctx.require(WORKING_DIRECTORY))
