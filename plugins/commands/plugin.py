"""显式选择的 Commands 服务入口。"""
from agent.plugin_composition import Context
from agent.plugin_composition.commands import COMMANDS

from .registry import PluginCommands

api_version = 3
name = "commands"
version = "1.0.0"
desc = "命令注册、短生命目录执行和未知效果恢复"


async def apply(ctx: Context) -> None:
    """由 provider 创建并提供按当前登记生成的命令目录。"""
    await ctx.provide(COMMANDS, PluginCommands(ctx))
