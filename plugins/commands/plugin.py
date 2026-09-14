"""显式选择的 Commands 服务入口。"""
from agent.plugin_composition import Context, SNAPSHOT_SEALING, SnapshotSealing
from agent.plugin_composition.commands import COMMANDS

from .registry import PluginCommands

api_version = 3
name = "commands"
version = "1.0.0"
desc = "命令注册、封存、执行和未知效果恢复"


async def apply(ctx: Context) -> None:
    """由 provider 创建目录并在组合发布前自行封存。"""
    commands = PluginCommands(ctx)
    await ctx.provide(COMMANDS, commands)

    def seal(_event: SnapshotSealing) -> None:
        commands.freeze()

    await ctx.on(SNAPSHOT_SEALING, seal)
