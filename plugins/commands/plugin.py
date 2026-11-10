"""显式选择的 Commands 服务入口。"""
from agent.plugin_composition import Context
from plugins.commands.contract import COMMANDS
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from .registry import PluginCommands

api_version = 3
name = "commands"
version = "1.0.0"
desc = "命令注册、短生命目录执行和未知效果恢复"


async def apply(ctx: Context) -> None:
    """由 provider 创建并提供按当前登记生成的命令目录。"""
    commands = PluginCommands(ctx)
    await ctx.provide(COMMANDS, commands)

    async def plugin_ui(child: Context) -> None:
        """每次打开建议菜单都读取当前命令登记，不保留旧 generation 的目录。"""
        async def query(method: str, payload: dict[str, object], *,
                        session_id: str | None, turn_id: str | None) -> dict[str, object]:
            if method != "commands.list" or payload:
                raise PluginUiRpcInvalidRequest("命令界面只支持读取命令目录")
            return {"items": [
                {"name": row.name, "description": row.description, "input_hint": row.input_hint}
                for row in commands.freeze().descriptors
            ]}

        await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js", slots=("composer.suggest",)), query=query,
        )

    await ctx.inject((UI_SLOTS,), plugin_ui, name="plugin-ui")
