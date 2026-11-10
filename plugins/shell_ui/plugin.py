from agent.plugin_composition import Context
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import UI
from . import inventory
api_version = 3
name = "shell-ui"
version = "1.0.0"
desc = "应用外框、设置工作区与插件列表"

inject = (UI, PLUGIN_UPDATES, RUNTIME_CATALOG)


async def apply(ctx: Context) -> None:
    _ = await ctx.provide(inventory.INVENTORY, inventory.bind(ctx))

    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(
            child, web="web_module.js",
            dashboard=lambda: inventory,
            requires=("web.root.v1", "shell.rail-actions.v1"),
            provides=("shell.pages.v1", "shell.rail-actions.v1", "shell.settings.v1", "shell.settings-plugins.v1"),
            contract_digests={
                "shell.rail-actions.v1": "911133603ab7d50f616775e0352f90506345d61ee12d1865a63d18a656bb045c",
                "shell.settings.v1": "1f4dd2eaee9118c799590a36745975150c46979a3f62b431f73aa29e949ce189",
                "shell.settings-plugins.v1": "1823b19c778297893495ca9193d688e24c99d079effbaf5ecf0d1148ae0a1aa2",
            },
        )

    _ = await ctx.inject((UI, inventory.INVENTORY), register_ui, name="ui")
