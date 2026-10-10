from importlib import import_module
from agent.plugin_composition.ui import UI
from agent.plugin_composition.messages import MESSAGE_CATALOG

api_version = 3
name = "workbench-ui"
version = "2.0.0"
inject = (UI, MESSAGE_CATALOG,)


async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__),
        requires=("conversation.tools.v1",),
        provides=("workbench.panels.v2",),
        contract_digests={
            "conversation.tools.v1": "ed47d69b84e946e27a2e297634e96bcc6afc72a3d3089caac1a14632703efb54",
            "workbench.panels.v2": "fb6417c9bf532c1fdb344767d06065d5d3293da85deb64eff1e8088889a33bcb",
        },
    )
