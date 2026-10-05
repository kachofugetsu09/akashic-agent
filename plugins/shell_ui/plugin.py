from agent.plugin_composition.ui import UI
api_version = 3
name = "shell-ui"
version = "1.0.0"

inject = (UI,)


async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("web.root.v1", "shell.rail-actions.v1"),
        provides=("shell.pages.v1", "shell.rail-actions.v1"),
        contract_digests={
            "shell.rail-actions.v1": "911133603ab7d50f616775e0352f90506345d61ee12d1865a63d18a656bb045c",
        },
    )
