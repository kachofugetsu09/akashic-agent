from agent.plugin_composition.ui import UI
api_version = 3
name = "shell-ui"
version = "1.0.0"

inject = (UI,)


async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("web.root.v1",),
        provides=("shell.pages.v1",),
    )
