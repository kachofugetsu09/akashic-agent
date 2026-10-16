from plugins.ui.contract import UI
api_version = 3
name = "conversation-ui"
version = "1.0.0"

inject = (UI,)


async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("shell.pages.v1",),
        provides=("conversation.tools.v2",),
        contract_digests={
            "conversation.tools.v2": "5aa099a3c88e6964834e95c4c1e122fdae25ba8b8cda816f5fe0a9d4ffd958d3",
        },
    )
