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
            "conversation.tools.v2": "5d7768495b92f6e365d37bae74dab1acb7f4335b8827a08623b933130e62b41b",
        },
    )
