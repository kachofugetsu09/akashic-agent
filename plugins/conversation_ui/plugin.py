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
            "conversation.tools.v2": "64a81cdc7c33305d8b7ea854073b9a7266d6c7258c88af65c5b5760dadb912ea",
        },
    )
