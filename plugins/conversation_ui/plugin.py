from agent.plugin_composition.ui import UI
api_version = 3
name = "conversation-ui"
version = "1.0.0"

inject = (UI,)


async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("shell.pages.v1",),
        provides=("conversation.tools.v1",),
        contract_digests={
            "conversation.tools.v1": "ed47d69b84e946e27a2e297634e96bcc6afc72a3d3089caac1a14632703efb54",
        },
    )
