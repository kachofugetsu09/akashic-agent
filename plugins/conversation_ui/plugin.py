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
            "conversation.tools.v2": "9d42ff15d101dd80177264d0e4220cc9192d770aaea6d38cc4e320c4d9035141",
        },
    )
