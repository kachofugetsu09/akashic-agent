from plugins.ui.contract import UI
api_version = 3
name = "shell-ui"
version = "1.0.0"

inject = (UI,)


async def apply(ctx):
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("web.root.v1", "shell.rail-actions.v1"),
        provides=("shell.pages.v1", "shell.rail-actions.v1", "shell.settings.v1", "shell.settings-plugins.v1"),
        contract_digests={
            "shell.rail-actions.v1": "911133603ab7d50f616775e0352f90506345d61ee12d1865a63d18a656bb045c",
            "shell.settings.v1": "a5040165b28b8126a1d55c1a80c8cc707ad55dd0e53cb337fce8c4c721272736",
            "shell.settings-plugins.v1": "a1762d8d7286d3f221181e2e427c0da062d63800179afe2788430268456de14b",
        },
    )
