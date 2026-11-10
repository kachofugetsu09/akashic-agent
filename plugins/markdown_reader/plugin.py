"""Read-only Markdown workbench using execution-host file access."""
from agent.plugin_composition import Context
from plugins.host_execution.contract import FILES
from plugins.working_directory.contract import WORKING_DIRECTORY
from plugins.ui.contract import UI
from . import dashboard

api_version = 3
name = "markdown_reader"
version = "1.0.0"
desc = "在右侧栏阅读工作目录或绝对路径的 Markdown 文档"
inject = (UI, FILES, WORKING_DIRECTORY)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(UI).register(
        ctx, web="web_module.js", dashboard=lambda: dashboard,
        requires=("conversation.tools.v2",),
        contract_digests={"conversation.tools.v2": "dcbf5db629d4fa6fed4019ad91663f9af94c5b9a02f9e67c94b0745e40f19049"},
    )
