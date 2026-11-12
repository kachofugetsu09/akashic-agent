"""Read-only Markdown workbench using execution-host file access."""
from agent.plugin_composition import Context
from plugins.host_execution.contract import FILES
from plugins.working_directory.contract import WORKING_DIRECTORY
from plugins.ui.contract import UI
from . import dashboard

api_version = 3
name = "markdown_reader"
version = "1.0.0"
desc = "在右侧工作台浏览执行主机工作区、会话目录或指定路径下的只读文件"
inject = (UI, FILES, WORKING_DIRECTORY)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(UI).register(
        ctx, web="web_module.js", dashboard=lambda: dashboard,
        requires=("conversation.tools.v2",),
        contract_digests={"conversation.tools.v2": "b3f5e9aea07351d69ab1ae55d2dcf18c5e352fc61b6354d4af34dbeca16787eb"},
    )
