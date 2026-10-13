"""最小页面插件：在 WebUI 侧栏加一个页面。web_module.js / .css 与本文件放在插件根目录。"""

from agent.plugin_composition import Context
from plugins.ui.contract import UI

api_version = 3
name = "hello_page"
version = "0.1.0"
desc = "WebUI 示例页面"
inject = (UI,)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(UI).register(ctx, web="web_module.js", requires=("shell.pages.v1",))
