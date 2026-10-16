"""工作台“预览”标签：在侧栏查看对话里的文本与表格附件。"""
from agent.plugin_composition import Context
from plugins.ui.contract import UI

from . import dashboard

api_version = 3
name = "preview_ui"
version = "1.0.0"
desc = "在对话工作台侧栏预览文本与表格附件"
author = "Akashic Core"
inject = ()
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context) -> None:
    # UI 缺席时只是没有侧栏预览，聊天端回到原有弹窗。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(
            child, web="web_module.js", dashboard=lambda: dashboard,
            requires=("conversation.tools.v2",),
            contract_digests={
                "conversation.tools.v2": "438131c0fee0caf2ba0c8509d9312ccda6c2c195ad4ce29c26da404e8d15292c",
            },
        )

    _ = await ctx.inject((UI,), register_ui, name="preview-ui")
