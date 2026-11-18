"""截屏隐私打码与对话截图：侧栏底栏动作、对话顶栏标记与回答工具栏的截图共享同一份浏览器端脱敏状态。"""
from agent.plugin_composition import Context
from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

api_version = 3
name = "privacy_mask"
version = "1.0.0"
desc = "截屏隐私打码：一键将侧栏会话标题化为骨架条、对话内容高斯模糊；也能把一段对话截成长图并打码"
author = "Akashic Core"
inject = (UI,)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(UI).register(
        ctx,
        web="web_module.js",
        requires=("shell.rail-actions.v1",),
        contract_digests={
            "shell.rail-actions.v1": "911133603ab7d50f616775e0352f90506345d61ee12d1865a63d18a656bb045c",
        },
    )

    async def plugin_ui(child: Context) -> None:
        async def query(
            method: str,
            payload: dict[str, object],
            *,
            session_id: str | None,
            turn_id: str | None,
        ) -> dict[str, object]:
            raise PluginUiRpcInvalidRequest(f"截屏打码插件不支持远程查询: {method}")

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child,
            PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css", slots=("header.end", "message.actions")),
            query=query,
        )

    _ = await ctx.inject((UI_SLOTS,), plugin_ui, name="privacy-mask-plugin-ui")
