"""技能：内置、工作区与用户目录中的 Skill 目录、提示词材料、load_skill 工具和输入框技能建议。"""
from agent.plugin_composition import Context
from plugins.assets.contract import INSTALLED_ASSETS
from plugins.context.contract import MATERIALS_V4
from plugins.host_execution.contract import PROCESSES
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from ._tool_boundary import TOOLS
from .skills import register_skills

api_version = 3
name = "skills"
version = "1.0.0"
desc = "提供技能目录与技能读取工具"
inject = (TOOLS, PROCESSES, MATERIALS_V4, INSTALLED_ASSETS)


async def apply(ctx: Context) -> None:
    _ = await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
    _ = await ctx.require(TOOLS).declare_group(ctx, always_on=True, description=desc)
    skills = await register_skills(ctx)

    # 输入框的 / 与 @ 技能建议是本插件自己的界面；宿主只提供插槽。
    async def plugin_ui(child: Context) -> None:
        async def query(method: str, payload: dict[str, object], *,
                        session_id: str | None, turn_id: str | None) -> dict[str, object]:
            if method != "skills.list" or payload:
                raise PluginUiRpcInvalidRequest("技能界面只支持读取技能目录")
            return {"items": list(await skills.list_skills())}

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js", slots=("composer.suggest",)), query=query,
        )

    _ = await ctx.inject((UI_SLOTS,), plugin_ui, name="plugin-ui")
