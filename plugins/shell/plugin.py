from importlib import import_module

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from plugins.context.contract import MATERIALS_V4
from plugins.host_execution.contract import PROCESSES
from plugins.ui.contract import UI_SLOTS, Configuration, ModelSettings, PluginUiDefinition, PluginUiRpcInvalidRequest, mount_settings

from ._tool_boundary import TOOLS
from .shell import Config, register_shell

api_version = 3
name = "shell"
version = "1.0.0"
desc = "执行命令并管理其进程"
inject = (PLUGIN_CONFIG, TOOLS, PROCESSES, MATERIALS_V4)
SETTINGS = ServiceKey[Configuration]("shell.settings.v1")


# 工作目录与后台作业通知都在调用时借用：缺席时退回配置目录、不提供后台运行。
async def apply(ctx: Context) -> None:
    _ = await ctx.require(TOOLS).declare_group(ctx, always_on=True, description=desc)
    config = Config.model_validate(ctx.config)
    _ = await register_shell(ctx, config)
    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, Config, ("timeout", "max_output_tokens")),
                         lambda: import_module(".dashboard", __package__))

    # 命令块是本插件工具结果的展示；宿主只把本插件注册的工具结果交给它，没有界面服务时不影响命令执行。
    async def plugin_ui(child: Context) -> None:
        async def query(method: str, payload: dict[str, object], *,
                        session_id: str | None, turn_id: str | None) -> dict[str, object]:
            raise PluginUiRpcInvalidRequest("命令结果展示不提供查询")

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css", slots=("tool.result",)),
            query=query,
        )

    _ = await ctx.inject((UI_SLOTS,), plugin_ui, name="plugin-ui")
