"""SearXNG owns its endpoint; standard_web owns the tool and provider selection."""
from importlib import import_module

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from plugins.standard_web.contract import WEB_PROVIDERS
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

from .search import SearxngConfig, SearxngSearch

api_version = 3
name = "web_search_searxng"
version = "1.0.0"
desc = "通过自建 SearXNG 搜索互联网；可配置服务地址"
inject = (PLUGIN_CONFIG,)

SETTINGS = ServiceKey[Configuration]("web_search_searxng.settings.v1")
FIELDS = ("endpoint", "timeout_seconds")


async def apply(ctx: Context) -> None:
    """Keep settings available before configuration or Web tool activation."""
    config = SearxngConfig.model_validate(ctx.config)

    async def run(child: Context) -> None:
        # An unconfigured instance must not make automatic provider selection ambiguous.
        if config.endpoint is not None:
            await child.require(WEB_PROVIDERS).register_search(
                child, SearxngSearch(config.endpoint, config.timeout_seconds), label="SearXNG")

    function = await ctx.inject((WEB_PROVIDERS,), run, name="function")

    async def status() -> dict[str, object]:
        if config.endpoint is None:
            return {"ready": False, "reason": "请填写 SearXNG 服务地址"}
        if function.state.value != "active":
            return {"ready": False, "reason": "搜索提供方未就绪，请检查 standard_web（Web 工具）及插件运行状态"}
        return {"ready": True, "reason": ""}

    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, SearxngConfig, FIELDS, status=status),
                         lambda: import_module(".dashboard", __package__))
