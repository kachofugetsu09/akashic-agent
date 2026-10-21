"""图片理解：主模型不能直接看图时，经插件自己选的看图模型读取图片并返回文本。"""
from importlib import import_module

from agent.plugin_composition import Context
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from plugins.host_execution.contract import FILES
from plugins.models.contract import CHAT_MODELS, MODEL_CATALOG, model_options
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

from ._tool_boundary import TOOLS
from .vision import VISION_MODEL, Config, register_vision

api_version = 3
name = "vision"
version = "1.1.0"
desc = "通过能看图的模型查看图片"
inject = (PLUGIN_CONFIG, TOOLS, FILES, CHAT_MODELS, MODEL_CATALOG)

SETTINGS = ServiceKey[Configuration]("vision.settings.v1")


async def apply(ctx: Context) -> None:
    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, Config, ("model",),
                         status=model_options(ctx, {"model": VISION_MODEL})),
                         lambda: import_module(".dashboard", __package__))
    _ = await ctx.require(TOOLS).declare_group(ctx, always_on=True, description=desc)
    await register_vision(ctx, Config.model_validate(ctx.config))
