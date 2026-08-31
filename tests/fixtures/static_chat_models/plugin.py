from __future__ import annotations

from agent.plugin_composition import CHAT_MODELS, Context
from tests.model_plugin_fakes import build_test_chat_models


api_version = 3
name = "static_chat_models"
version = "1.0.0"
inject = ()


async def apply(ctx: Context, config: object) -> None:
    _ = config
    _ = await ctx.provide(CHAT_MODELS, build_test_chat_models(object()))
