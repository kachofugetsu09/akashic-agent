from __future__ import annotations

from agent.plugin_composition import Context
from tests.model_plugin_fakes import provide_test_model_services


api_version = 3
name = "model_services"
version = "1.0.0"
inject = ()


async def apply(ctx: Context, config: object) -> None:
    _ = config
    await provide_test_model_services(ctx)
