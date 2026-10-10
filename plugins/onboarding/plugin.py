"""从普通配置贡献与当前依赖图生成引导，不持有业务 writer。"""
from __future__ import annotations

from importlib import import_module
from agent.plugin_composition import Context
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import UI
from plugins.onboarding.contract import ONBOARDING
from .projection import Registry

api_version = 3
name = "onboarding"
version = "1.0.0"
desc = "按依赖顺序配置已安装的功能"
inject = (RUNTIME_CATALOG,)


async def apply(ctx: Context) -> None:
    registry = Registry(ctx)
    await ctx.provide(ONBOARDING, registry)
    await ctx.inject((UI, ONBOARDING), register_ui, name="ui")


async def register_ui(ctx: Context) -> None:
    await ctx.require(UI).register(ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__), requires=("shell.settings.v1",),
        contract_digests={"shell.settings.v1": "a5040165b28b8126a1d55c1a80c8cc707ad55dd0e53cb337fce8c4c721272736"})
