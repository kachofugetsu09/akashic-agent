"""从普通配置贡献与当前依赖图生成引导，不持有业务 writer。"""
from __future__ import annotations

from importlib import import_module
from agent.plugin_composition import Context
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import UI
from plugins.onboarding.contract import ONBOARDING
from plugins.operations.contract import OPERATIONS_V1, OperationRequest, OperationResult, OperationSpec, OptionSpec
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
    _ = await ctx.inject((OPERATIONS_V1,), lambda child: register_ops(child, registry), name="operations")


async def register_ui(ctx: Context) -> None:
    await ctx.require(UI).register(ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__), requires=("shell.settings.v1",),
        contract_digests={"shell.settings.v1": "1f4dd2eaee9118c799590a36745975150c46979a3f62b431f73aa29e949ce189"})


# 引导目录与单步状态：与设置页同一 Registry，只读。
async def register_ops(ctx: Context, registry: Registry) -> None:
    async def catalog(_req: OperationRequest) -> OperationResult:
        return OperationResult(True, "引导目录已读取", data=await registry.catalog())

    async def status(req: OperationRequest) -> OperationResult:
        key = str(req.args["key"])
        try:
            data = await registry.status(key)
        except KeyError:
            return OperationResult(False, f"引导步骤不存在或已移除: {key}", error_code="not_found")
        return OperationResult(True, f"引导步骤 {key} 状态已读取", data=data)

    operations = ctx.require(OPERATIONS_V1)
    _ = await operations.register(ctx, OperationSpec(
        "onboarding.catalog", ("onboarding", "catalog"), "列出引导分组、步骤与能力卡", catalog,
        mirrors=("http:onboarding:GET /api/dashboard/onboarding/catalog",)))
    _ = await operations.register(ctx, OperationSpec(
        "onboarding.status", ("onboarding", "status"), "读取一个引导步骤的就绪状态", status,
        (OptionSpec("key", "步骤键（catalog 里的 key）", required=True),),
        mirrors=("http:onboarding:GET /api/dashboard/onboarding/status/{key:path}",)))
