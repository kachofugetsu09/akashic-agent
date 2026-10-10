"""Akasha 情景记忆 的开关、前置检查与表单提交归本插件。"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast
from pydantic import BaseModel
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.context import FiberHandle
from agent.plugin_composition.models import MODEL_CATALOG, ModelCatalogSnapshot, ModelAvailability
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from agent.plugin_contracts.configuration import Configuration
from agent.plugin_contracts.onboarding import ONBOARDING, Ability, PreviewLine, Step

SETTINGS = ServiceKey[Configuration]("akasha.settings.v1")


class Settings:
    def __init__(self, ctx: Context, model: type[BaseModel], function: FiberHandle):
        self.ctx, self.model, self.function = ctx, model, function
        self.models: Callable[[], ModelCatalogSnapshot] | None = None

    async def read(self) -> dict[str, object]:
        async with self.ctx.runtime_scope():
            config = self.model.model_validate(self.ctx.config).model_dump()
            graph = self.ctx.require(RUNTIME_CATALOG)(self.ctx)
            own = next(item for item in cast(list[dict[str, Any]], graph["plugins"]) if item["id"] == self.ctx.runtime.plugin_id)
            fiber = next(item for item in own["composition"]["fibers"] if item["fiber_id"] == self.function.fiber_id)
            missing = fiber["missing_services"]
            model = None if self.models is None else self.models()
            reason = ""
            blocked = bool(missing)
            if missing:
                reason = "缺少前置能力：" + "、".join(missing)
            elif model is None or not model.default_embedding_model_id or model.model(model.default_embedding_model_id).availability != ModelAvailability.AVAILABLE:
                reason = "请先在模型设置中选择向量模型"
            enabled = config["enabled"]
            return {**self.ctx.require(PLUGIN_CONFIG).read(self.ctx), "enabled": enabled,
                "ready": enabled is True and not reason and self.function.state.value == "active",
                "can_enable": not reason, "blocked": blocked, "reason": reason or fiber["error"] or "",
                "values": {}}

    async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]:
        """只接受用户决定与本插件字段，启用前重新核对前置。"""
        async with self.ctx.runtime_scope():
            if set(values) - {"enabled"} or type(values.get("enabled")) is not bool:
                raise ValueError("配置字段无效，请选择开启或关闭")
            current = await self.read()
            if values["enabled"] and not current["can_enable"]:
                raise ValueError(str(current["reason"]))
            config = self.model.model_validate({**self.ctx.config, **values}).model_dump()
            return await self.ctx.require(PLUGIN_CONFIG).apply(self.ctx, request_id=request_id,
                expected_input=expected_input, config=config)

    def receipt(self, request_id: str) -> dict[str, object]:
        return self.ctx.entrypoint(lambda: self.ctx.require(PLUGIN_CONFIG).receipt(self.ctx, request_id))()


async def mount(ctx: Context, model: type[BaseModel], function: FiberHandle) -> None:
    settings = Settings(ctx, model, function)
    await ctx.provide(SETTINGS, settings)
    async def models(child: Context):
        def attach():
            settings.models = child.entrypoint(child.require(MODEL_CATALOG).snapshot)
            return lambda: setattr(settings, "models", None)
        await child.effect(attach, label="model-readiness")
    await ctx.inject((MODEL_CATALOG,), models, name="model-status")
    async def contribute(child: Context):
        await child.require(ONBOARDING).group(child, "memory", "Akasha 情景记忆", Ability(
            pitch="记住你们聊过的事",
            benefit="对话会沉淀成情景记忆，之后聊到相关的事时它会自己想起来。需要一个向量模型。",
            preview=(PreviewLine("你", "上次说的那家拉面店叫什么来着？"),
                     PreviewLine("Akashic", "是「风云儿」，你上个月去过，说汤头偏咸但面很好。")),
        ))
        await child.require(ONBOARDING).register(child, Step("configure", "Akasha 情景记忆", "memory", "akasha-settings", settings.read, (function.fiber_id,)))
    await ctx.inject((ONBOARDING,), contribute, name="onboarding")
