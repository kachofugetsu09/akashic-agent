"""Wake 主动联系 的开关、前置检查与表单提交归本插件。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast
from pydantic import BaseModel
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.context import FiberHandle
from plugins.models.contract import ModelCatalogSnapshot
from plugins.models.contract import MODEL_CATALOG
from plugins.ledger.contract import MESSAGE_CATALOG
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import Configuration
from plugins.onboarding.contract import ONBOARDING, Ability, PreviewLine, Step
from plugins.delivery.contract import (
    DELIVERY_SENDERS,
    SenderDefinition,
)
from plugins.akasha.contract import SEMANTIC_INTEREST
from plugins.ledger.contract import Input

SETTINGS = ServiceKey[Configuration]("wake.settings.v1")


class Settings:
    def __init__(self, ctx: Context, model: type[BaseModel], function: FiberHandle):
        self.ctx, self.model, self.function = ctx, model, function
        self.models: Callable[[], ModelCatalogSnapshot] | None = None
        self.senders: Callable[[], tuple[Any, ...]] | None = None
        self.interest: Callable[[], str | None] | None = None
        self.interest_decision: Callable[[], bool | None] | None = None

    def targets(self) -> list[dict[str, str]]:
        if self.senders is None:
            return []
        senders = {item["name"] for item in self.senders() if item["available"]}
        targets: list[dict[str, str]] = []
        for session in self.ctx.require(MESSAGE_CATALOG).sessions(visibility="listed", limit=200).items:
            message = session.first_message
            if message is None or not isinstance(message.body, Input):
                continue
            preview = next((part.value.strip() for part in message.body.parts
                            if part.kind == "text" and isinstance(part.value, str) and part.value.strip()), "")
            for part in message.body.parts:
                if part.kind == "channel.origin" and isinstance(part.value, Mapping):
                    channel, recipient = part.value.get("channel"), part.value.get("chat_id")
                    if isinstance(channel, str) and channel in senders and isinstance(recipient, str):
                        targets.append({"channel": channel, "recipient": recipient, "session_id": session.session_id,
                                        "label": f"{channel} · {preview[:48] or recipient}"})
        return targets

    async def read(self) -> dict[str, object]:
        async with self.ctx.runtime_scope():
            config = self.model.model_validate(self.ctx.config).model_dump()
            graph = self.ctx.require(RUNTIME_CATALOG)(self.ctx)
            own = next(item for item in cast(list[dict[str, Any]], graph["plugins"]) if item["id"] == self.ctx.runtime.plugin_id)
            fiber = next(item for item in own["composition"]["fibers"] if item["fiber_id"] == self.function.fiber_id)
            # 当前目标离线不阻止在设置中改选另一个可用目标。
            selected = config["delivery"]
            selected_key = None if selected is None else SenderDefinition.key(selected["channel"]).name
            missing = [key for key in fiber["missing_services"] if key != selected_key]
            selected_missing = selected_key in fiber["missing_services"]
            model = None if self.models is None else self.models()
            selected_model = None if model is None else model.role_bindings.get("default")
            reason = ""
            blocked = bool(missing)
            if missing:
                reason = "缺少前置能力：" + "、".join(missing)
            elif self.interest is None:
                reason, blocked = "Akasha 兴趣能力未安装或不可用", True
            elif self.interest() is not None:
                reason = self.interest() or "Akasha 不可用"
                blocked = self.interest_decision is not None and self.interest_decision() is not True
            elif model is None or selected_model is None or model.model(selected_model).availability != 'available':
                reason = "请先选择默认聊天模型"
            elif not self.targets():
                reason = "还没有可用发送目标，请先开启发送渠道并建立对话"
            can_enable = not reason
            if selected_missing:
                blocked = True
                if not reason:
                    reason = "当前发送渠道不可用，请在功能设置中改选目标或重新开启该渠道"
            enabled = config["enabled"]
            return {**self.ctx.require(PLUGIN_CONFIG).read(self.ctx), "enabled": enabled,
                "ready": enabled is True and not reason and self.function.state.value == "active",
                "can_enable": can_enable, "blocked": blocked, "reason": reason or fiber["error"] or "",
                "values": {"delivery": config["delivery"], "timezone": config["timezone"]}, "targets": self.targets()}

    async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]:
        """只接受用户决定与本插件字段，启用前重新核对前置。"""
        async with self.ctx.runtime_scope():
            if set(values) - {"enabled", "delivery", "timezone"} or type(values.get("enabled")) is not bool:
                raise ValueError("配置字段无效，请选择开启或关闭")
            current = await self.read()
            if values["enabled"] and not current["can_enable"]:
                raise ValueError(str(current["reason"]))
            if values["enabled"]:
                target = values.get("delivery", self.ctx.config.get("delivery"))
                choices = [{key: value for key, value in item.items() if key != "label"} for item in self.targets()]
                if target not in choices:
                    raise ValueError("请选择现有对话的可用发送目标")
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
        await child.require(ONBOARDING).group(child, "proactive", "Wake 主动联系", Ability(
            pitch="在合适的时候主动找你",
            benefit="有具体理由时它会先开口，比如你提过的事有了进展。消息发到你选择的聊天渠道。",
            preview=(PreviewLine("Akashic · 主动消息", "你周三说想等那本书降价，现在电子版打五折了。"),),
        ))
        await child.require(ONBOARDING).register(child, Step("configure", "Wake 主动联系", "proactive", "wake-settings", settings.read, (function.fiber_id,)))
    await ctx.inject((ONBOARDING,), contribute, name="onboarding")
    async def senders(child: Context):
        def attach():
            settings.senders = child.entrypoint(child.require(DELIVERY_SENDERS).candidates)
            return lambda: setattr(settings, "senders", None)
        await child.effect(attach, label="sender-options")
    async def interest(child: Context):
        def attach():
            service = child.require(SEMANTIC_INTEREST)
            settings.interest = child.entrypoint(service.status)
            settings.interest_decision = child.entrypoint(service.decision)
            def stop():
                settings.interest = None
                settings.interest_decision = None
            return stop
        await child.effect(attach, label="interest-status")
    await ctx.inject((DELIVERY_SENDERS,), senders, name="sender-options")
    await ctx.inject((SEMANTIC_INTEREST,), interest, name="interest-status")
