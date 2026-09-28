"""qq_channel 的设置、校验与用户选择归本插件。"""
from __future__ import annotations

from importlib import import_module
from typing import Any, cast
from pydantic import BaseModel
from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.context import FiberHandle
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from agent.plugin_composition.config_input import save_credential
from agent.plugin_composition.ui import UI
from agent.plugin_contracts.configuration import Configuration
from agent.plugin_contracts.onboarding import ONBOARDING, Step

SETTINGS = ServiceKey[Configuration]("qq_channel.settings.v1")
FIELDS = ('bot_uin', 'allow_from', 'websocket_open_timeout_seconds')


class Settings:
    def __init__(self, ctx: Context, model: type[BaseModel], function: FiberHandle):
        self.ctx, self.model, self.function = ctx, model, function

    def choice(self):
        return {"enabled": self.ctx.config.get("enabled", False)}

    async def read(self) -> dict[str, object]:
        async with self.ctx.runtime_scope():
            config = self.model.model_validate(self.ctx.config)
            values = config.model_dump()
            graph = self.ctx.require(RUNTIME_CATALOG)(self.ctx)
            own = next(item for item in cast(list[dict[str, Any]], graph["plugins"]) if item["id"] == self.ctx.runtime.plugin_id)
            fiber = next(item for item in own["composition"]["fibers"] if item["fiber_id"] == self.function.fiber_id)
            missing = fiber["missing_services"]
            enabled = values["enabled"]
            return {**self.ctx.require(PLUGIN_CONFIG).read(self.ctx), "enabled": enabled,
                    "ready": enabled is True and self.function.state.value == "active",
                    "can_enable": not missing, "blocked": bool(missing), "reason": "缺少前置能力：" + "、".join(missing) if missing else (fiber["error"] or ""),
                    "values": {key: values[key] for key in FIELDS}, "has_token": bool(values.get("token"))}

    async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]:
        """校验后保留未编辑字段，关闭不会清除凭据。"""
        async with self.ctx.runtime_scope():
            if set(values) - {*FIELDS, "enabled", "token"}:
                raise ValueError("配置包含不支持的字段")
            if type(values.get("enabled")) is not bool:
                raise ValueError("请选择开启或关闭")
            if values["enabled"]:
                current = await self.read()
                if not current["can_enable"]:
                    raise ValueError(str(current["reason"]))
            config = dict(self.ctx.config)
            token = values.get("token")
            if token is not None and not isinstance(token, str):
                raise ValueError("token 必须是字符串")
            config.update({key: value for key, value in values.items() if key != "token"})
            pass
            if token:
                config["token"] = save_credential(self.ctx.data_root, token)
            checked = self.model.model_validate(config).model_dump(exclude={"token"})
            if "token" in config:
                checked["token"] = config["token"]
            return await self.ctx.require(PLUGIN_CONFIG).apply(self.ctx, request_id=request_id,
                expected_input=expected_input, config=checked)

    def receipt(self, request_id: str) -> dict[str, object]:
        return self.ctx.entrypoint(lambda: self.ctx.require(PLUGIN_CONFIG).receipt(self.ctx, request_id))()


async def mount(ctx: Context, model: type[BaseModel], function: FiberHandle) -> None:
    settings = Settings(ctx, model, function)
    await ctx.provide(SETTINGS, settings)
    async def contribute(child: Context):
        await child.require(ONBOARDING).register(child, Step("configure", "QQ 接入", "channels", "qq_channel-settings", settings.read, (function.fiber_id,)))
    async def ui(child: Context):
        await child.require(UI).register(child, web="web_module.js", dashboard=lambda: import_module(".dashboard", __package__), requires=("shell.pages.v1",))
    await ctx.inject((ONBOARDING,), contribute, name="onboarding")
    await ctx.inject((UI, SETTINGS), ui, name="settings-ui")
