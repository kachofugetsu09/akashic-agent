"""telegram_sender 的设置、校验与用户选择归本插件。"""
from __future__ import annotations

from importlib import import_module
from typing import Any, cast
from pydantic import BaseModel
from agent.plugin_composition import Context, ServiceKey, CREDENTIALS, CredentialRef
from agent.plugin_composition.context import FiberHandle
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from agent.plugin_composition.config_input import save_credential
from agent.plugin_composition.ui import UI
from agent.plugin_contracts.configuration import Configuration
from agent.plugin_contracts.onboarding import ONBOARDING, Step

SETTINGS = ServiceKey[Configuration]("telegram_sender.settings.v1")
FIELDS = ('channel', 'api_base', 'timeout_seconds')


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
            self.model.model_validate({**config, "enabled": False})
            if values["enabled"]:
                from contextlib import AsyncExitStack
                import aiohttp
                async with AsyncExitStack() as stack:
                    probe_token = token
                    if not probe_token and config.get("token") is not None:
                        ref = cast(CredentialRef, config["token"])
                        credentials = await stack.enter_async_context(self.ctx.require(CREDENTIALS).open(self.ctx, {"token": ref}))
                        probe_token = credentials.credential(ref)
                    if not probe_token:
                        raise ValueError("请填写 Bot token")
                    # A token can contain ':' and URL-safe characters; path/query injection is not a token.
                    import re
                    if re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", probe_token) is None:
                        raise ValueError("Bot token 格式无效")
                    try:
                        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8), trust_env=True) as client:
                            async with client.get(f"{config.get('api_base', 'https://api.telegram.org')}/bot{probe_token}/getMe") as response:
                                result = await response.json()
                                status = response.status
                    except (aiohttp.ClientError, TimeoutError, ValueError):
                        raise ValueError("Telegram 连接验证失败，请检查网络与 token") from None
                    if not 200 <= status < 300 or not isinstance(result, dict) or result.get("ok") is not True:
                        raise ValueError("Telegram token 无效，请重新填写")
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
        await child.require(ONBOARDING).register(child, Step("configure", "Telegram 发送", "channels", "telegram_sender-settings", settings.read, (function.fiber_id,)))
    async def ui(child: Context):
        await child.require(UI).register(child, web="web_module.js", dashboard=lambda: import_module(".dashboard", __package__), requires=("shell.settings-plugins.v1",),
                                         contract_digests={"shell.settings-plugins.v1": "1823b19c778297893495ca9193d688e24c99d079effbaf5ecf0d1148ae0a1aa2"})
    await ctx.inject((ONBOARDING,), contribute, name="onboarding")
    await ctx.inject((UI, SETTINGS), ui, name="settings-ui")
    from agent.plugin_contracts.delivery import DELIVERY_SENDERS
    async def candidate(child: Context):
        await child.require(DELIVERY_SENDERS).candidate(child, name=str(ctx.config.get("channel", "telegram")),
            title="Telegram 发送", route="telegram_sender-settings", status=settings.choice)
    await ctx.inject((DELIVERY_SENDERS,), candidate, name="sender-candidate")
