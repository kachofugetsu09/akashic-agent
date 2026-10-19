"""web_search_exa 的端点、凭据与搜索选项设置归本插件；启用与否由插件行决定。"""
from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from pydantic import ValidationError

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.config_input import save_credential
from agent.plugin_composition.context import FiberHandle
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.ui.contract import UI, Configuration

from .search import ExaConfig

SETTINGS = ServiceKey[Configuration]("web_search_exa.settings.v1")
FIELDS = ("endpoint", "search_type", "livecrawl", "timeout_seconds")
UI_CONTRACT = {"shell.settings-plugins.v1": "1823b19c778297893495ca9193d688e24c99d079effbaf5ecf0d1148ae0a1aa2"}


class Settings:
    def __init__(self, ctx: Context, function: FiberHandle):
        self.ctx, self.function = ctx, function

    # 插件已加载即视为开启；ready 反映提供方是否已登记到 standard_web。
    async def read(self) -> dict[str, object]:
        async with self.ctx.runtime_scope():
            config = ExaConfig.model_validate(self.ctx.config)
            values = config.model_dump(mode="json", exclude={"api_key"})
            graph = self.ctx.require(RUNTIME_CATALOG)(self.ctx)
            own = next(item for item in cast(list[dict[str, Any]], graph["plugins"]) if item["id"] == self.ctx.runtime.plugin_id)
            fiber = next(item for item in own["composition"]["fibers"] if item["fiber_id"] == self.function.fiber_id)
            missing = fiber["missing_services"]
            reason = "需要先启用 standard_web（Web 工具）" if missing else (fiber["error"] or "")
            return {**self.ctx.require(PLUGIN_CONFIG).read(self.ctx), "enabled": True,
                    "ready": self.function.state.value == "active", "can_enable": True, "blocked": False,
                    "reason": reason, "values": {key: values[key] for key in FIELDS},
                    "has_token": config.api_key is not None}

    # 1. 只接受已知字段；2. 空 key 保留原凭据；3. 校验通过后经宿主换代生效。
    async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]:
        async with self.ctx.runtime_scope():
            if set(values) - {*FIELDS, "enabled", "api_key"} or values.get("enabled", True) is not True:
                raise ValueError("配置包含不支持的字段")
            key = values.get("api_key")
            if key is not None and not isinstance(key, str):
                raise ValueError("API key 必须是字符串")
            config = dict(self.ctx.config)
            config.update({name: value for name, value in values.items() if name in FIELDS})
            check(config)
            if key and key.strip():
                config["api_key"] = save_credential(self.ctx.data_root, key.strip())
            checked = ExaConfig.model_validate(config)
            stored: dict[str, object] = checked.model_dump(mode="json", exclude={"api_key"})
            if checked.api_key is not None:
                stored["api_key"] = checked.api_key
            return await self.ctx.require(PLUGIN_CONFIG).apply(
                self.ctx, request_id=request_id, expected_input=expected_input, config=stored)

    def receipt(self, request_id: str) -> dict[str, object]:
        return self.ctx.entrypoint(lambda: self.ctx.require(PLUGIN_CONFIG).receipt(self.ctx, request_id))()


# 表单只显示校验原因，不带 pydantic 的错误类型前缀。
def check(config: dict[str, object]) -> None:
    try:
        _ = ExaConfig.model_validate(config)
    except ValidationError as error:
        raise ValueError("；".join(str(item["msg"]).removeprefix("Value error, ") for item in error.errors())) from None


async def mount(ctx: Context, function: FiberHandle) -> None:
    settings = Settings(ctx, function)
    await ctx.provide(SETTINGS, settings)

    async def ui(child: Context) -> None:
        await child.require(UI).register(child, web="web_module.js", dashboard=lambda: import_module(".dashboard", __package__),
                                         requires=("shell.settings-plugins.v1",), contract_digests=UI_CONTRACT)
    await ctx.inject((UI, SETTINGS), ui, name="settings-ui")
