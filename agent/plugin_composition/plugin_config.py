"""插件只能提交自身固定配置；正式采用和排空归宿主。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey


class ConfigHost(Protocol):
    def read_config_input(self, plugin_id: str) -> dict[str, object]: ...
    async def apply_config_input(self, plugin_id: str, request_id: str,
                                 expected_input: str, config: Mapping[str, object]) -> dict[str, object]: ...
    def read_config_update(self, plugin_id: str, request_id: str) -> dict[str, object]: ...


class PluginConfig:
    """按当前 Context 限制配置应用，不授予安装或其他插件的 writer。"""

    def __init__(self, host: ConfigHost | None):
        self._host = host

    def _owner(self, ctx: Context) -> tuple[ConfigHost, str]:
        ctx.require_runtime_owner(PLUGIN_CONFIG, self)
        if self._host is None:
            raise PermissionError("配置应用宿主不可用")
        return self._host, ctx.runtime.plugin_id

    def read(self, ctx: Context) -> dict[str, object]:
        host, owner = self._owner(ctx)
        return host.read_config_input(owner)

    async def apply(self, ctx: Context, *, request_id: str, expected_input: str,
                    config: Mapping[str, object]) -> dict[str, object]:
        host, owner = self._owner(ctx)
        return await host.apply_config_input(owner, request_id, expected_input, config)

    def receipt(self, ctx: Context, request_id: str) -> dict[str, object]:
        host, owner = self._owner(ctx)
        return host.read_config_update(owner, request_id)


PLUGIN_CONFIG = ServiceKey[PluginConfig]("core.plugin_config.v1")
