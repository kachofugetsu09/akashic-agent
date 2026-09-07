from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from agent.plugin_composition.context import Context
from agent.plugin_composition.channels import CredentialRef, ProviderClient, ProviderClientFactory
from agent.plugin_composition.model import ServiceKey


class CredentialClients:
    """只为实际插件 owner 打开其固定配置的短凭据租约。"""

    def __init__(self, factories: Mapping[str, ProviderClientFactory] | None):
        self._factories = None if factories is None else dict(factories)

    @asynccontextmanager
    async def open(self, ctx: Context, refs: Mapping[str, CredentialRef]) -> AsyncGenerator[ProviderClient]:
        owner = ctx.require_runtime_owner(CREDENTIALS, self)
        if self._factories is None:
            raise RuntimeError("candidate 验证期禁止读取正式凭据")
        if owner not in self._factories:
            raise PermissionError("插件没有声明凭据读取范围")
        client = await self._factories[owner].create(refs)
        try:
            yield client
        finally:
            await client.aclose()

    async def aclose(self) -> None:
        if self._factories is not None:
            for factory in self._factories.values():
                await factory.aclose()


CREDENTIALS = ServiceKey[CredentialClients]("core.credentials")
