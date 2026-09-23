from __future__ import annotations

from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from agent.plugin_composition.context import Context
from agent.plugin_composition.channels import CredentialRef, ProviderClient, ProviderClientFactory
from agent.plugin_composition.model import ServiceKey


class CredentialClients:
    """只为实际插件 owner 打开其固定配置的短凭据租约。"""

    def __init__(self, factories: Mapping[object, ProviderClientFactory] | None):
        self._factories: dict[tuple[str, str], ProviderClientFactory] | None = (
            None if factories is None else {
                key if isinstance(key, tuple) else (str(key), "*"): factory
                for key, factory in factories.items()
            }
        )

    def add_factory(
        self, plugin_id: str, generation_id: str, factory: ProviderClientFactory,
    ) -> None:
        """Add one generation's factory to the stable formal facade."""
        if self._factories is None:
            raise RuntimeError("candidate 验证期不能添加正式凭据授权")
        key = (plugin_id, generation_id)
        if key in self._factories:
            raise RuntimeError(f"凭据 factory 已存在: {plugin_id}/{generation_id}")
        self._factories[key] = factory

    async def remove_factory(self, plugin_id: str, generation_id: str) -> None:
        """Close and remove one exact generation factory."""
        if self._factories is None:
            return
        key = (plugin_id, generation_id)
        factory = self._factories.get(key)
        if factory is None:
            return
        await factory.aclose()
        self._factories.pop(key, None)

    async def create(self, ctx: Context, refs: Mapping[str, CredentialRef]) -> ProviderClient:
        """为实际贡献 Context 创建凭据句柄；调用方负责确认释放。"""
        owner = ctx.require_runtime_owner(CREDENTIALS, self)
        if self._factories is None:
            raise RuntimeError("candidate 验证期禁止读取正式凭据")
        key = (ctx.runtime.plugin_id, ctx.runtime.generation_id)
        factory = self._factories.get(key) or self._factories.get((ctx.runtime.plugin_id, "*"))
        if factory is None:
            raise PermissionError("插件没有当前固定输入的凭据授权")
        return await factory.create(refs)

    @asynccontextmanager
    async def open(self, ctx: Context, refs: Mapping[str, CredentialRef]) -> AsyncGenerator[ProviderClient]:
        client = await self.create(ctx, refs)
        try:
            yield client
        finally:
            await client.aclose()

    async def aclose(self) -> None:
        if self._factories is not None:
            for factory in self._factories.values():
                await factory.aclose()


CREDENTIALS = ServiceKey[CredentialClients]("core.credentials")
