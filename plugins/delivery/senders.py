from __future__ import annotations

import re
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.bindings import Bindings
from session.message import Message

from .api import Receipt, Sender, Text

Open = Callable[[], AbstractAsyncContextManager[Sender]]


class Adapter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: Text
    owner: Text
    idempotent: bool


@dataclass(frozen=True, slots=True)
class _Registration:
    context: Context
    descriptor: Adapter
    open: Open


class _SenderView:
    """短命发送入口随资源作用域关闭；不能启动收件循环。"""

    def __init__(self, target: Sender):
        self._target = target
        self._active = True

    def _check(self) -> None:
        if not self._active:
            raise RuntimeError("发送 binding scope 已释放")

    @property
    def idempotent(self) -> bool:
        self._check()
        return self._target.idempotent

    async def send(self, key: str, address: str, message: Message) -> Receipt:
        self._check()
        return await self._target.send(key, address, message)

    async def query(self, key: str, address: str) -> Receipt | None:
        self._check()
        return await self._target.query(key, address)

    def close(self) -> None:
        self._active = False


class Senders:
    """普通渠道的出站注册表；只打开固定目标，不创建 Channel 收件实例。"""

    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._registrations: dict[str, _Registration] = {}

    async def register(self, ctx: Context, *, name: str, idempotent: bool, open: Open) -> Effect:
        """open 只取得发送资源，不得发送正文或启动收件；配置随真实 owner 归档。"""
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("发送注册不能跨 composition Root")
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) is None:
            raise ValueError("发送 adapter 名称无效")
        descriptor = Adapter(name=name, owner=ctx.runtime.plugin_id, idempotent=idempotent)

        def setup() -> Callable[[], None]:
            if name in self._registrations:
                raise ValueError(f"发送 adapter 重复: {name}")
            self._registrations[name] = _Registration(ctx, descriptor, open)

            def cleanup() -> None:
                del self._registrations[name]
            return cleanup

        return await ctx.effect(setup, label="sender:" + name)

    def registered_names(self) -> tuple[str, ...]:
        """只读当前可用名称，不创建绑定或打开任何 provider 资源。"""
        return tuple(sorted(self._registrations))

    def bind(self, name: str, bindings: Bindings) -> str:
        registration = self._registrations[name]
        return bindings.bind(
            DELIVERY_SENDERS, registration.descriptor.model_dump(),
            contributors=(registration.context,),
        )

    def bind_all(self, bindings: Bindings) -> Mapping[str, str]:
        """固定当前可选发送者；归档工具按此集合选路，不读取之后的注册表。"""
        return {name: self.bind(name, bindings) for name in sorted(self._registrations)}

    @asynccontextmanager
    async def open(self, metadata: Mapping[str, object]) -> AsyncGenerator[Sender]:
        """归档自身核对目标和幂等合同；不读取当前渠道注册或运行快照。"""
        descriptor = Adapter.model_validate(dict(metadata))
        registration = self._registrations[descriptor.name]
        if registration.descriptor != descriptor:
            raise ValueError("发送 binding 与归档注册不一致")
        async with self._ctx.runtime_scope():
            async with registration.open() as target:
                if target.idempotent != descriptor.idempotent:
                    raise ValueError("发送幂等协议与固定描述不一致")
                view = _SenderView(target)
                try:
                    yield view
                finally:
                    view.close()


DELIVERY_SENDERS = ServiceKey[Senders]("delivery.senders.v1")


@asynccontextmanager
async def open_sender(bindings: Bindings, binding_id: str) -> AsyncGenerator[Sender]:
    async with bindings.open(binding_id, DELIVERY_SENDERS) as (senders, metadata):
        async with senders.open(metadata) as sender:
            yield sender
