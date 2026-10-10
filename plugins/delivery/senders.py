from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass

from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import Context, Effect
from plugins.ledger.contract import Bindings
from plugins.ledger.contract import Message
from plugins.delivery.contract import (
    DELIVERY_SENDERS as DELIVERY_SENDERS,
    SenderDefinition,
)

from .api import Receipt, Sender, SenderResult, Text

Open = Callable[[], AbstractAsyncContextManager[Sender]]


class _SavedSender(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    name: Text
    owner: Text
    idempotent: bool


@dataclass(frozen=True, slots=True)
class _Registration:
    context: Context
    descriptor: SenderDefinition
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
        return _receipt(await self._target.send(key, address, message))

    async def query(self, key: str, address: str) -> Receipt | None:
        self._check()
        result = await self._target.query(key, address)
        return None if result is None else _receipt(result)

    def close(self) -> None:
        self._active = False


def _receipt(result: SenderResult) -> Receipt:
    """在 sender 与 Delivery 的边界校验并归一化 provider 结果。"""
    return Receipt.model_validate({
        "status": result.status,
        "provider_ids": result.provider_ids,
        "error": result.error,
    })


class Senders:
    """普通渠道的出站注册表；只打开固定目标，不创建 Channel 收件实例。"""

    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._registrations: dict[str, _Registration] = {}
        self._candidates: dict[str, tuple[Context, str, str, Callable[[], Mapping[str, object]]]] = {}

    async def register(self, ctx: Context, *, name: str, idempotent: bool, open: Open) -> Effect:
        """open 只取得发送资源，不得发送正文或启动收件；配置随真实 owner 归档。"""
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("发送注册不能跨 composition Root")
        key = SenderDefinition.key(name)
        descriptor = SenderDefinition(name=name, owner=ctx.runtime.plugin_id, idempotent=idempotent)

        def setup() -> Callable[[], None]:
            if name in self._registrations:
                raise ValueError(f"发送 adapter 重复: {name}")
            self._registrations[name] = _Registration(ctx, descriptor, open)

            def cleanup() -> None:
                del self._registrations[name]
            return cleanup

        async def start():
            cleanup = setup()
            try:
                presence = await ctx.provide(key, descriptor)
            except BaseException:
                cleanup()
                raise
            async def close():
                await presence.aclose()
                cleanup()
            return close
        return await ctx.effect(start, label="sender:" + name)

    async def candidate(self, ctx: Context, *, name: str, title: str, route: str,
                        status: Callable[[], Mapping[str, object]]) -> Effect:
        """只登记可配置候选，不创建传输或发送权限。"""
        SenderDefinition.key(name)
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("发送候选不能跨运行图")
        def start():
            if name in self._candidates:
                raise ValueError(f"发送候选重复: {name}")
            self._candidates[name] = (ctx, title, route, ctx.entrypoint(status))
            return lambda: self._candidates.pop(name)
        return await ctx.effect(start, label=f"sender-candidate:{name}")

    def candidates(self) -> tuple[Mapping[str, object], ...]:
        rows = {name: {"name": name, "owner": item.descriptor.owner, "title": name,
                       "route": None, "enabled": True, "available": True}
                for name, item in self._registrations.items()}
        for name, (ctx, title, route, status) in self._candidates.items():
            rows[name] = {"name": name, "owner": ctx.runtime.plugin_id, "title": title,
                          "route": route, **status(), "available": name in self._registrations}
        return tuple(rows[name] for name in sorted(rows))

    def bind(self, name: str, bindings: Bindings) -> str:
        registration = self._registrations[name]
        return bindings.bind(
            DELIVERY_SENDERS, asdict(registration.descriptor),
            contributors=(registration.context,),
        )

    def bind_all(self, bindings: Bindings) -> Mapping[str, str]:
        """固定当前可选发送者；归档工具按此集合选路，不读取之后的注册表。"""
        return {name: self.bind(name, bindings) for name in sorted(self._registrations)}

    @asynccontextmanager
    async def open(self, metadata: Mapping[str, object]) -> AsyncGenerator[Sender]:
        """核对归档目标并在 Senders 与目标 owner scope 中打开 sender。"""
        saved = _SavedSender.model_validate(dict(metadata))
        descriptor = SenderDefinition(saved.name, saved.owner, saved.idempotent)
        registration = self._registrations[descriptor.name]
        if registration.descriptor != descriptor:
            raise ValueError("发送 binding 与归档注册不一致")
        async with self._ctx.runtime_scope():
            async with registration.context.runtime_scope():
                async with registration.open() as target:
                    if target.idempotent != descriptor.idempotent:
                        raise ValueError("发送幂等协议与固定描述不一致")
                    view = _SenderView(target)
                    try:
                        yield view
                    finally:
                        view.close()


@asynccontextmanager
async def open_sender(bindings: Bindings, binding_id: str) -> AsyncGenerator[Sender]:
    async with bindings.open(binding_id, DELIVERY_SENDERS) as (senders, metadata):
        async with senders.open(metadata) as sender:
            yield sender
