from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

from agent.plugin_composition.context import (
    CompositionRoot,
    Context,
    _current_runtime_scope,
    _lifecycle_binding,
)
from agent.plugin_composition.model import ServiceKey
from session.log import MessageLog
from session.message_codec import json_value

if TYPE_CHECKING:
    from agent.plugins.generation import PluginGeneration

_T = TypeVar("_T")


class Bindings:
    """保存业务选择与来源证据；打开当前 provider，由它校验业务兼容性。"""

    def __init__(
        self,
        log: MessageLog | None,
        root: CompositionRoot,
        generation_lookup: Callable[[Context], PluginGeneration] | None = None,
    ):
        self._storage = log
        self._root = root
        self._generation_lookup = generation_lookup

    @property
    def _log(self) -> MessageLog:
        if self._storage is None:
            raise RuntimeError("未提供 MessageLog，不能固定或打开持久 binding")
        return self._storage

    def bind(
        self,
        service: ServiceKey[Any],
        metadata: Mapping[str, object],
        *,
        contributors: tuple[Context, ...] = (),
    ) -> str:
        """从当前许可保存业务选择与来源身份，不保存代码或依赖快照。"""
        log = self._log
        current = _current_runtime_scope()
        if current is not None:
            owner_root = current._call._fiber.root  # pyright: ignore[reportPrivateUsage]
            owner_context = current._call._fiber.context  # pyright: ignore[reportPrivateUsage]
        else:
            lifecycle = _lifecycle_binding.get()
            if lifecycle is None or lifecycle[1] is not asyncio.current_task():
                raise RuntimeError("固定 binding 需要实际 OwnerCall 或 lifecycle owner")
            owner_root = lifecycle[0]._fiber.root  # pyright: ignore[reportPrivateUsage]
            owner_context = lifecycle[0]
        if owner_root is not self._root:
            raise RuntimeError("固定 binding 的所属 Root 不属于当前 OwnerCall")
        # 1. 调用者已经声明的依赖是权限边界。
        owner_context.require(service)
        root = self._root
        selected: set[str] = set()
        pending: list[Context] = []
        services: set[ServiceKey[Any]] = set()
        contexts: dict[int, Context] = {}
        origins: dict[str, str | None] = {}

        def provider_for(key: ServiceKey[Any], requester: Context):
            frozen = requester._fiber.dependency_store.get(  # pyright: ignore[reportPrivateUsage]
                key,
            )
            if frozen is not None:
                return frozen
            provider = root._providers.get(key)  # pyright: ignore[reportPrivateUsage]
            if provider is None or provider.owner is not requester._fiber:  # pyright: ignore[reportPrivateUsage]
                raise ValueError(f"调用者未声明服务依赖: {key.name}")
            return provider

        def include_context(context: Context) -> None:
            identity = id(context)
            if identity in contexts:
                return
            if self._generation_lookup is not None:
                generation = self._generation_lookup(context)
                contributor = generation.plugin_id
                origins[contributor] = generation.generation_id
            else:
                contributor = root.context_owner(context)
                if contributor is None:
                    raise ValueError("注册 Context 不属于当前 active Root")
            contexts[identity] = context
            selected.add(contributor)
            pending.append(context)

        def include_service(key: ServiceKey[Any], requester: Context) -> None:
            if key in services:
                return
            services.add(key)
            provider = provider_for(key, requester)
            provider_context = provider.owner.context
            runtime = provider.owner.runtime
            if runtime is None:
                # Core 服务沿已声明的依赖授权，不属于插件来源。
                return
            include_context(provider_context)
            if provider.binding_contributors is not None:
                for context in provider.binding_contributors():
                    include_context(context)

        for context in contributors:
            include_context(context)
        include_service(service, owner_context)
        if not selected:
            raise ValueError("Core 服务绑定需要实际目标注册 owner")
        while pending:
            context = pending.pop()
            for key in context._fiber.dependencies:  # pyright: ignore[reportPrivateUsage]
                include_service(key, context)
        # 2. 仅记录实际参与的插件身份，恢复时仍由当前 provider 校验业务选择。
        if self._generation_lookup is None:
            origins = {name: None for name in sorted(selected)}
        descriptor: dict[str, object] = {
            "version": 2,
            "origins": origins,
            "service": service.name,
            "metadata": metadata,
        }
        payload = json.dumps(
            json_value(descriptor),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        identity = hashlib.sha256(payload.encode()).hexdigest()
        log.save_binding(identity, descriptor)
        return identity

    def describe(self, identity: str, service: ServiceKey[Any]) -> Mapping[str, object]:
        """只读绑定的业务选择；展示或请求投影无需启动历史目标。"""
        return cast(Mapping[str, object], self._read_descriptor(identity, service)["metadata"])

    @asynccontextmanager
    async def open(
        self, identity: str, service: ServiceKey[_T]
    ) -> AsyncIterator[tuple[_T, Mapping[str, object]]]:
        """在当前 Root 打开真实 provider；业务兼容性由该服务的 open 检查。"""
        metadata = self.describe(identity, service)
        current = _current_runtime_scope()
        if current is not None and current._call._fiber.root is not self._root:  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError("打开 binding 的所属 Root 不属于当前 runtime scope")
        provider_context, value = self._root._service_provider(service)  # pyright: ignore[reportPrivateUsage]
        async with provider_context.runtime_scope():
            yield cast(_T, value), cast(Mapping[str, object], metadata)

    def _read_descriptor(
        self, identity: str, service: ServiceKey[Any]
    ) -> Mapping[str, object]:
        """读取并校验 binding descriptor 的共同结构。"""
        descriptor = self._log.read_binding(identity)
        if descriptor["version"] not in {1, 2}:
            raise ValueError("binding 版本或服务不匹配")
        # 旧 Commands 选择属于不可变事实；只解释存储表示，不改 hash 或注册旧 key。
        stored_service = descriptor["service"]
        if stored_service == "core.commands":
            stored_service = "commands.v1"
        if stored_service != service.name:
            raise ValueError("binding 版本或服务不匹配")
        metadata = descriptor["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("binding metadata 必须是对象")
        return descriptor


BINDINGS = ServiceKey[Bindings]("core.bindings")
