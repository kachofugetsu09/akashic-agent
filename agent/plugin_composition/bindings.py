from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, cast

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.context import (
    CompositionRoot,
    Context,
    _lifecycle_binding,
    _current_runtime_scope,
)
from session.log import MessageLog
from session.message_codec import json_value

if TYPE_CHECKING:
    from agent.plugin_composition.archive import PluginArchive
    from agent.plugins.generation import PluginGeneration

_T = TypeVar("_T")


class BindingScope:
    """只读取本次打开的精确服务，不能发布或改变正式 Root。"""

    def __init__(self, root: CompositionRoot):
        self._root = root
        self._active = True

    def require(self, key: ServiceKey[_T]) -> _T:
        if not self._active:
            raise RuntimeError("binding scope 已关闭")
        value = self._root.service_value(key)
        if value is None:
            raise RuntimeError(f"归档不提供服务: {key.name}")
        return value

    def _expire(self) -> None:
        self._active = False


class Bindings:
    """保存 binding 事实，并在调用者选定的 runtime scope 中打开服务。"""

    def __init__(
        self,
        log: MessageLog | None,
        archive: PluginArchive,
        root: CompositionRoot,
        generation_lookup: Callable[[Context], PluginGeneration] | None = None,
    ):
        self._storage = log
        self._archive = archive
        self._root = root
        self._generation_lookup = generation_lookup

    @property
    def _log(self) -> MessageLog:
        if self._storage is None:
            raise RuntimeError("candidate 验证期禁止固定或打开正式 binding")
        return self._storage

    def bind(
        self,
        service: ServiceKey[object],
        metadata: Mapping[str, object],
        *,
        contributors: tuple[Context, ...] = (),
    ) -> str:
        """从当前 OwnerCall 和真实 provider Context 固定实现。"""
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
        # 1. The caller's frozen dependency store is the authorization boundary.
        owner_context.require(service)
        root = self._root
        selected: set[str] = set()
        pending: list[Context] = []
        services: set[ServiceKey[object]] = set()
        contexts: dict[int, Context] = {}

        def provider_for(key: ServiceKey[object], requester: Context):
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
            if self._generation_lookup is not None:
                generation = self._generation_lookup(context)
                contributor = generation.plugin_id
            else:
                contributor = root.context_owner(context)
                if contributor is None:
                    raise ValueError("注册 Context 不属于当前 active Root")
            identity = id(context)
            if identity in contexts:
                return
            contexts[identity] = context
            selected.add(contributor)
            pending.append(context)

        def include_service(key: ServiceKey[object], requester: Context) -> None:
            if key in services:
                return
            services.add(key)
            provider = provider_for(key, requester)
            provider_context = provider.owner.context
            runtime = provider.owner.runtime
            if runtime is None:
                # Core providers are authorized by the requester's frozen
                # dependency store and are not archive components.
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
        components: set[str] = set()
        for context in contexts.values():
            if self._generation_lookup is None:
                raise RuntimeError("正式 binding 缺少 Manager generation lookup")
            generation = self._generation_lookup(context)
            if generation.archive_ref is None:
                raise RuntimeError(f"插件缺少加载时归档: {generation.plugin_id}")
            components.add(generation.archive_ref)
        root_ref = self._archive.save_descriptor({"components": tuple(sorted(components))})
        descriptor: dict[str, object] = {
            "version": 1,
            "root_ref": root_ref,
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

    def describe(self, identity: str, service: ServiceKey[object]) -> Mapping[str, object]:
        """只读绑定的业务选择；展示或请求投影无需启动归档目标。"""
        return cast(Mapping[str, object], self._read_descriptor(identity, service)["metadata"])

    @asynccontextmanager
    async def open(
        self, identity: str, service: ServiceKey[_T]
    ) -> AsyncIterator[tuple[_T, Mapping[str, object]]]:
        """在调用者已选的 Root 中打开 provider-owned 服务 scope。"""
        metadata = self.describe(identity, service)
        current = _current_runtime_scope()
        if current is not None and current._call._fiber.root is not self._root:  # pyright: ignore[reportPrivateUsage]
            raise RuntimeError("打开 binding 的所属 Root 不属于当前 runtime scope")
        provider_context, value = self._root._service_provider(service)  # pyright: ignore[reportPrivateUsage]
        async with provider_context.runtime_scope():
            yield cast(_T, value), cast(Mapping[str, object], metadata)

    def _read_descriptor(
        self, identity: str, service: ServiceKey[object]
    ) -> Mapping[str, object]:
        """读取并校验 binding descriptor 的共同结构。"""
        descriptor = self._log.read_binding(identity)
        if descriptor["version"] != 1 or descriptor["service"] != service.name:
            raise ValueError("binding 版本或服务不匹配")
        metadata = descriptor["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("binding metadata 必须是对象")
        return descriptor



BINDINGS = ServiceKey[Bindings]("core.bindings")
