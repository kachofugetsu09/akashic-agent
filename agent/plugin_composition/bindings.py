from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, TypeVar, cast

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.context import CompositionRoot, Context
from session.log import MessageLog
from session.message_codec import json_value

if TYPE_CHECKING:
    from agent.plugins.archive import PluginArchive

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
    """固定服务闭包与调用者的不可变选择；不保存业务执行状态。"""

    def __init__(
        self,
        log: MessageLog,
        archive: PluginArchive,
        open_components: Callable[
            [tuple[str, ...]], AbstractAsyncContextManager[BindingScope]
        ],
    ):
        self._log = log
        self._archive = archive
        self._open_components = open_components

    def bind(
        self,
        service: ServiceKey[object],
        metadata: Mapping[str, object],
        *,
        contributors: tuple[Context, ...] = (),
    ) -> str:
        """从当前真实 lease 固定实现，随后 Message 可原子引用此 binding。"""
        from agent.plugins.snapshot import get_current_runtime_lease

        lease = get_current_runtime_lease()
        if lease is None or lease.snapshot.composition_root is None:
            raise RuntimeError("固定 binding 需要实际 runtime scope")
        if lease.snapshot.composition_root.context.get(service) is None:
            raise RuntimeError(f"当前 scope 不提供服务: {service.name}")
        # 1. 服务 provider 与目标注册 owner 是闭包入口，依赖只向上展开。
        root = lease.snapshot.composition_root
        owners = root.plugin_service_owners()
        dependencies = root.plugin_dependencies()
        selected: set[str] = set()
        for context in contributors:
            contributor = root.context_owner(context)
            if contributor is None:
                raise ValueError("注册 Context 不属于当前 scope")
            selected.add(contributor)
        owner = owners.get(service)
        if owner is not None:
            selected.add(owner)
        if not selected:
            raise ValueError("Core 服务绑定需要实际目标注册 owner")
        pending = list(selected)
        while pending:
            plugin_id = pending.pop()
            if plugin_id not in lease.snapshot.generations:
                raise ValueError(f"注册 owner 不属于当前 scope: {plugin_id}")
            for key in dependencies[plugin_id]:
                provider = owners.get(key)
                if provider is not None and provider not in selected:
                    selected.add(provider)
                    pending.append(provider)
        components: list[str] = []
        for plugin_id in sorted(selected):
            generation = lease.snapshot.generations[plugin_id]
            if generation.archive_ref is None:
                raise RuntimeError(f"插件缺少加载时归档: {generation.plugin_id}")
            components.append(generation.archive_ref)
        root_ref = self._archive.save_descriptor({"components": components})
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
        self._log.save_binding(identity, descriptor)
        return identity

    @asynccontextmanager
    async def open(
        self, identity: str, service: ServiceKey[_T]
    ) -> AsyncIterator[tuple[_T, Mapping[str, object]]]:
        """只打开记录中的闭包；当前安装或默认 provider 不参与选择。"""
        descriptor = self._log.read_binding(identity)
        if descriptor["version"] != 1 or descriptor["service"] != service.name:
            raise ValueError("binding 版本或服务不匹配")
        root = self._archive.read_descriptor(cast(str, descriptor["root_ref"]))
        components = root["components"]
        metadata = descriptor["metadata"]
        if not isinstance(components, tuple) or not isinstance(metadata, Mapping):
            raise ValueError("binding descriptor 结构无效")
        async with self._open_components(cast(tuple[str, ...], components)) as scope:
            yield scope.require(service), cast(Mapping[str, object], metadata)


BINDINGS = ServiceKey[Bindings]("core.bindings")
