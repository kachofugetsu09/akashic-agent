from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeVar, cast

from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.context import CompositionRoot, Context, RuntimeScope
from session.log import MessageLog
from session.message_codec import json_value

if TYPE_CHECKING:
    from agent.plugin_composition.archive import PluginArchive

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
    ):
        self._storage = log
        self._archive = archive
        self._root = root

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
        """从当前真实 lease 固定实现，随后 Message 可原子引用此 binding。"""
        from agent.plugins.snapshot import get_current_runtime_lease

        log = self._log
        lease = get_current_runtime_lease()
        if lease is None or lease.snapshot.composition_root is None:
            raise RuntimeError("固定 binding 需要实际 runtime scope")
        if not self._root_is_selected(lease.snapshot.composition_root):
            raise RuntimeError("固定 binding 的所属 Root 不属于当前 runtime scope")
        if lease.snapshot.composition_root.context.get(service) is None:
            raise RuntimeError(f"当前 scope 不提供服务: {service.name}")
        # 1. 服务 provider 与目标注册 owner 是闭包入口，依赖只向上展开。
        root = lease.snapshot.composition_root
        owners = root.plugin_service_owners()
        dependencies = root.plugin_dependencies()
        selected: set[str] = set()
        pending: list[str] = []
        services: set[ServiceKey[object]] = set()

        def include_context(context: Context) -> None:
            contributor = root.context_owner(context)
            if contributor is None:
                raise ValueError("注册 Context 不属于当前 scope")
            include_owner(contributor)

        def include_owner(plugin_id: str) -> None:
            if plugin_id not in selected:
                selected.add(plugin_id)
                pending.append(plugin_id)

        def include_service(key: ServiceKey[object]) -> None:
            if key in services:
                return
            services.add(key)
            owner = owners.get(key)
            if owner is not None:
                include_owner(owner)
                for context in root.binding_contributors(key):
                    include_context(context)

        for context in contributors:
            include_context(context)
        include_service(service)
        if not selected:
            raise ValueError("Core 服务绑定需要实际目标注册 owner")
        while pending:
            plugin_id = pending.pop()
            if plugin_id not in lease.snapshot.generations:
                raise ValueError(f"注册 owner 不属于当前 scope: {plugin_id}")
            for key in dependencies[plugin_id]:
                include_service(key)
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
        log.save_binding(identity, descriptor)
        return identity

    def describe(self, identity: str, service: ServiceKey[object]) -> Mapping[str, object]:
        """只读绑定的业务选择；展示或请求投影无需启动归档目标。"""
        return cast(Mapping[str, object], self._read_descriptor(identity, service)["metadata"])

    @asynccontextmanager
    async def open(
        self, identity: str, service: ServiceKey[_T]
    ) -> AsyncIterator[tuple[_T, Mapping[str, object]]]:
        """在调用者已选的 Root 中打开服务；缺 scope 时只从所属 Root 获取一次。"""
        metadata = self.describe(identity, service)
        from agent.plugins.snapshot import get_current_runtime_lease

        current = get_current_runtime_lease()
        if current is None:
            lease = await self._root._acquire_runtime_scope()  # pyright: ignore[reportPrivateUsage]
        else:
            lease = current.fork()

        async with RuntimeScope(lease):
            root = lease.snapshot.composition_root
            if root is None:
                raise RuntimeError("打开 binding 需要实际 runtime scope")
            if root is not self._root:
                raise RuntimeError("打开 binding 的所属 Root 不属于当前 runtime scope")
            value = root.context.get(service)
            if value is None:
                raise RuntimeError(f"当前 runtime scope 不提供服务: {service.name}")
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
