from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

from agent.plugin_composition.context import Context
from agent.plugin_composition.model import ServiceKey
from .log import MessageLog
from plugins.ledger.contract import json_value

_T = TypeVar("_T")


class Bindings:
    """保存业务选择与来源证据；打开当前 provider，由它校验业务兼容性。"""

    def __init__(self, log: MessageLog, context: Context):
        self._log = log
        self._context = context

    def bind(
        self,
        service: ServiceKey[Any],
        metadata: Mapping[str, object],
        *,
        contributors: tuple[Context, ...] = (),
    ) -> str:
        """从当前许可保存业务选择与来源身份，不保存代码或依赖快照。"""
        log = self._log
        origins = self._context.service_origins(service, contributors=contributors)
        if not origins:
            raise ValueError("Core 服务绑定需要实际目标注册 owner")
        descriptor: dict[str, object] = {
            "version": 2,
            "origins": {item.plugin_id: item.generation_id for item in origins},
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
        async with self._context.open_service(service) as value:
            yield cast(_T, value), cast(Mapping[str, object], metadata)

    def _read_descriptor(
        self, identity: str, service: ServiceKey[Any]
    ) -> Mapping[str, object]:
        """读取并校验 binding descriptor 的共同结构。"""
        descriptor = self._log.read_binding(identity)
        if descriptor["version"] not in {1, 2}:
            raise ValueError("binding 版本或服务不匹配")
        # 历史名称由合同 owner 声明；只读原事实，不注册别名或改写 descriptor。
        if descriptor["service"] not in (service.name, *service.binding_names):
            raise ValueError("binding 版本或服务不匹配")
        metadata = descriptor["metadata"]
        if not isinstance(metadata, Mapping):
            raise ValueError("binding metadata 必须是对象")
        return descriptor

