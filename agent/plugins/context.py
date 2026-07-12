from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from infra.persistence.json_store import atomic_save_json, load_json

T = TypeVar("T")

if TYPE_CHECKING:
    from pydantic import BaseModel
    from agent.plugins.config import PluginConfig
    from agent.plugins.jobs import PluginLlmService
    from agent.plugins.scope import Cleanup, PluginScope, ScopedEventBus


@dataclass
class PluginContext:
    event_bus: "ScopedEventBus"
    tool_registry: Any
    plugin_id: str
    plugin_dir: Path
    data_dir: Path | None
    kv_store: "PluginKVStore"
    config: "BaseModel | PluginConfig | None" = None
    workspace: Path | None = None
    session_manager: Any = None
    memory_engine: Any = None
    llm: "PluginLlmService | None" = None
    scope: "PluginScope | None" = None
    generation_id: str = ""

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        if self.scope is None:
            raise RuntimeError(f"插件缺少资源作用域: {self.plugin_id}")
        return self.scope.create_task(coroutine, name=name)

    def defer(self, resource: str, cleanup: "Cleanup") -> None:
        if self.scope is None:
            raise RuntimeError(f"插件缺少资源作用域: {self.plugin_id}")
        self.scope.defer(resource, cleanup)

    def track_process(
        self,
        process: subprocess.Popen[Any],
        *,
        name: str,
        timeout: float = 5,
    ) -> None:
        if self.scope is None:
            raise RuntimeError(f"插件缺少资源作用域: {self.plugin_id}")
        self.scope.track_process(process, name=name, timeout=timeout)


class PluginKVStore:
    def __init__(self, path: Path, *, writable: bool = True) -> None:
        self._path = path
        self._writable = writable

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._require_writable()
        data = self._read()
        data[key] = value
        self._write(data)

    def increment(self, key: str, delta: int = 1) -> int:
        self._require_writable()
        data = self._read()
        new_val = int(data.get(key, 0)) + delta
        data[key] = new_val
        self._write(data)
        return new_val

    def _read(self) -> dict[str, Any]:
        data = load_json(
            self._path,
            default={},
            domain=f"plugin_kv:{self._path}",
        )
        if not isinstance(data, dict):
            raise ValueError(f"插件 KV 根节点必须是对象: {self._path}")
        return cast(dict[str, Any], data)

    def _write(self, data: dict[str, Any]) -> None:
        atomic_save_json(
            self._path,
            data,
            ensure_ascii=False,
            domain=f"plugin_kv:{self._path}",
        )

    def _require_writable(self) -> None:
        if not self._writable:
            raise RuntimeError("候选声明阶段禁止写入插件 KV")
