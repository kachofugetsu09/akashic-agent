from __future__ import annotations

import json
import subprocess
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

T = TypeVar("T")

if TYPE_CHECKING:
    from pydantic import BaseModel
    from agent.plugins.config import PluginConfig
    from agent.plugins.jobs import PluginLlmService
    from agent.plugins.scope import PluginScope, ScopedEventBus


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
    ) -> Any:
        if self.scope is None:
            raise RuntimeError(f"插件缺少资源作用域: {self.plugin_id}")
        return self.scope.create_task(coroutine, name=name)

    def defer(self, resource: str, cleanup: Any) -> None:
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
        if not self._path.exists():
            return {}
        return json.loads(self._path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _ = self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _require_writable(self) -> None:
        if not self._writable:
            raise RuntimeError("候选声明阶段禁止写入插件 KV")
