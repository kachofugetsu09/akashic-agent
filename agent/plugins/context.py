from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from infra.persistence.json_store import atomic_save_json, load_json

T = TypeVar("T")
_CLEANUP_WRITER_ID: ContextVar[str] = ContextVar(
    "plugin_cleanup_writer_id",
    default="",
)

if TYPE_CHECKING:
    from pydantic import BaseModel
    from agent.plugins.config import PluginConfig
    from agent.plugins.jobs import PluginLlmService
    from agent.plugins.scope import Cleanup, PluginScope, ScopedEventBus


# 插件兼容接口：现有插件直接读取这些字段并通过这里登记任务、进程和清理。
# 核心重构可以改变内部装配方式，但迁移插件前不得删除字段或改变阶段权限。
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
    _can_start_tasks: Callable[[], bool] | None = field(
        default=None,
        repr=False,
    )

    def create_task(
        self,
        coroutine: Coroutine[Any, Any, T],
        *,
        name: str | None = None,
    ) -> asyncio.Task[T]:
        if self.scope is None:
            raise RuntimeError(f"插件缺少资源作用域: {self.plugin_id}")
        if self._can_start_tasks is None or not self._can_start_tasks():
            coroutine.close()
            raise RuntimeError("prepare 阶段禁止启动后台任务")
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


@contextmanager
def allow_plugin_cleanup_writes(writer_id: str) -> Iterator[None]:
    """只允许当前清理 task 使用指定 generation 的 KV writer。"""

    token = _CLEANUP_WRITER_ID.set(writer_id)
    try:
        yield
    finally:
        _CLEANUP_WRITER_ID.reset(token)


class PluginKVStore:
    def __init__(
        self,
        path: Path,
        *,
        writable: bool = True,
        can_write: Callable[[], bool] | None = None,
        writer_id: str = "",
    ) -> None:
        self._path = path
        self._writable = writable
        self._can_write = can_write
        self._writer_id = writer_id

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
        cleanup_write = _CLEANUP_WRITER_ID.get() == self._writer_id
        if self._can_write is not None and not self._can_write() and not cleanup_write:
            raise RuntimeError(
                f"已退役 generation 禁止写入插件 KV: {self._writer_id}"
            )


class PreparedPluginKVStore(PluginKVStore):
    """Stage candidate KV changes until the generation is committed."""

    def __init__(
        self,
        path: Path,
        *,
        can_write: Callable[[], bool],
        writer_id: str,
    ) -> None:
        super().__init__(
            path,
            can_write=can_write,
            writer_id=writer_id,
        )
        self._prepared_data = super()._read()
        self._is_committed = False
        self._is_dirty = False

    def get(self, key: str, default: Any = None) -> Any:
        if self._is_committed:
            return super().get(key, default)
        return self._prepared_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if self._is_committed:
            super().set(key, value)
            return
        self._prepared_data[key] = value
        self._is_dirty = True

    def increment(self, key: str, delta: int = 1) -> int:
        if self._is_committed:
            return super().increment(key, delta)
        new_value = int(self._prepared_data.get(key, 0)) + delta
        self._prepared_data[key] = new_value
        self._is_dirty = True
        return new_value

    @property
    def dirty(self) -> bool:
        return self._is_dirty

    def commit(self) -> None:
        if self._is_committed:
            return
        if self._is_dirty:
            super()._write(self._prepared_data)
        self._is_committed = True
