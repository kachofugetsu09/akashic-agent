from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.plugins.config import PluginConfig


@dataclass
class PluginContext:
    event_bus: Any
    tool_registry: Any
    plugin_id: str
    plugin_dir: Path
    kv_store: "PluginKVStore"
    config: "PluginConfig | None" = None
    workspace: Path | None = None
    observe_db_path: Path | None = None


class PluginKVStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def get(self, key: str, default: Any = None) -> Any:
        return self._read().get(key, default)

    def set(self, key: str, value: Any) -> None:
        # 1. 读取现有数据，写入新值，落盘
        data = self._read()
        data[key] = value
        self._write(data)

    def increment(self, key: str, delta: int = 1) -> int:
        # 1. 读取 → 加 delta → 写回，返回新值
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
