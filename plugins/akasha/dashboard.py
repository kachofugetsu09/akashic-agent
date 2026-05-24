from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException

from plugins.akasha.config import AkashaConfig, load_akasha_config, resolve_akasha_db_path
from plugins.akasha.store import AkashaStore


def plugin_enabled(app: FastAPI) -> bool:
    return _active_memory_engine(app) == "akasha"


class AkashaInspectorReader:
    def __init__(self, store: AkashaStore) -> None:
        self._store = store
        self._lock = threading.RLock()

    def get_overview(self) -> dict[str, Any]:
        items, total = self._store.list_query_logs(page=1, page_size=1)
        latest = items[0]["ts"] if items else None
        return {"available": True, "total": total, "latest_at": latest}

    def list_turns(
        self,
        *,
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        return self._store.list_query_logs(
            session_key=session_key,
            q=q,
            page=page,
            page_size=page_size,
        )

    def get_turn(self, query_id: str) -> dict[str, Any] | None:
        raw = self._store.get_query_log(query_id)
        if raw is None:
            return None
        result = dict(raw)
        # 反序列化 JSON 列
        for json_key, out_key in [
            ("activation_items_json", "activation_items"),
            ("dense_items_json", "dense_items"),
            ("ripple_items_json", "ripple_items"),
        ]:
            raw_json = result.pop(json_key, "[]")
            try:
                parsed = json.loads(str(raw_json))
            except Exception:
                parsed = []
            result[out_key] = parsed if isinstance(parsed, list) else []
        return cast(dict[str, Any], result)


def register(app: FastAPI, plugin_dir: Path, workspace: Path) -> list[object]:
    _ = plugin_dir

    # 找到 akasha sidecar DB 路径（复用 engine 的 config 解析）。
    akasha_config = _load_akasha_config(workspace)
    if akasha_config is None:
        return []

    store = AkashaStore(resolve_akasha_db_path(workspace=workspace, akasha_config=akasha_config))
    reader = AkashaInspectorReader(store)

    @app.get("/api/dashboard/akasha-inspector/overview")
    def get_akasha_inspector_overview() -> dict[str, Any]:
        return reader.get_overview()

    @app.get("/api/dashboard/akasha-inspector/turns")
    def list_akasha_inspector_turns(
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = reader.list_turns(
            session_key=session_key,
            q=q,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/akasha-inspector/turns/{query_id:path}")
    def get_akasha_inspector_turn(query_id: str) -> dict[str, Any]:
        item = reader.get_turn(query_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Akasha 检索记录不存在")
        return item

    return [store]


def _active_memory_engine(app: FastAPI) -> str:
    memory_admin = getattr(app.state, "memory_admin", None)
    describe = getattr(memory_admin, "describe", None)
    if not callable(describe):
        return ""
    return str(describe().name)


def _load_akasha_config(workspace: Path) -> AkashaConfig | None:
    # 从插件目录的 config.local.toml 加载；读取失败返回默认配置。
    _ = workspace
    try:
        plugin_dir = Path(__file__).resolve().parent
        return load_akasha_config(plugin_dir=plugin_dir)
    except Exception:
        return AkashaConfig()
