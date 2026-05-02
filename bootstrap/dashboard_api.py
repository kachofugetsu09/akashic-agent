from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import importlib.util
import logging
import json
import re
import sqlite3
import sys
import threading
from datetime import timedelta
from typing import Any, Protocol, cast

import subprocess

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from proactive_v2.memory_optimizer import MemoryOptimizerBusy
from proactive_v2.state import ProactiveStateStore
from core.common.timekit import utcnow
from session.store import SessionStore
from memory2.store import MemoryStore2

logger = logging.getLogger(__name__)

_DASHBOARD_ACCESS_PREFIXES = ("/api/dashboard", "/assets", "/plugins/")


def _is_dashboard_access_record(record: logging.LogRecord) -> bool:
    args = record.args
    if not isinstance(args, tuple) or len(args) < 3:
        return False
    path = args[2]
    if not isinstance(path, str):
        return False
    return path == "/" or any(
        path.startswith(prefix) for prefix in _DASHBOARD_ACCESS_PREFIXES
    )


# dashboard 会频繁轮询，访问日志只在 debug 模式保留。
class _DashboardAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not _is_dashboard_access_record(record):
            return True
        debug_enabled = logging.getLogger().isEnabledFor(
            logging.DEBUG
        ) or logging.getLogger("uvicorn.access").isEnabledFor(logging.DEBUG)
        if not debug_enabled:
            return False
        record.levelno = logging.DEBUG
        record.levelname = "DEBUG"
        return True


def _install_dashboard_access_log_filter() -> None:
    access_logger = logging.getLogger("uvicorn.access")
    if any(
        isinstance(filter_, _DashboardAccessLogFilter)
        for filter_ in access_logger.filters
    ):
        return
    access_logger.addFilter(_DashboardAccessLogFilter())


class SessionUpdatePayload(BaseModel):
    metadata: dict[str, Any] | None = None
    last_consolidated: int | None = None
    last_user_at: str | None = None
    last_proactive_at: str | None = None


class SessionBatchDeletePayload(BaseModel):
    keys: list[str]
    cascade: bool = True


class SessionConsolidatePayload(BaseModel):
    archive_all: bool = False
    force: bool = True


class MessageUpdatePayload(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_chain: Any | None = None
    extra: dict[str, Any] | None = None
    ts: str | None = None


class MessageBatchDeletePayload(BaseModel):
    ids: list[str]


class MemoryUpdatePayload(BaseModel):
    status: str | None = None
    extra_json: dict[str, Any] | None = None
    source_ref: str | None = None
    happened_at: str | None = None
    emotional_weight: int | None = None


class MemoryBatchDeletePayload(BaseModel):
    ids: list[str]


class ProactiveDeletePayload(BaseModel):
    source_key: str | None = None
    item_ids: list[str] | None = None


class ManualConsolidator(Protocol):
    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
    ) -> bool: ...


class ManualMemoryOptimizer(Protocol):
    @property
    def is_running(self) -> bool: ...

    async def optimize(self) -> None: ...


class ProactiveDashboardReader:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def get_overview(self) -> dict[str, Any]:
        counts = {
            "seen_items": self._count("seen_items"),
            "deliveries": self._count("deliveries"),
            "rejection_cooldown": self._count("rejection_cooldown"),
            "semantic_items": self._count("semantic_items"),
            "kv_state": self._count("kv_state"),
            "session_state": self._count("session_state"),
            "context_only_timestamps": self._count("context_only_timestamps"),
            "tick_logs": self._count("tick_log"),
            "tick_steps": self._count("tick_step_log"),
        }
        with self._lock:
            recent_tick = self._db.execute(
                """
                SELECT tick_id, session_key, started_at, finished_at, gate_exit,
                       terminal_action, skip_reason, steps_taken, drift_entered
                FROM tick_log
                ORDER BY started_at DESC
                LIMIT 1
                """
            ).fetchone()
            last_send_at = self._db.execute(
                """
                SELECT sent_at
                FROM deliveries
                ORDER BY sent_at DESC
                LIMIT 1
                """
            ).fetchone()
            result_counts_rows = self._db.execute(
                """
                SELECT COALESCE(terminal_action, gate_exit, 'unknown') AS bucket, COUNT(*) AS total
                FROM tick_log
                GROUP BY COALESCE(terminal_action, gate_exit, 'unknown')
                """
            ).fetchall()
            flow_counts_rows = self._db.execute(
                """
                SELECT CASE WHEN drift_entered = 1 THEN 'drift' ELSE 'proactive' END AS bucket,
                       COUNT(*) AS total
                FROM tick_log
                GROUP BY CASE WHEN drift_entered = 1 THEN 'drift' ELSE 'proactive' END
                """
            ).fetchall()
        result_counts = {str(row["bucket"]): int(row["total"]) for row in result_counts_rows}
        flow_counts = {str(row["bucket"]): int(row["total"]) for row in flow_counts_rows}
        return {
            "counts": counts,
            "result_counts": result_counts,
            "flow_counts": flow_counts,
            "last_tick_at": recent_tick["started_at"] if recent_tick is not None else None,
            "last_send_at": last_send_at["sent_at"] if last_send_at is not None else None,
            "last_skip_reason": (
                recent_tick["skip_reason"]
                if recent_tick is not None and recent_tick["terminal_action"] != "reply"
                else None
            ),
            "recent_tick": self._row_to_tick_log(recent_tick) if recent_tick is not None else None,
        }

    def list_deliveries(
        self,
        *,
        session_key: str = "",
        sent_from: str = "",
        sent_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = self._build_filters(
            ("session_key = ?", session_key),
            ("sent_at >= ?", sent_from),
            ("sent_at <= ?", sent_to),
        )
        return self._list_rows(
            table="deliveries",
            where=where,
            params=params,
            order_by="sent_at DESC, session_key ASC, delivery_key ASC",
            page=page,
            page_size=page_size,
            columns="session_key, delivery_key, sent_at",
        )

    def list_seen_items(
        self,
        *,
        source_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = self._build_filters(("source_key = ?", source_key))
        return self._list_rows(
            table="seen_items",
            where=where,
            params=params,
            order_by="seen_at DESC, source_key ASC, item_id ASC",
            page=page,
            page_size=page_size,
            columns="source_key, item_id, seen_at",
        )

    def list_rejection_cooldown(
        self,
        *,
        source_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        where, params = self._build_filters(("source_key = ?", source_key))
        return self._list_rows(
            table="rejection_cooldown",
            where=where,
            params=params,
            order_by="rejected_at DESC, source_key ASC, item_id ASC",
            page=page,
            page_size=page_size,
            columns="source_key, item_id, rejected_at",
        )

    def list_semantic_items(
        self,
        *,
        window_hours: int = 168,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        cutoff = (utcnow() - timedelta(hours=max(window_hours, 1))).isoformat()
        where, params = self._build_filters(("ts >= ?", cutoff))
        return self._list_rows(
            table="semantic_items",
            where=where,
            params=params,
            order_by="ts DESC, id DESC",
            page=page,
            page_size=page_size,
            columns="id, source_key, item_id, text, ts",
        )

    def list_tick_logs(
        self,
        *,
        session_key: str = "",
        terminal_action: str = "",
        gate_exit: str = "",
        flow: str = "",
        started_from: str = "",
        started_to: str = "",
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "started_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        drift_only = ""
        if flow == "drift":
            drift_only = "1"
        elif flow == "proactive":
            drift_only = "0"
        safe_sort_by = sort_by if sort_by in {
            "session_key",
            "started_at",
            "finished_at",
            "terminal_action",
            "gate_exit",
            "steps_taken",
            "alert_count",
            "content_count",
            "context_count",
            "drift_entered",
        } else "started_at"
        safe_sort_order = "ASC" if str(sort_order).lower() == "asc" else "DESC"
        where, params = self._build_filters(
            ("session_key = ?", session_key),
            ("terminal_action = ?", terminal_action),
            ("gate_exit = ?", gate_exit),
            ("drift_entered = ?", drift_only),
            ("started_at >= ?", started_from),
            ("started_at <= ?", started_to),
        )
        items, total = self._list_rows(
            table="tick_log",
            where=where,
            params=params,
            order_by=f"{safe_sort_by} {safe_sort_order}, id DESC",
            page=page,
            page_size=page_size,
            columns=(
                "tick_id, session_key, started_at, finished_at, gate_exit, "
                "terminal_action, skip_reason, steps_taken, alert_count, "
                "content_count, context_count, interesting_ids, discarded_ids, "
                "cited_ids, drift_entered, final_message"
            ),
            row_mapper=self._row_to_tick_log,
        )
        return items, total

    def get_tick_log(self, tick_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                """
                SELECT tick_id, session_key, started_at, finished_at, gate_exit,
                       terminal_action, skip_reason, steps_taken, alert_count,
                       content_count, context_count, interesting_ids, discarded_ids,
                       cited_ids, drift_entered, final_message
                FROM tick_log
                WHERE tick_id = ?
                """,
                (tick_id,),
            ).fetchone()
        return self._row_to_tick_log(row) if row is not None else None

    def list_tick_steps(self, tick_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                """
                SELECT step_index, phase, tool_name, tool_call_id, tool_args_json,
                       tool_result_text, terminal_action_after, skip_reason_after,
                       interesting_ids_after, discarded_ids_after, cited_ids_after,
                       final_message_after
                FROM tick_step_log
                WHERE tick_id = ?
                ORDER BY step_index ASC, id ASC
                """,
                (tick_id,),
            ).fetchall()
        return [self._row_to_tick_step(row) for row in rows]

    def delete_seen_items(
        self,
        *,
        source_key: str = "",
        item_ids: list[str] | None = None,
    ) -> int:
        return self._delete_rows("seen_items", source_key=source_key, item_ids=item_ids)

    def delete_rejection_cooldown(
        self,
        *,
        source_key: str = "",
        item_ids: list[str] | None = None,
    ) -> int:
        return self._delete_rows(
            "rejection_cooldown",
            source_key=source_key,
            item_ids=item_ids,
        )

    def _delete_rows(
        self,
        table: str,
        *,
        source_key: str = "",
        item_ids: list[str] | None = None,
    ) -> int:
        if not source_key and not item_ids:
            raise ValueError("至少提供 source_key 或 item_ids")
        clauses: list[str] = []
        params: list[Any] = []
        if source_key:
            clauses.append("source_key = ?")
            params.append(source_key)
        if item_ids:
            placeholders = ", ".join("?" for _ in item_ids)
            clauses.append(f"item_id IN ({placeholders})")
            params.extend(item_ids)
        where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            result = self._db.execute(
                f"DELETE FROM {table}{where_sql}",
                tuple(params),
            )
            self._db.commit()
        return int(result.rowcount or 0)

    def _list_rows(
        self,
        *,
        table: str,
        where: str,
        params: tuple[Any, ...],
        order_by: str,
        page: int,
        page_size: int,
        columns: str,
        row_mapper=None,
    ) -> tuple[list[dict[str, Any]], int]:
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        offset = (safe_page - 1) * safe_size
        with self._lock:
            total_row = self._db.execute(
                f"SELECT COUNT(*) FROM {table}{where}",
                params,
            ).fetchone()
            rows = self._db.execute(
                f"""
                SELECT {columns}
                FROM {table}{where}
                ORDER BY {order_by}
                LIMIT ? OFFSET ?
                """,
                (*params, safe_size, offset),
            ).fetchall()
        total = int(total_row[0]) if total_row is not None else 0
        mapper = row_mapper or self._row_to_dict
        return [mapper(row) for row in rows], total

    def _build_filters(self, *filters: tuple[str, Any]) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        for clause, value in filters:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            clauses.append(clause)
            params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, tuple(params)

    def _count(self, table: str) -> int:
        with self._lock:
            row = self._db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row is not None else 0

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    @staticmethod
    def _decode_json_list(raw: Any) -> list[str]:
        text = str(raw or "").strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            return []
        if not isinstance(value, list):
            return []
        return [str(item) for item in value]

    def _row_to_tick_log(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._row_to_dict(row)
        payload["interesting_ids"] = self._decode_json_list(payload.get("interesting_ids"))
        payload["discarded_ids"] = self._decode_json_list(payload.get("discarded_ids"))
        payload["cited_ids"] = self._decode_json_list(payload.get("cited_ids"))
        payload["drift_entered"] = bool(payload.get("drift_entered"))
        return payload

    def _row_to_tick_step(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._row_to_dict(row)
        payload["tool_args"] = self._decode_json_object(payload.pop("tool_args_json", ""))
        payload["interesting_ids_after"] = self._decode_json_list(payload.get("interesting_ids_after"))
        payload["discarded_ids_after"] = self._decode_json_list(payload.get("discarded_ids_after"))
        payload["cited_ids_after"] = self._decode_json_list(payload.get("cited_ids_after"))
        return payload

    @staticmethod
    def _decode_json_object(raw: Any) -> dict[str, Any]:
        text = str(raw or "").strip()
        if not text:
            return {}
        try:
            value = json.loads(text)
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}


def _build_plugin_panel_js(project_root: Path, plugin_dir: Path) -> None:
    ts_path = plugin_dir / "dashboard_panel.ts"
    js_path = plugin_dir / "dashboard_panel.js"

    if not ts_path.exists():
        return

    # Skip if .js exists and is not older than .ts
    if js_path.exists() and js_path.stat().st_mtime >= ts_path.stat().st_mtime:
        return

    esbuild_bin = project_root / "node_modules" / ".bin" / "esbuild"
    if not esbuild_bin.exists():
        logger.warning(
            "esbuild 未找到 (%s)；请先运行 'npm install' 启用插件自动编译。如已有 .js 将继续使用。",
            esbuild_bin,
        )
        return

    try:
        result = subprocess.run(
            [
                str(esbuild_bin),
                str(ts_path),
                f"--outfile={js_path}",
                "--bundle=false",
                "--platform=browser",
                "--target=es2020",
                "--format=iife",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("插件面板已编译: %s", plugin_dir.name)
        else:
            logger.warning("插件面板编译失败 (%s):\n%s", plugin_dir.name, result.stderr)
    except Exception as exc:
        logger.warning("插件面板编译异常 (%s): %s", plugin_dir.name, exc)


def _load_plugin_dashboard(app: FastAPI, plugin_dir: Path, workspace: Path) -> None:
    dash_path = plugin_dir / "dashboard.py"
    module_name = f"akasic_dashboard_plugin_{plugin_dir.name}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, dash_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {dash_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        if hasattr(mod, "register"):
            mod.register(app, plugin_dir, workspace)
            logger.info("插件 dashboard 已挂载: %s", plugin_dir.name)
    except Exception as e:
        logger.warning("插件 dashboard 挂载失败 (%s): %s", plugin_dir.name, e)


def _preview_text(value: Any, limit: int) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def create_dashboard_app(
    workspace: Path,
    *,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
) -> FastAPI:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SessionStore(workspace / "sessions.db")
    memory_store: MemoryStore2 | None = None
    proactive_reader: ProactiveDashboardReader | None = None
    optimizer_task: asyncio.Task[None] | None = None
    optimizer_last_status = "idle"
    optimizer_last_error: str | None = None
    project_root = Path(__file__).resolve().parent.parent
    plugins_root = project_root / "plugins"
    static_dir = project_root / "static" / "dashboard"

    def get_memory_store() -> MemoryStore2:
        nonlocal memory_store
        if memory_store is None:
            memory_store = MemoryStore2(workspace / "memory" / "memory2.db")
        return memory_store

    def get_proactive_reader() -> ProactiveDashboardReader:
        nonlocal proactive_reader
        if proactive_reader is None:
            ProactiveStateStore(workspace / "proactive.db").close()
            proactive_reader = ProactiveDashboardReader(workspace / "proactive.db")
        return proactive_reader

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            store.close()
            if memory_store is not None:
                get_memory_store().close()
            if proactive_reader is not None:
                get_proactive_reader().close()

    app = FastAPI(title="Akashic Dashboard API", lifespan=lifespan)
    app.mount("/assets", StaticFiles(directory=static_dir), name="dashboard-assets")

    # Compile TypeScript plugin panels and mount plugin routes
    if plugins_root.is_dir():
        for _plugin_dir in sorted(plugins_root.iterdir()):
            if not _plugin_dir.is_dir():
                continue
            _build_plugin_panel_js(project_root, _plugin_dir)
            if (_plugin_dir / "dashboard.py").exists():
                _load_plugin_dashboard(app, _plugin_dir, workspace)

    @app.get("/")
    def dashboard_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/dashboard/plugins")
    def list_dashboard_plugins() -> list[dict[str, Any]]:
        if not plugins_root.is_dir():
            return []
        result = []
        for plugin_dir in sorted(plugins_root.iterdir()):
            if plugin_dir.is_dir() and (plugin_dir / "dashboard_panel.js").exists():
                result.append({"id": plugin_dir.name})
        return result

    @app.get("/plugins/{plugin_id}/panel.js")
    def get_plugin_panel_js(plugin_id: str) -> FileResponse:
        if "/" in plugin_id or ".." in plugin_id:
            raise HTTPException(status_code=400, detail="invalid plugin id")
        js_path = plugins_root / plugin_id / "dashboard_panel.js"
        if not js_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel not found")
        return FileResponse(js_path, media_type="application/javascript")

    @app.get("/plugins/{plugin_id}/panel.css")
    def get_plugin_panel_css(plugin_id: str) -> FileResponse:
        if "/" in plugin_id or ".." in plugin_id:
            raise HTTPException(status_code=400, detail="invalid plugin id")
        css_path = plugins_root / plugin_id / "dashboard_panel.css"
        if not css_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        return FileResponse(css_path, media_type="text/css")

    @app.get("/api/dashboard/sessions")
    def list_sessions(
        q: str = "",
        channel: str = "",
        updated_from: str = "",
        updated_to: str = "",
        has_proactive: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = store.list_sessions_for_dashboard(
            q=q,
            channel=channel,
            updated_from=updated_from,
            updated_to=updated_to,
            has_proactive=has_proactive,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/sessions/{session_key:path}/messages")
    def list_session_messages(
        session_key: str,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        if not store.session_exists(session_key):
            raise HTTPException(status_code=404, detail="session 不存在")
        items, total = store.list_messages_for_dashboard(
            session_key=session_key,
            q=q,
            role=role,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.post("/api/dashboard/sessions/{session_key:path}/consolidate")
    async def consolidate_session(
        session_key: str,
        payload: SessionConsolidatePayload | None = None,
    ) -> dict[str, Any]:
        archive_all = bool(payload.archive_all) if payload is not None else False
        force = bool(payload.force) if payload is not None else True
        if manual_consolidator is None:
            raise HTTPException(status_code=503, detail="manual consolidation 未启用")
        if not store.session_exists(session_key):
            raise HTTPException(status_code=404, detail="session 不存在")
        logger.info(
            "Manual memory consolidation requested: session=%s archive_all=%s force=%s",
            session_key,
            archive_all,
            force,
        )
        try:
            triggered = await manual_consolidator.trigger_memory_consolidation(
                session_key,
                archive_all=archive_all,
                force=force,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        meta = store.get_session_meta(session_key) or {"key": session_key}
        meta["message_count"] = store.count_messages(session_key)
        logger.info(
            "Manual memory consolidation response: session=%s triggered=%s last_consolidated=%s message_count=%s",
            session_key,
            triggered,
            meta.get("last_consolidated"),
            meta.get("message_count"),
        )
        return {
            "session_key": session_key,
            "archive_all": archive_all,
            "force": force,
            "triggered": triggered,
            "session": meta,
        }

    async def _run_memory_optimizer() -> None:
        nonlocal optimizer_last_error, optimizer_last_status
        assert manual_memory_optimizer is not None
        optimizer_last_status = "running"
        optimizer_last_error = None
        try:
            await manual_memory_optimizer.optimize()
            optimizer_last_status = "succeeded"
        except MemoryOptimizerBusy:
            optimizer_last_status = "skipped"
            logger.info("manual memory optimizer skipped because it is already running")
        except asyncio.CancelledError:
            optimizer_last_status = "failed"
            optimizer_last_error = "memory optimizer 已取消"
            raise
        except Exception as exc:
            optimizer_last_status = "failed"
            optimizer_last_error = str(exc)
            logger.exception("manual memory optimizer failed: %s", exc)

    @app.get("/api/dashboard/memory/optimizer")
    async def get_memory_optimizer_status() -> dict[str, Any]:
        running = bool(
            manual_memory_optimizer is not None
            and (
                (optimizer_task is not None and not optimizer_task.done())
                or manual_memory_optimizer.is_running
            )
        )
        return {
            "enabled": manual_memory_optimizer is not None,
            "running": running,
            "last_status": "running" if running else optimizer_last_status,
            "last_error": optimizer_last_error,
        }

    @app.post("/api/dashboard/memory/optimize", status_code=202)
    async def trigger_memory_optimizer() -> dict[str, Any]:
        nonlocal optimizer_last_error, optimizer_last_status, optimizer_task
        if manual_memory_optimizer is None:
            raise HTTPException(status_code=503, detail="memory optimizer 未启用")
        if (
            optimizer_task is not None
            and not optimizer_task.done()
        ) or manual_memory_optimizer.is_running:
            raise HTTPException(status_code=409, detail="memory optimizer 正在运行")
        logger.info("Manual memory optimizer triggered via dashboard")
        optimizer_last_status = "running"
        optimizer_last_error = None
        optimizer_task = asyncio.create_task(
            _run_memory_optimizer(),
            name="manual_memory_optimizer",
        )
        return {"status": "started", "message": "Memory optimizer started"}

    @app.post("/api/dashboard/sessions/batch-delete")
    def delete_sessions_batch(payload: SessionBatchDeletePayload) -> dict[str, Any]:
        try:
            deleted_count = store.delete_sessions_batch(
                payload.keys,
                cascade=payload.cascade,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted_count": deleted_count}

    @app.get("/api/dashboard/sessions/{session_key:path}")
    def get_session(session_key: str) -> dict[str, Any]:
        meta = store.get_session_meta(session_key)
        if meta is None:
            raise HTTPException(status_code=404, detail="session 不存在")
        meta["message_count"] = store.count_messages(session_key)
        return meta

    @app.patch("/api/dashboard/sessions/{session_key:path}")
    def update_session(
        session_key: str,
        payload: SessionUpdatePayload,
    ) -> dict[str, Any]:
        meta = store.update_session(
            session_key,
            metadata=payload.metadata,
            last_consolidated=payload.last_consolidated,
            last_user_at=payload.last_user_at,
            last_proactive_at=payload.last_proactive_at,
        )
        if meta is None:
            raise HTTPException(status_code=404, detail="session 不存在")
        meta["message_count"] = store.count_messages(session_key)
        return meta

    @app.delete("/api/dashboard/sessions/{session_key:path}")
    def delete_session(
        session_key: str,
        cascade: bool = Query(default=True),
    ) -> dict[str, Any]:
        try:
            deleted = store.delete_session(session_key, cascade=cascade)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="session 不存在")
        return {"deleted": True, "session_key": session_key}

    @app.get("/api/dashboard/messages")
    def list_messages(
        session_key: str | None = None,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = store.list_messages_for_dashboard(
            session_key=session_key,
            q=q,
            role=role,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/messages/{message_id:path}")
    def get_message(message_id: str) -> dict[str, Any]:
        message = store.get_message(message_id)
        if message is None:
            raise HTTPException(status_code=404, detail="message 不存在")
        return message

    @app.patch("/api/dashboard/messages/{message_id:path}")
    def update_message(
        message_id: str,
        payload: MessageUpdatePayload,
    ) -> dict[str, Any]:
        message = store.update_message(
            message_id,
            role=payload.role,
            content=payload.content,
            tool_chain=payload.tool_chain,
            extra=payload.extra,
            ts=payload.ts,
        )
        if message is None:
            raise HTTPException(status_code=404, detail="message 不存在")
        return message

    @app.delete("/api/dashboard/messages/{message_id:path}")
    def delete_message(message_id: str) -> dict[str, Any]:
        deleted = store.delete_message(message_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="message 不存在")
        return {"deleted": True, "id": message_id}

    @app.post("/api/dashboard/messages/batch-delete")
    def delete_messages_batch(payload: MessageBatchDeletePayload) -> dict[str, Any]:
        deleted_count = store.delete_messages_batch(payload.ids)
        return {"deleted_count": deleted_count}

    @app.get("/api/dashboard/memories")
    def list_memories(
        q: str = "",
        memory_type: str = "",
        status: str = "",
        source_ref: str = "",
        scope_channel: str = "",
        scope_chat_id: str = "",
        has_embedding: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = get_memory_store().list_items_for_dashboard(
            q=q,
            memory_type=memory_type,
            status=status,
            source_ref=source_ref,
            scope_channel=scope_channel,
            scope_chat_id=scope_chat_id,
            has_embedding=has_embedding,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
            "vec_enabled": get_memory_store()._vec_enabled,
            "vec_dim": get_memory_store()._vec_dim,
        }

    @app.get("/api/dashboard/memories/{memory_id:path}/similar")
    def list_similar_memories(
        memory_id: str,
        top_k: int = 8,
        memory_type: str = "",
        score_threshold: float = 0.0,
        include_superseded: bool = False,
    ) -> dict[str, Any]:
        try:
            items = get_memory_store().find_similar_items_for_dashboard(
                memory_id,
                top_k=top_k,
                memory_type=memory_type,
                score_threshold=score_threshold,
                include_superseded=include_superseded,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="memory 不存在") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "items": items,
            "total": len(items),
            "source_id": memory_id,
        }

    @app.get("/api/dashboard/memories/{memory_id:path}")
    def get_memory(
        memory_id: str,
        include_embedding: bool = False,
    ) -> dict[str, Any]:
        item = get_memory_store().get_item_for_dashboard(
            memory_id,
            include_embedding=include_embedding,
        )
        if item is None:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return item

    @app.patch("/api/dashboard/memories/{memory_id:path}")
    def update_memory(
        memory_id: str,
        payload: MemoryUpdatePayload,
    ) -> dict[str, Any]:
        try:
            item = get_memory_store().update_item_for_dashboard(
                memory_id,
                status=payload.status,
                extra_json=payload.extra_json,
                source_ref=payload.source_ref,
                happened_at=payload.happened_at,
                emotional_weight=payload.emotional_weight,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return item

    @app.delete("/api/dashboard/memories/{memory_id:path}")
    def delete_memory(memory_id: str) -> dict[str, Any]:
        deleted = get_memory_store().delete_item(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return {"deleted": True, "id": memory_id}

    @app.post("/api/dashboard/memories/batch-delete")
    def delete_memories_batch(payload: MemoryBatchDeletePayload) -> dict[str, Any]:
        deleted_count = get_memory_store().delete_items_batch(payload.ids)
        return {"deleted_count": deleted_count}

    @app.get("/api/dashboard/proactive/overview")
    def get_proactive_overview() -> dict[str, Any]:
        return get_proactive_reader().get_overview()

    @app.get("/api/dashboard/proactive/deliveries")
    def list_proactive_deliveries(
        session_key: str = "",
        sent_from: str = "",
        sent_to: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = get_proactive_reader().list_deliveries(
            session_key=session_key,
            sent_from=sent_from,
            sent_to=sent_to,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/seen_items")
    def list_proactive_seen_items(
        source_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = get_proactive_reader().list_seen_items(
            source_key=source_key,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/rejection_cooldown")
    def list_proactive_rejection_cooldown(
        source_key: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = get_proactive_reader().list_rejection_cooldown(
            source_key=source_key,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/semantic_items")
    def list_proactive_semantic_items(
        page: int = 1,
        page_size: int = 50,
        window_hours: int = 168,
    ) -> dict[str, Any]:
        items, total = get_proactive_reader().list_semantic_items(
            page=page,
            page_size=page_size,
            window_hours=window_hours,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
            "window_hours": max(1, window_hours),
        }

    @app.get("/api/dashboard/proactive/tick_logs")
    def list_proactive_tick_logs(
        session_key: str = "",
        terminal_action: str = "",
        gate_exit: str = "",
        flow: str = Query(default="", pattern="^(|drift|proactive)$"),
        started_from: str = "",
        started_to: str = "",
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "started_at",
        sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = get_proactive_reader().list_tick_logs(
            session_key=session_key,
            terminal_action=terminal_action,
            gate_exit=gate_exit,
            flow=flow,
            started_from=started_from,
            started_to=started_to,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/proactive/tick_logs/{tick_id}")
    def get_proactive_tick_log(tick_id: str) -> dict[str, Any]:
        item = get_proactive_reader().get_tick_log(tick_id)
        if item is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        return item

    @app.get("/api/dashboard/proactive/tick_logs/{tick_id}/steps")
    def list_proactive_tick_steps(tick_id: str) -> dict[str, Any]:
        item = get_proactive_reader().get_tick_log(tick_id)
        if item is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        steps = get_proactive_reader().list_tick_steps(tick_id)
        return {
            "items": steps,
            "total": len(steps),
            "tick_id": tick_id,
        }

    @app.delete("/api/dashboard/proactive/seen_items/batch")
    def delete_proactive_seen_items(payload: ProactiveDeletePayload) -> dict[str, Any]:
        try:
            deleted_count = get_proactive_reader().delete_seen_items(
                source_key=str(payload.source_key or "").strip(),
                item_ids=payload.item_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted_count": deleted_count}

    @app.delete("/api/dashboard/proactive/rejection_cooldown/batch")
    def delete_proactive_rejection_cooldown(
        payload: ProactiveDeletePayload,
    ) -> dict[str, Any]:
        try:
            deleted_count = get_proactive_reader().delete_rejection_cooldown(
                source_key=str(payload.source_key or "").strip(),
                item_ids=payload.item_ids,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"deleted_count": deleted_count}

    return app


def run_dashboard_api(
    *,
    workspace: Path,
    host: str = "0.0.0.0",
    port: int = 2236,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
) -> None:
    server = uvicorn.Server(
        _build_dashboard_uvicorn_config(
            workspace=workspace,
            host=host,
            port=port,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
        )
    )
    server.run()


def _build_dashboard_uvicorn_config(
    *,
    workspace: Path,
    host: str,
    port: int,
    manual_consolidator: ManualConsolidator | None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
) -> uvicorn.Config:
    config = uvicorn.Config(
        create_dashboard_app(
            workspace,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
        ),
        host=host,
        port=port,
        log_level="info",
    )
    _install_dashboard_access_log_filter()
    return config


def build_dashboard_server(
    *,
    workspace: Path,
    host: str = "0.0.0.0",
    port: int = 2236,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
) -> uvicorn.Server:
    config = _build_dashboard_uvicorn_config(
        workspace=workspace,
        host=host,
        port=port,
        manual_consolidator=manual_consolidator,
        manual_memory_optimizer=manual_memory_optimizer,
    )
    return uvicorn.Server(config)
