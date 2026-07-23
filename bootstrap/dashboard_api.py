from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import inspect
from pathlib import Path, PureWindowsPath
import importlib.util
import logging
import json
import sqlite3
import sys
import threading
import os
import shutil
from types import ModuleType
from typing import Any, Protocol, cast

import subprocess

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent.plugins.manifest import (
    load_package_manifest,
    load_plugin_manifest,
    plugins_root as resolve_plugins_root,
)
from agent.plugins.packages import enabled_plugin_packages
from agent.plugins.source_resolver import resolve_plugin_sources
from bootstrap.cleanup import run_cleanup_steps

from agent.memory import MemoryStore
from proactive_v2.memory_optimizer import MemoryOptimizerBusy
from core.memory.engine import MemoryAdminApi
from session.store import SessionStore

logger = logging.getLogger(__name__)

_DASHBOARD_ACCESS_PREFIXES = ("/api/dashboard", "/assets", "/plugins/")


def _dashboard_plugin_dirs(project_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    manifest = load_plugin_manifest()
    builtin_root = project_root / "plugins"
    if builtin_root.is_dir():
        for plugin_dir in sorted(builtin_root.iterdir()):
            if (
                not plugin_dir.is_dir()
                or manifest.get(plugin_dir.name, True) is False
            ):
                continue
            result[plugin_dir.name] = plugin_dir

    cache_root = resolve_plugins_root() / "cache"
    for source in resolve_plugin_sources([], installed_cache_root=cache_root):
        plugin_name = source.plugin_root.parent.name
        plugin_id = f"{plugin_name}@{source.marketplace}"
        if manifest.get(plugin_id, True) is False:
            continue
        plugin_root = source.plugin_root.resolve(strict=False)
        if not plugin_root.is_dir():
            continue
        result[plugin_id] = plugin_root
    for package_id, package in enabled_plugin_packages(
        project_root,
        load_package_manifest(),
    ).items():
        if package.dashboard:
            result[package_id] = package.root
    return result


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
            "deliveries": self._count("deliveries"),
            "session_state": self._count("session_state"),
            "context_only_timestamps": self._count("context_only_timestamps"),
            "tick_logs": self._count("tick_log"),
            "tick_steps": self._count("tick_step_log"),
        }
        with self._lock:
            recent_tick = self._db.execute("""
                SELECT tick_id, session_key, started_at, finished_at, gate_exit,
                       terminal_action, skip_reason, steps_taken, drift_entered
                FROM tick_log
                ORDER BY started_at DESC
                LIMIT 1
                """).fetchone()
            last_send_at = self._db.execute("""
                SELECT sent_at
                FROM deliveries
                ORDER BY sent_at DESC
                LIMIT 1
                """).fetchone()
            result_counts_rows = self._db.execute("""
                SELECT COALESCE(terminal_action, gate_exit, 'unknown') AS bucket, COUNT(*) AS total
                FROM tick_log
                GROUP BY COALESCE(terminal_action, gate_exit, 'unknown')
                """).fetchall()
            flow_counts_rows = self._db.execute("""
                SELECT CASE WHEN drift_entered = 1 THEN 'drift' ELSE 'proactive' END AS bucket,
                       COUNT(*) AS total
                FROM tick_log
                GROUP BY CASE WHEN drift_entered = 1 THEN 'drift' ELSE 'proactive' END
                """).fetchall()
        result_counts = {
            str(row["bucket"]): int(row["total"]) for row in result_counts_rows
        }
        flow_counts = {
            str(row["bucket"]): int(row["total"]) for row in flow_counts_rows
        }
        return {
            "counts": counts,
            "result_counts": result_counts,
            "flow_counts": flow_counts,
            "last_tick_at": (
                recent_tick["started_at"] if recent_tick is not None else None
            ),
            "last_send_at": (
                last_send_at["sent_at"] if last_send_at is not None else None
            ),
            "last_skip_reason": (
                recent_tick["skip_reason"]
                if recent_tick is not None and recent_tick["terminal_action"] != "reply"
                else None
            ),
            "recent_tick": (
                self._row_to_tick_log(recent_tick) if recent_tick is not None else None
            ),
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
        safe_sort_by = (
            sort_by
            if sort_by
            in {
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
            }
            else "started_at"
        )
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
                "cited_ids, drift_entered, final_message, proactive_effects_json"
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
                       cited_ids, drift_entered, final_message, proactive_effects_json
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
    def _decode_json_list(raw: object) -> list[str]:
        text = ProactiveDashboardReader._json_text(raw, "列表")
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("proactive dashboard JSON 列表损坏") from exc
        if not isinstance(value, list):
            raise ValueError("proactive dashboard JSON 列表类型错误")
        if not all(isinstance(item, str) for item in value):
            raise ValueError("proactive dashboard JSON 列表元素类型错误")
        return value

    def _row_to_tick_log(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._row_to_dict(row)
        payload["interesting_ids"] = self._decode_json_list(
            payload.get("interesting_ids")
        )
        payload["discarded_ids"] = self._decode_json_list(payload.get("discarded_ids"))
        payload["cited_ids"] = self._decode_json_list(payload.get("cited_ids"))
        payload["proactive_effects"] = self._decode_json_object_list(
            payload.pop("proactive_effects_json", "")
        )
        payload["drift_entered"] = bool(payload.get("drift_entered"))
        return payload

    @staticmethod
    def _decode_json_object_list(raw: object) -> list[dict[str, Any]]:
        text = ProactiveDashboardReader._json_text(raw, "对象列表")
        if not text:
            return []
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("proactive dashboard JSON 对象列表损坏") from exc
        if not isinstance(value, list):
            raise ValueError("proactive dashboard JSON 对象列表类型错误")
        if not all(isinstance(item, dict) for item in value):
            raise ValueError("proactive dashboard JSON 对象列表元素类型错误")
        return value

    def _row_to_tick_step(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = self._row_to_dict(row)
        payload["tool_args"] = self._decode_json_object(
            payload.pop("tool_args_json", "")
        )
        payload["interesting_ids_after"] = self._decode_json_list(
            payload.get("interesting_ids_after")
        )
        payload["discarded_ids_after"] = self._decode_json_list(
            payload.get("discarded_ids_after")
        )
        payload["cited_ids_after"] = self._decode_json_list(
            payload.get("cited_ids_after")
        )
        return payload

    @staticmethod
    def _decode_json_object(raw: object) -> dict[str, Any]:
        text = ProactiveDashboardReader._json_text(raw, "对象")
        if not text:
            return {}
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("proactive dashboard JSON 对象损坏") from exc
        if not isinstance(value, dict):
            raise ValueError("proactive dashboard JSON 对象类型错误")
        return value

    @staticmethod
    def _json_text(raw: object, label: str) -> str:
        if raw is None:
            return ""
        if not isinstance(raw, str):
            raise ValueError(f"proactive dashboard JSON {label}存储类型错误")
        return raw.strip()


_pending_plugins: set[tuple[Path, Path]] = set()
_pending_plugins_lock = threading.Lock()


def _esbuild_command(project_root: Path) -> list[str] | None:
    bin_name = "esbuild.cmd" if os.name == "nt" else "esbuild"
    local_bin = project_root / "node_modules" / ".bin" / bin_name
    if local_bin.exists():
        return [str(local_bin)]
    if os.name == "nt":
        cmd_bin = shutil.which("cmd.exe") or shutil.which("cmd")
        npx_bin = shutil.which("npx.cmd") or shutil.which("npx")
        if cmd_bin and npx_bin:
            return [cmd_bin, "/d", "/s", "/c", "npx", "--yes", "esbuild"]
        return None
    npx_bin = shutil.which("npx")
    if npx_bin:
        return [npx_bin, "--yes", "esbuild"]
    return None


def _build_plugin_panels_js(project_root: Path, plugin_dir: Path) -> None:
    esbuild_cmd: list[str] | None = None
    for ts_path in _iter_plugin_panel_sources(plugin_dir):
        js_path = ts_path.with_suffix(".js")
        if js_path.exists() and js_path.stat().st_mtime >= ts_path.stat().st_mtime:
            continue
        if esbuild_cmd is None:
            esbuild_cmd = _esbuild_command(project_root)
        if esbuild_cmd is None:
            with _pending_plugins_lock:
                _pending_plugins.add((project_root, plugin_dir))
            return
        _run_esbuild(esbuild_cmd, ts_path, js_path, f"{plugin_dir.name}/{ts_path.stem}")


def _iter_plugin_panel_sources(plugin_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in plugin_dir.glob("dashboard_panel*")
        if path.suffix in {".ts", ".tsx"}
    )


def _run_esbuild(cmd: list[str], ts_path: Path, js_path: Path, name: str) -> None:
    try:
        result = subprocess.run(
            [
                *cmd,
                str(ts_path),
                f"--outfile={js_path}",
                "--bundle",
                "--platform=browser",
                "--target=es2020",
                "--format=esm",
                "--jsx=automatic",
                "--external:react",
                "--external:react-dom",
                "--external:react-dom/client",
                "--external:react/jsx-runtime",
                "--external:@akashic/dashboard-ui",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("插件面板已编译: %s", name)
        else:
            logger.warning("插件面板编译失败 (%s):\n%s", name, result.stderr)
    except Exception as exc:
        logger.warning("插件面板编译异常 (%s): %s", name, exc)


def _resolve_plugin_dir(
    plugin_dirs: dict[str, Path],
    plugin_id: str,
) -> Path:
    if not plugin_id or "/" in plugin_id or "\\" in plugin_id:
        raise HTTPException(status_code=400, detail="invalid plugin id")
    win_path = PureWindowsPath(plugin_id)
    if Path(plugin_id).is_absolute() or win_path.drive or win_path.root:
        raise HTTPException(status_code=400, detail="invalid plugin id")
    plugin_dir = plugin_dirs.get(plugin_id)
    if plugin_dir is None:
        raise HTTPException(status_code=404, detail="plugin not found")
    return plugin_dir


def _validate_panel_name(panel_name: str, detail: str) -> None:
    """校验插件面板文件名，阻止跨平台路径穿越。"""
    if not panel_name.startswith("dashboard_panel"):
        raise HTTPException(status_code=404, detail=detail)
    if (
        any(separator in panel_name for separator in ("/", "\\"))
        or "\x00" in panel_name
    ):
        raise HTTPException(status_code=400, detail="invalid plugin panel name")


async def _compile_pending_plugins_async() -> None:
    with _pending_plugins_lock:
        if not _pending_plugins:
            return
        pending = tuple(
            sorted(_pending_plugins, key=lambda item: (str(item[0]), str(item[1])))
        )
        _pending_plugins.clear()
    first_root = pending[0][0]

    logger.info("正在安装前端构建工具 (npx esbuild)...")
    esbuild_cmd = _esbuild_command(first_root)
    if esbuild_cmd is None:
        logger.warning("esbuild unavailable: neither local install nor npx was found")
        return
    proc = await asyncio.create_subprocess_exec(
        *esbuild_cmd,
        "--version",
        cwd=str(first_root),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning(
            "npx esbuild 不可用 (%d)，插件面板未编译:\n%s",
            proc.returncode,
            stderr.decode("utf-8", errors="replace")[:500],
        )
        return
    version = stdout.decode("utf-8", errors="replace").strip()
    logger.info("npx esbuild 就绪 (%s)，开始编译插件面板...", version)
    for root, pdir in pending:
        for ts_path in _iter_plugin_panel_sources(pdir):
            js_path = ts_path.with_suffix(".js")
            if not (js_path.exists() and js_path.stat().st_mtime >= ts_path.stat().st_mtime):
                _run_esbuild(esbuild_cmd, ts_path, js_path, f"{pdir.name}/{ts_path.stem}")


def _load_plugin_dashboard(app: FastAPI, plugin_dir: Path, workspace: Path) -> list[object]:
    try:
        mod = _load_plugin_dashboard_module(plugin_dir)
        if hasattr(mod, "register"):
            result = mod.register(app, plugin_dir, workspace)
            logger.info("插件 dashboard 已挂载: %s", plugin_dir.name)
            return _dashboard_closeables(result)
    except Exception as e:
        logger.warning("插件 dashboard 挂载失败 (%s): %s", plugin_dir.name, e)
    return []


def _plugin_dashboard_enabled(app: FastAPI, plugin_dir: Path) -> bool:
    dash_path = plugin_dir / "dashboard.py"
    if not dash_path.exists():
        return False
    try:
        mod = _load_plugin_dashboard_module(plugin_dir)
    except Exception as e:
        logger.warning("插件 dashboard 检查失败 (%s): %s", plugin_dir.name, e)
        return False
    enabled = getattr(mod, "plugin_enabled", None)
    if not callable(enabled):
        return True
    return bool(enabled(app))


def _load_plugin_dashboard_module(plugin_dir: Path) -> ModuleType:
    dash_path = plugin_dir / "dashboard.py"
    module_name = _dashboard_module_name(plugin_dir)
    spec = importlib.util.spec_from_file_location(
        module_name,
        dash_path,
        submodule_search_locations=[str(plugin_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {dash_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _dashboard_module_name(plugin_dir: Path) -> str:
    raw = str(plugin_dir.resolve(strict=False))
    normalized = "".join(ch if ch.isalnum() else "_" for ch in raw)
    return f"akasic_dashboard_plugin_{normalized}"


def _dashboard_closeables(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        items = cast(list[object], value)
        return [
            item
            for item in items
            if _is_dashboard_closeable(item)
        ]
    if _is_dashboard_closeable(value):
        return [value]
    return []


def _is_dashboard_closeable(value: object) -> bool:
    return callable(getattr(value, "close", None))


async def _close_dashboard_value(value: object) -> None:
    """关闭 dashboard 资源，并等待异步 close 完成。"""
    close = getattr(value, "close", None)
    if callable(close):
        result = close()
        if inspect.isawaitable(result):
            await result


def create_dashboard_app(
    workspace: Path,
    *,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_admin: MemoryAdminApi,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> FastAPI:
    workspace.mkdir(parents=True, exist_ok=True)
    store = SessionStore(workspace / "sessions.db")
    optimizer_task: asyncio.Task[None] | None = None
    optimizer_last_status = "idle"
    optimizer_last_error: str | None = None
    plugin_closeables: list[object] = []
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "static" / "dashboard"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        compile_task = asyncio.create_task(_compile_pending_plugins_async())
        try:
            yield
        finally:
            async def _cancel_compile_task() -> None:
                cancelled = compile_task.cancel()
                if not cancelled:
                    await compile_task
                    return
                try:
                    await compile_task
                except asyncio.CancelledError:
                    return

            await run_cleanup_steps(
                ("plugin_panel_compile.cancel", _cancel_compile_task),
                ("dashboard.session_store.close", lambda: _close_dashboard_value(store)),
                ("dashboard.memory_admin.close", lambda: _close_dashboard_value(memory_admin)),
                *[
                    (
                        f"dashboard.plugin_closeable[{index}].close",
                        lambda value=closeable: _close_dashboard_value(value),
                    )
                    for index, closeable in enumerate(reversed(plugin_closeables))
                ],
            )

    app = FastAPI(title="Akashic Dashboard API", lifespan=lifespan)
    app.state.memory_admin = memory_admin
    app.state.memory_store = memory_store or MemoryStore(workspace)
    # Vite 构建产物被 gitignore，新 clone 或 CI 环境可能没有该目录。
    # 预先创建目录并在挂载时关闭目录检查，避免 app 创建依赖构建是否执行；
    # dashboard_index() 会在入口文件缺失时报告错误。
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="dashboard-assets",
    )
    plugin_dirs = _dashboard_plugin_dirs(project_root)

    # 编译 TypeScript 插件面板并挂载插件路由
    for _plugin_id, _plugin_dir in sorted(plugin_dirs.items()):
        if plugin_manager is None and not _plugin_dashboard_enabled(app, _plugin_dir):
            continue
        _build_plugin_panels_js(project_root, _plugin_dir)
        if (
            plugin_manager is None or (_plugin_dir / "package.toml").exists()
        ) and (_plugin_dir / "dashboard.py").exists():
            plugin_closeables.extend(
                _load_plugin_dashboard(app, _plugin_dir, workspace)
            )

    # Vite 会在 /assets 下生成带内容哈希的资源 URL，因此直接原样提供 index.html；
    # 不需要手动处理缓存失效。
    @app.get("/")
    def dashboard_index() -> Response:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            return Response(
                content="Dashboard 前端尚未构建，请先运行 `npm run build`。",
                media_type="text/plain; charset=utf-8",
                status_code=503,
            )
        html = index_file.read_text(encoding="utf-8")
        return Response(content=html, media_type="text/html")

    @app.get("/api/dashboard/plugins")
    def list_dashboard_plugins() -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for plugin_id, plugin_dir in sorted(_dashboard_plugin_dirs(project_root).items()):
            if not dashboard_plugin_enabled(plugin_id, plugin_dir):
                continue
            _build_plugin_panels_js(project_root, plugin_dir)
            panels: list[dict[str, Any]] = []
            for js_path in sorted(plugin_dir.glob("dashboard_panel*.js")):
                css_path = js_path.with_suffix(".css")
                panels.append({
                    "name": js_path.stem,
                    "js_version": str(js_path.stat().st_mtime_ns),
                    "has_css": css_path.exists(),
                })
            if panels:
                result.append({"id": plugin_id, "panels": panels})
        return result

    @app.get("/plugins/{plugin_id}/{panel_name}.js")
    def get_plugin_panel_js(plugin_id: str, panel_name: str) -> FileResponse:
        _validate_panel_name(panel_name, "plugin panel not found")
        plugin_dir = _resolve_plugin_dir(
            _dashboard_plugin_dirs(project_root),
            plugin_id,
        )
        if not dashboard_plugin_enabled(
            plugin_id,
            plugin_dir,
        ):
            raise HTTPException(status_code=404, detail="plugin panel not found")
        _build_plugin_panels_js(project_root, plugin_dir)
        js_path = plugin_dir / f"{panel_name}.js"
        if not js_path.exists():
            raise HTTPException(status_code=404, detail="plugin panel not found")
        return FileResponse(js_path, media_type="application/javascript")

    @app.get("/plugins/{plugin_id}/{panel_name}.css")
    def get_plugin_panel_css(plugin_id: str, panel_name: str) -> FileResponse:
        _validate_panel_name(panel_name, "plugin panel css not found")
        plugin_dir = _resolve_plugin_dir(
            _dashboard_plugin_dirs(project_root),
            plugin_id,
        )
        if not dashboard_plugin_enabled(
            plugin_id,
            plugin_dir,
        ):
            raise HTTPException(status_code=404, detail="plugin panel css not found")
        css_path = plugin_dir / f"{panel_name}.css"
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

    @app.get("/api/dashboard/memory/engine-info")
    def get_memory_engine_info() -> dict[str, Any]:
        desc = memory_admin.describe()
        return {"name": desc.name}

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
            optimizer_task is not None and not optimizer_task.done()
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
        items, total = memory_admin.list_items_for_dashboard(
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
            "vec_enabled": True,
            "vec_dim": 0,
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
            items = memory_admin.find_similar_items_for_dashboard(
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
        item = memory_admin.get_item_for_dashboard(
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
            item = memory_admin.update_item_for_dashboard(
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
        deleted = memory_admin.delete_item(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="memory 不存在")
        return {"deleted": True, "id": memory_id}

    @app.post("/api/dashboard/memories/batch-delete")
    def delete_memories_batch(payload: MemoryBatchDeletePayload) -> dict[str, Any]:
        deleted_count = memory_admin.delete_items_batch(payload.ids)
        return {"deleted_count": deleted_count}

    if plugin_manager is not None:
        from agent.plugins.dashboard_host import (
            DashboardBinding,
            PluginDashboardHost,
            SnapshotDashboardMiddleware,
        )

        dashboard_host = PluginDashboardHost(
            workspace=workspace,
            memory_admin=memory_admin,
            memory_store=app.state.memory_store,
            core_routes=tuple(app.routes),
        )
        snapshot = plugin_manager.current_snapshot
        if snapshot is not None:
            dashboard_host.prepare_initial_snapshot(snapshot)
        plugin_manager.bind_dashboard_preparer(dashboard_host.prepare_snapshot)
        app.add_middleware(
            SnapshotDashboardMiddleware,
            snapshot_store=plugin_manager.snapshot_store,
        )

        def dashboard_plugin_enabled(plugin_id: str, plugin_dir: Path) -> bool:
            if (plugin_dir / "package.toml").exists():
                return True
            _ = plugin_dir
            current = plugin_manager.current_snapshot
            return current is not None and any(
                isinstance(binding, DashboardBinding)
                and binding.plugin_id == plugin_id
                and binding.routes
                for binding in current.dashboard_bindings
            )
    else:
        def dashboard_plugin_enabled(plugin_id: str, plugin_dir: Path) -> bool:
            _ = plugin_id
            return _plugin_dashboard_enabled(app, plugin_dir)

    return app


def run_dashboard_api(
    *,
    workspace: Path,
    host: str = "0.0.0.0",
    port: int = 2236,
    manual_consolidator: ManualConsolidator | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_admin: MemoryAdminApi,
    memory_store: MemoryStore | None = None,
) -> None:
    server = uvicorn.Server(
        _build_dashboard_uvicorn_config(
            workspace=workspace,
            host=host,
            port=port,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_admin=memory_admin,
            memory_store=memory_store,
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
    memory_admin: MemoryAdminApi,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> uvicorn.Config:
    config = uvicorn.Config(
        create_dashboard_app(
            workspace,
            manual_consolidator=manual_consolidator,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_admin=memory_admin,
            memory_store=memory_store,
            plugin_manager=plugin_manager,
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
    memory_admin: MemoryAdminApi,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> uvicorn.Server:
    config = _build_dashboard_uvicorn_config(
        workspace=workspace,
        host=host,
        port=port,
        manual_consolidator=manual_consolidator,
        manual_memory_optimizer=manual_memory_optimizer,
        memory_admin=memory_admin,
        memory_store=memory_store,
        plugin_manager=plugin_manager,
    )
    return uvicorn.Server(config)
