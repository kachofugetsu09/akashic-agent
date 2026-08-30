from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from agent.memory import MemoryStore
from core.memory.optimizer import MemoryOptimizerBusy

logger = logging.getLogger(__name__)

_DASHBOARD_ACCESS_PREFIXES = ("/api/dashboard", "/assets")


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


class ManualMemoryOptimizer(Protocol):
    @property
    def is_running(self) -> bool: ...

    async def optimize(self) -> None: ...


def create_dashboard_app(
    workspace: Path,
    *,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> FastAPI:
    workspace.mkdir(parents=True, exist_ok=True)
    optimizer_task: asyncio.Task[None] | None = None
    optimizer_last_status = "idle"
    optimizer_last_error: str | None = None
    project_root = Path(__file__).resolve().parent.parent
    static_dir = project_root / "static" / "dashboard"

    app = FastAPI(title="Akashic Dashboard API")
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

    if plugin_manager is not None:
        from agent.plugins.dashboard_host import (
            PluginDashboardHost,
            SnapshotDashboardMiddleware,
        )

        dashboard_host = PluginDashboardHost(
            core_routes=tuple(app.routes),
        )
        snapshot = plugin_manager.current_snapshot
        if snapshot is not None:
            dashboard_host.prepare_initial_snapshot(snapshot)
        plugin_manager.bind_dashboard_preparer(
            dashboard_host.prepare_snapshot,
            validation_releaser=dashboard_host.release_validation,
        )
        app.add_middleware(
            SnapshotDashboardMiddleware,
            snapshot_store=plugin_manager.snapshot_store,
        )

    return app


def run_dashboard_api(
    *,
    workspace: Path,
    host: str = "0.0.0.0",
    port: int = 2236,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
) -> None:
    server = uvicorn.Server(
        _build_dashboard_uvicorn_config(
            workspace=workspace,
            host=host,
            port=port,
            uds=None,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_store=memory_store,
        )
    )
    server.run()


def _build_dashboard_uvicorn_config(
    *,
    workspace: Path,
    host: str | None,
    port: int | None,
    uds: str | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> uvicorn.Config:
    config = uvicorn.Config(
        create_dashboard_app(
            workspace,
            manual_memory_optimizer=manual_memory_optimizer,
            memory_store=memory_store,
            plugin_manager=plugin_manager,
        ),
        host=host or "127.0.0.1",
        port=port or 2236,
        uds=uds,
        log_level="info",
    )
    _install_dashboard_access_log_filter()
    return config


def build_dashboard_server(
    *,
    workspace: Path,
    host: str | None = None,
    port: int | None = None,
    uds: str | None = None,
    manual_memory_optimizer: ManualMemoryOptimizer | None = None,
    memory_store: MemoryStore | None = None,
    plugin_manager: object | None = None,
) -> uvicorn.Server:
    config = _build_dashboard_uvicorn_config(
        workspace=workspace,
        host=host,
        port=port,
        uds=uds,
        manual_memory_optimizer=manual_memory_optimizer,
        memory_store=memory_store,
        plugin_manager=plugin_manager,
    )
    return uvicorn.Server(config)
