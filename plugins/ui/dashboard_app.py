from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from core.common.file_io import run_file_io

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


def create_dashboard_app(static_dir: Path) -> FastAPI:
    """只读取固定代码制品中的构建资产；缺少入口时明确返回不可用。"""
    app = FastAPI(title="Akashic Dashboard API")
    # Vite 构建产物被 gitignore，新 clone 或 CI 环境可能没有该目录。
    # 挂载时不写代码目录，避免 app 创建依赖构建是否执行；
    # dashboard_index() 会在入口文件缺失时报告错误。
    app.mount("/dashboard/assets", StaticFiles(directory=static_dir, check_dir=False),
              name="dashboard-public-assets")
    app.mount(
        "/assets",
        StaticFiles(directory=static_dir, check_dir=False),
        name="dashboard-assets",
    )
    # Vite 会在 /assets 下生成带内容哈希的资源 URL，因此直接原样提供 index.html；
    # 不需要手动处理缓存失效。
    @app.get("/")
    @app.get("/dashboard")
    @app.get("/dashboard/")
    async def dashboard_index() -> Response:
        index_file = static_dir / "index.html"
        try:
            html = await run_file_io(lambda: index_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Response(
                content="Dashboard 前端尚未构建，请先运行 `npm run build`。",
                media_type="text/plain; charset=utf-8",
                status_code=503,
            )
        return Response(content=html, media_type="text/html")

    return app
