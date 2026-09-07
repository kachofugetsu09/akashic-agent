"""Expose Wake's durable attempts and Message-backed flow evidence."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext
from agent.plugins.snapshot import get_current_runtime_snapshot

from .message_plugin import WAKE_DASHBOARD
from .runtime import DashboardView


def _view() -> DashboardView:
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        raise RuntimeError("Wake Dashboard 请求缺少实际 runtime snapshot")
    view = snapshot.composition_root.context.require(WAKE_DASHBOARD)()
    if view is None:
        raise RuntimeError("Wake runtime 尚未启动")
    return view


def register(app: FastAPI, context: DashboardContext) -> None:
    """Register read-only routes owned by the live Wake runtime."""

    @app.get("/api/dashboard/wake/attempts")
    async def list_attempts(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        view = _view()
        rows = view.list_attempts(page_size, offset=(page - 1) * page_size)
        return {"items": rows, "total": view.count_attempts(), "page": page, "page_size": page_size}

    @app.get("/api/dashboard/wake/attempts/{attempt_id}")
    async def get_attempt(attempt_id: str) -> Mapping[str, object]:
        item = _view().get_attempt(attempt_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Wake 定时检查不存在")
        return item

    @app.get("/api/dashboard/wake/runs")
    async def list_runs(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        view = _view()
        rows = view.list_runs(page_size, offset=(page - 1) * page_size)
        return {"items": rows, "total": view.count_runs(), "page": page, "page_size": page_size}

    @app.get("/api/dashboard/wake/runs/{run_id}")
    async def get_run(run_id: str) -> Mapping[str, object]:
        item = _view().get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Wake 判断记录不存在")
        return item
