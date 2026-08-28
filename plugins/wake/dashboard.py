"""Expose Wake Timer attempts and decision runs through read-only routes."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext
from .state import WakeState


def register(app: FastAPI, context: DashboardContext) -> None:
    state = WakeState(context.data_root / "wake.sqlite3")

    @app.get("/api/dashboard/wake/attempts")
    def list_attempts(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        rows = state.list_attempts(page_size, offset=(page - 1) * page_size)
        return {
            "items": rows,
            "total": state.count_attempts(),
            "page": page,
            "page_size": page_size,
        }

    @app.get("/api/dashboard/wake/attempts/{attempt_id}")
    def get_attempt(attempt_id: str) -> Mapping[str, object]:
        item = state.get_attempt(attempt_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Wake 定时检查不存在")
        return item

    @app.get("/api/dashboard/wake/runs")
    def list_runs(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=25, ge=1, le=100),
    ) -> dict[str, object]:
        rows = state.list_runs(page_size, offset=(page - 1) * page_size)
        return {
            "items": rows,
            "total": state.count_runs(),
            "page": page,
            "page_size": page_size,
        }

    @app.get("/api/dashboard/wake/runs/{run_id}")
    def get_run(run_id: str) -> Mapping[str, object]:
        item = state.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Wake 判断记录不存在")
        return item
