from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from bootstrap.dashboard_api import ProactiveDashboardReader
from proactive_v2.state import ProactiveStateStore


def register(
    app: FastAPI,
    plugin_dir: Path,
    workspace: Path,
) -> ProactiveDashboardReader:
    ProactiveStateStore(workspace / "proactive.db").close()
    reader = ProactiveDashboardReader(workspace / "proactive.db")

    @app.get("/api/dashboard/proactive/overview")
    def overview() -> dict[str, Any]:
        return reader.get_overview()

    @app.get("/api/dashboard/proactive/deliveries")
    def deliveries(
        session_key: str = "", sent_from: str = "", sent_to: str = "",
        page: int = 1, page_size: int = 50,
    ) -> dict[str, Any]:
        items, total = reader.list_deliveries(
            session_key=session_key, sent_from=sent_from, sent_to=sent_to,
            page=page, page_size=page_size,
        )
        return {"items": items, "total": total, "page": max(1, page), "page_size": max(1, min(page_size, 200))}

    @app.get("/api/dashboard/proactive/tick_logs")
    def tick_logs(
        session_key: str = "", terminal_action: str = "", gate_exit: str = "",
        flow: str = Query(default="", pattern="^(|drift|proactive)$"),
        started_from: str = "", started_to: str = "", page: int = 1,
        page_size: int = 50, sort_by: str = "started_at", sort_order: str = "desc",
    ) -> dict[str, Any]:
        items, total = reader.list_tick_logs(
            session_key=session_key, terminal_action=terminal_action,
            gate_exit=gate_exit, flow=flow, started_from=started_from,
            started_to=started_to, page=page, page_size=page_size,
            sort_by=sort_by, sort_order=sort_order,
        )
        return {"items": items, "total": total, "page": max(1, page), "page_size": max(1, min(page_size, 200))}

    @app.get("/api/dashboard/proactive/tick_logs/{tick_id}")
    def tick_log(tick_id: str) -> dict[str, Any]:
        item = reader.get_tick_log(tick_id)
        if item is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        return item

    @app.get("/api/dashboard/proactive/tick_logs/{tick_id}/steps")
    def tick_steps(tick_id: str) -> dict[str, Any]:
        if reader.get_tick_log(tick_id) is None:
            raise HTTPException(status_code=404, detail="tick 不存在")
        items = reader.list_tick_steps(tick_id)
        return {"items": items, "total": len(items), "tick_id": tick_id}

    return reader
