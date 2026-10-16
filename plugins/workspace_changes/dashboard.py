"""改动统计的浏览器入口。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext

from .changes import WORKSPACE_CHANGES

inject = (WORKSPACE_CHANGES,)
_SESSION = Query(min_length=1, max_length=512)


def register(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/workspace_changes/uncommitted")
    async def uncommitted(session: str = _SESSION) -> dict[str, object]:
        return await context.require(WORKSPACE_CHANGES).uncommitted(session)

    @app.get("/api/dashboard/workspace_changes/turns")
    async def turns(session: str = _SESSION) -> dict[str, object]:
        return {"turns": await context.require(WORKSPACE_CHANGES).windows.turns(session)}

    @app.get("/api/dashboard/workspace_changes/turn")
    async def turn(session: str = _SESSION, turn: str = Query(min_length=1, max_length=512)) -> dict[str, object]:
        found = await context.require(WORKSPACE_CHANGES).windows.turn(session, turn)
        if found is None:
            raise HTTPException(404, "没有这一轮的改动记录")
        return found
