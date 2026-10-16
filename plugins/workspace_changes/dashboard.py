"""改动统计的浏览器入口。"""
from __future__ import annotations

from fastapi import FastAPI, Query

from agent.plugin_composition import DashboardContext

from .changes import OID, SCOPE, WORKSPACE_CHANGES

inject = (WORKSPACE_CHANGES,)


def register(app: FastAPI, context: DashboardContext) -> None:
    # 每个参数各用一个 Query 实例：共用实例时参数会绑定到同一个名字上。
    @app.get("/api/dashboard/workspace_changes/scope")
    async def scope(
        session: str = Query(min_length=1, max_length=512),
        scope: str = Query(pattern=f"^(?:{SCOPE.pattern})$"),
    ) -> dict[str, object]:
        return await context.require(WORKSPACE_CHANGES).scope(session, scope)

    @app.get("/api/dashboard/workspace_changes/patch")
    async def patch(
        session: str = Query(min_length=1, max_length=512),
        before: str = Query(pattern=f"^(?:{OID.pattern})$"),
        after: str = Query(pattern=f"^(?:{OID.pattern})$"),
        path: str = Query(min_length=1, max_length=4096),
    ) -> dict[str, object]:
        return await context.require(WORKSPACE_CHANGES).patch(session, before, after, path)
