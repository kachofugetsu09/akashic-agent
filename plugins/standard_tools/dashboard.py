"""终端标签的只读接口：列出本会话命令、取单条输出、停止后台命令。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext

from .command_log import COMMAND_LOG_VIEW, SLOTS, record_view

inject = (COMMAND_LOG_VIEW,)


def register(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/standard-tools/commands")
    def commands(session: str = Query(min_length=1, max_length=512)) -> dict[str, object]:
        records = context.require(COMMAND_LOG_VIEW).list(session)
        return {"session": session, "commands": [record_view(value, with_output=False) for value in records]}

    @app.get("/api/dashboard/standard-tools/commands/{slot}")
    def command(slot: int, session: str = Query(min_length=1, max_length=512)) -> dict[str, object]:
        records = context.require(COMMAND_LOG_VIEW).list(session)
        found = next((value for value in records if value["slot"] == slot), None)
        if found is None:
            raise HTTPException(404, "命令记录不存在或已被覆盖")
        return record_view(found, with_output=True)

    @app.post("/api/dashboard/standard-tools/commands/{slot}/stop")
    async def stop(slot: int, session: str = Query(min_length=1, max_length=512)) -> dict[str, object]:
        if not 0 <= slot < SLOTS:
            raise HTTPException(404, "命令记录不存在")
        return {"stopped": await context.require(COMMAND_LOG_VIEW).stop(session, slot)}

