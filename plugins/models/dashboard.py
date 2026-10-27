"""Expose the exact models generation to its ordinary Web module."""

from datetime import datetime

from fastapi import FastAPI, HTTPException

from agent.plugin_composition import DashboardContext
from plugins.models.contract import (
    MODEL_CACHE_USAGE,
    MODEL_CATALOG,
    MODEL_CALL_STATS,
)

from .model_settings_http import BoundModelControl, create_model_settings_router
from .selection import MODEL_SELECTION
from .settings import MODEL_SETTINGS


inject = (MODEL_CATALOG, MODEL_CALL_STATS, MODEL_SETTINGS, MODEL_SELECTION, MODEL_CACHE_USAGE)


def register(app: FastAPI, context: DashboardContext) -> None:
    """Register routes that resolve declared services in each request lease."""

    router = create_model_settings_router(
        BoundModelControl(context),
        prefix="/api/dashboard/models",
    )
    app.router.routes.extend(router.routes)

    # 设置·用量：since 是界面所选范围的本地零点；会话排行在侧栏面板，这里只取一行。
    @app.get("/api/dashboard/models/usage")
    async def usage(since: str) -> dict[str, object]:
        try:
            moment = datetime.fromisoformat(since)
        except ValueError as error:
            raise HTTPException(status_code=422, detail="since 须为 ISO 时刻") from error
        if moment.tzinfo is None:
            raise HTTPException(status_code=422, detail="since 须带时区")
        return await context.require(MODEL_CACHE_USAGE).insights(since, 1, "input")
