"""Expose the exact models generation to its ordinary Web module."""

from fastapi import FastAPI, HTTPException

from agent.plugin_composition import DashboardContext
from plugins.models.contract import (
    MODEL_CACHE_USAGE,
    MODEL_CATALOG,
    MODEL_CALL_STATS,
)

from .model_settings_http import BoundModelControl, create_model_settings_router, usage_overview
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

    @app.get("/api/dashboard/models/usage")
    async def usage(since: str) -> dict[str, object]:
        try:
            return await usage_overview(context.require(MODEL_CACHE_USAGE), since)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
