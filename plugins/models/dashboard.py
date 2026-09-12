"""Expose the exact models generation to its ordinary Web module."""

from fastapi import FastAPI

from agent.plugin_composition import DashboardContext
from agent.plugin_composition.models import MODEL_CATALOG, MODEL_CALL_STATS

from .model_settings_http import BoundModelControl, create_model_settings_router
from .selection import MODEL_SELECTION
from .settings import MODEL_SETTINGS


inject = (MODEL_CATALOG, MODEL_CALL_STATS, MODEL_SETTINGS, MODEL_SELECTION)


def register(app: FastAPI, context: DashboardContext) -> None:
    """Register routes that resolve declared services in each request lease."""

    router = create_model_settings_router(
        BoundModelControl(context),
        prefix="/api/dashboard/models",
    )
    app.router.routes.extend(router.routes)
