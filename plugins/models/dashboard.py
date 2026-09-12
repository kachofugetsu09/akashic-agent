"""Expose the exact models generation to its ordinary Web module."""

from agent.plugin_composition.model_settings_http import BoundModelControl

from .model_settings_http import create_model_settings_router


def register(app, context):
    _ = context
    router = create_model_settings_router(
        BoundModelControl(),
        prefix="/api/dashboard/models",
    )
    app.router.routes.extend(router.routes)
