from __future__ import annotations

from agent.plugin_composition import (
    CHAT_MODELS,
    EMBEDDINGS,
    MODEL_CATALOG,
    MODEL_DRIVERS,
    MODEL_SETTINGS,
    SNAPSHOT_SEALING,
    Context,
)

from .state import ModelsState
from .store import ModelsStore

api_version = 3
name = "models"
version = "1.0.0"
desc = "Provider-neutral model connections, selection, and execution"
author = "Akashic Core"
inject = ()
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ("model-registry.sqlite3",)
web_module = "web_module.js"
web_requires = ("shell.pages.v1",)
web_provides = ("models.connection-types.v1",)
web_contract_digests = {
    "models.connection-types.v1": "258bc92f1a3f7e15c8d5421c787a31f2fa1c76f2d9166c0e806f858e23853266",
}
dashboard_module = "dashboard.py"


async def apply(ctx: Context, config: object) -> None:
    """Publish five narrow views over one Root-local model state."""

    _ = config
    store = ModelsStore(
        ctx.workspace_file("model-registry.sqlite3"),
        backup_dir=ctx.runtime.workspace / "runtime" / "model-backups",
        writable=ctx.data_access == "read_write",
    )
    if ctx.data_access == "read_write":
        store.initialize()
    state = ModelsState(store, root_instance_token=ctx.root_instance_token)
    _ = await ctx.effect(
        lambda: state.close_auth_attempts,
        label="model-auth-attempts",
    )
    _ = await ctx.provide(MODEL_DRIVERS, state.drivers)
    _ = await ctx.provide(CHAT_MODELS, state.chat_models)
    _ = await ctx.provide(EMBEDDINGS, state.embeddings)
    _ = await ctx.provide(MODEL_CATALOG, state.catalog)
    _ = await ctx.provide(MODEL_SETTINGS, state.settings)
    _ = await ctx.on(SNAPSHOT_SEALING, state.seal)
