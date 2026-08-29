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
    _ = await ctx.provide(MODEL_DRIVERS, state.drivers)
    _ = await ctx.provide(CHAT_MODELS, state.chat_models)
    _ = await ctx.provide(EMBEDDINGS, state.embeddings)
    _ = await ctx.provide(MODEL_CATALOG, state.catalog)
    _ = await ctx.provide(MODEL_SETTINGS, state.settings)
    _ = await ctx.on(SNAPSHOT_SEALING, state.seal)
