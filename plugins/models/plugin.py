from __future__ import annotations

from agent.plugin_composition import (
    CHAT_MODELS,
    EMBEDDINGS,
    MODEL_CATALOG,
    MODEL_DRIVERS,
    SNAPSHOT_SEALING,
    Context,
)

from .litellm_catalog import LiteLlmCapabilityCatalog
from .state import ModelsState
from .store import ModelsStore
from .content import MODEL_CONTENT, ContentOwner
from .projection import (
    MODEL_CALLS,
    MODEL_CALL_HISTORY,
    MODEL_DISPLAY,
    MODEL_MESSAGE_CHECKS,
    MODEL_PROJECTION,
    MessageChecksOwner,
    ProjectionOwner,
    display_facts,
)
from .selection import MODEL_SELECTION, SelectionOwner
from .settings import MODEL_SETTINGS
from agent.plugin_composition.models import MODEL_CALL_STATS
from agent.plugin_composition.rpc import rpc_method_key

from .model_settings_http import BoundModelControl, rpc_methods

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
    "models.connection-types.v1": "005155186b59c61f0d67311ce2e0f06dba016d516ba32f3142f0eef754208a4f",
}
dashboard_module = "dashboard.py"


async def apply(ctx: Context, config: object) -> None:
    """Publish narrow views over one Root-local model state."""

    _ = config
    store = ModelsStore(
        ctx.workspace_file("model-registry.sqlite3"),
        backup_dir=ctx.runtime.workspace / "runtime" / "model-backups",
        writable=True,
    )
    store.initialize()
    state = ModelsState(
        store,
        root_instance_token=ctx.root_instance_token,
        context=ctx,
        capability_catalog=LiteLlmCapabilityCatalog(
            ctx.data_root / "litellm-capabilities.json",
            writable=True,
        ),
    )
    _ = await ctx.effect(
        lambda: state.close_auth_attempts,
        label="model-auth-attempts",
    )
    _ = await ctx.provide(MODEL_DRIVERS, state.drivers)
    _ = await ctx.provide(CHAT_MODELS, state.chat_models, binding_contributors=state.chat_contributors)
    _ = await ctx.provide(EMBEDDINGS, state.embeddings)
    _ = await ctx.provide(MODEL_CATALOG, state.catalog)
    _ = await ctx.provide(MODEL_SETTINGS, state.settings)
    _ = await ctx.provide(MODEL_CALLS, store.read_call)
    _ = await ctx.provide(MODEL_CALL_HISTORY, store.read_calls)
    _ = await ctx.provide(MODEL_CALL_STATS, store.read_call_stats)
    _ = await ctx.provide(MODEL_DISPLAY, display_facts)
    _ = await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    _ = await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    _ = await ctx.provide(MODEL_CONTENT, ContentOwner())
    _ = await ctx.provide(MODEL_SELECTION, SelectionOwner())
    for method, operation in rpc_methods(BoundModelControl(ctx)).items():
        _ = await ctx.provide(rpc_method_key(method), operation)
    _ = await ctx.on(SNAPSHOT_SEALING, state.seal)
