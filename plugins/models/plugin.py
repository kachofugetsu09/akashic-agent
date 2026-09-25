from __future__ import annotations

from importlib import import_module

from agent.plugin_composition import (
    CHAT_MODELS,
    EMBEDDINGS,
    MODEL_CATALOG,
    MODEL_DRIVERS,
    Context,
)
from agent.plugin_composition.models import MODEL_CALL_STATS
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_composition.ui import UI

from .content import MODEL_CONTENT, ContentOwner
from .litellm_catalog import LiteLlmCapabilityCatalog
from .model_settings_http import BoundModelControl, rpc_methods
from .projection import (
    MODEL_CALL_HISTORY,
    MODEL_CALLS,
    MODEL_DISPLAY,
    MODEL_MESSAGE_CHECKS,
    MODEL_PROJECTION,
    MessageChecksOwner,
    ProjectionOwner,
    display_facts,
)
from .selection import MODEL_SELECTION, SelectionOwner
from .settings import MODEL_SETTINGS
from .state import ModelsState
from .store import ModelsStore

api_version = 3
name = "models"
version = "1.0.0"
desc = "Provider-neutral model connections, selection, and execution"
author = "Akashic Core"
inject = ()
workspace_roots = ()
workspace_files = ("model-registry.sqlite3",)


async def apply(ctx: Context) -> None:
    """Publish narrow views over one Root-local model state."""


    store = ModelsStore(
        ctx.workspace_file("model-registry.sqlite3"),
        backup_dir=ctx.runtime.workspace / "runtime" / "model-backups",
        writable=True,
    )
    store.initialize()
    # 宿主租约在本 Root 退役时最后释放：effect 清理按注册逆序执行，
    # 最先注册使它晚于 auth attempt 清理与其他资源归还。
    _ = await ctx.effect(lambda: store.close, label="model-registry-host-lock")
    state = ModelsState(
        store,
        context=ctx,
        capability_catalog=LiteLlmCapabilityCatalog(
            ctx.data_root / "litellm-capabilities.json",
            writable=True,
        ),
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
    _ = await ctx.inject((UI, MODEL_CATALOG, MODEL_CALL_STATS, MODEL_SETTINGS, MODEL_SELECTION),
                         _register_ui, name="ui")


async def _register_ui(ctx: Context) -> None:
    """界面随 UI provider 换代，不牵动计算与持久状态。"""
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__),
        requires=("shell.pages.v1",),
        provides=("models.connection-types.v1",),
        contract_digests={
            "models.connection-types.v1": "005155186b59c61f0d67311ce2e0f06dba016d516ba32f3142f0eef754208a4f",
        },
    )
