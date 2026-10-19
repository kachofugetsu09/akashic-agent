from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from importlib import import_module
from typing import cast

from agent.plugin_composition import Context
from plugins.models.contract import EMBEDDINGS
from plugins.models.contract import (
    CHAT_MODELS,
    MODEL_CATALOG,
    MODEL_DRIVERS,
)
from plugins.models.contract import MODEL_CACHE_USAGE, MODEL_CALL_STATS
from plugins.gateway.contract import RpcMethod
from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition, hidden_part, part_display_key
from plugins.conversation.contract import input_part_key
from plugins.ledger.contract import MESSAGE_CATALOG, MESSAGE_WRITERS, ContentPart, ContentReferences, MessageReader
from plugins.models.contract import ChatModelSelection, ModelError, ModelUnavailableError
from plugins.channels.contract import InputPartRejected

from .content import MODEL_CONTENT, ContentOwner
from .views import CONTENT_VIEWS, ContentViews
from .litellm_catalog import LiteLlmCapabilityCatalog
from .model_settings_http import BoundModelControl, rpc_methods
from .projection import (
    MODEL_CALLS,
    MODEL_DISPLAY,
    MODEL_MESSAGE_CHECKS,
    MODEL_PROJECTION,
    MessageChecksOwner,
    ProjectionOwner,
    display_facts,
)
from .chat_ui import ChatUiQueries
from .selection import MODEL_SELECTION, SAVED_SELECTION_KEYS, SelectionOwner, check_selection, saved_from_part
from .settings import MODEL_SETTINGS
from .state import ModelsState
from .store import ModelsStore
from .usage import CacheUsage

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
    _ = await ctx.provide(MODEL_CALL_STATS, store.read_call_stats)
    usage = CacheUsage(ctx, store)
    _ = await ctx.provide(MODEL_CACHE_USAGE, usage)
    _ = await ctx.provide(MODEL_DISPLAY, display_facts)
    # 自己写入的内部材料不进聊天正文；宿主不维护按插件列举的隐藏名单。
    for kind in ("model.selection", "tool.selection"):
        _ = await ctx.provide(part_display_key(kind), hidden_part)
    _ = await ctx.provide(MODEL_PROJECTION, ProjectionOwner())
    _ = await ctx.provide(MODEL_MESSAGE_CHECKS, MessageChecksOwner())
    _ = await ctx.provide(MODEL_CONTENT, ContentOwner())
    views = ContentViews(ctx)
    _ = await ctx.provide(CONTENT_VIEWS, views, binding_contributors=views.binding_contributors)
    _ = await ctx.provide(MODEL_SELECTION, SelectionOwner())
    from plugins.onboarding.contract import ONBOARDING, Ability, Step
    async def status():
        async with ctx.runtime_scope():
            catalog = state.catalog.snapshot()
            model_id = catalog.role_bindings.get("default")
            ready = model_id is not None and catalog.model(model_id).availability == 'available'
            return {"ready": ready, "enabled": True if ready else None, "blocked": False,
                    "reason": "" if ready else "请选择连接方式并设置默认模型"}
    async def contribute(child: Context):
        await child.require(ONBOARDING).group(child, "models", "模型连接", Ability(
            pitch="连接一个对话模型",
            benefit="Akashic 本身不带模型。选一个你已经有账号的服务，连上之后它就能回答你。",
            required=True,
        ))
        await child.require(ONBOARDING).register(child, Step("connect", "模型连接", "models", "models", status))
    await ctx.inject((ONBOARDING,), contribute, name="onboarding")
    for method, operation in rpc_methods(BoundModelControl(ctx)).items():
        _ = await ctx.provide(RpcMethod.key(method), operation)
    _ = await ctx.inject((UI, MODEL_CATALOG, MODEL_CALL_STATS, MODEL_SETTINGS, MODEL_SELECTION, MODEL_CACHE_USAGE),
                         _register_ui, name="ui")

    _ = await ctx.inject((MESSAGE_WRITERS,), _input_selection(state), name="input-selection")

    # 回答尾注与输入框模型胶囊都只读本插件的视图；聊天界面经插件界面插槽查询，选择随消息作为 model.selection 片段回来。
    async def message_ui(child: Context) -> None:
        query = ChatUiQueries(catalog=state.read_catalog, selection=state.read_chat_selection, messages=child.require(MESSAGE_CATALOG), call_stats=store.read_call_stats,
                              message_tokens=usage.message_output_tokens, icon=state.connection_icon)
        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="plugin_ui.js", stylesheet="plugin_ui.css",
                                      slots=("turn.after_answer", "composer.toolbar", "composer.status")), query=query)

    _ = await ctx.inject((UI_SLOTS, MESSAGE_CATALOG), message_ui, name="message-ui")


# 客户端随消息附带的模型选择：校验与会话保存都归本插件，二者随消息写入能力一起出现，没有只校验不保存的窗口。
def _input_selection(state: ModelsState) -> Callable[[Context], Awaitable[None]]:
    # 形状不对、模型不可用或强度不支持都是该请求的拒收，回给发送者而不是当程序错误。
    def check(part: ContentPart, reader: MessageReader) -> ContentReferences:
        try:
            references = check_selection(part)
            value = cast(Mapping[str, str | None], part.value)
            _ = state.catalog.validate_chat_selection(ChatModelSelection(value["model_id"], value["reasoning_effort"]))
        except (TypeError, ValueError) as error:
            raise InputPartRejected(str(error)) from error
        except RuntimeError as error:
            unavailable = ModelError.read(error, ModelUnavailableError)
            if unavailable is None:
                raise
            raise InputPartRejected(unavailable.message) from error
        return references

    async def install(ctx: Context) -> None:
        _ = await ctx.require(MESSAGE_WRITERS).register_part_metadata(
            ctx, kind="model.selection", keys=SAVED_SELECTION_KEYS, update=saved_from_part,
        )
        _ = await ctx.provide(input_part_key("model.selection"), check)

    return install


async def _register_ui(ctx: Context) -> None:
    """界面随 UI provider 换代，不牵动计算与持久状态。"""
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__),
        requires=("shell.settings.v1",),
        provides=("models.connection-types.v1",),
        contract_digests={
            "models.connection-types.v1": "8c304d85090a65a4a66cd777a5c2a88e6908de147964dc91e7f84336979ea561",
            "shell.settings.v1": "1f4dd2eaee9118c799590a36745975150c46979a3f62b431f73aa29e949ce189",
        },
    )
