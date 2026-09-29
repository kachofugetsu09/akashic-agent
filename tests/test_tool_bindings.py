from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast
import pytest
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition import CompositionRoot, RUNTIME_STARTED, RUNTIME_STARTING, RUNTIME_STOPPING
from agent.plugin_composition.model import PluginRuntime, ServiceKey
from agent.plugin_composition.tasks import TASKS, PluginTasks
from plugins.tool_search.plugin import LoadTools
from plugins.tools.plugin import ToolCatalog, ToolRef

TOOLS = ServiceKey("tools.v1")

def _runtime(tmp_path, plugin_id):
    return PluginRuntime(
        plugin_id,
        plugin_id + ":1",
        tmp_path,
        tmp_path / plugin_id / "data",
        tmp_path / "workspace",
        {},
    )

async def _watch_lifecycle(ctx, events):
    await ctx.on(RUNTIME_STARTING, lambda _event: events.append("starting"))
    await ctx.on(RUNTIME_STARTED, lambda _event: events.append("started"))
    await ctx.on(RUNTIME_STOPPING, lambda _event: events.append("stopping"))

async def _mount_local_tools(root, tmp_path, events):
    contexts = []
    catalogs = []
    admissions = []
    plugin_tasks = PluginTasks()
    await root.context.provide(TASKS, plugin_tasks)

    async def apply(ctx):
        contexts.append(ctx)
        admission = ctx.require(TASKS).open(ctx)
        admissions.append(admission)
        catalog = ToolCatalog(ctx, admission)
        catalogs.append(catalog)
        await ctx.provide(TOOLS, catalog)
        await _watch_lifecycle(ctx, events)

    fiber = await root.mount(
        apply,
        name="tools-provider",
        inject=(TASKS,),
        runtime=_runtime(tmp_path, "tools-provider"),
    )
    return fiber, contexts[0], catalogs[0], plugin_tasks, admissions[0]

@pytest.mark.asyncio
async def test_prepare_and_authorize_follow_exact_registration_identity(tmp_path):
    """同名新工具不能继承旧注册的参数转换或限制。"""
    root = CompositionRoot("exact-tool-contributions")
    refs = {}
    contexts = {}
    tools_fiber, _tools_ctx, catalog, plugin_tasks, _admission = await _mount_local_tools(
        root, tmp_path, []
    )

    def runtime(plugin_id):
        return PluginRuntime(
            plugin_id, plugin_id + ":1", tmp_path, tmp_path, tmp_path, {}
        )

    @asynccontextmanager
    async def open_target(_state):
        raise AssertionError("identity test does not open targets")
        yield

    async def target(ctx, label):
        contexts[label] = ctx
        refs[label] = await catalog.register(
            ctx,
            name="example",
            description="same public description",
            parameters={"type": "object"},
            open=open_target,
        )

    try:
        first = await root.mount(
            lambda ctx: target(ctx, "first"), name="first", runtime=runtime("first")
        )
        async def contribute(ctx):
            async def prepare(arguments):
                return {**arguments, "prepared_by": "first"}

            async def authorize(arguments):
                _ = arguments

            await catalog.register_prepare(
                ctx, tool=refs["first"], name="first-prepare", prepare=prepare
            )
            await catalog.register_authorize(
                ctx, tool=refs["first"], name="first-authorize", authorize=authorize
            )

        _ = await root.mount(contribute, name="policy", runtime=runtime("policy"))

        class CapturingBindings:
            def __init__(self):
                self.metadata = None

            def bind(self, _key, metadata, *, contributors):
                self.metadata = metadata
                return "binding"

        captured = CapturingBindings()
        await catalog.bind(refs["first"], cast(Bindings, captured))
        assert captured.metadata["prepare"] == "first-prepare"
        assert captured.metadata["authorize"] == "first-authorize"

        stale = refs["first"]
        await first.dispose()
        await root.mount(
            lambda ctx: target(ctx, "second"), name="second", runtime=runtime("second")
        )
        await catalog.bind(refs["second"], cast(Bindings, captured))
        assert captured.metadata == {
            "tool": refs["second"].description,
            "prepare": None,
        }

        @asynccontextmanager
        async def open_resumable(_state):
            yield SimpleNamespace(idempotent=False)

        resumable = await catalog.register(
            contexts["second"], name="resumable", description="resumable tool",
            parameters={"type": "object"}, open=open_resumable, capture=lambda state: state,
        )
        legacy = {
            "tool": {
                **resumable.description,
                "risk": "read-only",
                "search_hint": "legacy display only",
            },
            "prepare": None,
            "state": {},
        }
        async with catalog.open(legacy) as opened:
            assert opened.idempotent is False
        await catalog.bind_saved(legacy, cast(Bindings, captured), configuration={})
        assert captured.metadata == {
            "tool": resumable.description,
            "prepare": None,
            "state": {},
        }

        forged = ToolRef(stale.name, stale.description)
        async def passthrough(value):
            return value
        for invalid in (stale, forged):
            with pytest.raises(RuntimeError, match="引用已经失效"):
                catalog.view(invalid)
            with pytest.raises(RuntimeError, match="引用已经失效"):
                await catalog.register_prepare(
                    root.context,
                    tool=invalid,
                    name="forged",
                    prepare=passthrough,
                )
    finally:
        await plugin_tasks.close()
        await tools_fiber.dispose()
        await root.dispose()

@pytest.mark.asyncio
async def test_parallel_flag_stays_out_of_binding_description(tmp_path):
    """重叠执行由 owner 声明，独立于冻结的工具描述和效果类别。"""
    root = CompositionRoot("parallel-flag")
    tools_fiber, tools_ctx, catalog, plugin_tasks, _admission = await _mount_local_tools(
        root, tmp_path, [],
    )

    @asynccontextmanager
    async def open_target(_state):
        yield object()

    try:
        async with tools_ctx.runtime_scope():
            ref = await catalog.register(
                tools_ctx, name="write_example", description="write",
                parameters={"type": "object"}, open=open_target, parallel=True,
            )
            assert "parallel" not in ref.description
            assert "risk" not in ref.description
            assert catalog.allows_parallel("write_example") is True
            assert catalog.allows_parallel("missing") is False
    finally:
        await plugin_tasks.close()
        await tools_fiber.dispose()
        await root.dispose()


@pytest.mark.asyncio
async def test_load_tools_returns_only_the_requested_frozen_group():
    """PLG-018: 准确插件 ID 不得选择其他已获授或未获授组。"""
    tool = LoadTools((
        {
            "plugin": "calendar@github",
            "purpose": "查看日历",
            "tools": (
                {
                    "type": "function",
                    "function": {
                        "name": "calendar_list",
                        "description": "列出日历",
                        "parameters": {"type": "object"},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calendar_events",
                        "description": "查看事件",
                        "parameters": {"type": "object"},
                    },
                },
            ),
        },
    ))

    loaded = await tool.invoke("load", {"plugin": "calendar@github"})
    assert loaded.outcome == "success"
    import json
    value = loaded.parts[0].value
    assert isinstance(value, str)
    payload = json.loads(value)
    assert payload["plugin"] == "calendar@github"
    assert [item["function"]["name"] for item in payload["tools"]] == [
        "calendar_list", "calendar_events",
    ]

    missing = await tool.invoke("load", {"plugin": "calendar"})
    assert missing.outcome == "error"
    missing_value = missing.parts[0].value
    assert isinstance(missing_value, str)
    assert "calendar_events" not in missing_value
