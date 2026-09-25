from contextlib import asynccontextmanager
from typing import cast
import pytest
from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition import CompositionRoot, RUNTIME_STARTED, RUNTIME_STARTING, RUNTIME_STOPPING
from agent.plugin_composition.model import PluginRuntime, ServiceKey
from agent.plugin_composition.tasks import TASKS, PluginTasks
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
        with pytest.raises(TypeError, match="搜索提示"):
            await catalog.register(
                contexts["first"],
                name="bad_hint",
                description="bad search hint",
                parameters={"type": "object"},
                open=open_target,
                search_hint=cast(str, object()),
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
        catalog.bind(refs["first"], cast(Bindings, captured))
        assert captured.metadata["prepare"] == "first-prepare"
        assert captured.metadata["authorize"] == "first-authorize"

        stale = refs["first"]
        await first.dispose()
        await root.mount(
            lambda ctx: target(ctx, "second"), name="second", runtime=runtime("second")
        )
        catalog.bind(refs["second"], cast(Bindings, captured))
        assert captured.metadata == {
            "tool": refs["second"].description,
            "prepare": None,
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
