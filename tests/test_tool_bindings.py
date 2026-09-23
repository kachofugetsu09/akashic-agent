import asyncio
from contextlib import asynccontextmanager
from collections.abc import Mapping
from functools import partial
from pathlib import Path
import shutil
from typing import cast

import pytest

from tests.fixtures.plugin_workspace import initialize_plugin_workspace

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition import (
    CompositionError,
    CompositionRoot,
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    RUNTIME_STOPPING,
)
from agent.plugin_composition.model import FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition.tasks import TASKS, PluginTasks, Tasks
from agent.plugin_contracts import CallRef
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.tools.api import Result, durable_call_key
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import (
    ALL_TOOLS,
    TOOL_DISPLAY_NAME,
    ToolCatalog,
    ToolRef,
    open_tool,
)
from session.log import MessageLog

TOOLS = ServiceKey("tools.v1")
LOCAL_TOOL_REF = ServiceKey("test.local.tool-ref")


def write_plugins(path):
    path.mkdir()
    for name in ("tools", "content"):
        shutil.copytree(Path(__file__).resolve().parents[1] / "plugins" / name,
                        path / name, ignore=shutil.ignore_patterns("__pycache__"))
    target = path / "target"
    target.mkdir()
    (target / "plugin.py").write_text("""
from contextlib import asynccontextmanager
from pathlib import Path
from agent.plugin_composition import ServiceKey
from dataclasses import dataclass
from agent.plugin_contracts import ContentPart
@dataclass(frozen=True)
class Result:
    outcome: str
    parts: tuple[ContentPart, ...]
api_version = 3
name = "target"
version = "1.0.0"
inject = (ServiceKey("tools.v1"),)
async def apply(ctx):
    class Target:
        idempotent = False
        async def prepare(self, arguments, source=None):
            if not isinstance(arguments["value"], str):
                raise ValueError("value must be text")
            return {"value": arguments["value"].strip()}
        async def invoke(self, key, arguments):
            log = ctx.data_root / "effects.txt"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as file:
                file.write(key + "\\n")
            return Result("success", (ContentPart("text", "A:" + arguments["value"]),))
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open_target(state):
        yield Target()
    ref = await ctx.require(inject[0]).register(
        ctx, name="example", description="Example target A",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},
        open=open_target,
    )
    await ctx.provide(ServiceKey("fixture.example-ref"), ref)
""")
    prepare = path / "prepare"
    prepare.mkdir()
    (prepare / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "prepare"
version = "1.0.0"
inject = (ServiceKey("tools.v1"), ServiceKey("fixture.example-ref"))
async def apply(ctx):
    async def prepare(arguments):
        return {"value": "restore:" + arguments["value"]}
    await ctx.require(inject[0]).register_prepare(
        ctx, tool=ctx.require(inject[1]), name="restore", prepare=prepare,
    )
""")


def add_authorize(path):
    policy = path / "authorize"
    policy.mkdir()
    (policy / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "authorize"
version = "1.0.0"
inject = (ServiceKey("tools.v1"), ServiceKey("fixture.example-ref"))
async def apply(ctx):
    async def authorize(arguments):
        if arguments["value"] == "restore:blocked":
            return "blocked by fixed policy"
    await ctx.require(inject[0]).register_authorize(
        ctx, tool=ctx.require(inject[1]), name="fixed-policy", authorize=authorize,
    )
""")


def manager(tmp_path, sources, log=None):
    return PluginManager(
        plugin_dirs=sources,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
        message_log=log,
    )


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
        runtime=_runtime(tmp_path, "tools-provider"),
    )
    return fiber, contexts[0], catalogs[0], plugin_tasks, admissions[0]


async def _mount_hard_tools_consumer(root, tmp_path):
    """Mount a real Tools consumer whose effect marks dependency unload."""
    unloading = asyncio.Event()

    async def apply(ctx):
        ctx.require(TOOLS)

        async def close_effect():
            unloading.set()

        await ctx.effect(lambda: close_effect, label="hard-tools-consumer")

    fiber = await root.mount(
        apply,
        name="hard-tools-consumer",
        inject=(TOOLS,),
        runtime=_runtime(tmp_path, "hard-tools-consumer"),
    )
    return fiber, unloading


@pytest.mark.asyncio
async def test_tool_drain_keeps_original_admission_during_tools_unload(tmp_path):
    """Drain waits through Tools unload without reopening its Context."""
    root = CompositionRoot("tools-drain-admission")
    tools_fiber, tools_ctx, catalog, plugin_tasks, admission = await _mount_local_tools(
        root, tmp_path, []
    )
    _consumer_fiber, consumer_unloading = await _mount_hard_tools_consumer(
        root, tmp_path
    )
    call = CallRef("old-call", 0)
    entered = asyncio.Event()
    release = asyncio.Event()
    old_task = None
    drain = None
    unloading = None
    tasks_closed = False

    async def old_effect(_task):
        entered.set()
        await release.wait()

    try:
        async with tools_ctx.runtime_scope():
            old_task = await admission.admit(
                ("effects", durable_call_key(call)),
                lambda slot: slot.start(old_effect),
            )
        await entered.wait()

        # The hard consumer observes dependency unload before Tools can reach
        # STOPPING; this is the real lifecycle barrier, not the owner's event.
        unloading = asyncio.create_task(tools_fiber.dispose())
        await consumer_unloading.wait()
        assert tools_fiber.state is FiberState.UNLOADING
        assert old_task is not None and not old_task.done

        # Drain runs in an independent Task with no Tools OwnerCall.  It must
        # wait for the already-admitted old call instead of reopening Context.
        drain = asyncio.create_task(catalog.drain_calls((call,)))
        barrier = asyncio.Event()
        asyncio.get_running_loop().call_soon(barrier.set)
        await barrier.wait()
        assert not drain.done()

        release.set()
        await drain
        await unloading
        assert old_task.done
    finally:
        release.set()
        pending = tuple(
            task for task in (drain, unloading) if task is not None and not task.done()
        )
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not tasks_closed:
            await plugin_tasks.close()
            tasks_closed = True
        await root.dispose()


@pytest.mark.asyncio
async def test_tool_drain_rejects_closed_admission(tmp_path):
    """A closed Core Task admission is an explicit drain failure."""
    root = CompositionRoot("closed-tools-drain")
    tools_fiber, _tools_ctx, catalog, plugin_tasks, _admission = await _mount_local_tools(
        root, tmp_path, []
    )
    closed = False
    try:
        await plugin_tasks.close()
        closed = True
        with pytest.raises(RuntimeError, match="Task 服务已关闭"):
            await catalog.drain_calls((CallRef("closed-call", 0),))
    finally:
        if not closed:
            await plugin_tasks.close()
        await tools_fiber.dispose()
        await root.dispose()


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


@pytest.mark.asyncio
async def test_display_name_reads_old_binding_without_opening_removed_tool(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            from agent.plugin_composition.bindings import BINDINGS
            ctx = snapshot.composition_root.context
            binding_id = ctx.require(TOOLS).bind(
                ctx.require(ALL_TOOLS)().select("example"), ctx.require(BINDINGS)
            )
        await host.terminate_all()
        shutil.rmtree(sources / "target")
        shutil.rmtree(sources / "prepare")
        restored = manager(tmp_path, [sources], log)
        try:
            await restored.load_all()
            async with lease_runtime_snapshot(restored.snapshot_store) as snapshot:
                lookup = snapshot.composition_root.context.require(TOOL_DISPLAY_NAME)
                assert lookup(binding_id) == "example"
                with pytest.raises(KeyError):
                    lookup("missing")
            assert not list((tmp_path / "workspace").rglob("effects.txt"))
        finally:
            await restored.terminate_all()
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_ordinary_tool_binding_runs_in_the_selected_stable_scope(
    tmp_path,
):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [sources])
    log = MessageLog(tmp_path / "sessions.db")
    tasks = Tasks()
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            binding_id = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
        authorized = []

        async def authorize(binding, arguments):
            authorized.append((binding, arguments))
            return {"permission": "current"}

        execution = ToolExecution(
            log.owner("tools"),
            tasks,
            partial(open_tool, bindings),
            authorize,
            task_key="tools"
        )
        result = await execution.execute("request", binding_id, {"value": "input "})
        assert result.parts[0].value == "A:restore:input"
        assert authorized == [(binding_id, {"value": "restore:input"})]
        assert (
            await execution.execute("request", binding_id, {"value": "input "})
        ).parts == result.parts
        effects = list((tmp_path / "workspace").rglob("effects.txt"))
        assert len(effects) == 1
        assert effects[0].read_text().splitlines() == ["program:request"]
        async with open_tool(bindings, binding_id) as expired:
            assert not expired.idempotent
        with pytest.raises(RuntimeError, match="释放"):
            await expired.invoke("escaped", {"value": "should not run"})
        assert effects[0].read_text().splitlines() == ["program:request"]
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_tool_configuration_is_owned_frozen_without_recapture(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    path = sources / "target/plugin.py"
    source = path.read_text().replace('"A:" + arguments["value"]', 'state["prefix"] + arguments["value"]')
    source = source.replace('    class Target:', '''    def capture(configuration):
        if set(configuration) != {"prefix"} or not isinstance(configuration["prefix"], str):
            raise ValueError("prefix configuration required")
        return {"prefix": configuration["prefix"].strip()}
    class Target:
        def __init__(self, captured):
            self.state = captured''').replace('state["prefix"] + arguments', 'self.state["prefix"] + arguments')
    source = source.replace('yield Target()', 'yield Target(state)').replace('open=open_target,', 'open=open_target, capture=capture,')
    path.write_text(source)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [sources])
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            ref = ctx.require(ALL_TOOLS)().select("example")
            with pytest.raises(ValueError, match="prefix configuration"):
                catalog.bind(ref, bindings)
            with pytest.raises(ValueError, match="prefix configuration"):
                catalog.bind(ref, bindings, configuration={"unexpected": True})
            options = {"prefix": " job-a: "}
            identity = catalog.bind(ref, bindings, configuration=options)
            options["prefix"] = "job-b:"
            assert bindings.describe(identity, TOOLS)["state"] == {"prefix": "job-a:"}
        async with open_tool(bindings, identity) as target:
            prepared = await target.prepare({"value": "input"})
            assert isinstance(prepared, Mapping)
            result = await target.invoke("fixed", prepared)
            assert result.parts[0].value == "job-a:restore:input"
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_binding_authorize_checks_final_arguments(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            old_binding = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
        await host.terminate_all()

        add_authorize(sources)
        host = manager(tmp_path, [sources], log)
        await host.load_all()
        caller_checks = []

        async def caller_authorize(binding, arguments):
            caller_checks.append((binding, arguments))
            return {"permission": "caller"}

        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            new_binding = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
            execution = catalog.execution(caller_authorize)
            with pytest.raises(ValueError, match="归档工具限制与 binding 不一致"):
                await execution.execute("old", old_binding, {"value": "blocked"})
            denied = await execution.execute("denied", new_binding, {"value": "blocked"})
            safe = await execution.execute("safe", new_binding, {"value": "ok"})

        assert denied.outcome == "denied"
        assert denied.parts[0].value == "blocked by fixed policy"
        assert safe.outcome == "success"
        assert caller_checks == [(new_binding, {"value": "restore:ok"})]
        effects = sorted(
            line
            for path in (tmp_path / "workspace").rglob("effects.txt")
            for line in path.read_text().splitlines()
        )
        assert effects == ["program:safe"]

        malformed = dict(bindings.describe(new_binding, TOOLS))
        malformed["authorize"] = None
        with pytest.raises(ValueError, match="限制字段无效") as error:
            async with catalog.open(malformed):
                raise AssertionError("损坏 binding 不应打开工具")
        assert type(error.value) is ValueError
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["removed", "renamed"])
async def test_binding_authorize_presence_and_name_must_match_current_registration(
    tmp_path, replacement
):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    add_authorize(sources)
    log = MessageLog(tmp_path / "sessions.db")
    initialize_plugin_workspace(tmp_path / "workspace")
    host = manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            ctx = snapshot.composition_root.context
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            catalog = ctx.require(TOOLS)
            old_binding = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
        await host.terminate_all()

        if replacement == "removed":
            shutil.rmtree(sources / "authorize")
        else:
            path = sources / "authorize" / "plugin.py"
            path.write_text(path.read_text().replace("fixed-policy", "renamed-policy"))

        host = manager(tmp_path, [sources], log)
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            assert snapshot.composition_root is not None
            ctx = snapshot.composition_root.context
            bindings = Bindings(log, host._archive, snapshot.composition_root)
            catalog = ctx.require(TOOLS)
            metadata = bindings.describe(old_binding, TOOLS)
            with pytest.raises(ValueError, match="归档工具限制与 binding 不一致") as error:
                async with catalog.open(metadata):
                    raise AssertionError("不兼容 binding 不应打开工具")
            assert type(error.value) is ValueError

            async def _allow(binding: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
                return {"permission": "caller"}

            execution = catalog.execution(_allow)
            with pytest.raises(ValueError, match="归档工具限制与 binding 不一致"):
                await execution.execute("incompatible", old_binding, {"value": "ok"})
        assert not list((tmp_path / "workspace").rglob("effects.txt"))
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_local_tool_open_protects_target_until_opener_cleanup(tmp_path):
    """Real target work keeps only its target and Tools owners in flight."""

    root = CompositionRoot("local-tools-open")
    tools_events = []
    unrelated_events = []
    tools_fiber, tools_ctx, catalog, plugin_tasks, _admission = await _mount_local_tools(
        root, tmp_path, tools_events
    )

    unrelated_contexts = []

    async def unrelated_apply(ctx):
        unrelated_contexts.append(ctx)
        await _watch_lifecycle(ctx, unrelated_events)

    unrelated_fiber = await root.mount(
        unrelated_apply,
        name="unrelated",
        runtime=_runtime(tmp_path, "unrelated"),
    )
    unrelated_ctx = unrelated_contexts[0]
    tools_events_before = tuple(tools_events)
    unrelated_events_before = tuple(unrelated_events)

    refs = []
    opener_entered = asyncio.Event()
    invoke_entered = asyncio.Event()
    query_finished = asyncio.Event()
    release_invoke = asyncio.Event()
    opener_cleanup_done = asyncio.Event()
    target_effect_done = asyncio.Event()
    cleanup_order = []
    target_effect_count = 0
    open_calls = 0

    async def target_apply(ctx):
        nonlocal target_effect_count

        def close_target():
            nonlocal target_effect_count
            target_effect_count += 1
            cleanup_order.append("target-effect")
            target_effect_done.set()

        await ctx.effect(lambda: close_target, label="target-resource")

        class Target:
            idempotent = False

            async def prepare(self, arguments, source=None):
                _ = source
                return arguments

            async def invoke(self, key, arguments):
                _ = (key, arguments)
                invoke_entered.set()
                await release_invoke.wait()
                return Result("success", ())

            async def query(self, key):
                _ = key
                query_finished.set()
                return None

        target = Target()

        @asynccontextmanager
        async def open_target(_state):
            nonlocal open_calls
            open_calls += 1
            opener_entered.set()
            try:
                yield target
            finally:
                captured = ctx.capture_runtime_scope()
                async with captured:
                    cleanup_order.append("opener")
                    opener_cleanup_done.set()

        refs.append(
            await ctx.require(TOOLS).register(
                ctx,
                name="blocked_target",
                description="A blocked local target",
                parameters={"type": "object"},
                open=open_target,
            )
        )

    target_fiber = await root.mount(
        target_apply,
        name="target",
        inject=(TOOLS,),
        runtime=_runtime(tmp_path, "target"),
    )
    metadata = {"tool": refs[0].description, "prepare": None}

    async def use_tool():
        async with catalog.open(metadata) as opened:
            await opened.invoke("invoke", {})
            await opened.query("query")

    open_task = asyncio.create_task(use_tool())
    await opener_entered.wait()
    await invoke_entered.wait()
    assert open_calls == 1

    dispose_task = asyncio.create_task(target_fiber.dispose())
    marker = asyncio.Event()
    asyncio.get_running_loop().call_soon(marker.set)
    await marker.wait()
    assert target_fiber.state is FiberState.UNLOADING
    assert target_effect_count == 0
    assert not opener_cleanup_done.is_set()
    assert tools_fiber.state is FiberState.ACTIVE
    assert unrelated_fiber.state is FiberState.ACTIVE
    assert tools_ctx is tools_fiber.context
    assert unrelated_ctx is unrelated_fiber.context
    assert tuple(tools_events) == tools_events_before
    assert tuple(unrelated_events) == unrelated_events_before

    tools_calls_before_failed_open = len(tools_ctx.fiber._fiber._in_flight_calls)
    with pytest.raises(CompositionError, match="当前 activation 不接纳"):
        async with catalog.open(metadata):
            raise AssertionError("UNLOADING target 不应进入 opener body")
    assert open_calls == 1
    assert len(tools_ctx.fiber._fiber._in_flight_calls) == tools_calls_before_failed_open

    release_invoke.set()
    await open_task
    await dispose_task
    assert query_finished.is_set()
    assert cleanup_order == ["opener", "target-effect"]
    assert target_effect_done.is_set()
    assert target_effect_count == 1
    assert not tools_ctx.fiber._fiber._in_flight_calls
    await plugin_tasks.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_local_tool_prepare_and_authorize_protect_contributor_scopes(tmp_path):
    """Preparation and authorization protect only their real callback owners."""

    root = CompositionRoot("local-tools-contributors")
    tools_events = []
    _tools_fiber, _tools_ctx, catalog, plugin_tasks, _admission = await _mount_local_tools(
        root, tmp_path, tools_events
    )
    refs = []
    target_opener_cleanup = asyncio.Event()

    async def target_apply(ctx):
        class Target:
            idempotent = False

            async def prepare(self, arguments, source=None):
                _ = source
                captured = ctx.capture_runtime_scope()
                async with captured:
                    return arguments

            async def invoke(self, key, arguments):
                _ = (key, arguments)
                return Result("success", ())

            async def query(self, key):
                _ = key
                return None

        target = Target()

        @asynccontextmanager
        async def open_target(_state):
            try:
                yield target
            finally:
                captured = ctx.capture_runtime_scope()
                async with captured:
                    target_opener_cleanup.set()

        ref = await ctx.require(TOOLS).register(
            ctx,
            name="contributor_target",
            description="A target with local contributors",
            parameters={"type": "object"},
            open=open_target,
        )
        refs.append(ref)
        await ctx.provide(LOCAL_TOOL_REF, ref)

    target_fiber = await root.mount(
        target_apply,
        name="target",
        inject=(TOOLS,),
        runtime=_runtime(tmp_path, "target"),
    )
    prepare_contexts = []
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    prepare_closed = asyncio.Event()
    prepare_close_count = 0

    async def prepare_apply(ctx):
        nonlocal prepare_close_count
        prepare_contexts.append(ctx)

        def close_prepare():
            nonlocal prepare_close_count
            prepare_close_count += 1
            prepare_closed.set()

        await ctx.effect(lambda: close_prepare, label="prepare-resource")

        async def prepare(arguments):
            assert ctx.fiber._fiber._call_owned_by_current_task() is not None
            if arguments["value"] == "error":
                raise ValueError("prepare callback failed")
            prepare_started.set()
            await release_prepare.wait()
            return {"value": arguments["value"].strip()}

        await ctx.require(TOOLS).register_prepare(
            ctx,
            tool=ctx.require(LOCAL_TOOL_REF),
            name="normalize",
            prepare=prepare,
        )

    prepare_fiber = await root.mount(
        prepare_apply,
        name="prepare",
        inject=(TOOLS, LOCAL_TOOL_REF),
        runtime=_runtime(tmp_path, "prepare"),
    )
    prepare_ctx = prepare_contexts[0]

    authorize_contexts = []
    authorize_started = asyncio.Event()
    authorize_closed = asyncio.Event()
    authorize_close_count = 0

    async def authorize_apply(ctx):
        nonlocal authorize_close_count
        authorize_contexts.append(ctx)

        def close_authorize():
            nonlocal authorize_close_count
            authorize_close_count += 1
            authorize_closed.set()

        await ctx.effect(lambda: close_authorize, label="authorize-resource")

        async def authorize(_arguments):
            assert ctx.fiber._fiber._call_owned_by_current_task() is not None
            authorize_started.set()
            await asyncio.Future()

        await ctx.require(TOOLS).register_authorize(
            ctx,
            tool=ctx.require(LOCAL_TOOL_REF),
            name="policy",
            authorize=authorize,
        )

    authorize_fiber = await root.mount(
        authorize_apply,
        name="authorize",
        inject=(TOOLS, LOCAL_TOOL_REF),
        runtime=_runtime(tmp_path, "authorize"),
    )
    authorize_ctx = authorize_contexts[0]
    metadata = {
        "tool": refs[0].description,
        "prepare": "normalize",
        "authorize": "policy",
    }
    assert not prepare_ctx.fiber._fiber._in_flight_calls
    assert not authorize_ctx.fiber._fiber._in_flight_calls

    async def run_prepare(arguments):
        async with catalog.open(metadata) as opened:
            return await opened.prepare(arguments)

    with pytest.raises(ValueError, match="prepare callback failed"):
        await run_prepare({"value": "error"})
    assert not prepare_ctx.fiber._fiber._in_flight_calls

    prepare_task = asyncio.create_task(run_prepare({"value": " input "}))
    await prepare_started.wait()
    prepare_dispose = asyncio.create_task(prepare_fiber.dispose())
    prepare_marker = asyncio.Event()
    asyncio.get_running_loop().call_soon(prepare_marker.set)
    await prepare_marker.wait()
    assert prepare_fiber.state is FiberState.UNLOADING
    assert prepare_close_count == 0
    release_prepare.set()
    assert await prepare_task == {"value": "input"}
    assert not prepare_ctx.fiber._fiber._in_flight_calls
    await prepare_dispose

    assert target_opener_cleanup.is_set()
    assert prepare_closed.is_set()
    assert prepare_close_count == 1

    authorize_task = asyncio.create_task(
        catalog.authorize(metadata, {"value": "input"})
    )
    await authorize_started.wait()
    authorize_dispose = asyncio.create_task(authorize_fiber.dispose())
    authorize_marker = asyncio.Event()
    asyncio.get_running_loop().call_soon(authorize_marker.set)
    await authorize_marker.wait()
    assert authorize_fiber.state is FiberState.UNLOADING
    assert authorize_close_count == 0
    authorize_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await authorize_task
    await authorize_dispose
    assert authorize_closed.is_set()
    assert authorize_close_count == 1
    assert not authorize_ctx.fiber._fiber._in_flight_calls
    assert target_fiber.state is FiberState.ACTIVE
    await plugin_tasks.close()
    await root.dispose()
