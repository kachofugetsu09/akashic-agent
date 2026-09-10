from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
import shutil
from typing import cast

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.model import PluginRuntime, ServiceKey
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.tools.execution import ToolExecution
from agent.plugin_contracts.tools import (
    ALL_TOOLS,
    TOOL_DISPLAY_NAME,
    ToolRef,
)
from plugins.tools.plugin import ToolCatalog, open_tool
from session.log import MessageLog

TOOLS = ServiceKey("tools.v1")


def write_plugins(path):
    path.mkdir()
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "plugins" / "tools",
        path / "tools",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    target = path / "target"
    target.mkdir()
    (target / "plugin.py").write_text("""
from contextlib import asynccontextmanager
from pathlib import Path
from agent.plugin_composition import ServiceKey
from plugins.tools.execution import Result
from session.message import ContentPart
api_version = 3
name = "target"
version = "1.0.0"
inject = (ServiceKey("tools.v1"),)
async def apply(ctx, config):
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
async def apply(ctx, config):
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
from agent.plugin_contracts.tool_api import Denied
api_version = 3
name = "authorize"
version = "1.0.0"
inject = (ServiceKey("tools.v1"), ServiceKey("fixture.example-ref"))
async def apply(ctx, config):
    async def authorize(arguments):
        if arguments["value"] == "restore:blocked":
            raise Denied("blocked by fixed policy")
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


@pytest.mark.asyncio
async def test_prepare_and_authorize_follow_exact_registration_identity(tmp_path):
    """同名新工具不能继承旧注册的参数转换或限制。"""
    root = CompositionRoot("exact-tool-contributions")
    refs = {}
    contexts = {}
    catalog = ToolCatalog(root.context)

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
        await root.dispose()


@pytest.mark.asyncio
async def test_display_name_reads_old_binding_without_opening_removed_tool(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
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
async def test_ordinary_tool_binding_restores_code_and_preparer_without_current_plugins(
    tmp_path,
):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    host = manager(tmp_path, [sources])
    log = MessageLog(tmp_path / "sessions.db")
    tasks = Tasks()
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            binding_id = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
        await host.terminate_all()
        shutil.rmtree(sources)
        restored = manager(tmp_path, [])
        bindings = Bindings(log, restored._archive, restored.open_binding)
        authorized = []

        async def authorize(binding, arguments):
            authorized.append((binding, arguments))
            return {"permission": "current"}

        execution = ToolExecution(
            log.owner("tools"),
            tasks,
            partial(open_tool, bindings),
            authorize,
            task_key="tools",
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
        await restored.terminate_all()
    finally:
        await tasks.close()
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_tool_configuration_is_owned_frozen_and_restored_without_recapture(tmp_path):
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
    host = manager(tmp_path, [sources])
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
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
        await host.terminate_all()
        shutil.rmtree(sources)
        host = manager(tmp_path, [])
        bindings = Bindings(log, host._archive, host.open_binding)
        async with open_tool(bindings, identity) as target:
            result = await target.invoke("fixed", await target.prepare({"value": "input"}))
            assert result.parts[0].value == "job-a:restore:input"
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_binding_authorize_checks_final_arguments_and_old_binding_keeps_old_policy(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            old_binding = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
        await host.terminate_all()

        add_authorize(sources)
        host = manager(tmp_path, [sources], log)
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        caller_checks = []

        async def caller_authorize(binding, arguments):
            caller_checks.append((binding, arguments))
            return {"permission": "caller"}

        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            new_binding = catalog.bind(
                ctx.require(ALL_TOOLS)().select("example"), bindings
            )
            execution = catalog.execution(caller_authorize)
            old_result = await execution.execute("old", old_binding, {"value": "blocked"})
            denied = await execution.execute("new", new_binding, {"value": "blocked"})
            safe = await execution.execute("safe", new_binding, {"value": "ok"})

        assert old_result.outcome == "success"
        assert denied.outcome == "denied"
        assert denied.parts[0].value == "blocked by fixed policy"
        assert safe.outcome == "success"
        assert caller_checks == [
            (old_binding, {"value": "restore:blocked"}),
            (new_binding, {"value": "restore:ok"}),
        ]
        effects = sorted(
            line
            for path in (tmp_path / "workspace").rglob("effects.txt")
            for line in path.read_text().splitlines()
        )
        assert effects == ["program:old", "program:safe"]
    finally:
        await host.terminate_all()
        log.close()
