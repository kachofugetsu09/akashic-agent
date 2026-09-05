from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.model import ServiceKey
from agent.plugin_composition.tasks import Tasks
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.tools.execution import ToolExecution
from plugins.tools.plugin import open_tool
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
    await ctx.require(inject[0]).register(
        ctx, name="example", description="Example target A",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},
        open=open_target,
    )
""")
    prepare = path / "prepare"
    prepare.mkdir()
    (prepare / "plugin.py").write_text("""
from agent.plugin_composition import ServiceKey
api_version = 3
name = "prepare"
version = "1.0.0"
inject = (ServiceKey("tools.v1"),)
async def apply(ctx, config):
    async def prepare(arguments):
        return {"value": "restore:" + arguments["value"]}
    await ctx.require(inject[0]).register_prepare(ctx, tool="example", name="restore", prepare=prepare)
""")


def manager(tmp_path, sources):
    return PluginManager(
        plugin_dirs=sources,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


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
            catalog = snapshot.composition_root.context.require(TOOLS)
            binding_id = catalog.bind("example", bindings)
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
