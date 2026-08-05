from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import py_compile
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.convertors import CONVERTOR_TYPES, StringConvertor

from agent.plugins.manager import PluginManager, _CandidateRejected
from agent.plugins.manifest import write_plugin_manifest
from agent.plugins.watcher import PluginWatcher
from agent.plugins.jobs import PluginJobRuntime
from agent.plugins.registry import plugin_registry
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
)
from agent.looping.core import AgentLoop
from agent.looping.session_lane import SessionLaneRegistry
from agent.background.subagent_manager import SubagentManager
from proactive_v2.loop import ProactiveLoop
from bootstrap.dashboard_api import create_dashboard_app
from agent.plugins.dashboard_host import (
    DashboardBinding,
    _plugin_routes,
    _require_routes_available,
)
from agent.skills import SkillsLoader
from agent.tools.registry import ToolRegistry
from agent.tools.base import Tool
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


def _write_plugin(root: Path, name: str, source: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
    return plugin_dir


def _manager(
    tmp_path: Path,
    *,
    tools: ToolRegistry | None = None,
    workspace: Path | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=tools,
        workspace=workspace or tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _write_mcp_server(plugin_dir: Path, tools: tuple[str, ...]) -> None:
    tool_items = ", ".join(
        repr(
            {
                "name": name,
                "description": name,
                "inputSchema": {"type": "object", "properties": {}},
            }
        )
        for name in tools
    )
    _ = (plugin_dir / "server.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['AKA_PLUGIN_DATA_DIR']) / 'mcp-lifecycle.log'\n"
        "log.parent.mkdir(parents=True, exist_ok=True)\n"
        "with log.open('a', encoding='utf-8') as f: f.write('started\\n')\n"
        "try:\n"
        "    for line in sys.stdin:\n"
        "        msg = json.loads(line)\n"
        "        if 'id' not in msg:\n"
        "            continue\n"
        "        method = msg.get('method')\n"
        "        result = {}\n"
        "        if method == 'initialize': result = {'protocolVersion': '2025-11-25'}\n"
        f"        elif method == 'tools/list': result = {{'tools': [{tool_items}]}}\n"
        "        elif method == 'tools/call': result = {'content': [{'type': 'text', 'text': '[]'}]}\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n"
        "finally:\n"
        "    with log.open('a', encoding='utf-8') as f: f.write('stopped\\n')\n",
        encoding="utf-8",
    )


def _workspace_mcp_spec(
    server_dir: Path,
    *,
    tool_name: str,
) -> dict[str, dict[str, object]]:
    server_dir.mkdir(parents=True, exist_ok=True)
    _write_mcp_server(server_dir, (tool_name,))
    return {
        "workspace": {
            "command": [sys.executable, str(server_dir / "server.py")],
            "env": {"AKA_PLUGIN_DATA_DIR": str(server_dir / "data")},
            "cwd": str(server_dir),
        }
    }


async def _wait_for_log(path: Path, expected: list[str]) -> None:
    for _ in range(100):
        if path.exists() and path.read_text(encoding="utf-8").splitlines() == expected:
            return
        await asyncio.sleep(0.01)
    assert path.read_text(encoding="utf-8").splitlines() == expected


@pytest.mark.asyncio
async def test_candidate_gate_publishes_unique_generation(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "candidate",
        "from agent.plugins import Plugin, PluginSemanticCheck\n"
        "class CandidatePlugin(Plugin):\n"
        "    name = 'candidate'\n"
        "    def static_semantic_checks(self):\n"
        "        return [PluginSemanticCheck('fixture', True, 'ok')]\n"
        "    async def prepare(self):\n"
        "        self.context.kv_store.set('generation', self.context.generation_id)\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    generation = manager.generation("candidate")
    gate = manager.latest_gate("candidate")
    assert generation is not None
    assert gate is not None and gate.status == "passed"
    assert generation.module_path.startswith("akasic_plugin_plugins_candidate__g")
    assert generation.generation_id == generation.instance.context.generation_id
    assert plugin_registry.get_instance("akasic_plugin_plugins_candidate") is generation.instance
    assert generation.contributions.manifest["name"] == "candidate"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "source", "failed_check"),
    [
        (
            "bad_api",
            "from agent.plugins import Plugin\n"
            "class BadApiPlugin(Plugin):\n"
            "    name = 'bad_api'\n"
            "    api_version = 1\n",
            "api_version",
        ),
        (
            "bad_semantic",
            "from agent.plugins import Plugin, PluginSemanticCheck\n"
            "class BadSemanticPlugin(Plugin):\n"
            "    name = 'bad_semantic'\n"
            "    def static_semantic_checks(self):\n"
            "        return [PluginSemanticCheck('model', False, 'missing')]\n",
            "semantic_checks",
        ),
        (
            "legacy_lifecycle",
            "from agent.plugins import Plugin\n"
            "class LegacyLifecyclePlugin(Plugin):\n"
            "    name = 'legacy_lifecycle'\n"
            "    async def initialize(self):\n"
            "        raise RuntimeError('legacy lifecycle ran')\n",
            "lifecycle_api",
        ),
        (
            "bad_source",
            "from agent.plugins import Plugin, ProactiveSourceSpec\n"
            "class BadSourcePlugin(Plugin):\n"
            "    name = 'bad_source'\n"
            "    def proactive_sources(self):\n"
            "        return [ProactiveSourceSpec(id='feed', channels=('content',), "
            "server='missing', fetch_tool='fetch')]\n",
            "proactive_sources",
        ),
        (
            "phase_cycle",
            "from agent.plugins import Plugin\n"
            "class A:\n"
            "    slot = 'plugin.a'\n"
            "    requires = ('plugin.b',)\n"
            "class B:\n"
            "    slot = 'plugin.b'\n"
            "    requires = ('plugin.a',)\n"
            "class PhaseCyclePlugin(Plugin):\n"
            "    name = 'phase_cycle'\n"
            "    def before_turn_modules(self): return [A(), B()]\n",
            "phase_graph",
        ),
        (
            "duplicate_tool",
            "from agent.plugins import Plugin, tool\n"
            "class DuplicateToolPlugin(Plugin):\n"
            "    name = 'duplicate_tool'\n"
            "    @tool(name='same')\n"
            "    async def first(self, event):\n"
            "        \"\"\"First.\"\"\"\n"
            "        return 'first'\n"
            "    @tool(name='same')\n"
            "    async def second(self, event):\n"
            "        \"\"\"Second.\"\"\"\n"
            "        return 'second'\n",
            "tool_names",
        ),
    ],
)
async def test_failed_candidate_never_prepares(
    tmp_path: Path,
    name: str,
    source: str,
    failed_check: str,
):
    plugin_dir = _write_plugin(tmp_path / "plugins", name, source)
    marker = plugin_dir / "initialized"
    with (plugin_dir / "plugin.py").open("a", encoding="utf-8") as file:
        _ = file.write(
            "    async def prepare(self):\n"
            f"        open({str(marker)!r}, 'w').write('bad')\n"
        )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)

    await manager.load_all()

    gate = manager.latest_gate(name)
    assert manager.loaded_count == 0
    assert manager.generation(name) is None
    assert gate is not None and gate.status == "failed"
    assert failed_check in {check.check_id for check in gate.checks if check.status == "failed"}
    assert not marker.exists()
    assert tools.get_registered_names() == set()
    assert not any(module.startswith(f"akasic_plugin_plugins_{name}__g") for module in sys.modules)


@pytest.mark.asyncio
async def test_import_failure_returns_gate_result(tmp_path: Path):
    _write_plugin(tmp_path / "plugins", "broken", "this is not python !!!\n")
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("broken")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "import"
    assert not any(module.startswith("akasic_plugin_plugins_broken__g") for module in sys.modules)


@pytest.mark.asyncio
async def test_candidate_declarations_are_collected_once(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "once",
        "from agent.plugins import Plugin\n"
        "calls = 0\n"
        "class Module:\n"
        "    slot = 'plugin.once'\n"
        "    requires = ()\n"
        "class OncePlugin(Plugin):\n"
        "    name = 'once'\n"
        "    def before_turn_modules(self):\n"
        "        global calls\n"
        "        calls += 1\n"
        "        return [Module()]\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    generation = manager.generation("once")
    assert generation is not None
    module = sys.modules[generation.module_path]
    assert module.calls == 1
    assert len(manager.before_turn_modules) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy_method", "legacy_declaration"),
    [
        (
            "mobile_ui_module",
            "    @classmethod\n"
            "    def mobile_ui_module(cls): return 'legacy.js'\n",
        ),
        (
            "mobile_ui_stylesheet",
            "    @classmethod\n"
            "    def mobile_ui_stylesheet(cls): return 'legacy.css'\n",
        ),
        (
            "mobile_ui_call",
            "    async def mobile_ui_call(self, method, payload): return {}\n",
        ),
    ],
)
async def test_legacy_mobile_ui_candidate_keeps_active_snapshot(
    tmp_path: Path,
    legacy_method: str,
    legacy_declaration: str,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "mobile_contract",
        "from agent.plugins import MobileUiContribution, Plugin\n"
        "class MobileContractPlugin(Plugin):\n"
        "    name = 'mobile_contract'\n"
        "    @classmethod\n"
        "    def mobile_ui(cls):\n"
        "        return MobileUiContribution(module='mobile.js', slots=('drawer.panel',))\n",
    )
    _ = (plugin_dir / "mobile.js").write_text("export const version = 'v2';\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("mobile_contract")
    active_snapshot = manager.current_snapshot
    assert active is not None and active_snapshot is not None

    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class MobileContractPlugin(Plugin):\n"
        "    name = 'mobile_contract'\n"
        + legacy_declaration,
        encoding="utf-8",
    )
    prepared = await manager.prepare_candidate("mobile_contract")

    gate = manager.latest_gate("mobile_contract")
    assert prepared is None
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "declarations"
    assert legacy_method in gate.failure_reason
    assert manager.generation("mobile_contract") is active
    assert manager.current_snapshot is active_snapshot
    assert active_snapshot.generations["mobile_contract"] is active
    assert active.contributions.mobile_ui_asset is not None
    assert active.contributions.mobile_ui_asset.module == "export const version = 'v2';\n"


@pytest.mark.asyncio
async def test_same_source_gets_new_generation_namespace_after_restart(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "repeat",
        "from agent.plugins import Plugin\n"
        "class RepeatPlugin(Plugin):\n"
        "    name = 'repeat'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    first = manager.generation("repeat")
    assert first is not None

    await manager.terminate_all()
    await manager.load_all()

    second = manager.generation("repeat")
    assert second is not None
    assert first.state == "retired"
    assert second.generation_id != first.generation_id
    assert second.module_path != first.module_path


@pytest.mark.asyncio
async def test_candidate_declaration_cannot_write_plugin_kv(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "readonly",
        "from agent.plugins import Plugin\n"
        "class ReadonlyPlugin(Plugin):\n"
        "    name = 'readonly'\n"
        "    def before_turn_modules(self):\n"
        "        self.context.kv_store.set('leaked', True)\n"
        "        return []\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("readonly")
    kv_path = tmp_path / "workspace" / "plugin-data" / "readonly-builtin" / ".kv.json"
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "declarations"
    assert not kv_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "declaration",
    (
        "raise RuntimeError('declaration failed')",
        "return 'not-a-list'",
        "return None",
    ),
)
async def test_invalid_declaration_fails_gate(tmp_path: Path, declaration: str):
    _write_plugin(
        tmp_path / "plugins",
        "invalid_declaration",
        "from agent.plugins import Plugin\n"
        "class InvalidDeclarationPlugin(Plugin):\n"
        "    name = 'invalid_declaration'\n"
        "    def before_turn_modules(self):\n"
        f"        {declaration}\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("invalid_declaration")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "declarations"
    assert manager.generation("invalid_declaration") is None


@pytest.mark.asyncio
async def test_candidate_phase_graph_includes_active_plugins(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "first",
        "from agent.plugins import Plugin\n"
        "class Module:\n"
        "    slot = 'plugin.shared'\n"
        "class FirstPlugin(Plugin):\n"
        "    name = 'first'\n"
        "    def before_turn_modules(self): return [Module()]\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    _write_plugin(
        tmp_path / "plugins",
        "second",
        "from agent.plugins import Plugin\n"
        "class Module:\n"
        "    slot = 'plugin.shared'\n"
        "class SecondPlugin(Plugin):\n"
        "    name = 'second'\n"
        "    def before_turn_modules(self): return [Module()]\n",
    )

    await manager.load_all()

    gate = manager.latest_gate("second")
    assert manager.generation("first") is not None
    assert manager.generation("second") is None
    assert gate is not None and gate.status == "failed"
    assert "phase_graph" in {
        check.check_id for check in gate.checks if check.status == "failed"
    }


@pytest.mark.asyncio
async def test_prepare_failure_replaces_passed_gate(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "init_failure",
        "from agent.plugins import Plugin\n"
        "class InitFailurePlugin(Plugin):\n"
        "    name = 'init_failure'\n"
        "    async def prepare(self): raise RuntimeError('init failed')\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("init_failure")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "prepare"
    assert manager.generation("init_failure") is None


@pytest.mark.asyncio
async def test_generation_module_tree_is_removed_on_config_failure_and_terminate(
    tmp_path: Path,
):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "module_tree",
        "from pydantic import BaseModel\n"
        "from agent.plugins import Plugin\n"
        "from . import child\n"
        "class Config(BaseModel):\n"
        "    required: str\n"
        "class ModuleTreePlugin(Plugin):\n"
        "    name = 'module_tree'\n"
        "    ConfigModel = Config\n",
    )
    _ = (plugin_dir / "child.py").write_text("value = 1\n", encoding="utf-8")
    config_dir = tmp_path / "workspace" / "plugin-data" / "module_tree-builtin"
    config_dir.mkdir(parents=True)
    _ = (config_dir / "config.local.toml").write_text("", encoding="utf-8")
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.latest_gate("module_tree").status == "failed"  # type: ignore[union-attr]
    assert not any("plugins_module_tree__g" in name for name in sys.modules)

    _ = (config_dir / "config.local.toml").write_text(
        "required = 'ok'\n",
        encoding="utf-8",
    )
    await manager.load_all()
    generation = manager.generation("module_tree")
    assert generation is not None
    assert f"{generation.module_path}.child" in sys.modules
    stable_child = importlib.import_module("akasic_plugin_plugins_module_tree.child")
    assert stable_child.value == 1

    await manager.terminate_all()

    assert not any("plugins_module_tree__g" in name for name in sys.modules)
    assert "akasic_plugin_plugins_module_tree.child" not in sys.modules


@pytest.mark.asyncio
async def test_candidate_is_not_published_before_prepare_finishes(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "pending",
        "import asyncio\n"
        "from agent.plugins import Plugin, tool\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class PendingPlugin(Plugin):\n"
        "    name = 'pending'\n"
        "    @tool(name='pending_tool')\n"
        "    async def pending_tool(self, event):\n"
        "        \"\"\"Pending tool.\"\"\"\n"
        "        return 'ready'\n"
        "    async def prepare(self):\n"
        "        self.context.event_bus.on(TurnCommitted, self.on_turn)\n"
        "        while not (self.context.plugin_dir / 'release').exists():\n"
        "            await asyncio.sleep(0.01)\n"
        "    def on_turn(self, event): return None\n",
    )
    tools = ToolRegistry()
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        tool_registry=tools,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )

    loading = asyncio.create_task(manager.load_all())
    while manager.latest_gate("pending") is None:
        await asyncio.sleep(0.01)

    assert manager.latest_gate("pending").status == "passed"  # type: ignore[union-attr]
    assert manager.generation("pending") is None
    assert tools.get_tool("pending_tool") is None
    assert event_bus.handler_count() == 0

    _ = (plugin_dir / "release").write_text("ok", encoding="utf-8")
    await loading

    assert manager.generation("pending") is not None
    assert tools.get_tool("pending_tool") is not None
    assert event_bus.handler_count() == 0


@pytest.mark.asyncio
async def test_prepare_same_plugin_keeps_active_generation_until_snapshot_publish(
    tmp_path: Path,
):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "replaceable",
        "from agent.plugins import Plugin, tool\n"
        "class ReplaceablePlugin(Plugin):\n"
        "    name = 'replaceable'\n"
        "    @tool(name='replaceable_tool')\n"
        "    async def run(self, event):\n"
        "        \"\"\"Replaceable tool.\"\"\"\n"
        "        return 'v1'\n",
    )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()

    active = manager.generation("replaceable")
    assert active is not None

    _ = (plugin_dir / "plugin.py").write_text("not valid python !!!\n", encoding="utf-8")
    rejected = await manager.prepare_candidate("replaceable")

    assert rejected is None
    assert manager.generation("replaceable") is active
    assert manager.prepared_generation("replaceable") is None
    assert manager.latest_gate("replaceable").status == "failed"  # type: ignore[union-attr]
    assert tools.get_tool("replaceable_tool") is not None

    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin, tool\n"
        "class ReplaceablePlugin(Plugin):\n"
        "    name = 'replaceable'\n"
        "    @tool(name='replaceable_tool')\n"
        "    async def run(self, event):\n"
        "        \"\"\"Replaceable tool.\"\"\"\n"
        "        return 'v2'\n"
        "    async def prepare(self):\n"
        "        self.context.kv_store.set('initialized_v2', True)\n",
        encoding="utf-8",
    )
    prepared = await manager.prepare_candidate("replaceable")

    assert prepared is not None and prepared.state == "prepared"
    assert manager.generation("replaceable") is active
    assert manager.prepared_generation("replaceable") is prepared
    assert manager.latest_gate("replaceable").status == "passed"  # type: ignore[union-attr]
    assert not (
        tmp_path / "workspace" / "plugin-data" / "replaceable-builtin" / ".kv.json"
    ).exists()
    assert tools.get_tool("replaceable_tool") is not None

    await manager.discard_prepared("replaceable")

    assert prepared.state == "discarded"
    assert manager.prepared_generation("replaceable") is None


@pytest.mark.asyncio
async def test_source_revision_includes_helper_changes(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "revision",
        "from agent.plugins import Plugin\n"
        "from . import helper\n"
        "class RevisionPlugin(Plugin):\n"
        "    name = 'revision'\n",
    )
    helper = plugin_dir / "helper.py"
    _ = helper.write_text("value = 1\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("revision")
    assert active is not None

    _ = helper.write_text("value = 2\n", encoding="utf-8")
    prepared = await manager.prepare_candidate("revision")

    assert prepared is not None
    assert prepared.source_revision != active.source_revision


@pytest.mark.asyncio
async def test_declared_paths_cannot_escape_plugin_root(tmp_path: Path):
    outside = tmp_path / "plugins" / "outside" / "skill"
    outside.mkdir(parents=True)
    _ = (outside / "SKILL.md").write_text("# outside\n", encoding="utf-8")
    _write_plugin(
        tmp_path / "plugins",
        "escaped",
        "from agent.plugins import Plugin\n"
        "class EscapedPlugin(Plugin):\n"
        "    name = 'escaped'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('../outside',)\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("escaped")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "declarations"


@pytest.mark.asyncio
async def test_proactive_lifecycle_structure_fails_static_gate(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "bad_lifecycle",
        "from agent.plugins import Plugin\n"
        "from proactive_v2.lifecycle import ProactiveLifecycleSpec\n"
        "class BadLifecyclePlugin(Plugin):\n"
        "    name = 'bad_lifecycle'\n"
        "    def proactive_lifecycles(self):\n"
        "        return [ProactiveLifecycleSpec(id='bad', modules=(object(),))]\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("bad_lifecycle")
    assert gate is not None and gate.status == "failed"
    assert "proactive_lifecycle_structure" in {
        check.check_id for check in gate.checks if check.status == "failed"
    }


@pytest.mark.asyncio
async def test_source_symlink_cannot_escape_plugin_root(tmp_path: Path):
    outside = tmp_path / "outside.py"
    _ = outside.write_text("value = 1\n", encoding="utf-8")
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "linked_source",
        "from agent.plugins import Plugin\n"
        "from . import helper\n"
        "class LinkedSourcePlugin(Plugin):\n"
        "    name = 'linked_source'\n",
    )
    (plugin_dir / "helper.py").symlink_to(outside)
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("linked_source")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "source_boundary"


@pytest.mark.asyncio
async def test_candidate_ignores_stale_bytecode_for_root_and_helper(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "fresh_source",
        "from agent.plugins import Plugin\n"
        "from . import helper\n"
        "class FreshSourcePlugin(Plugin):\n"
        "    name = 'fresh_source'\n"
        "    version = 'v1'\n"
        "    helper_value = helper.VALUE\n",
    )
    plugin_file = plugin_dir / "plugin.py"
    helper_file = plugin_dir / "helper.py"
    _ = helper_file.write_text("VALUE = 'v1'\n", encoding="utf-8")
    plugin_stat = plugin_file.stat()
    helper_stat = helper_file.stat()
    _ = py_compile.compile(str(plugin_file), doraise=True)
    _ = py_compile.compile(str(helper_file), doraise=True)
    manager = _manager(tmp_path)
    await manager.load_all()

    _ = plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace("'v1'", "'v2'"),
        encoding="utf-8",
    )
    _ = helper_file.write_text("VALUE = 'v2'\n", encoding="utf-8")
    os.utime(plugin_file, ns=(plugin_stat.st_atime_ns, plugin_stat.st_mtime_ns))
    os.utime(helper_file, ns=(helper_stat.st_atime_ns, helper_stat.st_mtime_ns))

    prepared = await manager.prepare_candidate("fresh_source")

    assert prepared is not None
    assert prepared.instance.version == "v2"  # type: ignore[attr-defined]
    assert prepared.instance.helper_value == "v2"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_return_to_active_revision_discards_stale_prepared(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "return_active",
        "from agent.plugins import Plugin\n"
        "class ReturnActivePlugin(Plugin):\n"
        "    name = 'return_active'\n"
        "    version = 'v1'\n",
    )
    plugin_file = plugin_dir / "plugin.py"
    active_source = plugin_file.read_text(encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    caplog.set_level(logging.INFO, logger="agent.plugins.manager")
    _ = plugin_file.write_text(
        active_source.replace("'v1'", "'v2'"),
        encoding="utf-8",
    )
    first_scan = await manager.prepare_changed()
    prepared = manager.prepared_generation("return_active")
    assert first_scan[0]["gate_status"] == "passed"
    assert prepared is not None

    _ = plugin_file.write_text(active_source, encoding="utf-8")
    second_scan = await manager.prepare_changed()

    assert second_scan[0]["gate_status"] == "active"
    assert manager.prepared_generation("return_active") is None
    assert prepared.state == "discarded"
    status_logs = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("plugin_candidate_status ")
    ]
    assert status_logs
    assert "counts=skills:" in status_logs[-1]
    assert "skill_descriptions" not in status_logs[-1]


@pytest.mark.asyncio
async def test_terminating_one_manager_keeps_newer_stable_alias(tmp_path: Path):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "shared_alias",
        "from agent.plugins import Plugin\n"
        "class SharedAliasPlugin(Plugin):\n"
        "    name = 'shared_alias'\n",
    )
    child = plugin_dir / "child.py"
    _ = child.write_text("value = 'v1'\n", encoding="utf-8")
    first_manager = _manager(tmp_path)
    second_manager = _manager(tmp_path)
    await first_manager.load_all()
    stable_alias = "akasic_plugin_plugins_shared_alias"
    first_child = importlib.import_module(f"{stable_alias}.child")
    assert first_child.value == "v1"
    _ = child.write_text("value = 'v2'\n", encoding="utf-8")
    await second_manager.load_all()
    second = second_manager.generation("shared_alias")
    assert second is not None
    second_child = importlib.import_module(f"{stable_alias}.child")
    assert second_child.value == "v2"
    assert second_child is not first_child

    await first_manager.terminate_all()

    assert plugin_registry.get_instance(stable_alias) is second.instance
    assert sys.modules[stable_alias] is sys.modules[second.module_path]

    await second_manager.terminate_all()


@pytest.mark.asyncio
async def test_skill_catalog_rejects_cross_plugin_duplicates(tmp_path: Path):
    first_dir = _write_plugin(
        tmp_path / "plugins",
        "first_skills",
        "from agent.plugins import Plugin\n"
        "class FirstSkillsPlugin(Plugin):\n"
        "    name = 'first_skills'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills',)\n",
    )
    first_skill = first_dir / "skills" / "shared"
    first_skill.mkdir(parents=True)
    _ = (first_skill / "SKILL.md").write_text(
        "---\ndescription: first\n---\nfirst\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path, workspace=tmp_path / "workspace")
    await manager.load_all()
    first = manager.generation("first_skills")
    assert first is not None and first.skill_catalog is not None
    assert first.skill_catalog.normal.get("shared").source_id == "first_skills"  # type: ignore[union-attr]

    second_dir = _write_plugin(
        tmp_path / "plugins",
        "second_skills",
        "from agent.plugins import Plugin\n"
        "class SecondSkillsPlugin(Plugin):\n"
        "    name = 'second_skills'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills',)\n",
    )
    second_skill = second_dir / "skills" / "shared"
    second_skill.mkdir(parents=True)
    _ = (second_skill / "SKILL.md").write_text(
        "---\ndescription: second\n---\nsecond\n",
        encoding="utf-8",
    )

    await manager.load_all()

    gate = manager.latest_gate("second_skills")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[-1].check_id == "skill_catalog"
    assert manager.generation("second_skills") is None
    generation_id = first.generation_id
    await manager.terminate_all()
    assert manager.skill_catalog(generation_id) is None


@pytest.mark.asyncio
async def test_skill_catalog_freezes_generation_and_ignores_old_root_link(
    tmp_path: Path,
):
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "skill_reload",
        "from agent.plugins import Plugin\n"
        "class SkillReloadPlugin(Plugin):\n"
        "    name = 'skill_reload'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills-v1',)\n",
    )
    v1_skill = plugin_dir / "skills-v1" / "shared"
    v1_skill.mkdir(parents=True)
    _ = (v1_skill / "SKILL.md").write_text(
        "---\ndescription: version one\n---\nbody v1\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace_skill = workspace / "skills" / "personal"
    workspace_skill.mkdir(parents=True)
    _ = (workspace_skill / "SKILL.md").write_text(
        "---\ndescription: workspace one\n---\nworkspace body v1\n",
        encoding="utf-8",
    )
    workspace_drift = workspace / "drift" / "skills" / "personal-drift"
    workspace_drift.mkdir(parents=True)
    _ = (workspace_drift / "SKILL.md").write_text(
        "---\ndescription: drift workspace one\n---\ndrift workspace v1\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    active = manager.generation("skill_reload")
    assert active is not None and active.skill_catalog is not None
    active_record = active.skill_catalog.normal.get("shared")
    assert active_record is not None
    active_workspace = active.skill_catalog.normal.get("personal")
    active_workspace_drift = active.skill_catalog.drift.get("personal-drift")
    assert active_workspace is not None
    assert active_workspace_drift is not None

    workspace_skills = workspace / "skills"
    workspace_skills.mkdir(parents=True, exist_ok=True)
    (workspace_skills / "shared").symlink_to(v1_skill, target_is_directory=True)
    _ = (v1_skill / "SKILL.md").write_text(
        "---\ndescription: changed old source\n---\nchanged old body\n",
        encoding="utf-8",
    )
    _ = (workspace_skill / "SKILL.md").write_text(
        "---\ndescription: workspace two\n---\nworkspace body v2\n",
        encoding="utf-8",
    )
    _ = (workspace_drift / "SKILL.md").write_text(
        "---\ndescription: drift workspace two\n---\ndrift workspace v2\n",
        encoding="utf-8",
    )
    v2_skill = plugin_dir / "skills-v2" / "shared"
    v2_skill.mkdir(parents=True)
    _ = (v2_skill / "SKILL.md").write_text(
        "---\ndescription: version two\n---\nbody v2\n",
        encoding="utf-8",
    )
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class SkillReloadPlugin(Plugin):\n"
        "    name = 'skill_reload'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills-v2',)\n",
        encoding="utf-8",
    )

    prepared = await manager.prepare_candidate("skill_reload")

    assert prepared is not None and prepared.skill_catalog is not None
    prepared_record = prepared.skill_catalog.normal.get("shared")
    assert prepared_record is not None
    assert active_record.description == "version one"
    assert "body v1" in active_record.skill_file.read_text(encoding="utf-8")
    assert prepared_record.description == "version two"
    assert prepared_record.source == "plugin"
    assert "body v2" in prepared_record.skill_file.read_text(encoding="utf-8")
    assert active_record.root_dir != prepared_record.root_dir
    prepared_workspace = prepared.skill_catalog.normal.get("personal")
    prepared_workspace_drift = prepared.skill_catalog.drift.get("personal-drift")
    assert prepared_workspace is not None
    assert prepared_workspace_drift is not None
    assert "workspace body v1" in active_workspace.content
    assert "workspace body v1" in active_workspace.skill_file.read_text(encoding="utf-8")
    assert "workspace body v2" in prepared_workspace.content
    assert "drift workspace v1" in active_workspace_drift.content
    assert "drift workspace v2" in prepared_workspace_drift.content
    active_snapshot = active.skill_catalog.snapshot_root
    prepared_snapshot = prepared.skill_catalog.snapshot_root

    await manager.discard_prepared("skill_reload")
    await manager.terminate_all()

    assert not active_snapshot.exists()
    assert not prepared_snapshot.exists()


@pytest.mark.asyncio
async def test_skill_catalog_cleanup_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "skill_cleanup",
        "from agent.plugins import Plugin\n"
        "class SkillCleanupPlugin(Plugin):\n"
        "    name = 'skill_cleanup'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    generation = manager.generation("skill_cleanup")
    assert generation is not None and generation.skill_catalog is not None
    snapshot_root = generation.skill_catalog.snapshot_root
    real_rmtree = shutil.rmtree

    def fail_snapshot_cleanup(path: Path, *args: Any, **kwargs: Any) -> None:
        if Path(path) == snapshot_root and not args and not kwargs:
            raise OSError("snapshot cleanup failed")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_snapshot_cleanup)
    await manager.terminate_all()

    assert any(
        failure.resource == "skill_catalog"
        and failure.error == "snapshot cleanup failed"
        for failure in manager.cleanup_failures
    )


@pytest.mark.asyncio
async def test_candidate_mcp_catalog_uses_stable_public_names_and_closes(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "mcp_ready",
        "from agent.plugins import (IntervalTrigger, McpServerSpec, Plugin, PluginJobSpec, "
        "PluginSemanticCheck, ProactiveSourceSpec)\n"
        "class McpReadyPlugin(Plugin):\n"
        "    name = 'mcp_ready'\n"
        "    def __init__(self):\n"
        "        self.job_spec = PluginJobSpec(id='refresh', triggers=[IntervalTrigger(3600)], handler=self.refresh)\n"
        "        self.source_spec = ProactiveSourceSpec(id='feed', channels=['content'], server='feed', fetch_tool='fetch_events', ack_tool='ack_events')\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='feed', command=('python', 'server.py'))]\n"
        "    def proactive_sources(self):\n"
        "        return [self.source_spec]\n"
        "    def jobs(self):\n"
        "        return [self.job_spec]\n"
        "    async def prepare(self):\n"
        "        self.job_spec.triggers.append(object())\n"
        "        self.source_spec.channels.append('invalid')\n"
        "    async def refresh(self, context):\n"
        "        return None\n"
        "    async def readiness_semantic_checks(self, context):\n"
        "        self.job_spec.triggers.append(object())\n"
        "        self.source_spec.channels.append('invalid')\n"
        "        value = await context.mcp_catalog.servers['feed'].client.call('fetch_events', {})\n"
        "        job = context.job_catalog.jobs['mcp_ready:refresh']\n"
        "        source = context.proactive_catalog.sources['mcp_ready:feed']\n"
        "        owned = getattr(job.spec.handler, '__self__', None) is self\n"
        "        frozen = len(job.spec.triggers) == 1\n"
        "        source_frozen = source.spec.channels == ('content',)\n"
        "        evidence = {'mcp': value, 'job_owned': owned, 'source': source.spec.id, 'frozen': frozen, 'source_frozen': source_frozen}\n"
        "        return [PluginSemanticCheck('candidate_feed', value == '[]' and owned and frozen and source_frozen, evidence)]\n",
    )
    _write_mcp_server(
        plugin_dir,
        ("fetch_events", "ack_events"),
    )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()
    active = manager.generation("mcp_ready")
    assert active is not None and active.job_catalog is not None
    active_job = manager.jobs[0]
    active_source = manager.proactive_sources[0]
    assert isinstance(active_job.spec.triggers, tuple)
    assert len(active_job.spec.triggers) == 1
    assert active_source.spec.channels == ("content",)

    prepared = await manager.prepare_candidate("mcp_ready")

    assert (
        prepared is not None
        and prepared.mcp_catalog is not None
        and prepared.job_catalog is not None
        and prepared.proactive_catalog is not None
    )
    catalog = prepared.mcp_catalog
    server = catalog.servers["feed"]
    assert server.client.name == f"feed@{prepared.generation_id}"
    assert catalog.tool_names == (
        "mcp_feed__ack_events",
        "mcp_feed__fetch_events",
    )
    assert not any(prepared.generation_id in name for name in catalog.tool_names)
    assert tools.get_registered_names() == set()
    assert await server.tools[1].execute() == "[]"
    assert tuple(catalog.servers) == ("feed",)
    assert tuple(prepared.job_catalog.jobs) == ("mcp_ready:refresh",)
    assert tuple(prepared.proactive_catalog.sources) == ("mcp_ready:feed",)
    assert prepared.proactive_catalog.sources["mcp_ready:feed"].spec.channels == (
        "content",
    )
    prepared_job = prepared.job_catalog.jobs["mcp_ready:refresh"]
    assert isinstance(prepared_job.spec.triggers, tuple)
    assert len(prepared_job.spec.triggers) == 1
    assert prepared_job.spec.handler.__self__ is prepared.instance  # type: ignore[attr-defined]
    assert manager.jobs == [active_job]
    assert active_job.spec.handler.__self__ is active.instance  # type: ignore[attr-defined]
    lifecycle = tmp_path / "workspace" / "plugin-data" / "mcp_ready-builtin" / "mcp-lifecycle.log"
    await _wait_for_log(lifecycle, ["started", "started"])

    await manager.discard_prepared("mcp_ready")

    await _wait_for_log(lifecycle, ["started", "started", "stopped"])
    assert manager.mcp_catalog(prepared.generation_id) is None
    assert manager.job_catalog(prepared.generation_id) is None
    assert manager.proactive_catalog(prepared.generation_id) is None
    assert server.client._process is None
    assert server.client._stderr_task is None
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_invalid_job_structure_fails_activity_catalog_gate(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "invalid_job",
        "from agent.plugins import IntervalTrigger, Plugin, PluginJobSpec\n"
        "class InvalidJobPlugin(Plugin):\n"
        "    name = 'invalid_job'\n"
        "    def jobs(self):\n"
        "        return [PluginJobSpec(id='bad', triggers=[IntervalTrigger(0)], handler=self.run)]\n"
        "    async def run(self, context):\n"
        "        return None\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.generation("invalid_job") is None
    gate = manager.latest_gate("invalid_job")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[-1].check_id == "activity_catalogs"
    assert manager.jobs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_tool", "semantic"])
async def test_candidate_mcp_readiness_failure_closes_process(
    tmp_path: Path,
    failure: str,
) -> None:
    readiness = (
        "    async def readiness_semantic_checks(self, context):\n"
        "        return [PluginSemanticCheck('remote', False, 'not ready')]\n"
        if failure == "semantic"
        else ""
    )
    fetch_tool = "missing" if failure == "missing_tool" else "fetch_events"
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "mcp_rejected",
        "from agent.plugins import (McpServerSpec, Plugin, PluginSemanticCheck, "
        "ProactiveSourceSpec)\n"
        "class McpRejectedPlugin(Plugin):\n"
        "    name = 'mcp_rejected'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='feed', command=('python', 'server.py'))]\n"
        "    def proactive_sources(self):\n"
        f"        return [ProactiveSourceSpec(id='feed', channels=('content',), server='feed', fetch_tool='{fetch_tool}')]\n"
        + readiness,
    )
    _write_mcp_server(plugin_dir, ("fetch_events",))
    invalid_source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class McpRejectedPlugin(Plugin):\n"
        "    name = 'mcp_rejected'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='feed', command=('python', 'server.py'))]\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    _ = (plugin_dir / "plugin.py").write_text(invalid_source, encoding="utf-8")

    prepared = await manager.prepare_candidate("mcp_rejected")

    assert prepared is None
    assert manager.prepared_generation("mcp_rejected") is None
    gate = manager.latest_gate("mcp_rejected")
    assert gate is not None and gate.status == "failed"
    failed_checks = {check.check_id for check in gate.checks if check.status == "failed"}
    assert (
        "mcp_readiness" if failure == "missing_tool" else "readiness_semantic_checks"
    ) in failed_checks
    lifecycle = tmp_path / "workspace" / "plugin-data" / "mcp_rejected-builtin" / "mcp-lifecycle.log"
    await _wait_for_log(lifecycle, ["started", "started", "stopped"])
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_mcp_handshake_failure_closes_process(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "mcp_handshake",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class McpHandshakePlugin(Plugin):\n"
        "    name = 'mcp_handshake'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='broken', command=('python', 'server.py'))]\n",
    )
    _ = (plugin_dir / "server.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "log = Path(os.environ['AKA_PLUGIN_DATA_DIR']) / 'mcp-lifecycle.log'\n"
        "log.parent.mkdir(parents=True, exist_ok=True)\n"
        "log.write_text('started\\nstopped\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()

    prepared = await manager.prepare_candidate("mcp_handshake")

    assert prepared is None
    gate = manager.latest_gate("mcp_handshake")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[-1].check_id == "mcp_readiness"
    lifecycle = tmp_path / "workspace" / "plugin-data" / "mcp_handshake-builtin" / "mcp-lifecycle.log"
    await _wait_for_log(lifecycle, ["started", "stopped"])
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_mcp_cleanup_failure_is_reported_and_catalog_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "mcp_cleanup",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class McpCleanupPlugin(Plugin):\n"
        "    name = 'mcp_cleanup'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='cleanup', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(plugin_dir, ("health",))
    manager = _manager(tmp_path)
    await manager.load_all()
    prepared = await manager.prepare_candidate("mcp_cleanup")
    assert prepared is not None and prepared.mcp_catalog is not None
    client = prepared.mcp_catalog.servers["cleanup"].client
    async def fail_before_disconnect() -> None:
        raise OSError("mcp cleanup failed")

    monkeypatch.setattr(client, "disconnect", fail_before_disconnect)
    await manager.discard_prepared("mcp_cleanup")

    assert manager.mcp_catalog(prepared.generation_id) is None
    assert client.connected is False
    assert any(
        failure.resource == "mcp_catalog"
        and "mcp cleanup failed" in failure.error
        for failure in manager.cleanup_failures
    )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_runtime_snapshot_lease_commit_and_abort(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot",
        "from agent.plugins import Plugin\n"
        "class A:\n"
        "    slot = 'snapshot.a'\n"
        "    requires = ()\n"
        "class B:\n"
        "    slot = 'snapshot.b'\n"
        "    requires = ('snapshot.a',)\n"
        "class SnapshotPlugin(Plugin):\n"
        "    name = 'snapshot'\n"
        "    def before_turn_modules(self): return [B(), A()]\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot")
    installed = manager.current_snapshot
    prepared = await manager.prepare_candidate("snapshot")
    assert active is not None and prepared is not None and installed is not None
    assert installed.generations["snapshot"] is active
    assert installed.state == "committed"
    assert manager.current_snapshot is installed
    assert [module.slot for module in installed.before_turn_modules] == [
        "snapshot.a",
        "snapshot.b",
    ]
    compiler = RuntimeSnapshotCompiler()
    v1 = compiler.compile({"snapshot": active}, catalog_generation=active)
    v2 = compiler.compile({"snapshot": prepared}, catalog_generation=prepared)
    drained: list[str] = []

    async def on_drained(snapshot) -> None:
        drained.append(snapshot.snapshot_id)

    store = RuntimeSnapshotStore(on_drained)
    store.install(v1)
    v1_lease = store.lease()
    assert v1_lease.snapshot is v1
    transaction = store.begin_publish(v2)
    assert store.current is v1
    with pytest.raises(RuntimeError, match="不可租用"):
        _ = store.lease(v2.snapshot_id)

    await store.abort(transaction)

    assert store.current is v1
    assert drained == [v2.snapshot_id]
    await v1_lease.release()
    assert active.lease_count == 0
    assert prepared.lease_count == 0

    next_v2 = compiler.compile({"snapshot": prepared}, catalog_generation=prepared)
    held_v1 = store.lease()
    committed = store.begin_publish(next_v2)
    await store.commit(committed)

    assert store.current is next_v2
    assert drained == [v2.snapshot_id]
    with pytest.raises(RuntimeError, match="不可租用"):
        _ = store.lease(v1.snapshot_id)
    await held_v1.release()
    await store.retry_drains()
    assert drained == [v2.snapshot_id, v1.snapshot_id]
    await store.close()
    assert drained == [v2.snapshot_id, v1.snapshot_id, next_v2.snapshot_id]
    await manager.discard_prepared("snapshot")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_admission_waits_while_current_is_quiesced(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_admission",
        "from agent.plugins import Plugin\n"
        "class SnapshotAdmissionPlugin(Plugin):\n"
        "    name = 'snapshot_admission'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    held = manager.snapshot_store.lease()
    quiescing = asyncio.create_task(manager.snapshot_store.quiesce_current())
    waiting = asyncio.create_task(manager.snapshot_store.acquire())
    await asyncio.sleep(0)
    assert not quiescing.done()
    assert not waiting.done()

    await held.release()
    assert await quiescing is snapshot
    assert not waiting.done()
    await manager.snapshot_store.resume(snapshot)
    admitted = await waiting
    assert admitted.snapshot is snapshot

    await admitted.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_proactive_quiesce_does_not_deadlock_with_paused_tick(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "proactive_admission",
        "from agent.plugins import Plugin\n"
        "class ProactiveAdmissionPlugin(Plugin):\n"
        "    name = 'proactive_admission'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    loop = object.__new__(ProactiveLoop)
    loop._runtime_snapshot_store = manager.snapshot_store
    loop._reload_lock = asyncio.Lock()
    stopped = asyncio.Event()

    async def stop_active_kernel() -> None:
        stopped.set()

    async def switch_snapshot(_snapshot) -> None:
        return None

    async def tick_bound() -> float:
        return 1.0

    loop._stop_active_kernel = stop_active_kernel
    loop._switch_snapshot = switch_snapshot
    loop._tick_bound = tick_bound
    snapshot = manager.snapshot_store.pause_admission()
    tick = asyncio.create_task(loop._tick())
    await asyncio.sleep(0)

    await asyncio.wait_for(loop.quiesce_for_reload(), timeout=0.2)

    assert stopped.is_set()
    assert not tick.done()
    await manager.snapshot_store.resume(snapshot)
    assert await tick == 1.0
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_managed_service_publish_drains_old_snapshot_before_switch(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "service_admission",
        "from agent.plugins import ManagedServiceSpec, Plugin\n"
        "class ServiceAdmissionPlugin(Plugin):\n"
        "    name = 'service_admission'\n"
        "    version = 'v1'\n"
        "    @classmethod\n"
        "    def managed_services(cls):\n"
        "        return [ManagedServiceSpec(id='worker', command=('python', 'service.py'))]\n",
    )
    _ = (plugin_dir / "service.py").write_text("pass\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    held = manager.snapshot_store.lease()
    switched = asyncio.Event()

    async def switch_services(plugin_id, old_services, new_services) -> None:
        assert plugin_id == "service_admission"
        assert old_services["worker"]["revision"] != new_services["worker"]["revision"]
        switched.set()

    manager.bind_service_switcher(switch_services)
    source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    _ = (plugin_dir / "plugin.py").write_text(
        source.replace("version = 'v1'", "version = 'v2'"),
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("service_admission") is not None
    publishing = asyncio.create_task(manager.publish_prepared("service_admission"))
    await asyncio.sleep(0)
    waiting = asyncio.create_task(manager.snapshot_store.acquire())
    await asyncio.sleep(0)
    assert not switched.is_set()
    assert not waiting.done()

    await held.release()
    result = await publishing
    admitted = await waiting

    assert result["publication_state"] == "committed"
    assert switched.is_set()
    assert admitted.snapshot is manager.current_snapshot
    await admitted.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_endpoint_failure_resumes_admission_before_candidate_terminate(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "service_cleanup",
        "from agent.plugins import ManagedServiceSpec, Plugin\n"
        "class ServiceCleanupPlugin(Plugin):\n"
        "    name = 'service_cleanup'\n"
        "    version = 'v1'\n"
        "    @classmethod\n"
        "    def managed_services(cls):\n"
        "        return [ManagedServiceSpec(id='worker', command=('python', 'service.py'))]\n",
    )
    _ = (plugin_dir / "service.py").write_text("pass\n", encoding="utf-8")
    manager = _manager(tmp_path)
    await manager.load_all()
    source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    _ = (plugin_dir / "plugin.py").write_text(
        source.replace("version = 'v1'", "version = 'v2'"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("service_cleanup")
    assert candidate is not None
    terminated = asyncio.Event()

    async def terminate() -> None:
        lease = await manager.snapshot_store.acquire()
        await lease.release()
        terminated.set()

    async def fail_switch(_plugin_id, _old, _new) -> None:
        raise RuntimeError("endpoint failed")

    candidate.instance.terminate = terminate  # type: ignore[method-assign]
    manager.bind_service_switcher(fail_switch)

    result = await asyncio.wait_for(
        manager.publish_prepared("service_cleanup"),
        timeout=1,
    )

    assert result["publication_state"] == "failed"
    assert terminated.is_set()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_runtime_snapshot_commit_does_not_wait_for_drain(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_owner",
        "from agent.plugins import Plugin\n"
        "class SnapshotOwnerPlugin(Plugin):\n"
        "    name = 'snapshot_owner'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_owner")
    prepared = await manager.prepare_candidate("snapshot_owner")
    assert active is not None and prepared is not None
    compiler = RuntimeSnapshotCompiler()
    v1 = compiler.compile({"snapshot_owner": active}, catalog_generation=active)
    v2 = compiler.compile(
        {"snapshot_owner": prepared},
        catalog_generation=prepared,
    )
    attempts = 0
    drain_started = asyncio.Event()
    allow_drain = asyncio.Event()

    async def fail_once(snapshot) -> None:
        nonlocal attempts
        attempts += 1
        drain_started.set()
        await allow_drain.wait()
        if attempts == 1:
            raise RuntimeError("drain failed")

    store = RuntimeSnapshotStore(fail_once)
    store.install(v1)
    lease = store.lease()
    with pytest.raises(RuntimeError, match="仍有 lease"):
        await store.close()
    assert store.current is v1
    await lease.release()
    transaction = store.begin_publish(v2)

    await asyncio.wait_for(store.commit(transaction), timeout=0.1)
    assert store.current is v2
    await drain_started.wait()
    assert v1.snapshot_id in store.retained_snapshot_ids
    allow_drain.set()
    await store.retry_drains()
    assert v1.snapshot_id not in store.retained_snapshot_ids
    other_store = RuntimeSnapshotStore()
    with pytest.raises(RuntimeError, match="全新 compiled"):
        other_store.install(v2)
    await store.close()
    await manager.discard_prepared("snapshot_owner")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_snapshot_compile_failure_does_not_publish_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(
        plugins,
        "first_snapshot",
        "from agent.plugins import Plugin\n"
        "class FirstSnapshotPlugin(Plugin):\n"
        "    name = 'first_snapshot'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    current = manager.current_snapshot
    assert current is not None
    _write_plugin(
        plugins,
        "second_snapshot",
        "from agent.plugins import Plugin\n"
        "class SecondSnapshotPlugin(Plugin):\n"
        "    name = 'second_snapshot'\n"
        "    async def prepare(self):\n"
        "        self.context.kv_store.set('initialized', True)\n",
    )
    compile_snapshot = manager._snapshot_compiler.compile

    def fail_second(generations, *, catalog_generation=None):
        if "second_snapshot" in generations:
            raise RuntimeError("snapshot compile failed")
        return compile_snapshot(
            generations,
            catalog_generation=catalog_generation,
        )

    monkeypatch.setattr(manager._snapshot_compiler, "compile", fail_second)

    await manager.load_all()

    assert manager.current_snapshot is current
    assert manager.loaded_count == 1
    assert manager.generation("second_snapshot") is None
    assert plugin_registry.get_instance("akasic_plugin_plugins_second_snapshot") is None
    state = tmp_path / "workspace" / "plugin-data" / "second_snapshot-builtin" / ".kv.json"
    assert not state.exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_passive_runtime_admission_holds_one_snapshot(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "passive_snapshot",
        "from agent.plugins import Plugin\n"
        "class PassiveSnapshotPlugin(Plugin):\n"
        "    name = 'passive_snapshot'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("passive_snapshot")
    prepared = await manager.prepare_candidate("passive_snapshot")
    assert active is not None and prepared is not None
    compiler = RuntimeSnapshotCompiler()
    v1 = compiler.compile({"passive_snapshot": active}, catalog_generation=active)
    v2 = compiler.compile(
        {"passive_snapshot": prepared},
        catalog_generation=prepared,
    )
    store = RuntimeSnapshotStore()
    store.install(v1)
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = store
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    async def process(msg, **kwargs):
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        assert snapshot is not None
        seen.append(snapshot.snapshot_id)
        entered.set()
        await release.wait()
        assert get_current_runtime_snapshot() is snapshot
        seen.append(snapshot.snapshot_id)
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:snapshot"))
    running = asyncio.create_task(loop._process_with_runtime_admission(message))
    await entered.wait()
    transaction = store.begin_publish(v2)
    await store.commit(transaction)
    release.set()

    assert await running == "done"
    assert seen == [v1.snapshot_id, v1.snapshot_id]
    await loop._process_with_runtime_admission(message)
    assert seen[-2:] == [v2.snapshot_id, v2.snapshot_id]
    await store.close()
    await manager.discard_prepared("passive_snapshot")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_passive_runtime_snapshot_does_not_leak_to_detached_task(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "detached_snapshot",
        "from agent.plugins import Plugin\n"
        "class DetachedSnapshotPlugin(Plugin):\n"
        "    name = 'detached_snapshot'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    generation = manager.generation("detached_snapshot")
    assert generation is not None
    snapshot = RuntimeSnapshotCompiler().compile(
        {"detached_snapshot": generation},
        catalog_generation=generation,
    )
    store = RuntimeSnapshotStore()
    store.install(snapshot)
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = store
    detached_seen: list[RuntimeSnapshot | None] = []
    detached_done = asyncio.Event()
    detached_release = asyncio.Event()
    detached_tasks: list[asyncio.Task[None]] = []

    async def detached() -> None:
        await detached_release.wait()
        from agent.plugins.snapshot import get_current_runtime_snapshot

        detached_seen.append(get_current_runtime_snapshot())
        detached_done.set()

    async def process(msg, **kwargs):
        from agent.plugins.snapshot import get_current_runtime_snapshot

        assert get_current_runtime_snapshot() is snapshot
        detached_tasks.append(asyncio.create_task(detached()))
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:detached-snapshot"))
    assert await loop._process_with_runtime_admission(message) == "done"
    assert snapshot.lease_count == 0
    detached_release.set()
    await detached_done.wait()
    assert detached_seen == [None]
    await detached_tasks[0]
    await store.close()
    await manager.terminate_all()


def _snapshot_publish_source(
    version: str,
    *,
    fail_prepare: bool = False,
    fail_activate: bool = False,
) -> str:
    prepare_body = (
        "        self.context.kv_store.set('initialized', 'v2')\n"
        "        raise RuntimeError('prepare failed')\n"
        if fail_prepare
        else f"        self.context.kv_store.set('initialized', '{version}')\n"
    )
    activate_body = (
        "        raise RuntimeError('activate failed')\n"
        if fail_activate
        else ""
    )
    return (
        "from agent.plugins import Plugin\n"
        "class SnapshotModule:\n"
        "    slot = 'snapshot_publish.before_turn'\n"
        "    requires = ('before_turn.emit',)\n"
        f"    version = '{version}'\n"
        "    async def run(self, frame): return frame\n"
        "class SnapshotPublishPlugin(Plugin):\n"
        "    name = 'snapshot_publish'\n"
        "    def before_turn_modules(self): return [SnapshotModule()]\n"
        "    async def prepare(self):\n"
        "        self.data_dir_hidden_during_prepare = self.context.data_dir is None\n"
        f"{prepare_body}"
        "    def activate(self):\n"
        "        if self.context.data_dir is None:\n"
        "            raise RuntimeError('plugin data unavailable during activation')\n"
        f"{activate_body}"
        "    def retire(self):\n"
        "        self.retired = True\n"
        "    async def terminate(self):\n"
        "        self.context.kv_store.set('terminated', True)\n"
    )


@pytest.mark.asyncio
async def test_publish_prepared_switches_snapshot_after_prepare(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_publish")
    old_snapshot = manager.current_snapshot
    assert active is not None and old_snapshot is not None
    assert active.instance.data_dir_hidden_during_prepare is True
    old_lease = manager.snapshot_store.lease()

    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None
    assert candidate.reload_tx_id is not None
    assert manager.reload_journal.get(candidate.reload_tx_id).phase == "prepared"

    assert candidate.prepare_started is False

    result = await manager.publish_prepared("snapshot_publish")

    assert result["publication_state"] == "committed"
    assert manager.generation("snapshot_publish") is candidate
    assert manager.current_snapshot is candidate.runtime_snapshot
    assert candidate.prepare_started is True
    assert candidate.instance.data_dir_hidden_during_prepare is True
    assert active.state == "retired"
    assert active.instance.retired is True
    assert old_lease.snapshot is old_snapshot
    assert old_lease.snapshot.before_turn_modules[0].version == "v1"
    next_lease = manager.snapshot_store.lease()
    assert next_lease.snapshot.before_turn_modules[0].version == "v2"
    state = tmp_path / "workspace" / "plugin-data" / "snapshot_publish-builtin" / ".kv.json"
    state_value: object = json.loads(state.read_text(encoding="utf-8"))
    assert isinstance(state_value, dict)
    assert state_value["initialized"] == "v2"
    with pytest.raises(RuntimeError, match="已退役 generation"):
        active.instance.context.kv_store.set("stale_write", True)
    assert "stale_write" not in json.loads(state.read_text(encoding="utf-8"))
    assert manager.reload_journal.get(candidate.reload_tx_id).phase == "draining"

    await next_lease.release()
    await old_lease.release()
    await manager.snapshot_store.retry_drains()
    assert manager.reload_journal.get(candidate.reload_tx_id).phase == "complete"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_publish_prepared_prepare_failure_keeps_active_snapshot(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_publish")
    old_snapshot = manager.current_snapshot
    assert active is not None and old_snapshot is not None
    state_path = (
        tmp_path
        / "workspace"
        / "plugin-data"
        / "snapshot_publish-builtin"
        / ".kv.json"
    )
    state_before = state_path.read_bytes()
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2", fail_prepare=True),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None

    result = await manager.publish_prepared("snapshot_publish")

    assert result["publication_state"] == "failed"
    assert manager.generation("snapshot_publish") is active
    assert manager.current_snapshot is old_snapshot
    assert manager.prepared_generation("snapshot_publish") is None
    assert candidate.state == "discarded"
    assert state_path.read_bytes() == state_before
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_startup_recovers_committed_reload_by_exact_source_revision(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v2"),
    )
    first_manager = _manager(tmp_path)
    await first_manager.load_all()
    generation = first_manager.generation("snapshot_publish")
    snapshot = first_manager.current_snapshot
    assert generation is not None and snapshot is not None
    source_revision = generation.source_revision
    await first_manager.terminate_all()

    tx_id = first_manager.reload_journal.begin(
        plugin_id="snapshot_publish",
        base_snapshot_id="snapshot-v1",
        generation_id="snapshot_publish:source-v2:2",
        source_revision=source_revision,
        config_revision=generation.config_revision,
    )
    first_manager.reload_journal.advance(
        tx_id,
        "prepared",
        candidate_snapshot_id=snapshot.snapshot_id,
    )
    first_manager.reload_journal.advance(tx_id, "validating")
    first_manager.reload_journal.advance(tx_id, "commit_started")

    restarted = _manager(tmp_path)
    await restarted.load_all()

    assert restarted.reload_journal.get(tx_id).phase == "recovered"
    assert restarted.generation("snapshot_publish") is not None
    await restarted.terminate_all()


@pytest.mark.asyncio
async def test_activate_failure_keeps_previous_snapshot_and_plugin_data(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_publish")
    old_snapshot = manager.current_snapshot
    assert active is not None and old_snapshot is not None
    state_path = (
        tmp_path
        / "workspace"
        / "plugin-data"
        / "snapshot_publish-builtin"
        / ".kv.json"
    )
    state_before = state_path.read_bytes()
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2", fail_activate=True),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None

    with pytest.raises(RuntimeError, match="activate failed"):
        await manager.publish_prepared("snapshot_publish")

    assert manager.current_snapshot is old_snapshot
    assert manager.generation("snapshot_publish") is active
    assert candidate.instance.context.data_dir is None
    assert candidate.scope.closed is True
    assert state_path.read_bytes() == state_before
    assert candidate.reload_tx_id is not None
    assert manager.reload_journal.get(candidate.reload_tx_id).phase == "aborted"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_startup_fails_when_committed_reload_source_is_unavailable(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v2"),
    )
    manager = _manager(tmp_path)
    tx_id = manager.reload_journal.begin(
        plugin_id="snapshot_publish",
        base_snapshot_id="snapshot-v1",
        generation_id="snapshot_publish:missing:2",
        source_revision="missing-source-revision",
        config_revision="",
    )
    manager.reload_journal.advance(
        tx_id,
        "prepared",
        candidate_snapshot_id="snapshot-v2",
    )
    manager.reload_journal.advance(tx_id, "validating")
    manager.reload_journal.advance(tx_id, "commit_started")

    with pytest.raises(
        RuntimeError,
        match="ReloadTransaction 恢复源码不一致",
    ):
        await manager.load_all()

    assert manager.generation("snapshot_publish") is None
    assert manager.reload_journal.get(tx_id).phase == "commit_started"
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_kv_commit_failure_keeps_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent.plugins.context import PreparedPluginKVStore

    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_publish")
    old_snapshot = manager.current_snapshot
    assert active is not None and old_snapshot is not None
    state_path = (
        tmp_path
        / "workspace"
        / "plugin-data"
        / "snapshot_publish-builtin"
        / ".kv.json"
    )
    state_before = state_path.read_bytes()
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None

    def fail_commit(_store: PreparedPluginKVStore) -> None:
        raise OSError("candidate KV commit failed")

    monkeypatch.setattr(PreparedPluginKVStore, "commit", fail_commit)

    with pytest.raises(OSError, match="candidate KV commit failed"):
        await manager.publish_prepared("snapshot_publish")

    assert manager.current_snapshot is old_snapshot
    assert manager.generation("snapshot_publish") is active
    assert manager.prepared_generation("snapshot_publish") is None
    assert candidate.scope.closed is True
    assert state_path.read_bytes() == state_before
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_prepare_cannot_start_generation_task_at_runtime(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "prepare_task",
        "from agent.plugins import Plugin\n"
        "class PrepareTaskPlugin(Plugin):\n"
        "    name = 'prepare_task'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("prepare_task")
    old_snapshot = manager.current_snapshot
    assert active is not None and old_snapshot is not None
    _ = (plugin_dir / "plugin.py").write_text(
        "import asyncio\n"
        "from agent.plugins import Plugin\n"
        "class PrepareTaskPlugin(Plugin):\n"
        "    name = 'prepare_task'\n"
        "    async def prepare(self):\n"
        "        self.context.create_task(self._run(), name='forbidden-prepare-task')\n"
        "    async def _run(self):\n"
        "        await asyncio.Event().wait()\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("prepare_task")
    assert candidate is not None

    result = await manager.publish_prepared("prepare_task")

    assert result["publication_state"] == "failed"
    assert manager.generation("prepare_task") is active
    assert manager.current_snapshot is old_snapshot
    assert candidate.scope.closed is True
    gate = manager.latest_gate("prepare_task")
    assert gate is not None
    assert gate.checks[-1].check_id == "prepare"
    assert "prepare 阶段禁止启动后台任务" in gate.failure_reason
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_candidate_is_hidden_until_commit_and_failure_keeps_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    active = manager.generation("snapshot_publish")
    old_snapshot = manager.current_snapshot
    assert active is not None and old_snapshot is not None
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None

    async def fail_terminate() -> None:
        raise RuntimeError("terminate failed")

    candidate.instance.terminate = fail_terminate  # type: ignore[attr-defined]

    invariant_entered = asyncio.Event()
    invariant_release = asyncio.Event()

    async def fail_invariant(generation, snapshot) -> None:
        invariant_entered.set()
        await invariant_release.wait()
        raise RuntimeError("post publish failed")

    monkeypatch.setattr(manager, "_post_publish_invariants", fail_invariant)

    publishing = asyncio.create_task(manager.publish_prepared("snapshot_publish"))
    await invariant_entered.wait()
    visible_snapshot = manager.current_snapshot
    ordinary_lease = manager.snapshot_store.lease()
    invariant_release.set()
    with pytest.raises(RuntimeError, match="post publish failed"):
        await publishing
    leased_snapshot = ordinary_lease.snapshot
    await ordinary_lease.release()

    assert visible_snapshot is old_snapshot
    assert leased_snapshot is old_snapshot
    assert manager.current_snapshot is old_snapshot
    assert manager.generation("snapshot_publish") is active
    assert manager.prepared_generation("snapshot_publish") is None
    assert candidate.state == "aborted"
    assert candidate.runtime_snapshot is not None
    assert candidate.runtime_snapshot.state == "aborted"
    assert candidate.scope.closed is True
    assert candidate.reload_tx_id is not None
    assert manager.reload_journal.get(candidate.reload_tx_id).phase == "aborted"
    assert any(
        failure.error == "terminate failed"
        for failure in manager.cleanup_failures
    )
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_post_publish_invariant_timeout_aborts_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None and old_snapshot is not None

    async def never_finishes(generation, snapshot) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_post_publish_invariants", never_finishes)
    manager.POST_PUBLISH_TIMEOUT_SECONDS = 0.01

    with pytest.raises(TimeoutError):
        await manager.publish_prepared("snapshot_publish")

    assert manager.current_snapshot is old_snapshot
    assert candidate.state == "aborted"
    assert candidate.scope.closed is True
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_cancelled_publish_never_leases_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_publish",
        _snapshot_publish_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_publish_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_publish")
    assert candidate is not None and old_snapshot is not None
    entered = asyncio.Event()

    async def wait_forever(generation, snapshot) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_post_publish_invariants", wait_forever)
    publishing = asyncio.create_task(manager.publish_prepared("snapshot_publish"))
    await entered.wait()
    lease = manager.snapshot_store.lease()
    assert lease.snapshot is old_snapshot
    publishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await publishing

    assert manager.current_snapshot is old_snapshot
    assert candidate.scope.closed is True
    await lease.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_dead_candidate_mcp_aborts_post_publish_invariant(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_mcp",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class SnapshotMcpPlugin(Plugin):\n"
        "    name = 'snapshot_mcp'\n"
        "    version = 'v1'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='snapshot', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(plugin_dir, ("version",))
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    plugin_file = plugin_dir / "plugin.py"
    _ = plugin_file.write_text(
        plugin_file.read_text(encoding="utf-8").replace("'v1'", "'v2'"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_mcp")
    assert candidate is not None and candidate.mcp_catalog is not None
    client = candidate.mcp_catalog.servers["snapshot"].client
    process = client._process
    assert process is not None
    process.kill()
    await process.wait()
    assert client.connected is False

    with pytest.raises(RuntimeError, match="MCP client 已断开"):
        await manager.publish_prepared("snapshot_mcp")

    assert manager.current_snapshot is old_snapshot
    assert candidate.scope.closed is True
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_reconcile_changed_publishes_multiple_plugins_from_latest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugins = tmp_path / "plugins"

    def source(name: str, version: str) -> str:
        class_name = "".join(part.title() for part in name.split("_"))
        return (
            "from agent.plugins import Plugin\n"
            f"class {class_name}Plugin(Plugin):\n"
            f"    name = '{name}'\n"
            f"    version = '{version}'\n"
        )

    first_dir = _write_plugin(plugins, "snapshot_first", source("snapshot_first", "v1"))
    second_dir = _write_plugin(
        plugins,
        "snapshot_second",
        source("snapshot_second", "v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    _ = (first_dir / "plugin.py").write_text(
        source("snapshot_first", "v2"),
        encoding="utf-8",
    )
    _ = (second_dir / "plugin.py").write_text(
        source("snapshot_second", "v2"),
        encoding="utf-8",
    )
    discover_calls = 0
    original_discover = manager.discover

    def count_discoveries() -> list[dict[str, str]]:
        nonlocal discover_calls
        discover_calls += 1
        return original_discover()

    monkeypatch.setattr(manager, "discover", count_discoveries)

    results = await manager.reconcile_changed()

    assert discover_calls == 1
    assert [result["publication_state"] for result in results] == [
        "committed",
        "committed",
    ]
    first = manager.generation("snapshot_first")
    second = manager.generation("snapshot_second")
    snapshot = manager.current_snapshot
    assert first is not None and second is not None and snapshot is not None
    assert first.instance.version == "v2"  # type: ignore[attr-defined]
    assert second.instance.version == "v2"  # type: ignore[attr-defined]
    assert snapshot.generations == {
        "snapshot_first": first,
        "snapshot_second": second,
    }
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_reconcile_changed_disables_and_reenables_plugin_capabilities(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "topology",
        "from agent.plugins import Plugin, tool\n"
        "class TopologyPlugin(Plugin):\n"
        "    name = 'topology'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills',)\n"
        "    @tool(name='topology_value')\n"
        "    async def value(self, event):\n"
        "        \"\"\"Return topology value.\"\"\"\n"
        "        return 'active'\n"
        "    def retire(self):\n"
        "        self.retired = True\n",
    )
    skill = plugin_dir / "skills" / "topology-skill"
    skill.mkdir(parents=True)
    _ = (skill / "SKILL.md").write_text(
        "---\ndescription: topology\n---\ntopology\n",
        encoding="utf-8",
    )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools, workspace=tmp_path / "workspace")
    plugins_home = tmp_path / "home"
    write_plugin_manifest({"topology": True}, plugins_home=plugins_home)
    await manager.load_all()
    assert tools.has_tool("topology_value")
    active = manager.generation("topology")
    assert active is not None

    write_plugin_manifest({"topology": False}, plugins_home=plugins_home)
    disabled = await manager.reconcile_changed()

    snapshot = manager.current_snapshot
    assert disabled[0]["publication_state"] == "disabled"
    assert snapshot is not None and snapshot.generations == {}
    assert not snapshot.tool_registry.has_tool("topology_value")
    assert "topology-skill" not in snapshot.plugin_skill_index.records
    assert manager.generation("topology") is None
    assert active.retire_started is True
    assert active.instance.retired is True

    write_plugin_manifest({"topology": True}, plugins_home=plugins_home)
    enabled = await manager.reconcile_changed()

    snapshot = manager.current_snapshot
    assert enabled[0]["publication_state"] == "committed"
    assert snapshot is not None and "topology" in snapshot.generations
    assert snapshot.tool_registry.has_tool("topology_value")
    assert "topology-skill" in snapshot.plugin_skill_index.records
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_repeated_disable_with_retained_snapshot_keeps_unique_id_and_alias(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "retained_topology",
        "from agent.plugins import Plugin\n"
        "class RetainedTopologyPlugin(Plugin):\n"
        "    name = 'retained_topology'\n",
    )
    manager = _manager(tmp_path)
    plugins_home = tmp_path / "home"
    write_plugin_manifest({"retained_topology": True}, plugins_home=plugins_home)
    await manager.load_all()
    alias = "akasic_plugin_plugins_retained_topology"

    write_plugin_manifest({"retained_topology": False}, plugins_home=plugins_home)
    await manager.reconcile_changed()
    first_disabled = manager.current_snapshot
    assert first_disabled is not None
    retained = manager.snapshot_store.lease()

    write_plugin_manifest({"retained_topology": True}, plugins_home=plugins_home)
    await manager.reconcile_changed()
    reenabled = manager.generation("retained_topology")
    assert reenabled is not None
    assert sys.modules[alias] is sys.modules[reenabled.module_path]

    write_plugin_manifest({"retained_topology": False}, plugins_home=plugins_home)
    await manager.reconcile_changed()
    second_disabled = manager.current_snapshot
    assert second_disabled is not None
    assert second_disabled.snapshot_id != first_disabled.snapshot_id

    await retained.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_current_plugin_views_ignore_retained_old_generation(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "current_view",
        "from agent.plugins import Plugin\n"
        "class CurrentViewPlugin(Plugin):\n"
        "    name = 'current_view'\n"
        "    version = 'v1'\n"
        "    def telegram_bot_commands(self):\n"
        "        return [(f'view-{self.version}', self.version)]\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    retained = manager.snapshot_store.lease()

    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class CurrentViewPlugin(Plugin):\n"
        "    name = 'current_view'\n"
        "    version = 'v2'\n"
        "    def telegram_bot_commands(self):\n"
        "        return [(f'view-{self.version}', self.version)]\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("current_view")
    assert candidate is not None
    await manager.publish_prepared("current_view")

    active = manager.active_plugins()
    assert len(active) == 1
    assert active[0].manifest["version"] == "v2"
    assert manager.telegram_bot_commands == [("view-v2", "v2")]
    assert manager.mobile_bot_commands == []

    write_plugin_manifest(
        {"current_view": False},
        plugins_home=tmp_path / "home",
    )
    await manager.reconcile_changed()
    assert manager.active_plugins() == []
    assert manager.telegram_bot_commands == []
    assert manager.mobile_bot_commands == []

    await retained.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_repeated_drain_joins_cancelled_request_until_old_turn_releases(
    tmp_path: Path,
) -> None:
    """复现 mobile turn 持有 lease 时卸载 context_pressure 的 drain。"""

    # 1. 当前 turn 持有包含 context_pressure 的旧快照。
    _write_plugin(
        tmp_path / "plugins",
        "context_pressure",
        "from agent.plugins import Plugin\n"
        "class ContextPressurePlugin(Plugin):\n"
        "    name = 'context_pressure'\n",
    )
    manager = _manager(tmp_path)
    plugins_home = tmp_path / "home"
    write_plugin_manifest({"context_pressure": True}, plugins_home=plugins_home)
    await manager.load_all()
    active = manager.generation("context_pressure")
    assert active is not None
    old_turn = manager.snapshot_store.lease()

    # 2. 第一次 drain 发布禁用快照后，被外层 Shell timeout 取消。
    write_plugin_manifest({"context_pressure": False}, plugins_home=plugins_home)
    first = asyncio.create_task(
        manager.reconcile_disabled_and_drain("context_pressure")
    )
    for _ in range(100):
        if manager.generation("context_pressure") is None:
            break
        await asyncio.sleep(0)
    else:
        pytest.fail("context_pressure 未进入 retired generation")
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    # 3. 重试不得因 active 已移除而假报 drained，必须等待旧 turn 结束。
    repeated = asyncio.create_task(
        manager.reconcile_disabled_and_drain("context_pressure")
    )
    await asyncio.sleep(0)
    assert not repeated.done()

    await old_turn.release()
    await asyncio.wait_for(repeated, timeout=1)
    assert active.scope.closed
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_drain_joins_generation_predeactivated_by_manifest_watcher(
    tmp_path: Path,
) -> None:
    """复现 watcher 抢先停用插件后卸载误报 drain 完成的竞争。"""

    # 1. 旧 turn 持有快照，watcher 先观察 disabled manifest 并停用插件。
    _write_plugin(
        tmp_path / "plugins",
        "context_pressure",
        "from agent.plugins import Plugin\n"
        "class ContextPressurePlugin(Plugin):\n"
        "    name = 'context_pressure'\n",
    )
    manager = _manager(tmp_path)
    plugins_home = tmp_path / "home"
    write_plugin_manifest({"context_pressure": True}, plugins_home=plugins_home)
    await manager.load_all()
    active = manager.generation("context_pressure")
    assert active is not None
    old_turn = manager.snapshot_store.lease()

    write_plugin_manifest({"context_pressure": False}, plugins_home=plugins_home)
    await manager.reconcile_changed()
    assert manager.generation("context_pressure") is None
    assert not active.scope.closed

    # 2. 卸载必须加入 watcher 登记的旧代 drain，不能因 active 为空提前完成。
    drain = asyncio.create_task(
        manager.reconcile_disabled_and_drain("context_pressure")
    )
    await asyncio.sleep(0)
    assert not drain.done()

    await old_turn.release()
    await asyncio.wait_for(drain, timeout=1)
    assert active.scope.closed
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_uninstall_waits_for_generation_retired_by_hot_reload(
    tmp_path: Path,
) -> None:
    """复现 G1 热替换为 G2 后卸载遗漏 G1 lease 的竞争。"""

    # 1. 旧 turn 持有 G1，watcher 发布 G2 并令 G1 进入退役状态。
    plugin_dir = tmp_path / "plugins"
    _write_plugin(
        plugin_dir,
        "context_pressure",
        "from agent.plugins import Plugin\n"
        "class ContextPressurePlugin(Plugin):\n"
        "    name = 'context_pressure'\n"
        "    version = 'v1'\n",
    )
    manager = _manager(tmp_path)
    plugins_home = tmp_path / "home"
    write_plugin_manifest({"context_pressure": True}, plugins_home=plugins_home)
    await manager.load_all()
    generation_one = manager.generation("context_pressure")
    assert generation_one is not None
    old_turn = manager.snapshot_store.lease()

    _ = (plugin_dir / "context_pressure" / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class ContextPressurePlugin(Plugin):\n"
        "    name = 'context_pressure'\n"
        "    version = 'v2'\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("context_pressure")
    assert candidate is not None
    await manager.publish_prepared("context_pressure")
    generation_two = manager.generation("context_pressure")
    assert generation_two is not None
    assert generation_two is not generation_one
    assert not generation_one.scope.closed

    # 2. 卸载 G2 必须同时等待仍被旧 turn 持有的 G1。
    write_plugin_manifest({"context_pressure": False}, plugins_home=plugins_home)
    drain = asyncio.create_task(
        manager.reconcile_disabled_and_drain("context_pressure")
    )
    await asyncio.sleep(0)
    assert not drain.done()

    await old_turn.release()
    await asyncio.wait_for(drain, timeout=1)
    assert generation_one.scope.closed
    assert generation_two.scope.closed
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_hot_reload_forgets_closed_retired_generation(tmp_path: Path) -> None:
    """热重载完成排空后不再保留旧 generation 引用。"""

    plugin_dir = tmp_path / "plugins"
    _write_plugin(
        plugin_dir,
        "reload_cleanup",
        "from agent.plugins import Plugin\n"
        "class ReloadCleanupPlugin(Plugin):\n"
        "    name = 'reload_cleanup'\n"
        "    version = 'v1'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    generation_one = manager.generation("reload_cleanup")
    assert generation_one is not None
    old_turn = manager.snapshot_store.lease()

    _ = (plugin_dir / "reload_cleanup" / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class ReloadCleanupPlugin(Plugin):\n"
        "    name = 'reload_cleanup'\n"
        "    version = 'v2'\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("reload_cleanup")
    assert candidate is not None
    await manager.publish_prepared("reload_cleanup")
    assert manager._draining_generations["reload_cleanup"] == [generation_one]

    await old_turn.release()
    await manager.snapshot_store.wait_for_generation_drained(generation_one)
    assert generation_one.scope.closed
    assert "reload_cleanup" not in manager._draining_generations
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_bot_command_catalogs_require_explicit_channel_declarations(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "a_telegram_only",
        "from agent.plugins import Plugin\n"
        "class TelegramOnlyPlugin(Plugin):\n"
        "    name = 'a_telegram_only'\n"
        "    def telegram_bot_commands(self):\n"
        "        return [('telegram_only', '仅 Telegram')]\n",
    )
    _write_plugin(
        tmp_path / "plugins",
        "b_shared",
        "from agent.plugins import Plugin\n"
        "class SharedCommandPlugin(Plugin):\n"
        "    name = 'b_shared'\n"
        "    def telegram_bot_commands(self):\n"
        "        return [('shared', '共享命令')]\n"
        "    def mobile_bot_commands(self):\n"
        "        return [('shared', '共享命令')]\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    assert manager.telegram_bot_commands == [
        ("telegram_only", "仅 Telegram"),
        ("shared", "共享命令"),
    ]
    assert manager.mobile_bot_commands == [("shared", "共享命令")]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_publish_cancellation_after_store_commit_keeps_manager_consistent(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "commit_cancel",
        "from agent.plugins import Plugin\n"
        "class CommitCancelPlugin(Plugin):\n"
        "    name = 'commit_cancel'\n"
        "    version = 'v1'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    _ = (plugin_dir / "plugin.py").write_text(
        source.replace("version = 'v1'", "version = 'v2'"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("commit_cancel")
    assert candidate is not None
    committed = asyncio.Event()
    release = asyncio.Event()
    original_commit = manager.snapshot_store.commit

    async def delayed_commit(
        transaction,
        *,
        before_open=None,
        after_open=None,
    ) -> None:
        await original_commit(
            transaction,
            before_open=before_open,
            after_open=after_open,
        )
        committed.set()
        await release.wait()

    manager.snapshot_store.commit = delayed_commit  # type: ignore[method-assign]
    publishing = asyncio.create_task(manager.publish_prepared("commit_cancel"))
    await committed.wait()
    publishing.cancel()
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await publishing

    active = manager.generation("commit_cancel")
    alias = "akasic_plugin_plugins_commit_cancel"
    assert active is candidate
    assert manager.prepared_generation("commit_cancel") is None
    assert manager.current_snapshot is candidate.runtime_snapshot
    assert sys.modules[alias] is sys.modules[candidate.module_path]
    manager.snapshot_store.commit = original_commit  # type: ignore[method-assign]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_reconcile_changed_adds_and_removes_discovered_plugin(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    _write_plugin(
        plugins,
        "anchor",
        "from agent.plugins import Plugin\n"
        "class AnchorPlugin(Plugin):\n"
        "    name = 'anchor'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    added_dir = _write_plugin(
        plugins,
        "added",
        "from agent.plugins import Plugin\n"
        "class AddedPlugin(Plugin):\n"
        "    name = 'added'\n",
    )

    added = await manager.reconcile_changed()
    assert added[0]["publication_state"] == "committed"
    assert manager.generation("added") is not None

    shutil.rmtree(added_dir)
    removed = await manager.reconcile_changed()
    assert removed[0]["publication_state"] == "disabled"
    assert manager.generation("added") is None
    assert manager.current_snapshot is not None
    assert set(manager.current_snapshot.generations) == {"anchor"}
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_plugin_watcher_reloads_source_without_signal(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "watched",
        "from agent.plugins import Plugin\n"
        "class WatchedPlugin(Plugin):\n"
        "    name = 'watched'\n"
        "    version = 'v1'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    baseline_revision = await asyncio.to_thread(manager.watch_revision)
    watcher = PluginWatcher(
        manager,
        baseline_revision=baseline_revision,
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    _ = (plugin_dir / "plugin.py").write_text(
        source.replace("version = 'v1'", "version = 'v2'"),
        encoding="utf-8",
    )

    for _ in range(100):
        generation = manager.generation("watched")
        if generation is not None and generation.instance.version == "v2":
            break
        await asyncio.sleep(0.01)

    generation = manager.generation("watched")
    assert generation is not None and generation.instance.version == "v2"
    watcher.stop()
    await task
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_plugin_watcher_scans_files_outside_event_loop_thread() -> None:
    event_loop_thread = threading.get_ident()

    class Manager:
        def __init__(self) -> None:
            self.scan_threads: list[int] = []

        def watch_revision(self) -> str:
            self.scan_threads.append(threading.get_ident())
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    for _ in range(100):
        if manager.scan_threads:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.scan_threads
    assert all(thread_id != event_loop_thread for thread_id in manager.scan_threads)


@pytest.mark.asyncio
async def test_plugin_watcher_reconciles_change_arriving_during_scan() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "a"
            self.calls = 0
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await self.release.wait()
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="a",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    manager.revision = "b"
    await manager.started.wait()
    manager.revision = "c"
    manager.release.set()
    for _ in range(100):
        if manager.calls >= 2:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.calls >= 2


@pytest.mark.asyncio
async def test_plugin_watcher_forced_wake_reconciles_unchanged_revision() -> None:
    class Manager:
        def __init__(self) -> None:
            self.called = asyncio.Event()

        def watch_revision(self) -> str:
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.called.set()
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=60,
    )
    task = asyncio.create_task(watcher.run())
    watcher.wake()
    await asyncio.wait_for(manager.called.wait(), timeout=1)

    watcher.stop()
    await task


@pytest.mark.asyncio
async def test_plugin_watcher_recovers_after_revision_scan_error() -> None:
    class Manager:
        def __init__(self) -> None:
            self.scans = 0
            self.calls = 0

        def watch_revision(self) -> str:
            self.scans += 1
            if self.scans == 1:
                raise PermissionError("transient")
            return "ready"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    for _ in range(100):
        if manager.calls:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_does_not_reconcile_after_recovered_scan_error() -> None:
    class Manager:
        def __init__(self) -> None:
            self.scans = 0
            self.calls = 0

        def watch_revision(self) -> str:
            self.scans += 1
            if self.scans == 2:
                raise OSError("transient")
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    for _ in range(100):
        if manager.scans >= 3:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.scans >= 3
    assert manager.calls == 0


@pytest.mark.asyncio
async def test_plugin_watcher_recovers_on_third_reconcile_attempt() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "broken"
            self.failures_remaining = 2
            self.calls = 0
            self.recovered = asyncio.Event()

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if self.failures_remaining:
                self.failures_remaining -= 1
                raise RuntimeError("transient callback failure")
            self.recovered.set()
            return []

    manager = Manager()
    notification_calls = 0
    notified = asyncio.Event()

    async def notify() -> None:
        nonlocal notification_calls
        notification_calls += 1
        notified.set()

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=notify,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.wait_for(manager.recovered.wait(), timeout=1)
    await asyncio.wait_for(notified.wait(), timeout=1)

    watcher.stop()
    await task
    assert manager.calls == 3
    assert notification_calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_recovers_after_reconcile_failure() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "broken"
            self.allow_reconcile = False
            self.failed = asyncio.Event()
            self.recovered = asyncio.Event()
            self.calls = 0

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if not self.allow_reconcile:
                self.failed.set()
                raise RuntimeError("callback failed")
            self.recovered.set()
            return []

    manager = Manager()
    reconciled = 0

    async def after_reconcile() -> None:
        nonlocal reconciled
        reconciled += 1

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=after_reconcile,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    await asyncio.wait_for(manager.failed.wait(), timeout=1)
    for _ in range(100):
        if manager.calls >= 3:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.03)
    assert manager.calls == 3
    assert reconciled == 0
    assert not task.done()

    manager.allow_reconcile = True
    manager.revision = "fixed"
    await asyncio.wait_for(manager.recovered.wait(), timeout=1)
    watcher.stop()
    await task
    assert manager.calls == 4
    assert reconciled == 1


@pytest.mark.asyncio
async def test_plugin_watcher_wake_retries_blocked_revision() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "broken"
            self.allow_reconcile = False
            self.calls = 0
            self.recovered = asyncio.Event()

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if not self.allow_reconcile:
                raise RuntimeError("persistent callback failure")
            self.recovered.set()
            return []

    manager = Manager()
    notification_calls = 0
    notified = asyncio.Event()

    async def notify() -> None:
        nonlocal notification_calls
        notification_calls += 1
        notified.set()

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=notify,
    )
    task = asyncio.create_task(watcher.run())
    for _ in range(100):
        if manager.calls >= 3:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.03)
    assert manager.calls == 3
    assert notification_calls == 0

    manager.allow_reconcile = True
    watcher.wake()
    await asyncio.wait_for(manager.recovered.wait(), timeout=1)
    await asyncio.wait_for(notified.wait(), timeout=1)
    watcher.stop()
    await task
    assert manager.calls == 4
    assert notification_calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_retries_notification_without_reconciling_again() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "stable"
            self.calls = 0

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            return []

    manager = Manager()
    notification_calls = 0
    notified = asyncio.Event()

    async def after_reconcile() -> None:
        nonlocal notification_calls
        notification_calls += 1
        if notification_calls == 1:
            raise RuntimeError("transient notification failure")
        notified.set()

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=after_reconcile,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    manager.revision = "changed"
    await asyncio.wait_for(notified.wait(), timeout=1)

    watcher.stop()
    await task
    assert manager.calls == 1
    assert notification_calls == 2


@pytest.mark.asyncio
async def test_plugin_watcher_confirms_disabled_result_against_stable_revision() -> None:
    class Manager:
        def __init__(self) -> None:
            self.calls = 0

        def watch_revision(self) -> str:
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return [{"plugin_id": "observe", "publication_state": "disabled"}]
            return [{"plugin_id": "observe", "publication_state": "committed"}]

    manager = Manager()
    notification_calls = 0

    async def notify() -> None:
        nonlocal notification_calls
        notification_calls += 1

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=notify,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    watcher.wake()
    for _ in range(100):
        if manager.calls >= 2:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.calls == 2
    assert notification_calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_notifies_explicit_disable_after_confirmation() -> None:
    class Manager:
        def __init__(self) -> None:
            self.calls = 0

        def watch_revision(self) -> str:
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return [{"plugin_id": "observe", "publication_state": "disabled"}]
            return []

    manager = Manager()
    notification_calls = 0

    async def notify() -> None:
        nonlocal notification_calls
        notification_calls += 1

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=notify,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    watcher.wake()
    for _ in range(100):
        if manager.calls >= 2 and notification_calls == 1:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.calls == 2
    assert notification_calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_retries_failed_disabled_confirmation() -> None:
    class Manager:
        def __init__(self) -> None:
            self.calls = 0

        def watch_revision(self) -> str:
            return "stable"

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            if self.calls == 1:
                return [{"plugin_id": "observe", "publication_state": "disabled"}]
            if self.calls == 2:
                raise RuntimeError("temporary scan failure")
            return []

    manager = Manager()
    notification_calls = 0

    async def notify() -> None:
        nonlocal notification_calls
        notification_calls += 1

    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
        after_reconcile=notify,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    watcher.wake()
    for _ in range(100):
        if manager.calls >= 3 and notification_calls == 1:
            break
        await asyncio.sleep(0.01)

    watcher.stop()
    await task
    assert manager.calls == 3
    assert notification_calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_propagates_cancellation_and_marks_stopped() -> None:
    class Manager:
        def __init__(self) -> None:
            self.revision = "stable"
            self.started = asyncio.Event()
            self.calls = 0

        def watch_revision(self) -> str:
            return self.revision

        async def reconcile_changed(self) -> list[dict[str, object]]:
            self.calls += 1
            self.started.set()
            await asyncio.Event().wait()
            return []

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    await asyncio.sleep(0)
    manager.revision = "changed"
    await asyncio.wait_for(manager.started.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await watcher.wait_stopped()
    assert manager.calls == 1


@pytest.mark.asyncio
async def test_plugin_watcher_stop_before_run_skips_initial_scan() -> None:
    class Manager:
        def __init__(self) -> None:
            self.scans = 0

        def watch_revision(self) -> str:
            self.scans += 1
            return "stable"

    manager = Manager()
    watcher = PluginWatcher(
        cast(PluginManager, manager),
        baseline_revision="stable",
        interval_seconds=0.01,
    )
    watcher.stop()

    await watcher.run()
    await watcher.wait_stopped()
    assert manager.scans == 0


@pytest.mark.asyncio
async def test_plugin_watcher_cancellation_before_start_does_not_leak_waiter() -> None:
    class Manager:
        def watch_revision(self) -> str:
            raise AssertionError("未启动的 watcher 不应扫描")

    watcher = PluginWatcher(
        cast(PluginManager, Manager()),
        baseline_revision="stable",
        interval_seconds=0.01,
    )
    task = asyncio.create_task(watcher.run())
    watcher.stop()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(watcher.wait_stopped(), timeout=1)


def _snapshot_tool_source(version: str) -> str:
    return (
        "from agent.plugins import Plugin, tool\n"
        "class SnapshotToolPlugin(Plugin):\n"
        "    name = 'snapshot_tool'\n"
        "    @tool(name='snapshot_tool_value')\n"
        "    async def value(self, event):\n"
        f"        \"\"\"snapshot tool {version}\"\"\"\n"
        f"        return '{version}'\n"
    )


@pytest.mark.asyncio
async def test_tool_schema_search_and_execute_share_snapshot_generation(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_tool",
        _snapshot_tool_source("v1"),
    )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()

    class LateMcpTool(Tool):
        name = "mcp_late__value"
        description = "late mcp"
        parameters = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            return "late"

    tools.register(
        LateMcpTool(),
        source_type="mcp",
        source_name="late",
    )
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_tool_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_tool")
    assert candidate is not None
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = manager.snapshot_store
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[tuple[str, str, str]] = []
    late_seen: list[str] = []

    async def process(msg, **kwargs):
        schema = tools.get_schemas(["snapshot_tool_value"])[0]
        search = tools.search("snapshot tool", top_k=1)[0]
        before = str(await tools.execute("snapshot_tool_value", {}))
        late_seen.append(str(await tools.execute("mcp_late__value", {})))
        entered.set()
        await release.wait()
        after = str(await tools.execute("snapshot_tool_value", {}))
        seen.append((schema["function"]["description"], search["name"], before))
        seen.append((schema["function"]["description"], search["name"], after))
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:snapshot-tool"))
    old_turn = asyncio.create_task(loop._process_with_runtime_admission(message))
    await entered.wait()
    await manager.publish_prepared("snapshot_tool")
    release.set()
    assert await old_turn == "done"
    await loop._process_with_runtime_admission(message)

    assert seen[:2] == [
        ("snapshot tool v1", "snapshot_tool_value", "v1"),
        ("snapshot tool v1", "snapshot_tool_value", "v1"),
    ]
    assert seen[2:] == [
        ("snapshot tool v2", "snapshot_tool_value", "v2"),
        ("snapshot tool v2", "snapshot_tool_value", "v2"),
    ]
    assert late_seen == ["工具 'mcp_late__value' 不存在", "late"]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_removed_plugin_mcp_server_releases_name_for_user_registry(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "removed_mcp",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class RemovedMcpPlugin(Plugin):\n"
        "    name = 'removed_mcp'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='removed_server', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(plugin_dir, ("legacy",))
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()

    class LegacyPluginMcpTool(Tool):
        name = "mcp_removed_server__legacy"
        description = "legacy"
        parameters = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            return "legacy"

    tools.register(
        LegacyPluginMcpTool(),
        source_type="mcp",
        source_name="removed_server",
    )
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class RemovedMcpPlugin(Plugin):\n"
        "    name = 'removed_mcp'\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("removed_mcp")
    assert candidate is not None and candidate.runtime_snapshot is not None
    registry = candidate.runtime_snapshot.tool_registry
    assert registry is not None
    assert registry.has_tool("mcp_removed_server__legacy") is True
    await manager.discard_prepared("removed_mcp")
    await manager.terminate_all()


def test_tool_registry_fork_preserves_registration_order() -> None:
    registry = ToolRegistry()

    class OrderedTool(Tool):
        name = "ordered_base"
        description = "ordered"
        parameters = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            return self.name

    expected = [f"ordered_{index:02d}" for index in range(20)]
    for name in expected:
        tool_class = type(f"Tool_{name}", (OrderedTool,), {"name": name})
        registry.register(tool_class())

    assert registry.fork().get_registered_order() == expected


@pytest.mark.asyncio
async def test_failed_candidate_mcp_name_does_not_hide_user_mcp(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "candidate_mcp_pollution",
        "from agent.plugins import Plugin\n"
        "class CandidateMcpPollutionPlugin(Plugin):\n"
        "    name = 'candidate_mcp_pollution'\n"
        "    version = 'v1'\n",
    )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()

    class UserMcpTool(Tool):
        name = "mcp_candidate_failed__user"
        description = "user mcp"
        parameters = {"type": "object", "properties": {}}

        async def execute(self) -> str:
            return "user"

    tools.register(
        UserMcpTool(),
        source_type="mcp",
        source_name="candidate_failed",
    )
    _ = (plugin_dir / "fail.py").write_text(
        "raise RuntimeError('candidate failed')\n",
        encoding="utf-8",
    )
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class CandidateMcpPollutionPlugin(Plugin):\n"
        "    name = 'candidate_mcp_pollution'\n"
        "    version = 'v2'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='candidate_failed', command=('python', 'fail.py'))]\n",
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("candidate_mcp_pollution") is None
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class CandidateMcpPollutionPlugin(Plugin):\n"
        "    name = 'candidate_mcp_pollution'\n"
        "    version = 'v3'\n",
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("candidate_mcp_pollution")
    assert candidate is not None and candidate.runtime_snapshot is not None
    registry = candidate.runtime_snapshot.tool_registry
    assert registry is not None
    assert registry.has_tool("mcp_candidate_failed__user") is True
    await manager.discard_prepared("candidate_mcp_pollution")
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_initial_plugin_mcp_tool_is_visible_in_first_snapshot(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "initial_mcp_snapshot",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class InitialMcpSnapshotPlugin(Plugin):\n"
        "    name = 'initial_mcp_snapshot'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='initial_snapshot', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(plugin_dir, ("version",))
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()
    generation = manager.generation("initial_mcp_snapshot")
    assert generation is not None and generation.mcp_catalog is not None
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = manager.snapshot_store
    seen: list[str] = []

    async def process(msg, **kwargs):
        seen.append(
            str(await tools.execute("mcp_initial_snapshot__version", {}))
        )
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:initial-mcp"))
    await loop._process_with_runtime_admission(message)

    assert seen == ["[]"]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_workspace_mcp_generation_publish_preserves_old_turn_lease(
    tmp_path: Path,
) -> None:
    from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    first = await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "workspace-v1", tool_name="version_v1"),
        revision="v1",
    )
    await manager.publish_workspace_mcp()
    old_client = first.catalog.servers["workspace"].client
    old_lease = await manager.snapshot_store.acquire()
    old_fork = old_lease.fork()
    token = bind_runtime_snapshot(old_lease)
    try:
        assert await tools.execute("mcp_workspace__version_v1", {}) == "[]"
        second = await manager.prepare_workspace_mcp(
            _workspace_mcp_spec(tmp_path / "workspace-v2", tool_name="version_v2"),
            revision="v2",
        )
        await manager.publish_workspace_mcp()
        assert manager.active_workspace_mcp is second
        assert old_client.connected is True
        assert await tools.execute("mcp_workspace__version_v1", {}) == "[]"
        assert tools.has_tool("mcp_workspace__version_v2") is False
        new_lease = await manager.snapshot_store.acquire()
        new_token = bind_runtime_snapshot(new_lease)
        try:
            assert await tools.execute("mcp_workspace__version_v2", {}) == "[]"
        finally:
            reset_runtime_snapshot(new_token)
            await new_lease.release()
    finally:
        reset_runtime_snapshot(token)
        await old_lease.release()

    assert old_client.connected is True
    await old_fork.release()
    await manager.snapshot_store.retry_drains()
    assert old_client.connected is False
    new_lease = await manager.snapshot_store.acquire()
    token = bind_runtime_snapshot(new_lease)
    try:
        assert tools.has_tool("mcp_workspace__version_v1") is False
        assert await tools.execute("mcp_workspace__version_v2", {}) == "[]"
    finally:
        reset_runtime_snapshot(token)
        await new_lease.release()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_workspace_mcp_candidate_failure_keeps_active_and_cleans_partial(
    tmp_path: Path,
) -> None:
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    first = await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "active", tool_name="active"),
        revision="active",
    )
    await manager.publish_workspace_mcp()
    current = manager.current_snapshot

    good_dir = tmp_path / "candidate-good"
    good_spec = _workspace_mcp_spec(good_dir, tool_name="candidate")
    specs = {
        "a_good": next(iter(good_spec.values())),
        "z_bad": {"command": [str(tmp_path / "missing-command")]},
    }
    with pytest.raises(FileNotFoundError):
        await manager.prepare_workspace_mcp(specs, revision="broken")

    await _wait_for_log(
        good_dir / "data" / "mcp-lifecycle.log",
        ["started", "stopped"],
    )
    assert manager.current_snapshot is current
    assert manager.active_workspace_mcp is first
    assert manager.prepared_workspace_mcp is None
    assert first.catalog.servers["workspace"].client.connected is True

    candidate = await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "candidate-dead", tool_name="dead"),
        revision="dead",
    )
    process = candidate.catalog.servers["workspace"].client._process
    assert process is not None
    process.kill()
    await process.wait()
    with pytest.raises(RuntimeError, match="client 已断开"):
        await manager.publish_workspace_mcp()
    assert candidate.scope.closed is True
    assert manager.current_snapshot is current
    assert manager.active_workspace_mcp is first
    assert manager.prepared_workspace_mcp is None
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_workspace_mcp_and_plugin_mcp_names_conflict_both_directions(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "plugin_owner",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class PluginOwner(Plugin):\n"
        "    name = 'plugin_owner'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='shared', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(plugin_dir, ("value",))
    manager = _manager(tmp_path, tools=ToolRegistry())
    await manager.load_all()
    workspace_spec = _workspace_mcp_spec(tmp_path / "workspace", tool_name="value")
    shared_spec = {"shared": next(iter(workspace_spec.values()))}
    with pytest.raises(RuntimeError, match="名称冲突"):
        await manager.prepare_workspace_mcp(shared_spec, revision="workspace")
    await manager.terminate_all()

    other = _manager(tmp_path / "other", tools=ToolRegistry())
    active_spec = _workspace_mcp_spec(tmp_path / "active-user", tool_name="value")
    active_spec = {"shared": next(iter(active_spec.values()))}
    await other.prepare_workspace_mcp(active_spec, revision="user")
    await other.publish_workspace_mcp()
    other_plugin = _write_plugin(
        tmp_path / "other" / "plugins",
        "plugin_candidate",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class PluginCandidate(Plugin):\n"
        "    name = 'plugin_candidate'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='shared', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(other_plugin, ("value",))
    assert await other.prepare_candidate("plugin_candidate") is None
    with pytest.raises(RuntimeError, match="workspace MCP 与插件声明冲突"):
        other.assert_no_workspace_mcp_plugin_conflicts()
    await other.terminate_all()


@pytest.mark.asyncio
async def test_plugin_publish_rebases_latest_workspace_mcp_generation(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, tools=ToolRegistry())
    await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "v1", tool_name="v1"), revision="v1"
    )
    await manager.publish_workspace_mcp()
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "late_plugin",
        "from agent.plugins import Plugin\n"
        "class LatePlugin(Plugin):\n"
        "    name = 'late_plugin'\n",
    )
    candidate = await manager.prepare_candidate("late_plugin")
    assert candidate is not None
    await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "v2", tool_name="v2"), revision="v2"
    )
    latest = await manager.publish_workspace_mcp()
    await manager.publish_prepared("late_plugin")
    assert manager.current_snapshot is not None
    assert manager.current_snapshot.workspace_mcp_generation is latest
    assert candidate.runtime_snapshot is manager.current_snapshot
    assert candidate.instance.context.tool_registry is manager.current_snapshot.tool_registry
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("publish_owner", ["workspace", "plugin"])
async def test_later_mcp_publish_revalidates_prepared_name_conflict(
    tmp_path: Path,
    publish_owner: str,
) -> None:
    manager = _manager(tmp_path, tools=ToolRegistry())
    workspace = _workspace_mcp_spec(tmp_path / "workspace", tool_name="value")
    await manager.prepare_workspace_mcp(
        {"shared": next(iter(workspace.values()))}, revision="workspace"
    )
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "prepared_plugin",
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class PreparedPlugin(Plugin):\n"
        "    name = 'prepared_plugin'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='shared', command=('python', 'server.py'))]\n",
    )
    _write_mcp_server(plugin_dir, ("value",))
    plugin = await manager.prepare_candidate("prepared_plugin")
    assert plugin is not None
    current = manager.current_snapshot
    with pytest.raises(RuntimeError, match="名称冲突"):
        if publish_owner == "workspace":
            await manager.publish_workspace_mcp()
        else:
            await manager.publish_prepared("prepared_plugin")
    assert manager.current_snapshot is current
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_workspace_mcp_publish_cancellation_aborts_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path, tools=ToolRegistry())
    first = await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "active-cancel", tool_name="active"),
        revision="active",
    )
    await manager.publish_workspace_mcp()
    current = manager.current_snapshot
    candidate = await manager.prepare_workspace_mcp(
        _workspace_mcp_spec(tmp_path / "candidate-cancel", tool_name="candidate"),
        revision="candidate",
    )
    entered = asyncio.Event()

    async def wait_invariant(snapshot: RuntimeSnapshot) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(manager, "_post_snapshot_invariants", wait_invariant)
    task = asyncio.create_task(manager.publish_workspace_mcp())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.current_snapshot is current
    assert manager.snapshot_store._pending is None
    assert manager.active_workspace_mcp is first
    assert manager.prepared_workspace_mcp is None
    assert candidate.scope.closed is True
    assert candidate.catalog.servers["workspace"].client.connected is False
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_prepared_plugin_mcp_namespace_rejects_second_candidate_early(
    tmp_path: Path,
) -> None:
    source = lambda class_name, name: (
        "from agent.plugins import McpServerSpec, Plugin\n"
        f"class {class_name}(Plugin):\n"
        f"    name = '{name}'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='shared_plugin', command=('python', 'server.py'))]\n"
    )
    for name, class_name in (("owner_a", "OwnerA"), ("owner_b", "OwnerB")):
        plugin_dir = _write_plugin(
            tmp_path / "plugins", name, source(class_name, name)
        )
        _write_mcp_server(plugin_dir, ("value",))
    manager = _manager(tmp_path, tools=ToolRegistry())
    assert await manager.prepare_candidate("owner_a") is not None
    assert await manager.prepare_candidate("owner_b") is None
    assert manager.prepared_generation("owner_a") is not None
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_plugin_publish_rebase_conflict_discards_stale_candidate(
    tmp_path: Path,
) -> None:
    def source(class_name: str, name: str) -> str:
        return (
            "from agent.plugins import McpServerSpec, Plugin\n"
            f"class {class_name}(Plugin):\n"
            f"    name = '{name}'\n"
            "    @classmethod\n"
            "    def mcp_servers(cls):\n"
            "        return [McpServerSpec(name='shared_plugin', command=('python', 'server.py'))]\n"
        )

    for name, class_name in (("owner_a", "OwnerA"), ("owner_b", "OwnerB")):
        plugin_dir = _write_plugin(tmp_path / "plugins", name, source(class_name, name))
        _write_mcp_server(plugin_dir, ("value",))
    manager = _manager(tmp_path, tools=ToolRegistry())
    stale = await manager.prepare_candidate("owner_b")
    assert stale is not None and stale.mcp_catalog is not None
    _ = manager._prepared_generations.pop("owner_b")
    owner = await manager.prepare_candidate("owner_a")
    assert owner is not None
    await manager.publish_prepared("owner_a")
    current = manager.current_snapshot
    manager._prepared_generations["owner_b"] = stale
    client = stale.mcp_catalog.servers["shared_plugin"].client
    with pytest.raises(_CandidateRejected):
        await manager.publish_prepared("owner_b")
    assert manager.prepared_generation("owner_b") is None
    assert client.connected is False
    assert stale.scope.closed is True
    assert manager.current_snapshot is current
    assert manager.generation("owner_a") is owner
    assert manager.generation("owner_b") is None
    assert manager.latest_gate("owner_b").checks[0].check_id == "publish_rebase"
    await manager.terminate_all()


def _snapshot_event_source(version: str) -> str:
    wait = (
        "        self.context.kv_store.set('event_entered_v1', True)\n"
        "        release = self.context.data_dir / 'release-event-v1'\n"
        "        while not release.exists():\n"
        "            await asyncio.sleep(0.01)\n"
        if version == "v1"
        else ""
    )
    return (
        "import asyncio\n"
        "from agent.plugins import Plugin\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class SnapshotEventPlugin(Plugin):\n"
        "    name = 'snapshot_event'\n"
        "    async def prepare(self):\n"
        "        self.context.event_bus.on(TurnCommitted, self.handle)\n"
        "    async def handle(self, event):\n"
        f"{wait}"
        f"        self.context.kv_store.increment('event_finished_{version}')\n"
    )


@pytest.mark.asyncio
async def test_queued_event_keeps_enqueued_snapshot_generation(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_event",
        _snapshot_event_source("v1"),
    )
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_event_source("v2"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("snapshot_event")
    assert candidate is not None
    event = TurnCommitted(
        session_key="cli:event",
        channel="cli",
        chat_id="event",
        input_message="event",
        persisted_user_message="event",
        assistant_response="event",
        tools_used=[],
    )
    event_bus.enqueue(event)
    state_path = tmp_path / "workspace" / "plugin-data" / "snapshot_event-builtin" / ".kv.json"
    for _ in range(100):
        if state_path.exists():
            state: object = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(state, dict) and state.get("event_entered_v1") is True:
                break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("v1 queued event did not start")

    await manager.publish_prepared("snapshot_event")
    _ = (state_path.parent / "release-event-v1").write_text(
        "released\n",
        encoding="utf-8",
    )
    await event_bus.drain()
    await event_bus.fanout(event)
    state = cast(
        dict[str, object],
        json.loads(state_path.read_text(encoding="utf-8")),
    )

    assert state["event_finished_v1"] == 1
    assert state["event_finished_v2"] == 1
    assert event_bus.handler_count() == 0
    await manager.terminate_all()
    await event_bus.aclose()


@pytest.mark.asyncio
async def test_snapshot_event_subscription_close_takes_effect_immediately(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_event",
        "from agent.plugins import Plugin\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class SnapshotEventPlugin(Plugin):\n"
        "    name = 'snapshot_event'\n"
        "    async def prepare(self):\n"
        "        self.subscription = self.context.event_bus.on(TurnCommitted, self.handle)\n"
        "    def handle(self, event):\n"
        "        self.context.kv_store.increment('events')\n",
    )
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    generation = manager.generation("snapshot_event")
    assert generation is not None
    generation.instance.subscription.close()
    event = TurnCommitted(
        session_key="cli:event",
        channel="cli",
        chat_id="event",
        input_message="event",
        persisted_user_message="event",
        assistant_response="event",
        tools_used=[],
    )

    await event_bus.fanout(event)

    state_path = tmp_path / "workspace/plugin-data/snapshot_event-builtin/.kv.json"
    assert not state_path.exists()
    with pytest.raises(RuntimeError, match="必须在 generation 开放前"):
        generation.instance.context.event_bus.on(TurnCommitted, lambda _: None)
    await event_bus.aclose()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_event_bus_shutdown_drains_snapshot_lease_before_plugins(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_event",
        _snapshot_event_source("v1"),
    )
    event_bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )
    await manager.load_all()
    event_bus.enqueue(
        TurnCommitted(
            session_key="cli:event",
            channel="cli",
            chat_id="event",
            input_message="event",
            persisted_user_message="event",
            assistant_response="event",
            tools_used=[],
        )
    )
    state_path = tmp_path / "workspace/plugin-data/snapshot_event-builtin/.kv.json"
    for _ in range(100):
        if state_path.exists() and json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("event_entered_v1") is True:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("queued event did not start")

    closing = asyncio.create_task(event_bus.aclose())
    await asyncio.sleep(0)
    assert closing.done() is False
    _ = (state_path.parent / "release-event-v1").write_text(
        "released\n",
        encoding="utf-8",
    )
    await closing
    await manager.terminate_all()
    assert manager.snapshot_store.current is None


@pytest.mark.asyncio
async def test_event_observers_keep_snapshot_binding_in_isolated_tasks() -> None:
    event_bus = EventBus()
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    event_bus.bind_runtime_snapshot_store(store)
    seen: list[str | None] = []

    async def observe_snapshot(_event: str) -> None:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        current = get_current_runtime_snapshot()
        seen.append(current.snapshot_id if current is not None else None)

    event_bus.on(str, observe_snapshot)

    await event_bus.observe("observe")
    await event_bus.fanout("fanout")
    event_bus.enqueue("queued")
    await event_bus.drain()

    assert seen == [snapshot.snapshot_id] * 3
    await event_bus.aclose()
    await store.close()


@pytest.mark.asyncio
async def test_cancelled_fanout_releases_unstarted_observer_leases() -> None:
    event_bus = EventBus()
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    event_bus.bind_runtime_snapshot_store(store)

    async def observe(_event: str) -> None:
        await asyncio.sleep(0)

    for _ in range(20):
        event_bus.on(str, observe)

    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await event_bus.fanout("cancel")
    await asyncio.sleep(0)

    assert snapshot.lease_count == 0
    await event_bus.aclose()
    await store.close()


@pytest.mark.asyncio
async def test_running_fanout_cancellation_releases_observer_leases() -> None:
    event_bus = EventBus()
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    event_bus.bind_runtime_snapshot_store(store)
    entered = asyncio.Event()

    async def observe(_event: str) -> None:
        entered.set()
        await asyncio.Event().wait()

    event_bus.on(str, observe)
    fanout = asyncio.create_task(event_bus.fanout("cancel"))
    await entered.wait()
    _ = fanout.cancel()
    with pytest.raises(asyncio.CancelledError):
        await fanout

    assert snapshot.lease_count == 0
    await event_bus.aclose()
    await store.close()


def _snapshot_job_source(version: str, *, event_trigger: bool = False) -> str:
    triggers = "[EventTrigger(TurnCommitted)]" if event_trigger else "[]"
    return (
        "from agent.plugins import EventTrigger, Plugin, PluginJobSpec\n"
        "from agent.plugins.snapshot import get_current_runtime_snapshot\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class SnapshotJobPlugin(Plugin):\n"
        "    name = 'snapshot_job'\n"
        "    def jobs(self):\n"
        f"        return [PluginJobSpec(id='refresh', triggers={triggers}, handler=self.refresh)]\n"
        "    async def refresh(self, context):\n"
        f"        self.context.kv_store.increment('job_{version}')\n"
        "        snapshot = get_current_runtime_snapshot()\n"
        f"        self.context.kv_store.set('snapshot_{version}', snapshot.snapshot_id if snapshot else None)\n"
    )


@pytest.mark.asyncio
async def test_job_queue_envelope_keeps_enqueued_snapshot_generation(
    tmp_path: Path,
) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_job",
        _snapshot_job_source("v1"),
    )
    event_bus = EventBus()
    llm = cast(Any, SimpleNamespace())
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        llm=llm,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache",
    )
    await manager.load_all()
    runtime = PluginJobRuntime(
        event_bus=event_bus,
        llm=llm,
        snapshot_store=manager.snapshot_store,
    )
    runtime.enqueue("snapshot_job:refresh", reason="manual")
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_job_source("v2"),
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("snapshot_job") is not None
    result = await manager.publish_prepared("snapshot_job")
    assert result["publication_state"] == "committed"
    running = asyncio.create_task(runtime.run())
    state_path = tmp_path / "workspace/plugin-data/snapshot_job-builtin/.kv.json"
    for _ in range(100):
        if state_path.exists() and json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("job_v1") == 1:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("v1 job did not run")

    runtime.enqueue("snapshot_job:refresh", reason="manual")
    for _ in range(100):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("job_v2") == 1:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("v2 job did not run")

    assert state["snapshot_v1"] != state["snapshot_v2"]
    runtime.stop()
    await running
    await event_bus.aclose()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_job_event_trigger_uses_event_snapshot_catalog(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_job",
        _snapshot_job_source("v1", event_trigger=True),
    )
    event_bus = EventBus()
    llm = cast(Any, SimpleNamespace())
    manager = PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=event_bus,
        llm=llm,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home/cache",
    )
    await manager.load_all()
    old_lease = manager.snapshot_store.lease()
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_job_source("v2", event_trigger=False),
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("snapshot_job") is not None
    await manager.publish_prepared("snapshot_job")
    runtime = PluginJobRuntime(
        event_bus=event_bus,
        llm=llm,
        snapshot_store=manager.snapshot_store,
    )
    running = asyncio.create_task(runtime.run())
    await asyncio.sleep(0)
    from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

    token = bind_runtime_snapshot(old_lease)
    try:
        await event_bus.fanout(
            TurnCommitted(
                session_key="cli:event",
                channel="cli",
                chat_id="event",
                input_message="event",
                persisted_user_message="event",
                assistant_response="event",
                tools_used=[],
            )
        )
    finally:
        reset_runtime_snapshot(token)
        await old_lease.release()
    state_path = tmp_path / "workspace/plugin-data/snapshot_job-builtin/.kv.json"
    for _ in range(100):
        if state_path.exists() and json.loads(
            state_path.read_text(encoding="utf-8")
        ).get("job_v1") == 1:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("v1 event job did not run")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state.get("job_v2") is None
    runtime.stop()
    await running
    await event_bus.aclose()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_proactive_tick_keeps_one_snapshot_generation(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_tick",
        "from agent.plugins import Plugin\n"
        "class SnapshotTickPlugin(Plugin):\n"
        "    name = 'snapshot_tick'\n"
        "    version = 'v1'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str] = []

    async def run_tick(_session_key: str) -> None:
        from agent.plugins.snapshot import get_current_runtime_snapshot

        snapshot = get_current_runtime_snapshot()
        assert snapshot is not None
        seen.append(snapshot.snapshot_id)
        if len(seen) == 1:
            entered.set()
            await release.wait()

    loop = object.__new__(ProactiveLoop)
    loop._runtime_snapshot_store = manager.snapshot_store
    loop._reload_lock = asyncio.Lock()
    loop._sense = SimpleNamespace(target_session_key=lambda: "cli:tick")
    loop._proactive_kernel = SimpleNamespace(run_tick=run_tick)

    async def switch_snapshot(_snapshot) -> None:
        return None

    loop._switch_snapshot = switch_snapshot
    first_tick = asyncio.create_task(loop._tick())
    await entered.wait()
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class SnapshotTickPlugin(Plugin):\n"
        "    name = 'snapshot_tick'\n"
        "    version = 'v2'\n",
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("snapshot_tick") is not None
    await manager.publish_prepared("snapshot_tick")
    new_snapshot = manager.current_snapshot
    assert new_snapshot is not None
    release.set()
    await first_tick
    await loop._tick()

    assert seen == [old_snapshot.snapshot_id, new_snapshot.snapshot_id]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_proactive_kernel_owns_snapshot_until_stopped(tmp_path: Path) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "snapshot_kernel",
        "from agent.plugins import Plugin\n"
        "class SnapshotKernelPlugin(Plugin):\n"
        "    name = 'snapshot_kernel'\n",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    kernel = SimpleNamespace(stop=AsyncMock())
    loop = object.__new__(ProactiveLoop)
    loop._kernel_started = False
    loop._active_kernel_lease = None
    loop._active_snapshot_id = None

    async def build_and_start(_snapshot, _lease):
        return kernel

    loop._build_and_start_kernel = build_and_start
    from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

    admission = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(admission)
    try:
        await loop._switch_snapshot(snapshot)
    finally:
        reset_runtime_snapshot(token)
        await admission.release()

    assert snapshot.lease_count == 1
    await loop._stop_active_kernel()
    assert snapshot.lease_count == 0
    kernel.stop.assert_awaited_once()
    await manager.terminate_all()


def _snapshot_hook_source(version: str) -> str:
    return (
        "from agent.plugins import Plugin, on_tool_pre\n"
        "class SnapshotHookPlugin(Plugin):\n"
        "    name = 'snapshot_hook'\n"
        "    @on_tool_pre(tool_name='target')\n"
        "    async def hook(self, event):\n"
        f"        return {{**event.arguments, 'version': '{version}'}}\n"
    )


@pytest.mark.asyncio
async def test_tool_hooks_follow_bound_snapshot_generation(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_hook",
        _snapshot_hook_source("v1"),
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    old_lease = manager.snapshot_store.lease()
    _ = (plugin_dir / "plugin.py").write_text(
        _snapshot_hook_source("v2"),
        encoding="utf-8",
    )
    assert await manager.prepare_candidate("snapshot_hook") is not None
    await manager.publish_prepared("snapshot_hook")
    executor = ToolExecutor()
    request = ToolExecutionRequest(
        call_id="hook",
        tool_name="target",
        arguments={},
        source="passive",
    )
    from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot

    async def execute(lease):
        token = bind_runtime_snapshot(lease)
        try:
            return await executor.execute(request, lambda _name, args: asyncio.sleep(0, result=args))
        finally:
            reset_runtime_snapshot(token)
            await lease.release()

    old_result = await execute(old_lease)
    new_result = await execute(manager.snapshot_store.lease())

    assert old_result.final_arguments == {"version": "v1"}
    assert new_result.final_arguments == {"version": "v2"}
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_subagent_shutdown_releases_unstarted_snapshot_lease() -> None:
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({})
    store.install(snapshot)
    snapshot_lease = store.lease()
    manager = object.__new__(SubagentManager)
    manager._running_tasks = {}
    manager._running_jobs = {}
    manager._sync_tasks = {}
    manager._cancel_announced = set()
    manager._snapshot_release_tasks = set()
    released: list[bool] = []
    admission_lease = SimpleNamespace(release=lambda: released.append(True))

    async def wait_forever() -> None:
        _ = await asyncio.Event().wait()

    task = asyncio.create_task(wait_forever())
    manager._running_tasks["job"] = task
    task.add_done_callback(
        lambda done: manager._finish_background_job(
            "job",
            snapshot_lease,
            cast(Any, admission_lease),
            done,
        )
    )

    await manager.shutdown()

    assert snapshot.lease_count == 0
    assert released == [True]
    await store.close()


@pytest.mark.asyncio
async def test_dashboard_routes_follow_snapshot_generation(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "snapshot_dashboard",
        "from agent.plugins import Plugin\n"
        "class SnapshotDashboardPlugin(Plugin):\n"
        "    name = 'snapshot_dashboard'\n"
        "    @classmethod\n"
        "    def dashboard_module(cls): return 'dashboard.py'\n",
    )

    def write_dashboard(version: str) -> None:
        _ = (plugin_dir / "dashboard.py").write_text(
            "from fastapi import FastAPI\n"
            "def register(app: FastAPI, plugin_dir, workspace):\n"
            "    @app.get('/api/dashboard/snapshot-version')\n"
            f"    def version(): return {{'version': '{version}'}}\n"
            "    class Closeable:\n"
            "        def close(self):\n"
            f"            (workspace / 'dashboard-{version}-closed').write_text('closed')\n"
            "    return Closeable()\n",
            encoding="utf-8",
        )

    write_dashboard("v1")
    manager = _manager(tmp_path)
    await manager.load_all()
    old_snapshot = manager.current_snapshot
    assert old_snapshot is not None
    old_generation = old_snapshot.generations["snapshot_dashboard"]
    old_lease = manager.snapshot_store.lease()
    app = create_dashboard_app(
        tmp_path / "workspace",
        memory_admin=cast(Any, SimpleNamespace()),
        plugin_manager=manager,
    )
    client = TestClient(app)
    assert client.get("/api/dashboard/snapshot-version").json() == {"version": "v1"}
    write_dashboard("v2")
    assert await manager.prepare_candidate("snapshot_dashboard") is not None
    await manager.publish_prepared("snapshot_dashboard")

    assert client.get("/api/dashboard/snapshot-version").json() == {"version": "v2"}
    old_binding = old_snapshot.dashboard_bindings[0]
    assert TestClient(old_binding.app).get(  # type: ignore[attr-defined]
        "/api/dashboard/snapshot-version"
    ).json() == {"version": "v1"}
    assert not (tmp_path / "workspace" / "dashboard-v1-closed").exists()
    await old_lease.release()
    await manager.snapshot_store.retry_drains()
    assert (tmp_path / "workspace" / "dashboard-v1-closed").exists()
    assert old_generation.scope.closed
    assert f"{old_generation.module_path}.dashboard" not in sys.modules
    client.close()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_dashboard_candidate_cannot_override_core_route(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "dashboard_conflict",
        "from agent.plugins import Plugin\n"
        "class DashboardConflictPlugin(Plugin):\n"
        "    name = 'dashboard_conflict'\n"
        "    @classmethod\n"
        "    def dashboard_module(cls): return 'dashboard.py'\n",
    )

    def write_dashboard(path: str, *, async_close: bool = False) -> None:
        cleanup = (
            "    class Closeable:\n"
            "        async def close(self):\n"
            "            (workspace / 'async-dashboard-closed').write_text('closed')\n"
            "    return Closeable()\n"
            if async_close
            else ""
        )
        _ = (plugin_dir / "dashboard.py").write_text(
            "def register(app, plugin_dir, workspace):\n"
            f"    @app.get('{path}')\n"
            "    def route(): return {'owner': 'plugin'}\n"
            f"{cleanup}",
            encoding="utf-8",
        )

    write_dashboard("/api/dashboard/plugin-owned")
    manager = _manager(tmp_path)
    await manager.load_all()
    old_generation = manager.generation("dashboard_conflict")
    app = create_dashboard_app(
        tmp_path / "workspace",
        memory_admin=cast(Any, SimpleNamespace()),
        plugin_manager=manager,
    )
    write_dashboard("/api/dashboard/sessions", async_close=True)
    assert await manager.prepare_candidate("dashboard_conflict") is not None

    result = await manager.publish_prepared("dashboard_conflict")

    assert result["publication_state"] == "failed"
    assert manager.generation("dashboard_conflict") is old_generation
    assert (tmp_path / "workspace" / "async-dashboard-closed").exists()
    assert TestClient(app).get("/api/dashboard/plugin-owned").json() == {
        "owner": "plugin"
    }
    write_dashboard("/api/dashboard/{rest:path}")
    assert await manager.prepare_candidate("dashboard_conflict") is not None
    wildcard_result = await manager.publish_prepared("dashboard_conflict")
    assert wildcard_result["publication_state"] == "failed"
    assert manager.generation("dashboard_conflict") is old_generation
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_dashboard_candidate_cannot_override_other_plugin(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    first_dir = _write_plugin(
        root,
        "dashboard_first",
        "from agent.plugins import Plugin\n"
        "class DashboardFirstPlugin(Plugin):\n"
        "    name = 'dashboard_first'\n"
        "    @classmethod\n"
        "    def dashboard_module(cls): return 'dashboard.py'\n",
    )
    second_dir = _write_plugin(
        root,
        "dashboard_second",
        "from agent.plugins import Plugin\n"
        "class DashboardSecondPlugin(Plugin):\n"
        "    name = 'dashboard_second'\n"
        "    @classmethod\n"
        "    def dashboard_module(cls): return 'dashboard.py'\n",
    )

    def write_dashboard(directory: Path, path: str) -> None:
        _ = (directory / "dashboard.py").write_text(
            "def register(app, plugin_dir, workspace):\n"
            f"    @app.get('{path}')\n"
            f"    def route(): return {{'owner': '{directory.name}'}}\n",
            encoding="utf-8",
        )

    write_dashboard(first_dir, "/api/dashboard/items/{id}")
    write_dashboard(second_dir, "/api/dashboard/second")
    manager = _manager(tmp_path)
    await manager.load_all()
    old_generation = manager.generation("dashboard_second")
    _ = create_dashboard_app(
        tmp_path / "workspace",
        memory_admin=cast(Any, SimpleNamespace()),
        plugin_manager=manager,
    )
    write_dashboard(second_dir, "/api/dashboard/items/{name}")
    assert await manager.prepare_candidate("dashboard_second") is not None

    result = await manager.publish_prepared("dashboard_second")

    assert result["publication_state"] == "failed"
    assert manager.generation("dashboard_second") is old_generation
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_dashboard_allows_static_route_before_dynamic_route(tmp_path: Path) -> None:
    plugin_dir = _write_plugin(
        tmp_path / "plugins",
        "dashboard_ordered",
        "from agent.plugins import Plugin\n"
        "class DashboardOrderedPlugin(Plugin):\n"
        "    name = 'dashboard_ordered'\n"
        "    @classmethod\n"
        "    def dashboard_module(cls): return 'dashboard.py'\n",
    )
    _ = (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace):\n"
        "    @app.get('/api/dashboard/items/overview')\n"
        "    def overview(): return {'route': 'overview'}\n"
        "    @app.get('/api/dashboard/items/{item_id}')\n"
        "    def detail(item_id): return {'route': item_id}\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path)
    await manager.load_all()
    app = create_dashboard_app(
        tmp_path / "workspace",
        memory_admin=cast(Any, SimpleNamespace()),
        plugin_manager=manager,
    )
    client = TestClient(app)

    assert client.get("/api/dashboard/items/overview").json() == {
        "route": "overview"
    }
    assert client.get("/api/dashboard/items/42").json() == {"route": "42"}
    client.close()
    await manager.terminate_all()


def test_dashboard_rejects_custom_path_convertor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CustomConvertor(StringConvertor):
        regex = "(?:x|z)"

    monkeypatch.setitem(CONVERTOR_TYPES, "custom_gate", CustomConvertor())
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/dashboard/{value:custom_gate}")
    def route() -> dict[str, bool]:
        return {"ok": True}

    with pytest.raises(RuntimeError, match="内建 path converter"):
        _plugin_routes(app.routes)


@pytest.mark.parametrize("wildcard_methods", [None, set()])
def test_dashboard_treats_missing_methods_as_wildcard(
    wildcard_methods: set[str] | None,
) -> None:
    core_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    plugin_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @core_app.api_route("/api/dashboard/{rest:path}", methods=["GET"])
    def core_route() -> dict[str, bool]:
        return {"core": True}

    @plugin_app.get("/api/dashboard/sessions")
    def plugin_route() -> dict[str, bool]:
        return {"plugin": True}

    core_routes = _plugin_routes(core_app.routes)
    core_routes[0].methods = wildcard_methods
    binding = DashboardBinding(
        plugin_id="wildcard",
        app=plugin_app,
        routes=_plugin_routes(plugin_app.routes),
    )

    with pytest.raises(RuntimeError, match="dashboard route 冲突"):
        _require_routes_available(binding, list(core_routes))


@pytest.mark.asyncio
async def test_skill_body_stays_on_snapshot_generation(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "snapshot_skill"
    for version in ("v1", "v2"):
        skill_dir = plugin_dir / f"skills-{version}" / "snapshot-skill"
        skill_dir.mkdir(parents=True)
        _ = (skill_dir / "SKILL.md").write_text(
            f"---\ndescription: snapshot skill {version}\n---\nbody {version}\n",
            encoding="utf-8",
        )

    def source(version: str) -> str:
        return (
            "from agent.plugins import Plugin\n"
            "class SnapshotSkillPlugin(Plugin):\n"
            "    name = 'snapshot_skill'\n"
            "    @classmethod\n"
            "    def skill_roots(cls):\n"
            f"        return ('skills-{version}',)\n"
        )

    _ = (plugin_dir / "plugin.py").write_text(source("v1"), encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    workspace_skills = workspace / "skills"
    workspace_skills.mkdir()
    (workspace_skills / "snapshot-skill").symlink_to(
        plugin_dir / "skills-v1" / "snapshot-skill",
        target_is_directory=True,
    )
    _ = (plugin_dir / "plugin.py").write_text(source("v2"), encoding="utf-8")
    candidate = await manager.prepare_candidate("snapshot_skill")
    assert candidate is not None
    skills = SkillsLoader(workspace, runtime_catalog="normal")
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = manager.snapshot_store
    entered = asyncio.Event()
    release = asyncio.Event()
    seen: list[str | None] = []

    async def process(msg, **kwargs):
        seen.append(skills.load_skill_body("snapshot-skill"))
        entered.set()
        await release.wait()
        seen.append(skills.load_skill_body("snapshot-skill"))
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:snapshot-skill"))
    old_turn = asyncio.create_task(loop._process_with_runtime_admission(message))
    await entered.wait()
    await manager.publish_prepared("snapshot_skill")
    _ = (plugin_dir / "skills-v1" / "snapshot-skill" / "SKILL.md").write_text(
        "---\ndescription: mutated source\n---\nmutated v1\n",
        encoding="utf-8",
    )
    release.set()
    assert await old_turn == "done"
    await loop._process_with_runtime_admission(message)

    assert seen[:2] == ["body v1", "body v1"]
    assert seen[2:] == ["body v2", "body v2"]
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_workspace_skill_updates_without_plugin_snapshot_reload(
    tmp_path: Path,
) -> None:
    _write_plugin(
        tmp_path / "plugins",
        "workspace_skill_snapshot",
        "from agent.plugins import Plugin\n"
        "class WorkspaceSkillSnapshotPlugin(Plugin):\n"
        "    name = 'workspace_skill_snapshot'\n",
    )
    workspace = tmp_path / "workspace"
    skill_dir = workspace / "skills" / "workspace-live"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    _ = skill_file.write_text(
        "---\ndescription: workspace live\n---\nworkspace v1\n",
        encoding="utf-8",
    )
    manager = _manager(tmp_path, workspace=workspace)
    await manager.load_all()
    snapshot = manager.current_snapshot
    skills = SkillsLoader(workspace, runtime_catalog="normal")
    loop = object.__new__(AgentLoop)
    loop._session_lanes = SessionLaneRegistry()
    loop._runtime_snapshot_store = manager.snapshot_store
    seen: list[str | None] = []

    async def process(msg, **kwargs):
        seen.append(skills.load_skill_body("workspace-live"))
        return "done"

    loop._process = process
    message = cast(Any, SimpleNamespace(session_key="cli:workspace-skill"))
    await loop._process_with_runtime_admission(message)
    _ = skill_file.write_text(
        "---\ndescription: workspace live\n---\nworkspace v2\n",
        encoding="utf-8",
    )
    await loop._process_with_runtime_admission(message)

    assert manager.current_snapshot is snapshot
    assert seen == ["workspace v1", "workspace v2"]
    await manager.terminate_all()
