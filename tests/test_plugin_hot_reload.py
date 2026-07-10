from __future__ import annotations

import asyncio
import importlib
import os
import py_compile
import shutil
import sys
from pathlib import Path

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus


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
        workspace=workspace,
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
        f"        if method == 'tools/list': result = {{'tools': [{tool_items}]}}\n"
        "        elif method == 'tools/call': result = {'content': [{'type': 'text', 'text': '[]'}]}\n"
        "        print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n"
        "finally:\n"
        "    with log.open('a', encoding='utf-8') as f: f.write('stopped\\n')\n",
        encoding="utf-8",
    )


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
        "    async def initialize(self):\n"
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
            "    api_version = 2\n",
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
async def test_failed_candidate_never_initializes(
    tmp_path: Path,
    name: str,
    source: str,
    failed_check: str,
):
    plugin_dir = _write_plugin(tmp_path / "plugins", name, source)
    marker = plugin_dir / "initialized"
    with (plugin_dir / "plugin.py").open("a", encoding="utf-8") as file:
        _ = file.write(
            "    async def initialize(self):\n"
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
    kv_path = tmp_path / "home" / "data" / "readonly-builtin" / ".kv.json"
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
async def test_initialize_failure_replaces_passed_gate(tmp_path: Path):
    _write_plugin(
        tmp_path / "plugins",
        "init_failure",
        "from agent.plugins import Plugin\n"
        "class InitFailurePlugin(Plugin):\n"
        "    name = 'init_failure'\n"
        "    async def initialize(self): raise RuntimeError('init failed')\n",
    )
    manager = _manager(tmp_path)

    await manager.load_all()

    gate = manager.latest_gate("init_failure")
    assert gate is not None and gate.status == "failed"
    assert gate.checks[0].check_id == "initialize"
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
    config_dir = tmp_path / "home" / "data" / "module_tree-builtin"
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
async def test_candidate_is_not_published_before_initialize_finishes(tmp_path: Path):
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
        "    async def initialize(self):\n"
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
    assert event_bus.handler_count() == 1


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
        "    async def initialize(self):\n"
        "        self.context.kv_store.set('initialized_v2', True)\n",
        encoding="utf-8",
    )
    prepared = await manager.prepare_candidate("replaceable")

    assert prepared is not None and prepared.state == "prepared"
    assert manager.generation("replaceable") is active
    assert manager.prepared_generation("replaceable") is prepared
    assert manager.latest_gate("replaceable").status == "passed"  # type: ignore[union-attr]
    assert not (
        tmp_path / "home" / "data" / "replaceable-builtin" / ".kv.json"
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
async def test_return_to_active_revision_discards_stale_prepared(tmp_path: Path):
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

    def fail_snapshot_cleanup(path: Path, *args: object, **kwargs: object) -> None:
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
        "from agent.plugins import McpServerSpec, Plugin, ProactiveSourceSpec\n"
        "class McpReadyPlugin(Plugin):\n"
        "    name = 'mcp_ready'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='feed', command=('python', 'server.py'))]\n"
        "    def proactive_sources(self):\n"
        "        return [ProactiveSourceSpec(id='feed', channels=('content',), "
        "server='feed', fetch_tool='fetch_events', ack_tool='ack_events', "
        "poll_tool='poll_events')]\n",
    )
    _write_mcp_server(
        plugin_dir,
        ("fetch_events", "ack_events", "poll_events"),
    )
    tools = ToolRegistry()
    manager = _manager(tmp_path, tools=tools)
    await manager.load_all()

    prepared = await manager.prepare_candidate("mcp_ready")

    assert prepared is not None and prepared.mcp_catalog is not None
    catalog = prepared.mcp_catalog
    server = catalog.servers["feed"]
    assert server.client.name == f"feed@{prepared.generation_id}"
    assert catalog.tool_names == (
        "mcp_feed__ack_events",
        "mcp_feed__fetch_events",
        "mcp_feed__poll_events",
    )
    assert not any(prepared.generation_id in name for name in catalog.tool_names)
    assert tools.get_registered_names() == set()
    assert await server.tools[1].execute() == "[]"
    lifecycle = tmp_path / "home" / "data" / "mcp_ready-builtin" / "mcp-lifecycle.log"
    await _wait_for_log(lifecycle, ["started"])

    await manager.discard_prepared("mcp_ready")

    await _wait_for_log(lifecycle, ["started", "stopped"])
    assert manager.mcp_catalog(prepared.generation_id) is None
    assert server.client._process is None
    assert server.client._stderr_task is None
    await manager.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing_tool", "semantic"])
async def test_candidate_mcp_readiness_failure_closes_process(
    tmp_path: Path,
    failure: str,
) -> None:
    readiness = (
        "    def readiness_semantic_checks(self):\n"
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
    manager = _manager(tmp_path)
    await manager.load_all()

    prepared = await manager.prepare_candidate("mcp_rejected")

    assert prepared is None
    assert manager.prepared_generation("mcp_rejected") is None
    gate = manager.latest_gate("mcp_rejected")
    assert gate is not None and gate.status == "failed"
    failed_checks = {check.check_id for check in gate.checks if check.status == "failed"}
    assert (
        "mcp_readiness" if failure == "missing_tool" else "readiness_semantic_checks"
    ) in failed_checks
    lifecycle = tmp_path / "home" / "data" / "mcp_rejected-builtin" / "mcp-lifecycle.log"
    await _wait_for_log(lifecycle, ["started", "stopped"])
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
    lifecycle = tmp_path / "home" / "data" / "mcp_handshake-builtin" / "mcp-lifecycle.log"
    await _wait_for_log(lifecycle, ["started", "stopped"])
    await manager.terminate_all()
