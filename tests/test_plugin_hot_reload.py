from __future__ import annotations

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


def _manager(tmp_path: Path, *, tools: ToolRegistry | None = None) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=tools,
        installed_cache_root=tmp_path / "home" / "cache",
    )


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
    assert second.generation_id != first.generation_id
    assert second.module_path != first.module_path
