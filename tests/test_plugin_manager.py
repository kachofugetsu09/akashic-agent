from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# 预热 agent.core 导入链，避免 agent.lifecycle.types 触发循环导入
from agent.core.passive_turn import ContextStore as _  # noqa: F401
from agent.lifecycle.types import AfterStepCtx, AfterToolResultCtx, BeforeToolCallCtx, BeforeTurnCtx
from agent.plugins.manager import PluginManager
from agent.plugins.jobs import PluginJobRuntime
from agent.plugins.registry import plugin_registry
from agent.tool_hooks import ToolHook
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.events import MemoryWritten, RetrievalCompleted, RetrievalHitSummary
from proactive_v2.lifecycle import ProactiveLifecycleSpec

# ── fixtures ──────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "plugins"
TEST_PLUGIN_HOME = Path(tempfile.gettempdir()) / f"akasic-plugin-tests-{os.getpid()}"


@pytest.fixture(autouse=True)
def _clean_registry():
    # 每个测试前后清空全局 registry，避免插件状态跨测试污染
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    shutil.rmtree(TEST_PLUGIN_HOME, ignore_errors=True)
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    shutil.rmtree(TEST_PLUGIN_HOME, ignore_errors=True)


def _make_manager(plugin_dirs: list[Path], *, event_bus: EventBus, tools: ToolRegistry | None = None) -> PluginManager:
    return PluginManager(
        plugin_dirs=plugin_dirs,
        event_bus=event_bus,
        tool_registry=tools,
        installed_cache_root=TEST_PLUGIN_HOME / "cache",
    )


class _FakePluginLlm:
    async def generate_text(self, **kwargs: Any) -> str:
        return f"generated:{kwargs.get('prompt')}"


def _before_turn_ctx(**overrides: object) -> BeforeTurnCtx:
    defaults: dict = dict(
        session_key="test:123",
        channel="cli",
        chat_id="123",
        content="hello",
        timestamp=datetime.now(),
        retrieved_memory_block="",
        retrieval_trace_raw=None,
        history_messages=(),
    )
    defaults.update(overrides)
    return BeforeTurnCtx(**defaults)


def _after_step_ctx(**overrides: object) -> AfterStepCtx:
    defaults: dict = dict(
        session_key="test:123",
        channel="cli",
        chat_id="123",
        iteration=0,
        context_tokens_estimate=0,
        tools_called=(),
        partial_reply="",
        tools_used_so_far=(),
        tool_chain_partial=(),
        partial_thinking=None,
        has_more=False,
    )
    defaults.update(overrides)
    return AfterStepCtx(**defaults)


# ── 加载测试 ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_hello_plugin():
    bus = EventBus()
    mgr = _make_manager([FIXTURES_DIR], event_bus=bus)
    await mgr.load_all()
    assert mgr.loaded_count >= 1
    loaded_names = {m["name"] for m in mgr.discover()}
    assert "hello" in loaded_names


@pytest.mark.asyncio
async def test_load_default_proactive_lifecycle():
    bus = EventBus()
    plugin_dir = Path(__file__).parents[1] / "plugins" / "default_proactive"
    mgr = _make_manager([plugin_dir], event_bus=bus)

    await mgr.load_all()

    assert len(mgr.proactive_lifecycles) == 1
    lifecycle = mgr.proactive_lifecycles[0]
    assert isinstance(lifecycle, ProactiveLifecycleSpec)
    assert lifecycle.id == "default"
    assert len(mgr.proactive_module_factories) == 1
    assert len(mgr.proactive_runtime_factories) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin_name", ["proactive_flow", "drift_flow"])
async def test_load_proactive_flow_factory(plugin_name: str):
    bus = EventBus()
    plugin_dir = Path(__file__).parents[1] / "plugins" / plugin_name
    mgr = _make_manager([plugin_dir], event_bus=bus)

    await mgr.load_all()

    assert len(mgr.proactive_module_factories) == 1


@pytest.mark.asyncio
async def test_builtin_plugins_complete_default_proactive_lifecycle():
    bus = EventBus()
    plugin_root = Path(__file__).parents[1] / "plugins"
    mgr = _make_manager([plugin_root], event_bus=bus)

    await mgr.load_all()

    assert len(mgr.proactive_lifecycles) == 1
    assert len(mgr.proactive_runtime_factories) == 1
    assert len(mgr.proactive_module_factories) == 3


@pytest.mark.asyncio
async def test_duplicate_plugin_name_first_wins():
    # 同名插件目录放两份，second 应被跳过
    bus = EventBus()
    mgr = _make_manager([FIXTURES_DIR, FIXTURES_DIR], event_bus=bus)
    await mgr.load_all()
    # discover 跨两个同名目录，seen_names 跨目录共享 → 只加载一次
    assert mgr.loaded_count == len({m["name"] for m in mgr.discover()})


@pytest.mark.asyncio
async def test_installed_plugin_shadows_builtin_with_same_name(tmp_path: Path):
    builtin_root = tmp_path / "plugins"
    installed_root = tmp_path / "cache"
    builtin_plugin = builtin_root / "shadow"
    installed_plugin = installed_root / "github" / "shadow" / "0.1.0"
    for plugin_dir in (builtin_plugin, installed_plugin):
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text(
            "from agent.plugins import Plugin\n"
            "class ShadowPlugin(Plugin):\n"
            "    name = 'shadow'\n",
            encoding="utf-8",
        )
    bus = EventBus()
    mgr = PluginManager(
        plugin_dirs=[builtin_root],
        installed_cache_root=installed_root,
        event_bus=bus,
    )
    mods = mgr.discover()
    assert [mod["source_type"] for mod in mods] == ["installed"]
    await mgr.load_all()
    assert [plugin.plugin_id for plugin in mgr.active_plugins()] == ["shadow@github"]


@pytest.mark.asyncio
async def test_plugin_job_runtime_runs_event_job():
    bus = EventBus()
    llm = _FakePluginLlm()
    mgr = PluginManager(
        plugin_dirs=[FIXTURES_DIR],
        event_bus=bus,
        llm=llm,
        installed_cache_root=TEST_PLUGIN_HOME / "cache",
    )
    await mgr.load_all()
    job = next(job for job in mgr.jobs if job.plugin_id == "jobber")
    runtime = PluginJobRuntime(
        event_bus=bus,
        llm=llm,
        jobs=[job],
    )
    task = asyncio.create_task(runtime.run())
    await asyncio.sleep(0)
    await bus.fanout(
        TurnCommitted(
            session_key="cli:test",
            channel="cli",
            chat_id="test",
            input_message="hi",
            persisted_user_message="hi",
            assistant_response="ok",
            tools_used=[],
        )
    )
    await asyncio.sleep(0.05)
    runtime.stop()
    await task

    assert job.plugin_context.kv_store.get("last_job") == {
        "text": "generated:hello",
        "reason": "event",
        "has_event": True,
        "context_llm": True,
    }


# ── lifecycle hook 触发测试 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_before_turn_hook_fires():
    # FIXTURES_DIR 是包含 hello/ 子目录的父目录
    bus = EventBus()
    mgr = _make_manager([FIXTURES_DIR], event_bus=bus)
    await mgr.load_all()

    ctx = _before_turn_ctx()
    result = await bus.emit(ctx)
    assert result.extra_metadata.get("hello_touched") is True


@pytest.mark.asyncio
async def test_after_step_tap_hook_fires():
    bus = EventBus()
    mgr = _make_manager([FIXTURES_DIR], event_bus=bus)
    await mgr.load_all()

    # 从已加载的 hello 模块取 after_step_calls，断言 handler 真实执行
    import sys
    hello_mod = next(
        m for k, m in sys.modules.items()
        if k.startswith("akasic_plugin_") and k.endswith("_hello")
    )
    hello_mod.after_step_calls.clear()

    ctx = _after_step_ctx(session_key="test:123")
    await bus.fanout(ctx)
    assert "test:123" in hello_mod.after_step_calls


@pytest.mark.asyncio
async def test_counter_increments_extra_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        # counter 插件写 .kv.json，用临时目录隔离
        fixture_counter = FIXTURES_DIR / "counter"
        tmp_counter = Path(tmp) / "counter"
        shutil.copytree(fixture_counter, tmp_counter)

        # 清除可能从 fixture 复制过来的残留 .kv.json
        kv = tmp_counter / ".kv.json"
        kv.unlink(missing_ok=True)

        bus = EventBus()
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        ctx1 = _before_turn_ctx()
        r1 = await bus.emit(ctx1)
        assert r1.extra_metadata["turn_count"] == 1

        ctx2 = _before_turn_ctx()
        r2 = await bus.emit(ctx2)
        assert r2.extra_metadata["turn_count"] == 2


# ── kv_store 持久化测试 ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kv_store_persists_across_manager_instances():
    with tempfile.TemporaryDirectory() as tmp:
        fixture_counter = FIXTURES_DIR / "counter"
        tmp_counter = Path(tmp) / "counter"
        shutil.copytree(fixture_counter, tmp_counter)
        (tmp_counter / ".kv.json").unlink(missing_ok=True)

        # 第一个 manager 写入
        bus1 = EventBus()
        mgr1 = _make_manager([Path(tmp)], event_bus=bus1)
        await mgr1.load_all()
        await bus1.emit(_before_turn_ctx())

        # 第二个 manager 从同路径加载，计数应继续
        plugin_registry._handlers._handlers.clear()
        plugin_registry._classes.clear()
        plugin_registry._instances.clear()

        bus2 = EventBus()
        mgr2 = _make_manager([Path(tmp)], event_bus=bus2)
        await mgr2.load_all()
        ctx = _before_turn_ctx()
        result = await bus2.emit(ctx)
        assert result.extra_metadata["turn_count"] == 2

        kv_path = TEST_PLUGIN_HOME / "data" / "counter-builtin" / ".kv.json"
        assert kv_path.exists()
        data = json.loads(kv_path.read_text())
        assert data["turn_count"] == 2


# ── 程序化身份声明测试 ────────────────────────────────────────────────────────


def _get_instance(name_or_id: str) -> Any:
    # 从 registry 按 plugin_id 或 name 找到已加载的实例
    for inst in plugin_registry._instances.values():
        if getattr(inst, "name", None) == name_or_id:
            return inst
        ctx = getattr(inst, "context", None)
        if ctx and getattr(ctx, "plugin_id", None) == name_or_id:
            return inst
    raise KeyError(f"no loaded plugin with name/id={name_or_id!r}")


@pytest.mark.asyncio
async def test_plugin_uses_class_attributes():
    bus = EventBus()
    # 用包含 manifested/ 子目录的父目录
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "manifested", Path(tmp) / "manifested")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        instance = _get_instance("manifested")
        assert instance.name == "manifested"
        assert instance.version == "1.0.0"
        assert instance.desc == "programmatic declaration"
        assert instance.author == "tester"
        assert instance.context.plugin_id == "manifested"


@pytest.mark.asyncio
async def test_active_plugins_exposes_programmatic_metadata():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "manifested", Path(tmp) / "manifested")
        mgr = _make_manager([Path(tmp)], event_bus=bus)

        await mgr.load_all()

        active = mgr.active_plugins()
        assert len(active) == 1
        assert active[0].plugin_id == "manifested"
        assert active[0].plugin_dir == Path(tmp) / "manifested"
        assert active[0].manifest["name"] == "manifested"


@pytest.mark.asyncio
async def test_loads_installed_programmatic_plugin():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp) / "cache"
        plugin_root = cache_root / "lab" / "feed" / "0.1.0"
        plugin_root.mkdir(parents=True)
        (plugin_root / "skills" / "feed-manage").mkdir(parents=True)
        (plugin_root / "skills" / "feed-manage" / "SKILL.md").write_text(
            "---\nname: feed-manage\ndescription: feed\n---\nbody\n",
            encoding="utf-8",
        )
        (plugin_root / "plugin.py").write_text(
            "from agent.plugins import McpServerSpec, Plugin\n"
            "class FeedPlugin(Plugin):\n"
            "    name = 'feed'\n"
            "    version = '1.0.0'\n"
            "    @classmethod\n"
            "    def skill_roots(cls): return ('skills',)\n"
            "    @classmethod\n"
            "    def mcp_servers(cls):\n"
            "        return [McpServerSpec(name='feed', command=('run_mcp.py',))]\n",
            encoding="utf-8",
        )
        (plugin_root / "run_mcp.py").write_text("print('ok')\n", encoding="utf-8")

        mgr = PluginManager(
            plugin_dirs=[],
            installed_cache_root=cache_root,
            event_bus=bus,
        )
        await mgr.load_all()

        active = mgr.active_plugins()
        assert len(active) == 1
        assert active[0].plugin_id == "feed@lab"
        assert active[0].skill_roots == (plugin_root / "skills",)
        assert "feed" in active[0].mcp_servers
        assert mgr.loaded_count == 1


@pytest.mark.asyncio
async def test_sync_manifest_covers_builtin_and_installed_plugins(tmp_path: Path):
    bus = EventBus()
    builtin_root = tmp_path / "plugins"
    shutil.copytree(FIXTURES_DIR / "hello", builtin_root / "hello")

    cache_root = tmp_path / "cache"
    installed_root = cache_root / "lab" / "feed" / "0.1.0"
    installed_root.mkdir(parents=True)
    (installed_root / "skills" / "feed-manage").mkdir(parents=True)
    (installed_root / "skills" / "feed-manage" / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed\n---\nbody\n",
        encoding="utf-8",
    )
    (installed_root / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class FeedPlugin(Plugin):\n"
        "    name = 'feed'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def skill_roots(cls): return ('skills',)\n",
        encoding="utf-8",
    )

    mgr = PluginManager(
        plugin_dirs=[builtin_root],
        installed_cache_root=cache_root,
        event_bus=bus,
    )
    await mgr.load_all()

    manifest_path = mgr.sync_manifest(plugins_home=tmp_path / ".akashic-plugin")
    import tomllib
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["plugins"]) == {"feed@lab", "hello"}
    assert manifest["plugins"]["feed@lab"]["enabled"] is True


@pytest.mark.asyncio
async def test_active_plugins_excludes_inactive_memory_plugin(
    tmp_path: Path,
) -> None:
    bus = EventBus()
    mgr = PluginManager(
        plugin_dirs=[Path(__file__).parents[1] / "plugins"],
        event_bus=bus,
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
        installed_cache_root=tmp_path / ".akashic-plugin" / "cache",
    )
    await mgr.load_all()

    assert {plugin.plugin_id for plugin in mgr.active_plugins()} >= {"akasha"}
    assert "default_memory" not in {plugin.plugin_id for plugin in mgr.active_plugins()}


@pytest.mark.asyncio
async def test_installed_plugin_reads_own_toml_config(tmp_path: Path):
    bus = EventBus()
    cache_root = tmp_path / "cache"
    installed_root = cache_root / "github" / "demo" / "0.1.0"
    installed_root.mkdir(parents=True)
    (installed_root / "plugin.py").write_text(
        "from pydantic import BaseModel\n"
        "from agent.plugins import Plugin\n"
        "class DemoConfig(BaseModel):\n"
        "    value: int = 0\n"
        "class DemoModule:\n"
        "    slot = 'demo.before_turn'\n"
        "    requires = ()\n"
        "    def __init__(self, value: int) -> None:\n"
        "        self.value = value\n"
        "    async def run(self, frame):\n"
        "        return frame\n"
        "class DemoPlugin(Plugin):\n"
        "    name = 'demo'\n"
        "    version = '1.0.0'\n"
        "    ConfigModel = DemoConfig\n"
        "    def before_turn_modules(self):\n"
        "        return [DemoModule(self.context.config.value)]\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data" / "demo-github"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text("value = 7\n", encoding="utf-8")

    mgr = PluginManager(
        plugin_dirs=[],
        installed_cache_root=cache_root,
        event_bus=bus,
    )
    await mgr.load_all()

    assert mgr.loaded_count == 1
    assert len(mgr.before_turn_modules) == 1
    assert getattr(mgr.before_turn_modules[0], "value") == 7


@pytest.mark.asyncio
async def test_no_manifest_uses_class_attributes():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "hello", Path(tmp) / "hello")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        instance = _get_instance("hello")
        assert instance.name == "hello"
        assert instance.version == "0.1.0"
        assert instance.context.plugin_id == "hello"


# ── 工具注册测试 ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_tool_registration():
    bus = EventBus()
    tools = ToolRegistry()
    mgr = _make_manager([FIXTURES_DIR], event_bus=bus, tools=tools)
    await mgr.load_all()

    registered = set(tools._tools.keys())
    assert "get_weather" in registered


@pytest.mark.asyncio
async def test_tool_execute_returns_string():
    bus = EventBus()
    tools = ToolRegistry()
    mgr = _make_manager([FIXTURES_DIR], event_bus=bus, tools=tools)
    await mgr.load_all()

    result = await tools.execute("get_weather", {"city": "巴黎"})
    assert "巴黎" in str(result)


@pytest.mark.asyncio
async def test_collects_before_turn_plugin_modules():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        plugin_dir = Path(tmp) / "phase_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.py").write_text(
            """
from agent.plugins import Plugin


class EarlyModule:
    requires = ("session:session",)

    async def run(self, frame):
        return frame


class LateModule:
    requires = ("session:ctx",)

    async def run(self, frame):
        return frame

class PromptTopModule:
    async def run(self, frame):
        return frame

class PromptBottomModule:
    async def run(self, frame):
        return frame

class BeforeReasoningBeforeEmitModule:
    async def run(self, frame):
        return frame

class BeforeReasoningAfterEmitModule:
    async def run(self, frame):
        return frame

class BeforeStepBeforeEmitModule:
    async def run(self, frame):
        return frame

class BeforeStepAfterEmitModule:
    async def run(self, frame):
        return frame

class AfterStepBeforeFanoutModule:
    async def run(self, frame):
        return frame

class AfterStepAfterFanoutModule:
    async def run(self, frame):
        return frame

class AfterReasoningBeforeEmitModule:
    async def run(self, frame):
        return frame

class AfterReasoningBeforePersistModule:
    async def run(self, frame):
        return frame

class AfterTurnBeforeCommitModule:
    async def run(self, frame):
        return frame

class AfterTurnBeforeFanoutModule:
    async def run(self, frame):
        return frame


class PhasePlugin(Plugin):
    name = "phase_plugin"

    def before_turn_modules(self):
        return [EarlyModule(), LateModule()]

    def before_reasoning_modules(self):
        return [BeforeReasoningBeforeEmitModule(), BeforeReasoningAfterEmitModule()]

    def prompt_render_modules(self):
        return [PromptTopModule(), PromptBottomModule()]

    def before_step_modules(self):
        return [BeforeStepBeforeEmitModule(), BeforeStepAfterEmitModule()]

    def after_step_modules(self):
        return [AfterStepBeforeFanoutModule(), AfterStepAfterFanoutModule()]

    def after_reasoning_modules(self):
        return [AfterReasoningBeforeEmitModule(), AfterReasoningBeforePersistModule()]

    def after_turn_modules(self):
        return [AfterTurnBeforeCommitModule(), AfterTurnBeforeFanoutModule()]
""".strip(),
            encoding="utf-8",
        )
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        assert [m.__class__.__name__ for m in mgr.before_turn_modules] == [
            "EarlyModule",
            "LateModule",
        ]
        assert [m.__class__.__name__ for m in mgr.before_reasoning_modules] == [
            "BeforeReasoningBeforeEmitModule",
            "BeforeReasoningAfterEmitModule",
        ]
        assert [m.__class__.__name__ for m in mgr.prompt_render_modules] == [
            "PromptTopModule",
            "PromptBottomModule",
        ]
        assert [m.__class__.__name__ for m in mgr.before_step_modules] == [
            "BeforeStepBeforeEmitModule",
            "BeforeStepAfterEmitModule",
        ]
        assert [m.__class__.__name__ for m in mgr.after_step_modules] == [
            "AfterStepBeforeFanoutModule",
            "AfterStepAfterFanoutModule",
        ]
        assert [m.__class__.__name__ for m in mgr.after_reasoning_modules] == [
            "AfterReasoningBeforeEmitModule",
            "AfterReasoningBeforePersistModule",
        ]
        assert [m.__class__.__name__ for m in mgr.after_turn_modules] == [
            "AfterTurnBeforeCommitModule",
            "AfterTurnBeforeFanoutModule",
        ]


# ── config.local.toml 测试 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_model_defaults_injected_into_context():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "configured", Path(tmp) / "configured")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()
        instance = _get_instance("configured")
        assert instance.context.config is not None
        assert instance.context.config.api_key == "test-key"
        assert instance.context.config.max_results == 10
        assert instance.context.config.enabled is True


@pytest.mark.asyncio
async def test_missing_conf_schema_leaves_config_none():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "hello", Path(tmp) / "hello")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()
        instance = _get_instance("hello")
        assert instance.context.config is None


@pytest.mark.asyncio
async def test_plugin_toml_overrides_defaults():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "configured", Path(tmp) / "configured")
        data_dir = TEST_PLUGIN_HOME / "data" / "configured-builtin"
        data_dir.mkdir(parents=True)
        (data_dir / "config.local.toml").write_text(
            'api_key = "override-key"\nenabled = false\n',
            encoding="utf-8",
        )
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()
        instance = _get_instance("configured")
        assert instance.context.config is not None
        assert instance.context.config.api_key == "override-key"   # overridden
        assert instance.context.config.max_results == 10            # still default
        assert instance.context.config.enabled is False             # overridden


@pytest.mark.asyncio
async def test_plugin_disabled_marker_skips_plugin():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "configured", Path(tmp) / "configured")
        (Path(tmp) / "configured" / "plugin.disabled").write_text("", encoding="utf-8")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        assert mgr.loaded_count == 0
        with pytest.raises(KeyError):
            _get_instance("configured")


@pytest.mark.asyncio
async def test_no_plugin_toml_keeps_model_defaults():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "configured", Path(tmp) / "configured")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()
        instance = _get_instance("configured")
        assert instance.context.config is not None
        assert instance.context.config.api_key == "test-key"       # from schema default
        assert instance.context.config.max_results == 10
        assert instance.context.config.enabled is True


# ── on_tool_call / on_tool_result 测试 ───────────────────────────────────────


def _before_tool_call_ctx(**overrides: object) -> BeforeToolCallCtx:
    defaults: dict = dict(
        session_key="test:123",
        channel="cli",
        chat_id="123",
        tool_name="get_weather",
        arguments={"city": "Tokyo"},
    )
    defaults.update(overrides)
    return BeforeToolCallCtx(**defaults)


def _after_tool_result_ctx(**overrides: object) -> AfterToolResultCtx:
    defaults: dict = dict(
        session_key="test:123",
        channel="cli",
        chat_id="123",
        tool_name="get_weather",
        arguments={"city": "Tokyo"},
        result="Tokyo: 22°C",
        status="success",
    )
    defaults.update(overrides)
    return AfterToolResultCtx(**defaults)


@pytest.mark.asyncio
async def test_on_tool_call_fires_before_tool_execution():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "audit", Path(tmp) / "audit")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        instance = _get_instance("audit")
        instance.before_tool_calls.clear()  # type: ignore[union-attr]

        await bus.fanout(_before_tool_call_ctx(tool_name="get_weather"))
        assert "get_weather" in instance.before_tool_calls  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_on_tool_result_fires_after_tool_execution():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "audit", Path(tmp) / "audit")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        instance = _get_instance("audit")
        instance.after_tool_results.clear()  # type: ignore[union-attr]

        await bus.fanout(_after_tool_result_ctx(tool_name="get_weather", status="success"))
        assert ("get_weather", "success") in instance.after_tool_results  # type: ignore[union-attr]


# ── 接线集成测试：通过真实 DefaultReasoner 触发 on_tool_call / on_tool_result ──


@pytest.mark.asyncio
async def test_tool_hooks_fire_through_real_reasoner():
    """验证 passive_turn.py 中 BeforeToolCallCtx / AfterToolResultCtx 的真实接线。

    使用 FakeLLM：第一次返回 get_weather 工具调用，第二次返回文本结束循环。
    接线删除后此测试会失败，bus.fanout 手动测试不能替代它。
    """
    from agent.core.passive_turn import DefaultReasoner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.looping.ports import LLMConfig, LLMServices
    from agent.provider import LLMResponse, ToolCall

    # 1. 构造 fake LLM provider：首轮调 get_weather，次轮返回文本
    class FakeProvider:
        _call = 0

        async def chat(self, messages, tools, model, max_tokens, **kwargs) -> LLMResponse:
            self._call += 1
            if self._call == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="get_weather", arguments={"city": "Tokyo"})],
                )
            return LLMResponse(content="Tokyo is sunny.")

    fake_provider = FakeProvider()

    # 2. 注册 audit + weather 插件，共享 bus
    bus = EventBus()
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "audit", Path(tmp) / "audit")
        shutil.copytree(FIXTURES_DIR / "weather", Path(tmp) / "weather")
        mgr = _make_manager([Path(tmp)], event_bus=bus, tools=tools)
        await mgr.load_all()

        audit = _get_instance("audit")
        audit.before_tool_calls.clear()  # type: ignore[union-attr]
        audit.after_tool_results.clear()  # type: ignore[union-attr]

        # 3. 创建 DefaultReasoner，注入同一 bus
        reasoner = DefaultReasoner(
            llm=LLMServices(provider=fake_provider, light_provider=fake_provider),  # type: ignore[arg-type]
            llm_config=LLMConfig(max_iterations=5),
            tools=tools,
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            memory_window=40,
            event_bus=bus,
        )

        # 4. 直接调用 run()，绕过 ContextBuilder / session 依赖
        await reasoner.run(
            [{"role": "user", "content": "Tokyo weather?"}],
            tool_event_session_key="test:int",
            tool_event_channel="cli",
            tool_event_chat_id="0",
        )

        # 5. 验证插件确实被触发
        assert "get_weather" in audit.before_tool_calls  # type: ignore[union-attr]
        assert any(name == "get_weather" for name, _ in audit.after_tool_results)  # type: ignore[union-attr]


# ── @on_tool_pre 插件 hook 测试 ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_on_tool_pre_rewrites_rm_to_mv():
    """加载 shell_restore 插件，执行 shell rm，断言 arguments 被改写为 mv。"""
    bus = EventBus()
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus, tools=tools)
        await mgr.load_all()

        from agent.tool_hooks.executor import ToolExecutor
        from agent.tool_hooks.types import ToolExecutionRequest
        executor = ToolExecutor(mgr.tool_hooks)

        captured: dict[str, Any] = {}

        async def fake_invoker(name: str, args: dict[str, Any]) -> str:
            captured.update(args)
            return "ok"

        req = ToolExecutionRequest(
            call_id="c1",
            tool_name="shell",
            arguments={"command": "rm /tmp/a.txt"},
            source="passive",
            session_key="test:1",
        )
        result = await executor.execute(req, fake_invoker)
        assert result.status == "success"
        assert "command" in captured
        # shlex.join 产物：mv -- <targets>... <restore_dir>
        assert captured["command"].startswith("mv -- /tmp/a.txt ")
        assert Path(shlex.split(captured["command"])[-1]).name == "restore"
        # 确认 pre_hook trace 记录了匹配
        assert any(
            item.hook_name.startswith("plugin:") and item.matched
            for item in result.pre_hook_trace
        )


@pytest.mark.asyncio
async def test_on_tool_pre_skips_non_shell_tool():
    """非 shell 工具不触发 rm→mv 改写。"""
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        from agent.tool_hooks.executor import ToolExecutor
        from agent.tool_hooks.types import ToolExecutionRequest
        executor = ToolExecutor(mgr.tool_hooks)

        captured: dict[str, Any] = {}

        async def fake_invoker(name: str, args: dict[str, Any]) -> str:
            captured.update(args)
            return "ok"

        req = ToolExecutionRequest(
            call_id="c2",
            tool_name="read",
            arguments={"file_path": "/tmp/a.txt"},
            source="passive",
            session_key="test:1",
        )
        result = await executor.execute(req, fake_invoker)
        assert captured.get("file_path") == "/tmp/a.txt"  # unchanged


@pytest.mark.asyncio
async def test_on_tool_pre_skips_non_rm_command():
    """shell echo hi 不触发改写。"""
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        from agent.tool_hooks.executor import ToolExecutor
        from agent.tool_hooks.types import ToolExecutionRequest
        executor = ToolExecutor(mgr.tool_hooks)

        captured: dict[str, Any] = {}

        async def fake_invoker(name: str, args: dict[str, Any]) -> str:
            captured.update(args)
            return "ok"

        req = ToolExecutionRequest(
            call_id="c3",
            tool_name="shell",
            arguments={"command": "echo hi"},
            source="passive",
            session_key="test:1",
        )
        result = await executor.execute(req, fake_invoker)
        assert captured.get("command") == "echo hi"  # unchanged


@pytest.mark.asyncio
async def test_on_tool_pre_rewrites_rm_rf():
    """rm -rf 带选项 → 改写。"""
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        from agent.tool_hooks.executor import ToolExecutor
        from agent.tool_hooks.types import ToolExecutionRequest
        executor = ToolExecutor(mgr.tool_hooks)

        captured: dict[str, Any] = {}

        async def fake_invoker(name: str, args: dict[str, Any]) -> str:
            captured.update(args)
            return "ok"

        req = ToolExecutionRequest(
            call_id="c",
            tool_name="shell",
            arguments={"command": "rm -rf /tmp/a.txt"},
            source="passive",
            session_key="test:1",
        )
        await executor.execute(req, fake_invoker)
        assert captured["command"].startswith("mv -- /tmp/a.txt ")
        assert Path(shlex.split(captured["command"])[-1]).name == "restore"


@pytest.mark.asyncio
async def test_on_tool_pre_rewrites_sudo_rm():
    """sudo rm → 保留 sudo 前缀改写。"""
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        from agent.tool_hooks.executor import ToolExecutor
        from agent.tool_hooks.types import ToolExecutionRequest
        executor = ToolExecutor(mgr.tool_hooks)

        captured: dict[str, Any] = {}

        async def fake_invoker(name: str, args: dict[str, Any]) -> str:
            captured.update(args)
            return "ok"

        req = ToolExecutionRequest(
            call_id="c",
            tool_name="shell",
            arguments={"command": "sudo rm /tmp/b.txt"},
            source="passive",
            session_key="test:1",
        )
        await executor.execute(req, fake_invoker)
        assert captured["command"].startswith("sudo mv -- /tmp/b.txt ")
        assert Path(shlex.split(captured["command"])[-1]).name == "restore"


@pytest.mark.asyncio
async def test_on_tool_pre_fires_through_real_reasoner():
    """真实 DefaultReasoner 链路：仅用插件 hook 改写 rm→mv。"""
    from agent.core.passive_turn import DefaultReasoner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.looping.ports import LLMConfig, LLMServices
    from agent.provider import LLMResponse, ToolCall
    from agent.tool_hooks.executor import ToolExecutor

    class FakeProvider:
        _called = False

        async def chat(self, messages, tools, model, max_tokens, **kwargs) -> LLMResponse:
            if not self._called:
                self._called = True
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "rm /tmp/a.txt"})],
                )
            return LLMResponse(content="done")

    bus = EventBus()
    tools = ToolRegistry()
    captured_commands: list[str] = []

    from agent.tools.base import Tool as AgentTool

    class FakeShell(AgentTool):
        name = "shell"
        description = "fake shell"
        parameters = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

        async def execute(self, **kwargs: Any) -> str:
            captured_commands.append(str(kwargs.get("command", "")))
            return "ok"

    tools.register(FakeShell(), risk="destructive", always_on=True)

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus, tools=tools)
        await mgr.load_all()

        reasoner = DefaultReasoner(
            llm=LLMServices(provider=FakeProvider(), light_provider=FakeProvider()),  # type: ignore[arg-type]
            llm_config=LLMConfig(max_iterations=2),
            tools=tools,
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            memory_window=40,
            event_bus=bus,
        )
        # 替换默认空 hook executor，仅用插件 hook
        reasoner._tool_executor = ToolExecutor(mgr.tool_hooks)

        await reasoner.run(
            [{"role": "user", "content": "delete /tmp/a.txt"}],
            tool_event_session_key="test:pk",
            tool_event_channel="cli",
            tool_event_chat_id="0",
        )

        assert len(captured_commands) == 1
        assert captured_commands[0].startswith("mv -- /tmp/a.txt ")
        assert Path(shlex.split(captured_commands[0])[-1]).name == "restore"


@pytest.mark.asyncio
async def test_add_tool_hooks_propagates_to_tool_executor():
    """验证 DefaultReasoner.add_tool_hooks 确实把 hook 装进了 ToolExecutor。"""
    from agent.core.passive_turn import DefaultReasoner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.looping.ports import LLMConfig

    bus = EventBus()
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "shell_restore", Path(tmp) / "shell_restore")
        mgr = _make_manager([Path(tmp)], event_bus=bus, tools=tools)
        await mgr.load_all()
        assert len(mgr.tool_hooks) > 0

        reasoner = DefaultReasoner(
            llm=None,  # type: ignore[arg-type]
            llm_config=LLMConfig(max_iterations=5),
            tools=tools,
            discovery=ToolDiscoveryState(),
            tool_search_enabled=False,
            memory_window=40,
            event_bus=bus,
        )
        # 默认空 hook
        assert len(reasoner._tool_executor._hooks) == 0
        # 注入插件 hook
        reasoner.add_tool_hooks(mgr.tool_hooks)
        assert len(reasoner._tool_executor._hooks) > 0


@pytest.mark.asyncio
async def test_core_runtime_start_wires_plugin_tool_hooks_to_loop_and_spawn():
    from bootstrap.tools import CoreRuntime

    class FakePluginManager:
        def __init__(self) -> None:
            self.tool_hooks = [object()]
            self.before_turn_modules = [object()]
            self.before_reasoning_modules = [object()]
            self.prompt_render_modules = [object()]
            self.before_step_modules = [object()]
            self.after_step_modules = [object()]
            self.after_reasoning_modules = [object()]
            self.after_turn_modules = [object()]
            self.loaded_count = 0

        async def load_all(self) -> None:
            self.loaded_count = 1

    class FakeLoop:
        def __init__(self) -> None:
            self.received_hooks: list[ToolHook] | None = None
            self.received_before_turn: list[object] | None = None
            self.received_before_reasoning: list[object] | None = None
            self.received_prompt_render: list[object] | None = None
            self.received_before_step: list[object] | None = None
            self.received_after_step: list[object] | None = None
            self.received_after_reasoning: list[object] | None = None
            self.received_after_turn: list[object] | None = None

        def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
            self.received_hooks = list(hooks)

        def add_before_turn_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_before_turn = list(modules)

        def add_before_reasoning_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_before_reasoning = list(modules)

        def add_prompt_render_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_prompt_render = list(modules)

        def add_before_step_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_before_step = list(modules)

        def add_after_step_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_after_step = list(modules)

        def add_after_reasoning_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_after_reasoning = list(modules)

        def add_after_turn_plugin_modules(
            self,
            modules: list[object],
        ) -> None:
            self.received_after_turn = list(modules)

    class FakeSpawnTool:
        def __init__(self) -> None:
            self.received_hooks: list[ToolHook] | None = None

        def add_tool_hooks(self, hooks: list[ToolHook]) -> None:
            self.received_hooks = list(hooks)

    class FakeMcpRegistry:
        def start_connect_all_background(self) -> None:
            return None

        async def shutdown(self) -> None:
            return None

    spawn_tool = FakeSpawnTool()
    loop = FakeLoop()
    plugin_manager = FakePluginManager()

    runtime = CoreRuntime(
        config=SimpleNamespace(peer_agents=[]),  # type: ignore[arg-type]
        http_resources=SimpleNamespace(local_service=None),  # type: ignore[arg-type]
        loop=loop,  # type: ignore[arg-type]
        bus=SimpleNamespace(),  # type: ignore[arg-type]
        event_bus=SimpleNamespace(aclose=lambda: None),  # type: ignore[arg-type]
        tools=SimpleNamespace(get_tool=lambda name: spawn_tool if name == "spawn" else None),  # type: ignore[arg-type]
        push_tool=SimpleNamespace(),  # type: ignore[arg-type]
        session_manager=SimpleNamespace(),  # type: ignore[arg-type]
        scheduler=SimpleNamespace(),  # type: ignore[arg-type]
        provider=SimpleNamespace(),  # type: ignore[arg-type]
        light_provider=None,
        mcp_registry=FakeMcpRegistry(),  # type: ignore[arg-type]
        memory_runtime=SimpleNamespace(),  # type: ignore[arg-type]
        presence=SimpleNamespace(),  # type: ignore[arg-type]
        peer_process_manager=None,
        peer_poller=None,
        plugin_manager=plugin_manager,  # type: ignore[arg-type]
    )

    await runtime.start()

    assert plugin_manager.loaded_count == 1
    assert loop.received_before_turn == plugin_manager.before_turn_modules
    assert loop.received_before_reasoning == plugin_manager.before_reasoning_modules
    assert loop.received_prompt_render == plugin_manager.prompt_render_modules
    assert loop.received_before_step == plugin_manager.before_step_modules
    assert loop.received_after_step == plugin_manager.after_step_modules
    assert loop.received_after_reasoning == plugin_manager.after_reasoning_modules
    assert loop.received_after_turn == plugin_manager.after_turn_modules
    assert loop.received_hooks == plugin_manager.tool_hooks
    assert spawn_tool.received_hooks == plugin_manager.tool_hooks
