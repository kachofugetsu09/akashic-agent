from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# 预热 agent.core 导入链，避免 agent.lifecycle.types 触发循环导入
from agent.core.passive_turn import ContextStore as _  # noqa: F401
from agent.lifecycle.types import AfterStepCtx, BeforeTurnCtx
from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus


# ── fixtures ──────────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "plugins"


@pytest.fixture(autouse=True)
def _clean_registry():
    # 每个测试前后清空全局 registry，避免插件状态跨测试污染
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


def _make_manager(plugin_dirs: list[Path], *, event_bus: EventBus, tools: ToolRegistry | None = None) -> PluginManager:
    return PluginManager(plugin_dirs=plugin_dirs, event_bus=event_bus, tool_registry=tools)


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
async def test_duplicate_plugin_name_first_wins():
    # 同名插件目录放两份，second 应被跳过
    bus = EventBus()
    mgr = _make_manager([FIXTURES_DIR, FIXTURES_DIR], event_bus=bus)
    await mgr.load_all()
    # discover 跨两个同名目录，seen_names 跨目录共享 → 只加载一次
    assert mgr.loaded_count == len({m["name"] for m in mgr.discover()})


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
        import shutil
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
        import shutil
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

        kv_path = tmp_counter / ".kv.json"
        assert kv_path.exists()
        data = json.loads(kv_path.read_text())
        assert data["turn_count"] == 2


# ── manifest.yaml 测试 ────────────────────────────────────────────────────────


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
async def test_manifest_overrides_class_attributes():
    bus = EventBus()
    # 用包含 manifested/ 子目录的父目录
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "manifested", Path(tmp) / "manifested")
        mgr = _make_manager([Path(tmp)], event_bus=bus)
        await mgr.load_all()

        instance = _get_instance("manifest_name")
        assert instance.name == "manifest_name"
        assert instance.version == "0.2.0"
        assert instance.desc == "from manifest"
        assert instance.author == "tester"
        assert instance.context.plugin_id == "manifest_name"


@pytest.mark.asyncio
async def test_no_manifest_uses_class_attributes():
    bus = EventBus()
    import tempfile, shutil
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
