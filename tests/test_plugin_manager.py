from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# 预热 agent.core 导入链，避免 agent.lifecycle.types 触发循环导入
from agent.core.passive_turn import ContextStore as _  # noqa: F401
from agent.config_models import Config
from agent.lifecycle.types import AfterStepCtx, AfterToolResultCtx, BeforeToolCallCtx, BeforeTurnCtx
from agent.plugins.context import PluginKVStore, PreparedPluginKVStore
from agent.plugins.manager import PluginManager
from agent.plugins.manifest import write_package_manifest
from agent.plugins.jobs import PluginJobRuntime, PluginJobSpec, RegisteredPluginJob
from agent.plugins.registry import plugin_registry
from agent.plugins.scope import PluginScope
from agent.tool_hooks import ToolHook
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.search_backend import KeywordSearchBackend
from bus.event_bus import EventBus
from bus.events_lifecycle import TurnCommitted
from core.memory.events import MemoryWritten, RetrievalCompleted, RetrievalHitSummary
from proactive_v2.lifecycle import ProactiveLifecycleSpec
from tests.provider_fakes import ProviderContextBudgetStub

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
        workspace=TEST_PLUGIN_HOME / "workspace",
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
async def test_wake_proactive_manifest_disables_legacy_flow_group():
    from agent.plugins.manifest import write_plugin_manifest

    plugins_root = Path(__file__).parents[1] / "plugins"
    write_plugin_manifest(
        {
            "default_proactive": False,
            "proactive_flow": False,
            "drift_flow": False,
            "wake_proactive": True,
        },
        plugins_home=TEST_PLUGIN_HOME,
    )
    mgr = _make_manager(
        [
            plugins_root / "default_proactive",
            plugins_root / "proactive_flow",
            plugins_root / "drift_flow",
            plugins_root / "wake_proactive",
        ],
        event_bus=EventBus(),
    )

    await mgr.load_all()

    assert mgr.loaded_count == 1
    assert [item.id for item in mgr.proactive_lifecycles] == ["wake"]
    assert len(mgr.proactive_module_factories) == 1
    assert len(mgr.proactive_runtime_factories) == 1
    await mgr.terminate_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("plugin_name", ["proactive_flow", "drift_flow"])
async def test_load_proactive_flow_factory(plugin_name: str):
    bus = EventBus()
    plugin_dir = Path(__file__).parents[1] / "plugins" / plugin_name
    mgr = _make_manager([plugin_dir], event_bus=bus)

    await mgr.load_all()

    assert len(mgr.proactive_module_factories) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("package_id", "lifecycle_id"),
    (("default-proactive", "default"), ("wake-proactive", "wake")),
)
async def test_builtin_package_completes_proactive_lifecycle(
    package_id: str,
    lifecycle_id: str,
):
    bus = EventBus()
    plugin_root = Path(__file__).parents[1] / "plugins"
    write_package_manifest({package_id: True}, plugins_home=TEST_PLUGIN_HOME)
    mgr = _make_manager([plugin_root], event_bus=bus)

    try:
        await mgr.load_all()

        assert [item.id for item in mgr.proactive_lifecycles] == [lifecycle_id]
        assert [
            item.lifecycle_id for item in mgr.proactive_runtime_factories
        ] == [lifecycle_id]
        assert len(mgr.proactive_module_factories) == 3
    finally:
        await mgr.terminate_all()


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
        workspace=tmp_path / "workspace",
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
        workspace=TEST_PLUGIN_HOME / "workspace",
        installed_cache_root=TEST_PLUGIN_HOME / "cache",
    )
    await mgr.load_all()
    job = next(job for job in mgr.jobs if job.plugin_id == "jobber")
    runtime = PluginJobRuntime(
        event_bus=bus,
        llm=llm,
        jobs=[job],
    )
    handler_count = bus.handler_count()
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
    assert bus.handler_count() == handler_count

    restarted = asyncio.create_task(runtime.run())
    await asyncio.sleep(0)
    assert not restarted.done()
    runtime.stop()
    await restarted
    assert bus.handler_count() == handler_count


@pytest.mark.asyncio
async def test_plugin_job_runtime_restarts_after_stop_during_job():
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_context: Any) -> None:
        started.set()
        await release.wait()

    job = RegisteredPluginJob(
        plugin_id="blocking",
        plugin_context=None,
        spec=PluginJobSpec(id="blocking", triggers=[], handler=handler),
    )
    runtime = PluginJobRuntime(
        event_bus=EventBus(),
        llm=_FakePluginLlm(),
        jobs=[job],
    )
    running = asyncio.create_task(runtime.run())
    await asyncio.sleep(0)
    runtime.enqueue("blocking:blocking", reason="test")
    await started.wait()

    runtime.stop()
    release.set()
    await running

    restarted = asyncio.create_task(runtime.run())
    await asyncio.sleep(0)
    assert not restarted.done()
    runtime.stop()
    await restarted


@pytest.mark.asyncio
async def test_plugin_job_runtime_coalesces_legacy_queue():
    calls = 0

    async def handler(_context: Any) -> None:
        nonlocal calls
        calls += 1

    job = RegisteredPluginJob(
        plugin_id="coalesce",
        plugin_context=None,
        spec=PluginJobSpec(id="refresh", triggers=[], handler=handler),
    )
    runtime = PluginJobRuntime(
        event_bus=EventBus(),
        llm=_FakePluginLlm(),
        jobs=[job],
    )
    runtime.enqueue("coalesce:refresh", reason="test")
    runtime.enqueue("coalesce:refresh", reason="test")
    running = asyncio.create_task(runtime.run())
    await asyncio.sleep(0.05)
    runtime.stop()
    await running

    assert calls == 1

@pytest.mark.asyncio
async def test_event_subscription_can_be_closed():
    bus = EventBus()
    called: list[str] = []
    subscription = bus.on(str, lambda event: called.append(event))

    await bus.fanout("first")
    subscription.close()
    await bus.fanout("second")

    assert called == ["first"]
    assert bus.handler_count() == 0


@pytest.mark.asyncio
async def test_observe_keeps_current_handler_snapshot():
    bus = EventBus()
    called: list[str] = []
    first = None

    def close_first(_event: str) -> None:
        called.append("first")
        assert first is not None
        first.close()

    first = bus.on(str, close_first)
    _ = bus.on(str, lambda _event: called.append("second"))

    await bus.observe("event")

    assert called == ["first", "second"]


@pytest.mark.asyncio
async def test_plugin_scope_cleans_in_reverse_order_after_failure():
    scope = PluginScope("scope-test")
    cleaned: list[str] = []

    def fail() -> None:
        cleaned.append("fail")
        raise RuntimeError("cleanup failed")

    scope.defer("first", lambda: cleaned.append("first"))
    scope.defer("failure", fail)
    scope.defer("last", lambda: cleaned.append("last"))

    failures = await scope.aclose()

    assert cleaned == ["last", "fail", "first"]
    assert [(item.resource, item.error) for item in failures] == [
        ("failure", "cleanup failed")
    ]
    assert scope.resource_count == 0


def test_plugin_scope_rejects_non_callable_cleanup() -> None:
    scope = PluginScope("invalid-cleanup")
    cleanup: Any = None

    with pytest.raises(
        TypeError,
        match="插件清理动作不可调用: invalid-cleanup:broken",
    ):
        scope.defer("broken", cleanup)

    assert scope.resource_count == 0


@pytest.mark.asyncio
async def test_plugin_scope_continues_after_cancelled_cleanup():
    scope = PluginScope("cancelled-cleanup")
    cleaned: list[str] = []

    def cancelled() -> None:
        raise asyncio.CancelledError

    scope.defer("last", lambda: cleaned.append("last"))
    scope.defer("cancelled", cancelled)
    scope.defer("first", lambda: cleaned.append("first"))

    failures = await scope.aclose()

    assert cleaned == ["first", "last"]
    assert [failure.resource for failure in failures] == ["cancelled"]


@pytest.mark.asyncio
async def test_plugin_scope_reports_failed_task_and_is_idempotent():
    scope = PluginScope("task-failure")
    cleaned: list[str] = []

    async def fail() -> None:
        raise RuntimeError("task failed")

    task = scope.create_task(fail(), name="worker")
    with pytest.raises(RuntimeError, match="task failed"):
        await task
    scope.defer("marker", lambda: cleaned.append("marker"))

    failures = await scope.aclose()

    assert cleaned == ["marker"]
    assert [(failure.resource, failure.error) for failure in failures] == [
        ("task:worker", "task failed")
    ]
    assert await scope.aclose() == []


@pytest.mark.asyncio
async def test_plugin_scope_reports_task_failure_before_close(caplog):
    scope = PluginScope("task-runtime-failure")

    async def fail() -> None:
        raise RuntimeError("runtime task failed")

    with caplog.at_level(logging.ERROR, logger="agent.plugins.scope"):
        task = scope.create_task(fail(), name="runtime-worker")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert task.done()
    record = next(
        record
        for record in caplog.records
        if record.name == "agent.plugins.scope"
    )
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert "runtime task failed" in caplog.text
    failures = await scope.aclose()
    assert [(failure.resource, failure.error) for failure in failures] == [
        ("task:runtime-worker", "runtime task failed")
    ]


@pytest.mark.asyncio
async def test_plugin_scope_finishes_cleanup_after_external_cancellation():
    scope = PluginScope("cancelled-close")
    entered = asyncio.Event()
    release = asyncio.Event()
    cancelled_inside_cleanup = False
    cleaned: list[str] = []

    async def cleanup() -> None:
        nonlocal cancelled_inside_cleanup
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled_inside_cleanup = True
            await release.wait()

    scope.defer("slow", cleanup)
    scope.defer("marker", lambda: cleaned.append("marker"))
    closing = asyncio.create_task(scope.aclose())
    await entered.wait()
    closing.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await closing

    assert cancelled_inside_cleanup is False
    assert cleaned == ["marker"]
    assert scope.resource_count == 0
    assert await scope.aclose() == []


@pytest.mark.asyncio
async def test_plugin_scope_handles_async_process_exit_and_timeout_kill():
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminate_calls = 0
            self.kill_calls = 0
            self.wait_calls = 0
            self._exit = asyncio.Event()

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9
            self._exit.set()

        async def wait(self) -> int:
            self.wait_calls += 1
            await self._exit.wait()
            assert self.returncode is not None
            return self.returncode

    exited = FakeProcess()
    exited.returncode = 0
    exited._exit.set()
    scope = PluginScope("async-process-exit")
    scope.track_async_process(cast(Any, exited), name="exited", timeout=0.01)
    assert await scope.aclose() == []
    assert exited.terminate_calls == 0
    assert exited.kill_calls == 0
    assert exited.wait_calls == 1

    timed_out = FakeProcess()
    scope = PluginScope("async-process-timeout")
    scope.track_async_process(cast(Any, timed_out), name="timed-out", timeout=0.01)
    assert await scope.aclose() == []
    assert timed_out.terminate_calls == 1
    assert timed_out.kill_calls == 1
    assert timed_out.wait_calls == 2


@pytest.mark.asyncio
async def test_plugin_scope_terminates_process():
    scope = PluginScope("process-test")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    scope.track_process(process, name="sleep")

    failures = await scope.aclose()

    assert failures == []
    assert process.poll() is not None


@pytest.mark.asyncio
async def test_closed_plugin_scope_does_not_create_resources():
    scope = PluginScope("closed")
    bus = EventBus()
    _ = await scope.aclose()

    with pytest.raises(RuntimeError, match="作用域已关闭"):
        _ = scope.subscribe(bus, str, lambda _event: None)

    async def wait() -> None:
        await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="作用域已关闭"):
        _ = scope.create_task(wait())

    assert bus.handler_count() == 0


@pytest.mark.asyncio
async def test_plugin_manager_scope_cleans_legacy_resources(tmp_path: Path):
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "scoped"
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from agent.plugins import Plugin\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class ScopedPlugin(Plugin):\n"
            "    name = 'scoped'\n"
            "    async def prepare(self):\n"
            "        self.context.event_bus.on(TurnCommitted, self._handle)\n"
            "    def activate(self):\n"
            "        self.task = self.context.create_task(self._run(), name='scoped-worker')\n"
        "        self.context.defer('marker', lambda: self.context.kv_store.set('closed', True))\n"
        "    async def terminate(self):\n"
        "        raise RuntimeError('terminate failed')\n"
        "    async def _run(self):\n"
        "        await asyncio.Event().wait()\n"
        "    def _handle(self, event):\n"
        "        return None\n",
        encoding="utf-8",
    )
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    for _ in range(20):
        await manager.load_all()
        instance = plugin_registry.get_instance("akasic_plugin_plugins_scoped")
        assert instance is not None
        task = instance.task
        assert bus.handler_count() == 0

        await manager.terminate_all()

        assert bus.handler_count() == 0
        assert task.done()
    data_dir = tmp_path / "workspace" / "plugin-data" / "scoped-builtin"
    assert json.loads((data_dir / ".kv.json").read_text(encoding="utf-8")) == {
        "closed": True
    }
    assert len(manager.cleanup_failures) == 20


@pytest.mark.asyncio
async def test_plugin_manager_consumes_scope_failures_after_terminate_cancellation(
    tmp_path: Path,
):
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "cancelled_close"
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "import asyncio\n"
        "from agent.plugins import Plugin\n"
        "entered = asyncio.Event()\n"
        "release = asyncio.Event()\n"
        "class CancelledClosePlugin(Plugin):\n"
        "    name = 'cancelled_close'\n"
        "    async def prepare(self):\n"
        "        async def slow_cleanup():\n"
        "            entered.set()\n"
        "            await release.wait()\n"
        "        def fail_cleanup():\n"
        "            raise RuntimeError('scope failure')\n"
        "        self.context.defer('slow', slow_cleanup)\n"
        "        self.context.defer('failure', fail_cleanup)\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    module = sys.modules["akasic_plugin_plugins_cancelled_close"]
    closing = asyncio.create_task(manager.terminate_all())
    await module.entered.wait()
    closing.cancel()
    module.release.set()

    with pytest.raises(asyncio.CancelledError):
        await closing

    assert manager.loaded_count == 0
    assert any(
        failure.resource == "failure" and failure.error == "scope failure"
        for failure in manager.cleanup_failures
    )


@pytest.mark.asyncio
async def test_plugin_prepare_failure_calls_terminate(tmp_path: Path):
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "failing"
    plugin_dir.mkdir(parents=True)
    marker = tmp_path / "terminated"
    _ = (plugin_dir / "plugin.py").write_text(
        "import asyncio\n"
        "from pathlib import Path\n"
        "from contextlib import suppress\n"
        "from agent.plugins import Plugin\n"
        "tasks = []\n"
        "class FailingPlugin(Plugin):\n"
        "    name = 'failing'\n"
        "    async def prepare(self):\n"
        "        self.task = asyncio.create_task(asyncio.Event().wait())\n"
        "        tasks.append(self.task)\n"
        "        raise RuntimeError('init failed')\n"
        "    async def terminate(self):\n"
        "        self.task.cancel()\n"
        "        with suppress(asyncio.CancelledError):\n"
        "            await self.task\n"
        f"        Path({str(marker)!r}).write_text(str(self.task.done()))\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()

    assert manager.loaded_count == 0
    assert marker.read_text(encoding="utf-8") == "True"


@pytest.mark.asyncio
async def test_plugin_prepare_cancellation_rolls_back(tmp_path: Path):
    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "cancelled"
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "import asyncio\n"
        "from contextlib import suppress\n"
        "from agent.plugins import Plugin\n"
        "started = asyncio.Event()\n"
        "terminate_started = asyncio.Event()\n"
        "release_terminate = asyncio.Event()\n"
        "tasks = []\n"
        "class CancelledPlugin(Plugin):\n"
        "    name = 'cancelled'\n"
        "    async def prepare(self):\n"
        "        self.task = asyncio.create_task(asyncio.Event().wait())\n"
        "        tasks.append(self.task)\n"
        "        started.set()\n"
        "        await asyncio.Event().wait()\n"
        "    async def terminate(self):\n"
        "        terminate_started.set()\n"
        "        await release_terminate.wait()\n"
        "        self.task.cancel()\n"
        "        with suppress(asyncio.CancelledError):\n"
        "            await self.task\n",
        encoding="utf-8",
    )
    bus = EventBus()
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=bus,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    loading = asyncio.create_task(manager.load_all())
    module_names: list[str] = []
    while not module_names:
        await asyncio.sleep(0)
        module_names = [
            name
            for name in sys.modules
            if name.startswith("akasic_plugin_plugins_cancelled__g")
        ]
    module = sys.modules[module_names[0]]
    await module.started.wait()

    loading.cancel()
    await module.terminate_started.wait()
    loading.cancel()
    await asyncio.sleep(0)
    assert not loading.done()
    module.release_terminate.set()
    with pytest.raises(asyncio.CancelledError):
        await loading

    assert manager.loaded_count == 0
    assert plugin_registry.get_instance("akasic_plugin_plugins_cancelled") is None
    assert module.tasks[0].done()
    assert bus.handler_count() == 0


@pytest.mark.asyncio
async def test_plugin_tool_registration_failure_rolls_back(tmp_path: Path):
    class FailingBackend(KeywordSearchBackend):
        def __init__(self) -> None:
            super().__init__()
            self.add_count = 0

        def add(self, document: Any) -> None:
            self.add_count += 1
            if self.add_count == 2:
                raise RuntimeError("index failed")
            super().add(document)

    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "tool_failure"
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin, tool\n"
        "class ToolFailurePlugin(Plugin):\n"
        "    name = 'tool_failure'\n"
        "    @tool(name='first')\n"
        "    async def first(self, event):\n"
        "        \"\"\"First tool.\"\"\"\n"
        "        return 'first'\n"
        "    @tool(name='second')\n"
        "    async def second(self, event):\n"
        "        \"\"\"Second tool.\"\"\"\n"
        "        return 'second'\n",
        encoding="utf-8",
    )
    backend = FailingBackend()
    tools = ToolRegistry(backend=backend)
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=EventBus(),
        tool_registry=tools,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()

    assert manager.loaded_count == 0
    assert tools.get_registered_names() == set()
    assert backend.add_count == 0
    gate = manager.latest_gate("tool_failure")
    assert gate is not None
    assert gate.status == "failed"
    assert any(check.check_id == "runtime_snapshot" for check in gate.checks)


@pytest.mark.asyncio
async def test_plugin_duplicate_tool_preserves_existing_tool(tmp_path: Path):
    class ExistingTool(Tool):
        name = "existing"
        description = "Existing tool."
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> str:
            return "existing"

    plugin_root = tmp_path / "plugins"
    plugin_dir = plugin_root / "duplicate_tool"
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin, tool\n"
        "class DuplicateToolPlugin(Plugin):\n"
        "    name = 'duplicate_tool'\n"
        "    @tool(name='existing')\n"
        "    async def existing(self, event):\n"
        "        \"\"\"Duplicate tool.\"\"\"\n"
        "        return 'plugin'\n",
        encoding="utf-8",
    )
    tools = ToolRegistry()
    existing = ExistingTool()
    tools.register(existing)
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=EventBus(),
        tool_registry=tools,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()

    assert manager.loaded_count == 0
    assert tools.get_tool("existing") is existing


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

        kv_path = TEST_PLUGIN_HOME / "workspace" / "plugin-data" / "counter-builtin" / ".kv.json"
        assert kv_path.exists()
        data = json.loads(kv_path.read_text())
        assert data["turn_count"] == 2


def test_kv_store_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / ".kv.json"
    _ = path.write_text("[]", encoding="utf-8")
    store = PluginKVStore(path)

    with pytest.raises(ValueError, match="插件 KV 根节点必须是对象"):
        store.get("turn_count")


def test_unmodified_candidate_kv_commit_does_not_create_state_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / ".kv.json"
    store = PreparedPluginKVStore(
        path,
        can_write=lambda: True,
        writer_id="candidate:1",
    )

    store.commit()

    assert not path.exists()


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
            "        return [McpServerSpec(name='feed', command=('python', 'run_mcp.py'))]\n",
            encoding="utf-8",
        )
        (plugin_root / "run_mcp.py").write_text(
            "import json, sys\n"
            "for line in sys.stdin:\n"
            "    msg = json.loads(line)\n"
            "    if 'id' not in msg: continue\n"
            "    method = msg.get('method')\n"
            "    result = {'protocolVersion': '2025-11-25'} if method == 'initialize' else {'tools': []}\n"
            "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n",
            encoding="utf-8",
        )

        mgr = PluginManager(
            plugin_dirs=[],
            installed_cache_root=cache_root,
            event_bus=bus,
            workspace=Path(tmp) / "workspace",
        )
        await mgr.load_all()

        active = mgr.active_plugins()
        assert len(active) == 1
        assert active[0].plugin_id == "feed@lab"
        assert active[0].skill_roots == (plugin_root / "skills",)
        assert "feed" in active[0].mcp_servers
        assert mgr.loaded_count == 1
        await mgr.terminate_all()


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
        workspace=tmp_path / "workspace",
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
    snapshot = mgr.current_snapshot
    assert snapshot is not None
    active_generations = snapshot.active_generations()
    assert {generation.plugin_id for generation in active_generations} >= {"akasha"}
    assert "default_memory" not in {
        generation.plugin_id for generation in active_generations
    }
    assert not any(
        root.name == "skills"
        and root.parent.name == "drift"
        and generation.plugin_id == "default_memory"
        for generation in active_generations
        for root in generation.contributions.drift_skill_roots
    )


@pytest.mark.asyncio
async def test_active_plugin_check_propagates_plugin_failure(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "broken_active"
    plugin_dir.mkdir(parents=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class BrokenActivePlugin(Plugin):\n"
        "    name = 'broken_active'\n"
        "    def is_active(self):\n"
        "        raise RuntimeError('active check failed')\n",
        encoding="utf-8",
    )
    manager = _make_manager([tmp_path / "plugins"], event_bus=EventBus())
    await manager.load_all()

    with pytest.raises(RuntimeError, match="插件 active 状态检查失败") as manager_error:
        manager.active_plugins()
    assert isinstance(manager_error.value.__cause__, RuntimeError)
    assert str(manager_error.value.__cause__) == "active check failed"
    snapshot = manager.current_snapshot
    assert snapshot is not None
    with pytest.raises(
        RuntimeError,
        match="插件 active 状态检查失败: broken_active",
    ) as snapshot_error:
        snapshot.active_generations()
    assert isinstance(snapshot_error.value.__cause__, RuntimeError)
    assert str(snapshot_error.value.__cause__) == "active check failed"

    await manager.terminate_all()


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
    data_dir = tmp_path / "workspace" / "plugin-data" / "demo-github"
    data_dir.mkdir(parents=True)
    (data_dir / "config.local.toml").write_text("value = 7\n", encoding="utf-8")

    mgr = PluginManager(
        plugin_dirs=[],
        installed_cache_root=cache_root,
        event_bus=bus,
        workspace=tmp_path / "workspace",
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
    slot = "plugin.early"
    requires = ("session:session",)

    async def run(self, frame):
        return frame


class LateModule:
    slot = "plugin.late"
    requires = ("session:ctx",)

    async def run(self, frame):
        return frame

class PromptTopModule:
    slot = "plugin.prompt_top"
    async def run(self, frame):
        return frame

class PromptBottomModule:
    slot = "plugin.prompt_bottom"
    async def run(self, frame):
        return frame

class BeforeReasoningBeforeEmitModule:
    slot = "plugin.before_reasoning_before_emit"
    async def run(self, frame):
        return frame

class BeforeReasoningAfterEmitModule:
    slot = "plugin.before_reasoning_after_emit"
    async def run(self, frame):
        return frame

class BeforeStepBeforeEmitModule:
    slot = "plugin.before_step_before_emit"
    async def run(self, frame):
        return frame

class BeforeStepAfterEmitModule:
    slot = "plugin.before_step_after_emit"
    async def run(self, frame):
        return frame

class AfterStepBeforeFanoutModule:
    slot = "plugin.after_step_before_fanout"
    async def run(self, frame):
        return frame

class AfterStepAfterFanoutModule:
    slot = "plugin.after_step_after_fanout"
    async def run(self, frame):
        return frame

class AfterReasoningBeforeEmitModule:
    slot = "plugin.after_reasoning_before_emit"
    async def run(self, frame):
        return frame

class AfterReasoningBeforePersistModule:
    slot = "plugin.after_reasoning_before_persist"
    async def run(self, frame):
        return frame

class AfterTurnBeforeCommitModule:
    slot = "plugin.after_turn_before_commit"
    async def run(self, frame):
        return frame

class AfterTurnBeforeFanoutModule:
    slot = "plugin.after_turn_before_fanout"
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
        data_dir = TEST_PLUGIN_HOME / "workspace" / "plugin-data" / "configured-builtin"
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
async def test_manifest_disables_builtin_plugin():
    bus = EventBus()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(FIXTURES_DIR / "configured", Path(tmp) / "configured")
        from agent.plugins.manifest import write_plugin_manifest

        write_plugin_manifest(
            {"configured": False},
            plugins_home=TEST_PLUGIN_HOME,
        )
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
    class FakeProvider(ProviderContextBudgetStub):
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

    class FakeProvider(ProviderContextBudgetStub):
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

    startup_order: list[str] = []

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
            startup_order.append("plugins")
            self.loaded_count = 1

        def assert_no_workspace_mcp_plugin_conflicts(self) -> None:
            return None

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

    class FakeMcpWatcher:
        async def reconcile(self) -> None:
            startup_order.append("mcp")
        async def run(self) -> None:
            return None
        def stop(self) -> None:
            return None
        async def wait_stopped(self) -> None:
            return None

    spawn_tool = FakeSpawnTool()
    loop = FakeLoop()
    plugin_manager = FakePluginManager()

    runtime = CoreRuntime(
        config=Config(
            provider="openai",
            model="m",
            api_key="k",
            system_prompt="s",
        ),
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
        workspace_mcp_watcher=FakeMcpWatcher(),  # type: ignore[arg-type]
        workspace_mcp_watcher_task=None,
        memory_runtime=SimpleNamespace(),  # type: ignore[arg-type]
        presence=SimpleNamespace(),  # type: ignore[arg-type]
        plugin_manager=plugin_manager,  # type: ignore[arg-type]
    )

    await runtime.start()

    assert startup_order == ["mcp", "plugins"]
    assert plugin_manager.loaded_count == 1
    assert loop.received_before_turn is None
    assert loop.received_before_reasoning is None
    assert loop.received_prompt_render is None
    assert loop.received_before_step is None
    assert loop.received_after_step is None
    assert loop.received_after_reasoning is None
    assert loop.received_after_turn is None
    assert loop.received_hooks == plugin_manager.tool_hooks
    assert spawn_tool.received_hooks == plugin_manager.tool_hooks


@pytest.mark.asyncio
async def test_core_runtime_stop_closes_session_manager(tmp_path: Path):
    from bootstrap.tools import CoreRuntime

    from session.manager import SessionManager

    async def _noop() -> None:
        return None

    session_manager = SessionManager(tmp_path)
    runtime = CoreRuntime(
        config=Config(
            provider="openai",
            model="m",
            api_key="k",
            system_prompt="s",
        ),
        http_resources=SimpleNamespace(),  # type: ignore[arg-type]
        loop=SimpleNamespace(),  # type: ignore[arg-type]
        bus=SimpleNamespace(),  # type: ignore[arg-type]
        event_bus=SimpleNamespace(aclose=_noop),  # type: ignore[arg-type]
        tools=SimpleNamespace(get_tool=lambda _name: None),  # type: ignore[arg-type]
        push_tool=SimpleNamespace(),  # type: ignore[arg-type]
        session_manager=session_manager,
        scheduler=SimpleNamespace(),  # type: ignore[arg-type]
        provider=SimpleNamespace(),  # type: ignore[arg-type]
        light_provider=None,
        workspace_mcp_watcher=SimpleNamespace(stop=lambda: None),  # type: ignore[arg-type]
        workspace_mcp_watcher_task=None,
        memory_runtime=SimpleNamespace(),  # type: ignore[arg-type]
        presence=SimpleNamespace(),  # type: ignore[arg-type]
        plugin_manager=None,
    )

    await runtime.stop()

    assert session_manager._store._closed is True
