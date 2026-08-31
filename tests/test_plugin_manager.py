from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

# 预热 agent.core 导入链，避免 agent.lifecycle.types 触发循环导入
from agent.core.passive_turn import ContextStore as _  # noqa: F401
from agent.config_models import Config
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.manager import PluginManager
from agent.plugins.scope import PluginScope
from bus.event_bus import EventBus

TEST_PLUGIN_HOME = Path(tempfile.gettempdir()) / f"akasic-plugin-tests-{os.getpid()}"


@pytest.fixture(autouse=True)
def _clean_plugin_home():
    """Clear the isolated plugin home around each test."""

    shutil.rmtree(TEST_PLUGIN_HOME, ignore_errors=True)
    yield
    shutil.rmtree(TEST_PLUGIN_HOME, ignore_errors=True)


def _write_v3_plugin(
    plugin_dir: Path,
    *,
    name: str | None = None,
    version: str = "1.0.0",
    source: str = "def apply(ctx, config):\n    return None\n",
    static_manifest: bool = True,
) -> Path:
    """Write one v3 namespace plugin and its static artifact identity."""

    # 1. Write the exact module-level namespace consumed by ComposablePlugin.
    plugin_name = plugin_dir.name if name is None else name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.py").write_text(
        f"api_version = 3\nname = {plugin_name!r}\nversion = {version!r}\n\n{source}",
        encoding="utf-8",
    )

    # 2. Installed artifacts also need an import-free static identity manifest.
    if static_manifest:
        (plugin_dir / "akashic.plugin.toml").write_text(
            "schema_version = 1\n"
            f"name = {json.dumps(plugin_name)}\n"
            f"version = {json.dumps(version)}\n"
            "api_version = 3\n"
            'entrypoint = "plugin.py"\n',
            encoding="utf-8",
        )
    return plugin_dir


def _write_installed_v3_plugin(
    cache_root: Path,
    *,
    marketplace: str,
    name: str,
    version: str = "1.0.0",
    source: str = "def apply(ctx, config):\n    return None\n",
) -> Path:
    plugin_base = cache_root / marketplace / name
    artifact_id = f"{version}-test"
    plugin_root = _write_v3_plugin(
        plugin_base / ".artifacts" / artifact_id,
        name=name,
        version=version,
        source=source,
    )
    pointer = ArtifactPointer(f".artifacts/{artifact_id}")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    return plugin_root


def _make_manager(
    plugin_dirs: list[Path],
    *,
    event_bus: EventBus,
    workspace: Path | None = None,
    installed_cache_root: Path | None = None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=plugin_dirs,
        event_bus=event_bus,
        workspace=workspace or TEST_PLUGIN_HOME / "workspace",
        installed_cache_root=installed_cache_root or TEST_PLUGIN_HOME / "cache",
    )


async def test_load_hello_plugin(tmp_path: Path):
    plugin_root = tmp_path / "plugins"
    _write_v3_plugin(plugin_root / "hello", name="hello", version="0.1.0")
    mgr = _make_manager(
        [plugin_root],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    try:
        await mgr.load_all()
        assert mgr.loaded_count == 1
        assert {item["name"] for item in mgr.discover()} == {"hello"}
    finally:
        await mgr.terminate_all()


@pytest.mark.asyncio
async def test_duplicate_plugin_name_first_wins(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_v3_plugin(first_root / "duplicate", name="duplicate")
    _write_v3_plugin(second_root / "duplicate", name="duplicate")
    mgr = _make_manager(
        [first_root, second_root],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    try:
        await mgr.load_all()
        assert mgr.loaded_count == len({item["name"] for item in mgr.discover()})
    finally:
        await mgr.terminate_all()


@pytest.mark.asyncio
async def test_installed_plugin_shadows_builtin_with_same_name(tmp_path: Path):
    builtin_root = tmp_path / "plugins"
    installed_root = tmp_path / "cache"
    _write_v3_plugin(builtin_root / "shadow", name="shadow")
    _write_installed_v3_plugin(
        installed_root,
        marketplace="github",
        name="shadow",
        version="0.1.0",
    )
    mgr = PluginManager(
        plugin_dirs=[builtin_root],
        installed_cache_root=installed_root,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
    )

    try:
        assert [item["source_type"] for item in mgr.discover()] == ["installed"]
        await mgr.load_all()
        assert [plugin.plugin_id for plugin in mgr.active_plugins()] == [
            "shadow@github"
        ]
    finally:
        await mgr.terminate_all()


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
        record for record in caplog.records if record.name == "agent.plugins.scope"
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
async def test_plugin_manager_scope_cleans_v3_resources(tmp_path: Path):
    plugin_dir = _write_v3_plugin(
        tmp_path / "plugins" / "scoped",
        source=(
            "import asyncio\n\n"
            "task = None\n\n"
            "async def apply(ctx, config):\n"
            "    global task\n"
            '    task = await ctx.spawn(asyncio.Event().wait(), name="scoped-worker")\n'
        ),
    )
    manager = _make_manager(
        [plugin_dir.parent],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()
    module = sys.modules["akasic_plugin_plugins_scoped"]
    assert module.task is not None
    assert not module.task.done()

    await manager.terminate_all()

    assert module.task.done()
    assert manager.loaded_count == 0
    assert manager.cleanup_failures == []


@pytest.mark.asyncio
async def test_active_plugins_exposes_v3_metadata(tmp_path: Path):
    plugin_dir = _write_v3_plugin(
        tmp_path / "plugins" / "manifested",
        name="manifested",
        version="1.0.0",
        source=(
            'desc = "v3 declaration"\n'
            'author = "tester"\n'
            'skill_roots = ("skills",)\n\n'
            "def apply(ctx, config):\n"
            "    return None\n"
        ),
    )
    (plugin_dir / "skills").mkdir()
    mgr = _make_manager(
        [plugin_dir.parent],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    try:
        await mgr.load_all()
        active = mgr.active_plugins()
        assert len(active) == 1
        assert active[0].plugin_id == "manifested"
        assert active[0].plugin_dir == plugin_dir
        assert active[0].manifest == {
            "name": "manifested",
            "version": "1.0.0",
            "desc": "v3 declaration",
            "author": "tester",
        }
        assert active[0].skill_roots == (plugin_dir / "skills",)
        generation = mgr.generation("manifested")
        assert generation is not None and generation.static_manifest is not None
        assert generation.static_manifest.api_version == 3
        assert generation.static_manifest.entrypoint == "plugin.py"
    finally:
        await mgr.terminate_all()


@pytest.mark.asyncio
async def test_loads_installed_v3_plugin(tmp_path: Path):
    cache_root = tmp_path / "cache"
    plugin_root = _write_installed_v3_plugin(
        cache_root,
        marketplace="lab",
        name="feed",
        version="1.0.0",
        source=(
            'skill_roots = ("skills",)\n\n'
            "def apply(ctx, config):\n"
            "    return None\n"
        ),
    )
    (plugin_root / "skills" / "feed-manage").mkdir(parents=True)
    (plugin_root / "skills" / "feed-manage" / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed\n---\nbody\n",
        encoding="utf-8",
    )
    mgr = PluginManager(
        plugin_dirs=[],
        installed_cache_root=cache_root,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
    )

    try:
        await mgr.load_all()
        active = mgr.active_plugins()
        assert len(active) == 1
        assert active[0].plugin_id == "feed@lab"
        assert active[0].skill_roots == (plugin_root / "skills",)
        assert mgr.loaded_count == 1
        generation = mgr.generation("feed@lab")
        assert generation is not None and generation.static_manifest is not None
        assert generation.static_manifest.name == "feed"
        assert generation.static_manifest.api_version == 3
    finally:
        await mgr.terminate_all()


@pytest.mark.asyncio
async def test_sync_manifest_covers_builtin_and_installed_plugins(tmp_path: Path):
    builtin_root = tmp_path / "plugins"
    _write_v3_plugin(builtin_root / "hello", name="hello", version="0.1.0")

    cache_root = tmp_path / "cache"
    installed_root = _write_installed_v3_plugin(
        cache_root,
        marketplace="lab",
        name="feed",
        version="1.0.0",
        source=(
            'skill_roots = ("skills",)\n\n'
            "def apply(ctx, config):\n"
            "    return None\n"
        ),
    )
    (installed_root / "skills" / "feed-manage").mkdir(parents=True)
    (installed_root / "skills" / "feed-manage" / "SKILL.md").write_text(
        "---\nname: feed-manage\ndescription: feed\n---\nbody\n",
        encoding="utf-8",
    )
    mgr = PluginManager(
        plugin_dirs=[builtin_root],
        installed_cache_root=cache_root,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
    )

    try:
        await mgr.load_all()
        manifest_path = mgr.sync_manifest(plugins_home=tmp_path / ".akashic-plugin")
        import tomllib

        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest["plugins"]) == {"feed@lab", "hello"}
        assert manifest["plugins"]["feed@lab"]["enabled"] is True
    finally:
        await mgr.terminate_all()


@pytest.mark.asyncio
async def test_active_plugin_check_failure_is_recorded(tmp_path: Path) -> None:
    _write_v3_plugin(
        tmp_path / "plugins" / "broken_active",
        source=(
            "def is_active(services):\n"
            '    raise RuntimeError("active check failed")\n\n'
            "def apply(ctx, config):\n"
            "    return None\n"
        ),
    )
    manager = _make_manager(
        [tmp_path / "plugins"],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()

    assert manager.loaded_count == 0
    gate = manager.latest_gate("broken_active")
    assert gate is not None and gate.status == "failed"
    assert any("active check failed" in str(check.evidence) for check in gate.checks)
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manifest_disables_builtin_plugin(tmp_path: Path):
    plugin_root = tmp_path / "plugins"
    _write_v3_plugin(plugin_root / "configured", name="configured")
    from agent.plugins.manifest import write_plugin_manifest

    write_plugin_manifest(
        {"configured": False},
        plugins_home=TEST_PLUGIN_HOME,
    )
    mgr = _make_manager(
        [plugin_root],
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=TEST_PLUGIN_HOME / "cache",
    )

    try:
        await mgr.load_all()
        assert mgr.loaded_count == 0
        assert mgr.active_plugins() == []
    finally:
        await mgr.terminate_all()


@pytest.mark.asyncio
async def test_core_runtime_stop_closes_session_manager(tmp_path: Path):
    from bootstrap.tools import CoreRuntime
    from session.manager import SessionManager

    async def _noop() -> None:
        return None

    session_manager = SessionManager(tmp_path)
    runtime = CoreRuntime(
        config=Config(system_prompt="s"),
        http_resources=SimpleNamespace(),  # type: ignore[arg-type]
        loop=SimpleNamespace(shutdown_compaction=_noop),  # type: ignore[arg-type]
        bus=SimpleNamespace(),  # type: ignore[arg-type]
        event_bus=SimpleNamespace(aclose=_noop),  # type: ignore[arg-type]
        tools=SimpleNamespace(get_tool=lambda _name: None),  # type: ignore[arg-type]
        push_tool=SimpleNamespace(),  # type: ignore[arg-type]
        session_manager=session_manager,
        presence=SimpleNamespace(),  # type: ignore[arg-type]
        plugin_manager=None,
    )

    await runtime.stop()

    assert session_manager._store._closed is True
