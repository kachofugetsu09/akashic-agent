from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent.mcp.declarations import load_workspace_mcp_declarations
from agent.mcp.watcher import WorkspaceMcpWatcher
from agent.plugins.manager import PluginManager
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from bootstrap.app import AppRuntime
from bootstrap.tools import CoreRuntime
from agent.config_models import Config
from session.manager import SessionManager


def _server(root: Path, tool: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "server.py"
    path.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "log=Path(__file__).with_name('lifecycle.log')\n"
        "with log.open('a') as f: f.write('started\\n')\n"
        "try:\n"
        " for line in sys.stdin:\n"
        "  m=json.loads(line)\n"
        "  if 'id' not in m: continue\n"
        "  method=m.get('method')\n"
        "  result={'protocolVersion':'2025-11-25'} if method=='initialize' else "
        f"({{'tools':[{{'name':'{tool}','description':'{tool}','inputSchema':{{'type':'object','properties':{{}}}}}}]}} "
        "if method=='tools/list' else {'content':[{'type':'text','text':'ok'}]})\n"
        "  print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':result}),flush=True)\n"
        "finally:\n with log.open('a') as f: f.write('stopped\\n')\n",
        encoding="utf-8",
    )
    return path


def _declare(root: Path, name: str, server: Path, *, watch: str = "") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    watch_line = f'watch_paths = ["{watch}"]\n' if watch else ""
    path = root / f"{name}.toml"
    path.write_text(
        "schema_version = 1\n"
        f'name = "{name}"\n'
        f'command = ["{sys.executable}", "{server}"]\n'
        f"{watch_line}",
        encoding="utf-8",
    )
    return path


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(),
        workspace=tmp_path,
        installed_cache_root=tmp_path / "cache",
    )


def test_declarations_validate_and_hash_watch_directory(tmp_path: Path) -> None:
    mcp_root = tmp_path / "workspace" / "mcp"
    declarations = mcp_root / "servers"
    watched = mcp_root / "fitbit-mcp"
    watched.mkdir(parents=True)
    (watched / "a.txt").write_text("a", encoding="utf-8")
    declaration = _declare(
        declarations,
        "docs",
        _server(declarations, "read"),
        watch="../fitbit-mcp",
    )
    declaration.write_text(
        declaration.read_text(encoding="utf-8") + 'cwd = "../fitbit-mcp"\n',
        encoding="utf-8",
    )
    first = load_workspace_mcp_declarations(declarations, mcp_root=mcp_root)
    assert first.specs["docs"]["cwd"] == str(watched)
    (watched / "b.txt").write_text("b", encoding="utf-8")
    second = load_workspace_mcp_declarations(declarations, mcp_root=mcp_root)
    assert first.revision != second.revision
    (declarations / "ignored.txt").write_text("ignored", encoding="utf-8")
    assert (
        load_workspace_mcp_declarations(declarations, mcp_root=mcp_root).revision
        == second.revision
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (mcp_root / "escape").symlink_to(outside, target_is_directory=True)
    declaration.write_text(
        'schema_version=1\nname="docs"\ncommand=["x"]\n'
        'watch_paths=["../escape/file"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="watch_paths 越界"):
        load_workspace_mcp_declarations(declarations, mcp_root=mcp_root)
    (declarations / "docs.toml").write_text(
        'schema_version=1\nname="wrong"\ncommand=["x"]\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version/name"):
        load_workspace_mcp_declarations(declarations)


@pytest.mark.asyncio
async def test_missing_watch_path_create_and_delete_each_reload(tmp_path: Path) -> None:
    mcp_root = tmp_path / "workspace" / "mcp"
    declarations = mcp_root / "servers"
    watched = mcp_root / "fitbit-mcp" / "server.py"
    _declare(
        declarations,
        "docs",
        _server(tmp_path / "server", "read"),
        watch="../fitbit-mcp/server.py",
    )
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(
        manager,
        declarations,
        mcp_root=mcp_root,
        interval_seconds=0.01,
    )
    await watcher.reconcile()
    first = manager.active_workspace_mcp
    assert first is not None
    task = asyncio.create_task(watcher.run())

    watched.parent.mkdir()
    watched.write_text("created", encoding="utf-8")
    for _ in range(100):
        if manager.active_workspace_mcp is not first:
            break
        await asyncio.sleep(0.01)
    second = manager.active_workspace_mcp
    assert second is not None and second is not first

    watched.unlink()
    for _ in range(100):
        if manager.active_workspace_mcp is not second:
            break
        await asyncio.sleep(0.01)
    assert manager.active_workspace_mcp is not second
    watcher.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_watch_directory_framing_prevents_path_content_collision(
    tmp_path: Path,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    watched = declarations / "watched"
    watched.mkdir(parents=True)
    first_file = watched / "a"
    first_file.write_text("bc", encoding="utf-8")
    _declare(
        declarations,
        "docs",
        _server(tmp_path / "server", "read"),
        watch="watched",
    )
    first_revision = load_workspace_mcp_declarations(declarations).revision
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations, interval_seconds=0.01)
    await watcher.reconcile()
    first_generation = manager.active_workspace_mcp
    assert first_generation is not None
    task = asyncio.create_task(watcher.run())

    first_file.unlink()
    (watched / "ab").write_text("c", encoding="utf-8")
    second_revision = load_workspace_mcp_declarations(declarations).revision
    assert second_revision != first_revision
    for _ in range(100):
        if manager.active_workspace_mcp is not first_generation:
            break
        await asyncio.sleep(0.01)
    assert manager.active_workspace_mcp is not first_generation

    watcher.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_real_workspace_mcp_declaration_reconcile_and_empty_delete(
    tmp_path: Path,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations)

    await watcher.reconcile()
    _declare(declarations, "docs", _server(tmp_path / "v1", "v1"))
    await watcher.reconcile()
    first = manager.active_workspace_mcp
    assert first is not None
    old_lease = await manager.snapshot_store.acquire()
    _declare(declarations, "docs", _server(tmp_path / "v2", "v2"))
    await watcher.reconcile()
    assert first.catalog.servers["docs"].client.connected is True
    await old_lease.release()
    await manager.snapshot_store.retry_drains()
    assert first.catalog.servers["docs"].client.connected is False
    for path in declarations.glob("*.toml"):
        path.unlink()
    declarations.rmdir()
    await watcher.reconcile()
    assert manager.active_workspace_mcp is not None
    assert not manager.active_workspace_mcp.catalog.servers
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_bad_initial_batch_cleans_partial_processes(tmp_path: Path) -> None:
    declarations = tmp_path / "mcp" / "servers"
    good = _server(tmp_path / "a-good", "good")
    _declare(declarations, "a_good", good)
    _declare(declarations, "z_bad", tmp_path / "missing-command")
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations)
    with pytest.raises(ConnectionError):
        await watcher.reconcile()
    for _ in range(100):
        lines = (good.parent / "lifecycle.log").read_text().splitlines()
        if lines == ["started", "stopped"]:
            break
        await asyncio.sleep(0.01)
    assert lines == ["started", "stopped"]
    assert manager.active_workspace_mcp is None
    assert manager.prepared_workspace_mcp is None
    assert not manager._mcp_host._catalogs
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_real_watcher_rolls_back_bad_batch_and_recovers_latest_fix(
    tmp_path: Path,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    v1 = _server(tmp_path / "v1", "v1")
    declaration = _declare(declarations, "docs", v1)
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations, interval_seconds=0.01)
    await watcher.reconcile()
    active = manager.active_workspace_mcp
    assert active is not None
    task = asyncio.create_task(watcher.run())

    declaration.write_text("schema_version = 1\nname = [\n", encoding="utf-8")
    for _ in range(100):
        if watcher.last_error is not None:
            break
        await asyncio.sleep(0.01)
    assert watcher.last_error is not None
    assert manager.active_workspace_mcp is active
    assert active.catalog.servers["docs"].client.connected is True

    v2 = _server(tmp_path / "v2", "v2")
    _declare(declarations, "docs", v2)
    for _ in range(100):
        current = manager.active_workspace_mcp
        if (
            current is not None
            and current is not active
            and watcher.last_error is None
        ):
            break
        await asyncio.sleep(0.01)
    current = manager.active_workspace_mcp
    assert current is not None and current is not active
    assert tuple(current.catalog.servers["docs"].remote_tool_names) == ("v2",)
    assert watcher.last_error is None

    watcher.stop()
    await watcher.wait_stopped()
    await task
    await manager.terminate_all()
    assert task.done()
    assert current.catalog.servers["docs"].client.connected is False


@pytest.mark.asyncio
async def test_real_watcher_restarts_on_watch_content_only(tmp_path: Path) -> None:
    declarations = tmp_path / "mcp" / "servers"
    watched = declarations / "source"
    watched.mkdir(parents=True)
    marker = watched / "version.txt"
    marker.write_text("v1", encoding="utf-8")
    _declare(
        declarations,
        "docs",
        _server(tmp_path / "server", "read"),
        watch="source",
    )
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations, interval_seconds=0.01)
    await watcher.reconcile()
    first = manager.active_workspace_mcp
    assert first is not None
    task = asyncio.create_task(watcher.run())

    await asyncio.sleep(0.04)
    assert manager.active_workspace_mcp is first
    marker.write_text("v2", encoding="utf-8")
    for _ in range(100):
        if manager.active_workspace_mcp is not first:
            break
        await asyncio.sleep(0.01)
    second = manager.active_workspace_mcp
    assert second is not None and second is not first
    await asyncio.sleep(0.04)
    assert manager.active_workspace_mcp is second

    watcher.stop()
    await watcher.wait_stopped()
    await task
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_real_watcher_keeps_active_generation_on_partial_start_failure(
    tmp_path: Path,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    _declare(declarations, "docs", _server(tmp_path / "v1", "v1"))
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations, interval_seconds=0.01)
    await watcher.reconcile()
    active = manager.active_workspace_mcp
    assert active is not None
    task = asyncio.create_task(watcher.run())

    candidate = _server(tmp_path / "v2", "v2")
    _declare(declarations, "docs", candidate)
    bad = _declare(declarations, "z_bad", tmp_path / "missing.py")
    for _ in range(100):
        if watcher.last_error is not None:
            break
        await asyncio.sleep(0.01)
    assert watcher.last_error is not None
    assert manager.active_workspace_mcp is active
    assert active.catalog.servers["docs"].client.connected is True
    for _ in range(100):
        lines = (candidate.parent / "lifecycle.log").read_text().splitlines()
        if lines == ["started", "stopped"]:
            break
        await asyncio.sleep(0.01)
    assert lines == ["started", "stopped"]

    bad.unlink()
    for _ in range(100):
        if manager.active_workspace_mcp is not active and watcher.last_error is None:
            break
        await asyncio.sleep(0.01)
    assert manager.active_workspace_mcp is not active
    assert watcher.last_error is None

    watcher.stop()
    await watcher.wait_stopped()
    await task
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_core_stop_cancels_blocked_real_candidate_before_publish(
    tmp_path: Path,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    _declare(declarations, "docs", _server(tmp_path / "v1", "v1"))
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations, interval_seconds=0.01)
    await watcher.reconcile()
    active = manager.active_workspace_mcp
    assert active is not None

    entered = asyncio.Event()
    release = asyncio.Event()
    original_prepare = manager.prepare_workspace_mcp

    async def blocked_prepare(
        specs: dict[str, dict[str, Any]], *, revision: str
    ) -> None:
        await original_prepare(specs, revision=revision)
        entered.set()
        await release.wait()

    manager.prepare_workspace_mcp = blocked_prepare  # type: ignore[method-assign]
    _declare(declarations, "docs", _server(tmp_path / "v2", "v2"))
    watcher_task = asyncio.create_task(watcher.run())
    watcher.wake()
    await entered.wait()
    candidate = manager.prepared_workspace_mcp
    assert candidate is not None
    candidate_client = candidate.catalog.servers["docs"].client
    active_before_terminate: list[bool] = []
    original_terminate = manager.terminate_all

    async def tracked_terminate() -> None:
        active_before_terminate.append(manager.active_workspace_mcp is active)
        await original_terminate()

    async def _shutdown_compaction() -> None:
        return None

    manager.terminate_all = tracked_terminate  # type: ignore[method-assign]

    runtime = CoreRuntime(
        config=Config(provider="openai", model="m", api_key="k", system_prompt="s"),
        http_resources=object(),  # type: ignore[arg-type]
        loop=SimpleNamespace(shutdown_compaction=_shutdown_compaction),  # type: ignore[arg-type]
        bus=object(),  # type: ignore[arg-type]
        event_bus=EventBus(),
        tools=ToolRegistry(),
        push_tool=object(),  # type: ignore[arg-type]
        session_manager=SessionManager(tmp_path / "sessions"),
        scheduler=object(),  # type: ignore[arg-type]
        provider=object(),  # type: ignore[arg-type]
        light_provider=None,
        workspace_mcp_watcher=watcher,
        workspace_mcp_watcher_task=watcher_task,
        memory_runtime=object(),  # type: ignore[arg-type]
        presence=object(),  # type: ignore[arg-type]
        plugin_manager=manager,
    )
    await asyncio.wait_for(runtime.stop(), timeout=1)

    assert watcher_task.done()
    assert manager.prepared_workspace_mcp is None
    assert candidate_client.connected is False
    assert active_before_terminate == [True]


@pytest.mark.asyncio
async def test_app_start_bad_batch_releases_workspace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    _declare(declarations, "bad", tmp_path / "missing.py")
    manager = _manager(tmp_path)
    watcher = WorkspaceMcpWatcher(manager, declarations)

    async def noop() -> None:
        return None

    class Core:
        loop = bus = tools = push_tool = session_manager = scheduler = object()
        provider = presence = object()
        light_provider = None
        event_bus = EventBus()
        plugin_manager = manager
        memory_runtime = type("Memory", (), {"aclose": staticmethod(noop)})()

        async def start(self) -> None:
            await watcher.reconcile()

        async def stop(self) -> None:
            await manager.terminate_all()

    monkeypatch.setattr("bootstrap.app.build_core_runtime", lambda *_, **__: Core())
    runtime = AppRuntime(object(), tmp_path)  # type: ignore[arg-type]
    with pytest.raises(ConnectionError):
        await runtime.start()
    assert runtime._workspace_lock._stream is None
    assert manager.prepared_workspace_mcp is None
    assert not manager._mcp_host._catalogs


@pytest.mark.asyncio
async def test_watcher_keeps_latest_change_and_deduplicates_failed_revision(
    tmp_path: Path,
) -> None:
    declarations = tmp_path / "mcp" / "servers"
    server = _server(tmp_path, "value")
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    class Manager:
        async def prepare_workspace_mcp(
            self, specs: dict[str, dict[str, Any]], *, revision: str
        ) -> None:
            command = str(next(iter(specs.values()))["command"][-1])
            calls.append(command)
            if command.endswith("v2.py"):
                entered.set()
                await release.wait()
            if command.endswith("bad.py"):
                raise RuntimeError("bad")

        async def publish_workspace_mcp(self) -> Any:
            return SimpleNamespace(
                generation_id=f"test:{len(calls)}",
                catalog=SimpleNamespace(servers={"docs": object()}, tool_names=()),
            )

    watcher = WorkspaceMcpWatcher(Manager(), declarations, interval_seconds=0.01)  # type: ignore[arg-type]
    _declare(declarations, "docs", server)
    await watcher.reconcile()
    _declare(declarations, "docs", tmp_path / "v2.py")
    task = asyncio.create_task(watcher.run())
    watcher.wake()
    await entered.wait()
    _declare(declarations, "docs", tmp_path / "v3.py")
    release.set()
    for _ in range(100):
        if calls and calls[-1].endswith("v3.py"):
            break
        await asyncio.sleep(0.01)
    assert calls[-1].endswith("v3.py")
    _declare(declarations, "docs", tmp_path / "bad.py")
    await asyncio.sleep(0.04)
    failed_count = len(calls)
    await asyncio.sleep(0.04)
    assert len(calls) == failed_count
    watcher.wake()
    await asyncio.sleep(0.02)
    assert len(calls) == failed_count + 1
    watcher.stop()
    await watcher.wait_stopped()
    await task
