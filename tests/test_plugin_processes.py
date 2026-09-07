import asyncio
import os
from pathlib import Path
import sys

import pytest

from agent.plugin_composition import PROCESSES, PluginProcesses, ProcessCleanupError, ServiceKey
from agent.plugin_composition.bindings import Bindings
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from agent.tools.unified_exec import (
    ExecutionCleanupFailure, ExecutionCleanupReport, ShellProcessManager, UnknownExecutionError,
)
from bus.event_bus import EventBus
from session.log import MessageLog


def write_plugins(path):
    for name in ("first", "second"):
        folder = path / name
        folder.mkdir(parents=True)
        (folder / "plugin.py").write_text(f'''
from agent.plugin_composition import PROCESSES, ServiceKey
api_version = 3
name = "{name}"
version = "1.0.0"
inject = (PROCESSES,)
async def apply(ctx, config):
    await ctx.provide(ServiceKey("fixture.processes.{name}"), ctx)
''')


def launch(processes, context, key, path, *, interactive=False):
    script = (
        "import os; print(os.getpid(), flush=True); print('GOT:' + input(), flush=True)"
        if interactive else "import os, signal; print(os.getpid(), flush=True); signal.pause()"
    )
    return processes.exec_command(
        context, key, command="controlled process", argv=[sys.executable, "-c", script],
        cwd=path, env=os.environ.copy(), tty=interactive, yield_time_ms=250,
        max_output_tokens=100, hard_timeout_s=30,
    )


def manager(tmp_path, log):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    return PluginManager([sources], event_bus=EventBus(), workspace=tmp_path / "workspace",
                         installed_cache_root=tmp_path / "cache", message_log=log)


@pytest.mark.asyncio
async def test_formal_and_archived_processes_share_backend_and_isolate_actual_owner(tmp_path, monkeypatch):
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, log)
    built = []

    def factory():
        backend = ShellProcessManager()
        built.append(backend)
        return backend

    monkeypatch.setattr(host._plugin_processes, "_factory", factory)
    try:
        await host.load_all()
        assert built == []
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            first = root.require(ServiceKey("fixture.processes.first"))
            second = root.require(ServiceKey("fixture.processes.second"))
            processes = root.require(PROCESSES)
            bindings = Bindings(log, host._archive, host.open_binding)
            binding = bindings.bind(ServiceKey("fixture.processes.first"), {})
            started = await launch(processes, first, "job", tmp_path, interactive=True)
            assert started.execution_id is not None
            async with bindings.open(binding, ServiceKey("fixture.processes.first")) as (archived, _):
                assert archived is not first
                assert archived.require(PROCESSES) is processes
                completed = await processes.write_stdin(
                    archived, "job", execution_id=started.execution_id, chars="PING\n",
                    yield_time_ms=1000, max_output_tokens=100,
                )
                assert b"GOT:PING" in completed.output
                assert completed.exit_code == 0
                started = await launch(processes, archived, "job", tmp_path)
            # 归档 scope 已释放，实际进程仍由同一宿主管理。
            assert started.execution_id is not None
            pid = int(started.output)
            for context, key in ((first, "other-job"), (second, "job")):
                with pytest.raises(UnknownExecutionError):
                    await processes.write_stdin(context, key, execution_id=started.execution_id,
                                                chars="", yield_time_ms=5000, max_output_tokens=100)
                assert not await processes.terminate_execution(context, key, started.execution_id)
            os.kill(pid, 0)
            cleaned = await processes.terminate_owner(first, "job")
            assert cleaned.cleaned_execution_ids == (started.execution_id,)
            assert not cleaned.failures
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            with pytest.raises(RuntimeError, match="正式进程"):
                await launch(PluginProcesses(formal=False, factory=factory), first, "candidate", tmp_path)
            assert len(built) == 1
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_process_shutdown_failure_retains_original_backend_until_cleanup(tmp_path, monkeypatch):
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            context = root.require(ServiceKey("fixture.processes.first"))
            processes = root.require(PROCESSES)
            started = await launch(processes, context, "job", tmp_path)
            assert started.execution_id is not None
            pid = int(started.output)
            backend = processes._manager

            async def fail():
                return ExecutionCleanupReport((started.execution_id,), (), (
                    ExecutionCleanupFailure(started.execution_id, "OSError", "controlled failure"),))

            with monkeypatch.context() as patch:
                patch.setattr(backend, "shutdown", fail)
                with pytest.raises(ProcessCleanupError) as error:
                    await processes.close()
                assert error.value.report.failed_execution_ids == (started.execution_id,)
                assert processes._manager is backend
                with pytest.raises(RuntimeError, match="尚未确认清理"):
                    processes.start()
                with pytest.raises(RuntimeError, match="正式进程"):
                    await launch(processes, context, "late", tmp_path)
                os.kill(pid, 0)
            await processes.close()
            assert processes._manager is None
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_shutdown_waits_for_admitted_spawn_before_cleaning_backend(tmp_path, monkeypatch, cancel):
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, log)
    backend = ShellProcessManager()
    entered, release, closing = asyncio.Event(), asyncio.Event(), asyncio.Event()
    execute = backend.exec_command

    async def blocked(**arguments):
        entered.set()
        await release.wait()
        return await execute(**arguments)

    monkeypatch.setattr(backend, "exec_command", blocked)
    monkeypatch.setattr(host._plugin_processes, "_factory", lambda: backend)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            root = snapshot.composition_root.context
            context = root.require(ServiceKey("fixture.processes.first"))
            processes = root.require(PROCESSES)
            wait = processes._drained.wait

            async def observed_wait():
                closing.set()
                return await wait()

            monkeypatch.setattr(processes._drained, "wait", observed_wait)
            scope = context.capture_runtime_scope()

            async def run():
                async with scope:
                    return await launch(processes, context, "job", tmp_path)

            pending = asyncio.create_task(run())
            admitted = asyncio.create_task(entered.wait())
            await asyncio.wait((pending, admitted), return_when=asyncio.FIRST_COMPLETED)
            if pending.done():
                admitted.cancel()
                await pending
            close = asyncio.create_task(processes.close())
            await closing.wait()
            assert not close.done()
            with pytest.raises(RuntimeError, match="正式进程"):
                await launch(processes, context, "late", tmp_path)
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                release.set()
                result = await pending
                pid = int(result.output)
            await close
            assert await backend.active_execution_ids() == []
            if not cancel:
                with pytest.raises(ProcessLookupError):
                    os.kill(pid, 0)
    finally:
        release.set()
        await host.terminate_all()
        await backend.shutdown()
        log.close()
