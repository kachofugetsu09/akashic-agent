"""新 Core boot 的宿主候选清理；不执行 Docker，也不重放 start。"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agent.plugins._operation import OperationBusyError, OperationTimeoutError
from agent.plugins.manager import PluginManager
from agent.plugins.selection import SelectionFormatError
from agent.workloads.client import WorkloadEffectUnknown
from agent.workloads.controller import WorkloadControllerServer
from agent.plugin_composition.execution import WorkloadLease, WorkloadStopReceipt
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


def test_controller_sigterm_releases_own_socket_and_lock(tmp_path):
    """The public CLI must release its listener before another owner can start."""

    workspace = tmp_path / "workspace"
    for part in ("plugin-data", "runtime/plugin-validation"):
        (workspace / part).mkdir(parents=True)
    sentinel = workspace / "plugin-data" / "keep"
    sentinel.write_bytes(b"workload data stays\n")
    socket_path = tmp_path / "run" / "controller.sock"
    state_path = tmp_path / "state" / "leases.json"
    command = [
        sys.executable, "-m", "agent.workloads.controller",
        "--workspace", str(workspace), "--socket", str(socket_path),
        "--docker-socket", str(tmp_path / "unused-docker.sock"),
        "--state", str(state_path), "--network", "test-net",
        "--allowed-uid", str(os.getuid()), "--socket-uid", str(os.getuid()),
        "--socket-gid", str(os.getgid()), "--workload-uid", str(os.getuid()),
        "--workload-gid", str(os.getgid()),
    ]
    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(f"Controller exited before readiness: {process.returncode}")
            if socket_path.is_socket():
                with socket.socket(socket.AF_UNIX) as probe:
                    try:
                        probe.connect(str(socket_path))
                    except ConnectionRefusedError:
                        pass
                    else:
                        probe.sendall(b'{"version":1,"action":"probe","body":{}}\n')
                        assert b'"ok":false' in probe.recv(4096)
                        break
            time.sleep(0.01)
        else:
            pytest.fail("Controller listener did not become ready")

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        assert not socket_path.exists()
        with state_path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        assert sentinel.read_bytes() == b"workload data stays\n"
        assert not state_path.exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def controller_server(tmp_path):
    workspace = tmp_path / "workspace"
    for part in ("plugin-data", "runtime/plugin-validation"):
        (workspace / part).mkdir(parents=True)
    state_path = tmp_path / "state" / "leases.json"
    socket_path = tmp_path / "run" / "controller.sock"
    return WorkloadControllerServer(
        workspace=workspace, socket_path=socket_path,
        docker_socket=tmp_path / "unused-docker.sock", state_path=state_path,
        network="test-net", allowed_uid=os.getuid(), socket_uid=os.getuid(),
        socket_gid=os.getgid(), workload_uid=os.getuid(), workload_gid=os.getgid(),
    ), socket_path, state_path


async def wait_for_controller(socket_path):
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        try:
            return await asyncio.open_unix_connection(socket_path)
        except (FileNotFoundError, ConnectionRefusedError):
            await asyncio.sleep(0.01)
    pytest.fail("Controller listener did not become ready")


@pytest.mark.asyncio
async def test_controller_cancel_closes_idle_request_and_own_socket(tmp_path):
    server, socket_path, state_path = controller_server(tmp_path)
    serving = asyncio.create_task(server.serve())
    reader, writer = await wait_for_controller(socket_path)
    try:
        serving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(serving, 5)
        assert await asyncio.wait_for(reader.read(), 2) == b""
        assert not socket_path.exists()
        with state_path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        writer.close()
        await writer.wait_closed()
        if not serving.done():
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.asyncio
async def test_controller_cancel_closes_accepted_request_before_handle_starts(
    tmp_path, monkeypatch,
):
    """A queued handler still owns its accepted Unix connection on stop."""
    server, socket_path, state_path = controller_server(tmp_path)
    serving = asyncio.create_task(server.serve())
    handle_entered = asyncio.Event()
    accepted = asyncio.Event()
    real_handle = server._handle
    real_accept = server._accept

    async def checked_handle(reader, writer):
        handle_entered.set()
        await real_handle(reader, writer)

    def accept_then_cancel(reader, writer):
        # Queue serve's cancellation ahead of the new handler's first step.
        serving.cancel()
        real_accept(reader, writer)
        accepted.set()

    monkeypatch.setattr(server, "_handle", checked_handle)
    monkeypatch.setattr(server, "_accept", accept_then_cancel)
    reader, writer = await wait_for_controller(socket_path)
    try:
        await asyncio.wait_for(accepted.wait(), 2)
        done, _ = await asyncio.wait({serving}, timeout=1)
        assert serving in done
        assert not handle_entered.is_set()
        with pytest.raises(asyncio.CancelledError):
            await serving
        assert await asyncio.wait_for(reader.read(), 1) == b""
        assert not socket_path.exists()
        assert not state_path.exists()
        with state_path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        writer.close()
        await writer.wait_closed()
        if not serving.done():
            serving.cancel()
            done, _ = await asyncio.wait({serving}, timeout=2)
            assert serving in done


@pytest.mark.asyncio
async def test_controller_cancel_waits_for_accepted_effect(tmp_path, monkeypatch):
    """A lost reply must not cancel an already accepted external effect."""

    server, socket_path, _state_path = controller_server(tmp_path)
    entered, release = asyncio.Event(), asyncio.Event()
    completed = []

    async def delayed_request(method, path, *, expected, body=None):
        entered.set()
        await release.wait()
        completed.append((method, path))
        return []

    monkeypatch.setattr(server._engine, "request", delayed_request)
    serving = asyncio.create_task(server.serve())
    _reader, writer = await wait_for_controller(socket_path)
    try:
        writer.write(
            (f'{{"version":1,"action":"cleanup_candidates","body":'
             f'{{"workspace_id":"{server._workspace_id}"}}}}\n').encode()
        )
        await writer.drain()
        await asyncio.wait_for(entered.wait(), 2)
        serving.cancel()
        await asyncio.sleep(0)
        assert not serving.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(serving, 5)
        assert len(completed) == 1
        assert completed[0][0] == "GET"
        assert not socket_path.exists()
    finally:
        release.set()
        writer.close()
        await writer.wait_closed()
        if not serving.done():
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.asyncio
async def test_controller_accepts_already_removed_own_socket(tmp_path):
    server, socket_path, _state_path = controller_server(tmp_path)
    serving = asyncio.create_task(server.serve())
    reader, writer = await wait_for_controller(socket_path)
    writer.write(b'{"version":1,"action":"probe","body":{}}\n')
    await writer.drain()
    assert b'"ok":false' in await reader.readline()
    writer.close()
    await writer.wait_closed()
    socket_path.unlink()
    serving.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(serving, 5)
    assert not socket_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["file", "symlink", "socket"])
async def test_controller_refuses_foreign_socket_path_replacement(tmp_path, replacement):
    server, socket_path, state_path = controller_server(tmp_path)
    serving = asyncio.create_task(server.serve())
    reader, writer = await wait_for_controller(socket_path)
    writer.write(b'{"version":1,"action":"probe","body":{}}\n')
    await writer.drain()
    assert b'"ok":false' in await reader.readline()
    writer.close()
    await writer.wait_closed()

    owned_path = socket_path.with_name("owned.sock")
    socket_path.rename(owned_path)
    foreign = None
    if replacement == "file":
        socket_path.write_bytes(b"foreign file")
    elif replacement == "symlink":
        socket_path.symlink_to(tmp_path / "foreign-target")
    else:
        foreign = socket.socket(socket.AF_UNIX)
        foreign.bind(str(socket_path))
    before = socket_path.lstat()
    try:
        serving.cancel()
        with pytest.raises(RuntimeError, match="已被替换"):
            await asyncio.wait_for(serving, 5)
        after = socket_path.lstat()
        assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
        with state_path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
    finally:
        if foreign is not None:
            foreign.close()
        owned_path.unlink()
        if not serving.done():
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


def receipt(workspace_id):
    return WorkloadStopReceipt(
        WorkloadLease(workspace_id, "old-plugin", "desktop", "candidate",
                      "old-request", "old-resource", "old-container", "old-spec"),
        container_absent=True, mounts_released=True,
    )


class Controller:
    def __init__(self):
        self.calls = []
        self.receipts = None

    async def cleanup_candidates(self, workspace_id):
        self.calls.append(workspace_id)
        return (receipt(workspace_id),) if self.receipts is None else self.receipts

    async def start(self, request):
        raise AssertionError("boot cleanup 不得重放 start")

    async def stop(self, lease):
        raise AssertionError("boot 必须沿 Controller 的候选清理原子操作")


def setup(tmp_path, controller, *, initialize=True):
    """为正式 boot 固定独立 workspace 和能观察 apply 的普通插件。"""
    source = tmp_path / "plugins" / "probe"
    source.mkdir(parents=True)
    (source / "plugin.py").write_text(
        "api_version = 3\nname = 'probe'\nversion = '1.0'\n"
        "async def apply(ctx):\n"
        "    marker = ctx.runtime.workspace / 'applied.txt'\n"
        "    with marker.open('a') as stream:\n"
        "        stream.write('applied\\n')\n",
    )
    workspace = tmp_path / "workspace"
    if initialize:
        initialize_plugin_workspace(workspace)
    return manager(tmp_path, controller)


def manager(tmp_path, controller):
    return PluginManager(
        [tmp_path / "plugins"], event_bus=EventBus(), workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache", workload_controller=controller,
    )


async def close(owner):
    await owner.terminate_all()


@pytest.mark.asyncio
async def test_new_boot_cleans_once_before_apply_and_root_changes_do_not(tmp_path):
    controller = Controller()
    owner = setup(tmp_path, controller)
    expected = hashlib.sha256(str((tmp_path / "workspace").resolve()).encode()).hexdigest()[:16]
    try:
        await owner.load_all()
        assert controller.calls == [expected]
        assert (tmp_path / "workspace" / "applied.txt").read_text() == "applied\n"
        live_root = owner.live_root
        old_generation = owner.generation("probe")
        assert live_root is not None and old_generation is not None
        with pytest.raises(RuntimeError, match="不能重复启动"):
            await owner.load_all()
        source = tmp_path / "plugins/probe/plugin.py"
        source.write_text(source.read_text() + "\n# changed input\n")
        changed = await owner.reconcile_changed()
        assert changed[0]["publication_state"] == "active"
        assert owner.live_root is live_root
        assert owner.generation("probe") is not old_generation
        assert controller.calls == [expected]
    finally:
        await close(owner)
    # 真正的新 Manager 从同一 durable stable 启动，再清理一次旧 boot 候选。
    second = manager(tmp_path, controller)
    try:
        await second.load_all()
        assert controller.calls == [expected, expected]
    finally:
        await close(second)


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [None, "broken", '{"version":1,"root_ref":"missing"}'])
async def test_invalid_selection_never_calls_controller_or_apply(tmp_path, raw):
    controller = Controller()
    owner = setup(tmp_path, controller, initialize=False)
    if raw is not None:
        owner._selection.path.parent.mkdir(parents=True, exist_ok=True)
        owner._selection.path.write_text(raw)
    try:
        with pytest.raises(SelectionFormatError):
            await owner.load_all()
        assert controller.calls == []
        assert not (tmp_path / "workspace" / "applied.txt").exists()
    finally:
        await close(owner)

@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["container", "mounts", "workspace", "formal"])
async def test_bad_cleanup_receipt_blocks_boot_and_keeps_failure_owner(tmp_path, failure):
    controller = Controller()
    owner = setup(tmp_path, controller)
    valid = receipt(owner._workload_workspace_id)
    bad = {
        "container": replace(valid, container_absent=False),
        "mounts": replace(valid, mounts_released=False),
        "workspace": replace(valid, lease=replace(valid.lease, workspace_id="foreign")),
        "formal": replace(valid, lease=replace(valid.lease, mode="formal")),
    }[failure]
    controller.receipts = (valid, bad)
    try:
        with pytest.raises(RuntimeError, match="boot cleanup 未确认") as error:
            await owner.load_all()
        assert owner._operation.task.exception() is error.value
        assert "old-container" in str(error.value)
        assert controller.receipts == (valid, bad)
        assert owner.live_root is None
        assert owner._selection.read() is None
        assert not (tmp_path / "workspace" / "applied.txt").exists()
    finally:
        await close(owner)


@pytest.mark.asyncio
async def test_unknown_cleanup_result_is_not_success_or_replayed(tmp_path):
    unknown = WorkloadEffectUnknown("cleanup reply lost")
    class UnknownController(Controller):
        async def cleanup_candidates(self, workspace_id):
            self.calls.append(workspace_id)
            raise unknown
    controller = UnknownController()
    owner = setup(tmp_path, controller)
    try:
        with pytest.raises(WorkloadEffectUnknown) as error:
            await owner.load_all()
        assert error.value is unknown
        assert owner._operation.task.exception() is unknown
        assert len(controller.calls) == 1
        assert owner.live_root is None
        assert not (tmp_path / "workspace" / "applied.txt").exists()
    finally:
        await close(owner)


@pytest.mark.asyncio
async def test_cancelled_boot_retains_pending_task_and_never_applies_after_late_receipt(tmp_path):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    class DelayedController(Controller):
        async def cleanup_candidates(self, workspace_id):
            self.calls.append(workspace_id)
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            return (receipt(workspace_id),)
    controller = DelayedController()
    owner = setup(tmp_path, controller)
    boot = asyncio.create_task(owner.load_all())
    try:
        await entered.wait()
        assert not (tmp_path / "workspace" / "applied.txt").exists()
        operation = owner._operation
        boot.cancel()
        with pytest.raises(asyncio.CancelledError):
            await boot
        await cancelled.wait()
        assert operation.revoked and not operation.task.done()
        with pytest.raises(OperationBusyError):
            await owner.load_all()
        assert owner._operation is operation
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation.task
        assert owner.live_root is None
        assert owner._selection.read() is None
        assert not (tmp_path / "workspace" / "applied.txt").exists()
        assert len(controller.calls) == 1
    finally:
        release.set()
        await asyncio.gather(boot, return_exceptions=True)
        await close(owner)


@pytest.mark.asyncio
async def test_expired_boot_does_not_apply_after_confirmed_cleanup(tmp_path):
    class ExpiredController(Controller):
        async def cleanup_candidates(self, workspace_id):
            self.calls.append(workspace_id)
            owner._operation.deadline = asyncio.get_running_loop().time()
            return (receipt(workspace_id),)
    owner = setup(tmp_path, ExpiredController())
    try:
        with pytest.raises(OperationTimeoutError):
            await owner.load_all()
        assert owner.live_root is None
        assert not (tmp_path / "workspace" / "applied.txt").exists()
    finally:
        await close(owner)
