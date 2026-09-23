"""新 Core boot 的宿主候选清理；不执行 Docker，也不重放 start。"""
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace

import pytest

from agent.plugins._operation import OperationBusyError, OperationTimeoutError
from agent.plugins.manager import PluginManager
from agent.plugins.selection import SelectionFormatError
from agent.workloads.client import WorkloadEffectUnknown
from agent.plugin_composition.execution import WorkloadLease, WorkloadStopReceipt
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


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
