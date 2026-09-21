"""候选初始化失败与校验目录清理的所有权合同。"""
import json
import os
import shutil
from pathlib import Path

import pytest

from agent.plugins.artifacts import read_pointers
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.test_plugin_update_rollback import prepare


@pytest.mark.asyncio
async def test_reconcile_candidate_init_failure_is_queryable_and_settles_only_on_discard(
    tmp_path, monkeypatch,
):
    """登台前初始化失败：错误登记在安装 owner，latest 指针不回退，不自动重试。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        stable = host.current_snapshot
        plugin_base = old.installed_path.parents[1]
        original_pointers = read_pointers(plugin_base)
        # 1. 只登台安装制品与 armed 更新，装配留给 reconcile。
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )
        staged_pointers = read_pointers(plugin_base)
        assert staged_pointers != original_pointers

        calls = 0
        original_load = host._load_one

        async def fail_load(mod, **kwargs):
            nonlocal calls
            if "probe" in str(mod.get("plugin_root", "")):
                calls += 1
                raise ValueError("injected init failure")
            return await original_load(mod, **kwargs)

        monkeypatch.setattr(host, "_load_one", fail_load)
        results = await host.reconcile_changed()

        # 2. 原错误登记在 armed 更新上；latest staging 指针不因失败被丢弃。
        failed = [
            item for item in results
            if item.get("plugin_id") == "probe@lab"
            and item.get("preparation_state") == "failed"
        ]
        assert failed and "injected init failure" in str(failed[0].get("error"))
        update = host.reload_journal.armed_update_for_plugin("probe@lab")
        assert update is not None and update.update_id == result.update_id
        assert "injected init failure" in update.error
        assert update.phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert host.current_snapshot is stable

        # 3. 已记录的初始化失败不自动重试。
        monkeypatch.setattr(host, "_load_one", fail_load)
        again = await host.reconcile_changed()
        assert calls == 1
        assert any(
            item.get("plugin_id") == "probe@lab"
            and item.get("preparation_state") == "failed"
            and "injected init failure" in str(item.get("error"))
            for item in again
        )

        # 4. 显式 discard 仍按原协议结算安装恢复点与指针。
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
        assert host.current_snapshot is stable
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_journal_failure_after_commit_keeps_real_error_and_recovery_owner(
    tmp_path, monkeypatch,
):
    """promote 提交后的 durable 收尾失败：保留真实错误与已提交 Root owner。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        stable = host.current_snapshot
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        tx_id = host.reload_journal.update(result.update_id).reload_tx_id
        assert tx_id is not None

        # 1. 提交已成功、仅 draining 收尾的 journal 写入失败。
        original_advance = host.reload_journal.advance
        def fail_advance(failed_tx, phase, **kwargs):
            if phase == "draining":
                raise OSError("injected journal failure")
            return original_advance(failed_tx, phase, **kwargs)
        monkeypatch.setattr(host.reload_journal, "advance", fail_advance)

        # 2. 真实原错误原样抛出，不被 AssertionError 掩盖。
        with pytest.raises(OSError, match="injected journal failure"):
            await host.switch_ready("probe@lab")

        # 3. 已发生发布不回滚：新 Root 仍是 current，事务保留为恢复 owner。
        publication = host._publication
        assert publication is not None and publication.must_retain
        promoted = host.current_snapshot
        assert promoted is publication.candidate and promoted is not stable
        assert promoted.accepting_leases is False
        assert stable.state == "retired"
        assert host.reload_journal.get(tx_id).phase == "committed"

        # 4. 未结算的 drain/保留 owner 期间新安装被拒，不静默接管已提交 owner。
        with pytest.raises(RuntimeError, match="drain 失败"):
            await host.install_candidate(
                source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
            )
    finally:
        await host.terminate_all()

    # 5. 恢复后结算既有事务并可接受新更新；外部效果按新启动只发生一次。
    recovered = PluginManager([], event_bus=EventBus(), workspace=workspace,
                              installed_cache_root=home / "cache")
    try:
        await recovered.load_all()
        current = recovered.current_snapshot
        assert current is not None
        assert current.generations["probe@lab"].runtime_snapshot is current
        record = recovered.reload_journal.get(tx_id)
        # 重启结算把 committed 事务推到终态，不再停留等待 drain。
        assert record.phase in {"recovered", "complete"}
        result2, _ = await recovered.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        assert result2.update_id != result.update_id
    finally:
        await recovered.terminate_all()


@pytest.mark.asyncio
async def test_candidate_validation_cleanup_failure_is_retained_and_retryable(
    tmp_path, monkeypatch,
):
    """候选校验目录清理失败保留 owner，原样抛出，显式重试完成。"""
    import agent.plugins.manager as manager_module

    source, home, workspace, _ = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        candidate_root = host.latest_snapshot.composition_root
        validation_parent = workspace / "runtime" / "plugin-validation"
        validation_root = next(iter(validation_parent.iterdir()))
        assert validation_root.is_dir()

        calls: list[object] = []
        original_remove = manager_module._remove_candidate_validation_root

        def fail_once(root, ws):
            calls.append(root)
            if len(calls) == 1:
                raise OSError("injected cleanup failure")
            return original_remove(root, ws)

        with monkeypatch.context() as patch:
            patch.setattr(
                manager_module, "_remove_candidate_validation_root", fail_once,
            )
            with pytest.raises(Exception):
                await host.discard_update(result.update_id)
        # 清理失败保持可查：deferred cleanup owner 仍持有该资源，目录未被吞掉。
        retained = [
            name for name, _ in candidate_root._internal_cleanups
            if name.startswith("validation-root:")
        ]
        assert retained
        assert validation_root.exists()
        assert host.reload_journal.update(result.update_id).phase == "armed"

        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert not validation_root.exists()
        assert calls == [validation_root]
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_candidate_data_dir_is_empty_and_isolated_from_formal_data(tmp_path):
    """正式 plugin-data 不复制进候选；候选数据目录初始为空且不穿透符号链接。"""
    source, home, workspace, old = prepare(tmp_path)
    formal = old.data_path
    (formal / "formal.txt").write_text("formal durable data")
    escape = tmp_path / "escape-target"
    escape.mkdir()
    (escape / "stolen.txt").write_text("must not be reached")
    (formal / "escape").symlink_to(escape, target_is_directory=True)

    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        result, _ = await host.install_candidate(
            source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
        )
        generation = host.latest_snapshot.generations["probe@lab"]
        candidate_data = generation.data_dir
        assert candidate_data.resolve() != formal.resolve()
        assert "plugin-validation" in candidate_data.parts
        assert not (candidate_data / "formal.txt").exists()
        assert not (candidate_data / "escape").exists()
        # 候选数据目录边界校验拒绝逃逸与符号链接路径。
        for entry in candidate_data.iterdir():
            assert not entry.is_symlink()
        await host.discard_update(result.update_id)
        assert (formal / "formal.txt").read_text() == "formal durable data"
        assert (escape / "stolen.txt").read_text() == "must not be reached"
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_init_and_cleanup_failure_blocks_discard_until_owner_cleaned(
    tmp_path, monkeypatch,
):
    """初始化失败叠加清理失败：Root 保留为 owner，discard 不能假结算指针。"""
    import agent.plugins.manager as manager_module

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        stable = host.current_snapshot
        plugin_base = old.installed_path.parents[1]
        original_pointers = read_pointers(plugin_base)
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )
        staged_pointers = read_pointers(plugin_base)
        assert staged_pointers != original_pointers

        # 1. 双故障：候选 Root 装配失败且 validation 目录清理失败。
        original_compile = host._snapshot_compiler.compile
        def fail_compile(*args, **kwargs):
            raise ValueError("injected init failure")
        monkeypatch.setattr(host._snapshot_compiler, "compile", fail_compile)

        cleanups: list[object] = []
        original_remove = manager_module._remove_candidate_validation_root
        def fail_remove(root, ws):
            cleanups.append(root)
            raise OSError("injected cleanup failure")
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", fail_remove,
        )

        results = await host.reconcile_changed()
        assert any(
            item.get("plugin_id") == "probe@lab"
            and item.get("preparation_state") == "failed"
            for item in results
        )
        update = host.reload_journal.armed_update_for_plugin("probe@lab")
        assert update is not None and update.update_id == result.update_id
        assert update.reload_tx_id is None
        # 清理失败保留实际 Root owner，且关联到本插件的 armed 更新。
        assert host._building_roots
        assert host._failed_candidate_roots.get("probe@lab")

        # 2. 显式 discard 先重试 owner 清理；清理仍失败则拒绝结算指针。
        with pytest.raises(Exception):
            await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert host._building_roots

        # 3. 修复清理后重试 discard：先完成 owner 清理，再结算指针。
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", original_remove,
        )
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
        assert not host._building_roots
        assert not host._failed_candidate_roots.get("probe@lab")
        assert host.current_snapshot is stable
        assert len(cleanups) >= 1
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_install_candidate_double_fault_keeps_update_armed(
    tmp_path, monkeypatch,
):
    """install_candidate 登台失败叠加清理失败：except 不得直接结算 armed 更新。"""
    import agent.plugins.manager as manager_module

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        plugin_base = old.installed_path.parents[1]
        original_pointers = read_pointers(plugin_base)

        def fail_compile(*args, **kwargs):
            raise ValueError("injected init failure")
        monkeypatch.setattr(host._snapshot_compiler, "compile", fail_compile)
        keep_failing = True
        original_remove = manager_module._remove_candidate_validation_root
        def fail_remove(root, ws):
            if keep_failing:
                raise OSError("injected cleanup failure")
            return original_remove(root, ws)
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", fail_remove,
        )

        # 1. 双故障下安装入口原样抛出初始化与清理两份错误，更新保持 armed 而非 rolled_back。
        with pytest.raises(Exception, match="构建和清理均失败"):
            await host.install_candidate(
                source=str(source), marketplace="lab", ref_name="", sparse_paths=[],
            )
        update = host.reload_journal.armed_update_for_plugin("probe@lab")
        assert update is not None
        assert update.phase == "armed" and update.reload_tx_id is None
        assert "candidate preparation failed" in update.error
        staged_pointers = read_pointers(plugin_base)
        assert staged_pointers != original_pointers
        assert host._failed_candidate_roots.get("probe@lab")
        validation_parent = workspace / "runtime" / "plugin-validation"
        validation_root = next(iter(validation_parent.iterdir()))
        assert validation_root in {o.validation_root for o in host.reload_journal.candidate_cleanup(update.update_id)}

        # 2. 清理义务未清完时 discard 被拒，指针不结算。
        with pytest.raises(Exception):
            await host.discard_update(update.update_id)
        assert host.reload_journal.update(update.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert validation_root.exists()

        # 3. 修复清理后显式 discard：先完成真实清理，再结算指针。
        keep_failing = False
        await host.discard_update(update.update_id)
        assert host.reload_journal.update(update.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
        assert not validation_root.exists()
        assert host.reload_journal.candidate_cleanup(update.update_id) == ()
    finally:
        monkeypatch.undo()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_init_cleanup_failure_discard_recovers_after_restart(
    tmp_path, monkeypatch,
):
    """双故障下 discard 被拒后重启：armed 更新保留，清理完成后才结算。"""
    import agent.plugins.manager as manager_module

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    plugin_base = old.installed_path.parents[1]
    original_pointers = read_pointers(plugin_base)
    try:
        await host.load_all()
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )
        staged_pointers = read_pointers(plugin_base)

        def fail_compile(*args, **kwargs):
            raise ValueError("injected init failure")
        monkeypatch.setattr(host._snapshot_compiler, "compile", fail_compile)
        def fail_remove(root, ws):
            raise OSError("injected cleanup failure")
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", fail_remove,
        )

        await host.reconcile_changed()
        with pytest.raises(Exception):
            await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
    finally:
        # 解除清理故障后 terminate 能完成保留 Root 的真实清理。
        monkeypatch.undo()
        await host.terminate_all()

    # 重启后 armed 更新仍在；无 retained Root，显式 discard 结算指针。
    recovered = PluginManager([], event_bus=EventBus(), workspace=workspace,
                              installed_cache_root=home / "cache")
    try:
        await recovered.load_all()
        update = recovered.reload_journal.update(result.update_id)
        assert update.phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        await recovered.discard_update(result.update_id)
        assert recovered.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
    finally:
        await recovered.terminate_all()


@pytest.mark.asyncio
async def test_live_old_host_blocks_journal_recovery_and_settlement(
    tmp_path, monkeypatch,
):
    """旧宿主同进程存活：新 Manager 不得凭 journal 抢删其确切校验目录。"""
    import agent.plugins.manager as manager_module

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    plugin_base = old.installed_path.parents[1]
    original_pointers = read_pointers(plugin_base)
    try:
        await host.load_all()
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )
        staged_pointers = read_pointers(plugin_base)

        def fail_compile(*args, **kwargs):
            raise ValueError("injected init failure")
        monkeypatch.setattr(host._snapshot_compiler, "compile", fail_compile)
        keep_failing = True
        original_remove = manager_module._remove_candidate_validation_root
        def fail_remove(root, ws):
            if keep_failing:
                raise OSError("injected cleanup failure")
            return original_remove(root, ws)
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", fail_remove,
        )

        await host.reconcile_changed()
        update = host.reload_journal.armed_update_for_plugin("probe@lab")
        assert update is not None
        obligations = host.reload_journal.candidate_cleanup(result.update_id)
        validation_root = obligations[0].validation_root
        assert validation_root.exists()
        # 义务记录必须携带创建它的确切宿主身份。
        assert obligations[0].owner_pid == os.getpid()
        assert obligations[0].owner_boot_id == host._host_boot_id
    finally:
        monkeypatch.undo()

    # 旧宿主对象仍存活：同进程新 Manager 没有旧 owner 退出证据，拒绝接管清理。
    recovered = PluginManager([], event_bus=EventBus(), workspace=workspace,
                              installed_cache_root=home / "cache")
    try:
        await recovered.load_all()
        assert recovered.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert validation_root.exists()
        with pytest.raises(RuntimeError, match="旧宿主仍在本进程存活"):
            await recovered.discard_update(result.update_id)
        assert recovered.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert validation_root.exists()
        assert recovered.reload_journal.candidate_cleanup(result.update_id) != ()
    finally:
        await recovered.terminate_all()

    # 旧 owner 自行 dispose：同一义务沿原宿主结算，而非被新 owner 抢删。
    await host.discard_update(result.update_id)
    assert host.reload_journal.update(result.update_id).phase == "rolled_back"
    assert read_pointers(plugin_base) == original_pointers
    assert not validation_root.exists()
    await host.terminate_all()


@pytest.mark.asyncio
async def test_sigkill_subprocess_recovers_durable_validation_root(
    tmp_path, monkeypatch,
):
    """真实独立子进程 SIGKILL：旧宿主进程死亡证据齐备后才删除确切目录并结算。"""
    import os
    import signal
    import subprocess
    import sys

    source, home, workspace, old = prepare(tmp_path)
    plugin_base = old.installed_path.parents[1]
    original_pointers = read_pointers(plugin_base)
    result = install_git_plugin(
        workspace=workspace, source=str(source), marketplace="lab",
        plugins_home=home, stage_candidate=True,
    )
    staged_pointers = read_pointers(plugin_base)

    child = tmp_path / "sigkill_child.py"
    child.write_text('''
import asyncio, json, os, sys
sys.path.insert(0, {repo!r})
from pathlib import Path
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus

async def main():
    host = PluginManager([], event_bus=EventBus(), workspace=Path(sys.argv[1]),
                         installed_cache_root=Path(sys.argv[2]) / "cache")
    await host.load_all()
    def fail_compile(*args, **kwargs):
        raise ValueError("injected init failure")
    host._snapshot_compiler.compile = fail_compile
    import agent.plugins.manager as manager_module
    def fail_remove(root, ws):
        raise OSError("injected cleanup failure")
    manager_module._remove_candidate_validation_root = fail_remove
    await host.reconcile_changed()
    update = host.reload_journal.armed_update_for_plugin("probe@lab")
    obligations = host.reload_journal.candidate_cleanup(update.update_id)
    print(json.dumps({{
        "update_id": update.update_id,
        "root": str(obligations[0].validation_root),
        "pid": os.getpid(),
        "boot_id": host._host_boot_id,
    }}), flush=True)
    await asyncio.Event().wait()

asyncio.run(main())
'''.format(repo=str(Path(__file__).resolve().parents[1])))
    proc = subprocess.Popen(
        [sys.executable, str(child), str(workspace), str(home)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line = proc.stdout.readline() if proc.stdout is not None else ""
        assert line, (proc.stderr.read() if proc.stderr is not None else "")
        evidence = json.loads(line)
        assert evidence["update_id"] == result.update_id
        assert evidence["pid"] == proc.pid
        validation_root = Path(evidence["root"])
        assert validation_root.exists()
        # 真实 SIGKILL：无 finally、无 dispose，旧宿主进程直接死亡。
        proc.kill()
        proc.wait()
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

    recovered = PluginManager([], event_bus=EventBus(), workspace=workspace,
                              installed_cache_root=home / "cache")
    try:
        await recovered.load_all()
        assert recovered.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert validation_root.exists()
        # 旧宿主进程已死（pid 不存在）：新 Manager 删除确切旧目录后才结算指针。
        await recovered.discard_update(result.update_id)
        assert recovered.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
        assert not validation_root.exists()
        assert recovered.reload_journal.candidate_cleanup(result.update_id) == ()
    finally:
        await recovered.terminate_all()


@pytest.mark.asyncio
async def test_record_candidate_cleanup_failure_still_cleans_root_and_keeps_error(
    tmp_path, monkeypatch,
):
    """清理义务记录失败：Root 仍走同一阶段真实清理，原错不丢，更新如实回退。"""
    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        plugin_base = old.installed_path.parents[1]
        original_pointers = read_pointers(plugin_base)
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )

        def fail_record(*args, **kwargs):
            raise OSError("injected record failure")
        monkeypatch.setattr(
            host.reload_journal, "record_candidate_cleanup", fail_record,
        )

        # 1. 记录失败的原错登记在安装 owner；reconcile 按插件收敛为 failed 结果。
        results = await host.reconcile_changed()
        failed = [
            item for item in results
            if item.get("plugin_id") == "probe@lab"
            and item.get("preparation_state") == "failed"
        ]
        assert failed and "injected record failure" in str(failed[0].get("error"))

        # 2. Root 被真实清理：无在轨构建 Root、无 retained owner、无残留目录。
        assert not host._building_roots
        assert "probe@lab" not in host._failed_candidate_roots
        validation_parent = workspace / "runtime" / "plugin-validation"
        assert not validation_parent.exists() or not any(validation_parent.iterdir())
        # 义务从未入账；更新保持 armed 并携带原错，由显式 discard 结算。
        update = host.reload_journal.update(result.update_id)
        assert update.phase == "armed"
        assert "injected record failure" in update.error
        assert host.reload_journal.candidate_cleanup(result.update_id) == ()
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
    finally:
        monkeypatch.undo()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_record_and_cleanup_double_fault_retains_owner_and_blocks_settlement(
    tmp_path, monkeypatch,
):
    """记录失败叠加清理失败：owner 保留、armed 更新拒绝指针结算，修复后 discard。"""
    import agent.plugins.manager as manager_module

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        plugin_base = old.installed_path.parents[1]
        original_pointers = read_pointers(plugin_base)
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )
        staged_pointers = read_pointers(plugin_base)

        def fail_record(*args, **kwargs):
            raise OSError("injected record failure")
        monkeypatch.setattr(
            host.reload_journal, "record_candidate_cleanup", fail_record,
        )
        keep_failing = True
        original_remove = manager_module._remove_candidate_validation_root
        def fail_remove(root, ws):
            if keep_failing:
                raise OSError("injected cleanup failure")
            return original_remove(root, ws)
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", fail_remove,
        )

        # 1. 双故障：record 原错与清理失败一起登记，Root owner 保留到 armed 更新。
        results = await host.reconcile_changed()
        failed = [
            item for item in results
            if item.get("plugin_id") == "probe@lab"
            and item.get("preparation_state") == "failed"
        ]
        assert failed and "构建和清理均失败" in str(failed[0].get("error"))
        update = host.reload_journal.update(result.update_id)
        assert update.phase == "armed"
        assert "构建和清理均失败" in update.error
        assert host._failed_candidate_roots.get("probe@lab")
        assert read_pointers(plugin_base) == staged_pointers
        # 义务没入账但 Root 保留：指针结算仍被 retained owner 阻断。
        with pytest.raises(Exception):
            await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "armed"

        # 2. 修复后显式 discard：旧 owner 真实清理完成后才结算指针。
        keep_failing = False
        await host.discard_update(result.update_id)
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
    finally:
        monkeypatch.undo()
        await host.terminate_all()


@pytest.mark.asyncio
async def test_rollback_updates_raises_on_unsettled_candidate_cleanup(
    tmp_path, monkeypatch,
):
    """rollback_updates 对未清完的清理义务必须显式失败，不得静默 continue。"""
    from agent.plugins.reload_journal import (
        CandidateCleanupPendingError,
        ReloadJournal,
    )

    source, home, workspace, old = prepare(tmp_path)
    host = PluginManager([], event_bus=EventBus(), workspace=workspace,
                         installed_cache_root=home / "cache")
    try:
        await host.load_all()
        plugin_base = old.installed_path.parents[1]
        original_pointers = read_pointers(plugin_base)
        result = install_git_plugin(
            workspace=workspace, source=str(source), marketplace="lab",
            plugins_home=home, stage_candidate=True,
        )
        staged_pointers = read_pointers(plugin_base)
        update = host.reload_journal.update(result.update_id)
        assert update.phase == "armed"

        # 1. 伪造一条仍欠的清理义务：rollback 必须显式失败而非正常返回。
        pending_dir = (
            workspace / "runtime" / "plugin-validation" / "pending-evidence"
        )
        pending_dir.mkdir(parents=True)
        journal = ReloadJournal(workspace)
        journal.record_candidate_cleanup(
            result.update_id, "probe@lab", pending_dir / "workspace",
            owner_boot_id="other-host", owner_pid=os.getpid(),
        )
        with pytest.raises(CandidateCleanupPendingError) as caught:
            journal.rollback_updates(
                home, update_id=result.update_id, error="explicit discard",
            )
        assert caught.value.update_ids == (result.update_id,)
        # 2. 更新保持 armed、指针不结算、错误被如实标注。
        update = host.reload_journal.update(result.update_id)
        assert update.phase == "armed"
        assert "candidate validation cleanup pending" in update.error
        assert read_pointers(plugin_base) == staged_pointers
        # 3. 义务清完后同一入口正常结算。
        journal.clear_candidate_cleanup(
            result.update_id, pending_dir / "workspace",
        )
        journal.rollback_updates(
            home, update_id=result.update_id, error="explicit discard",
        )
        assert host.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
    finally:
        await host.terminate_all()
