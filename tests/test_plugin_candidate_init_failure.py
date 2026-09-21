"""候选初始化失败与校验目录清理的所有权合同。"""
import shutil

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
        assert validation_root in host.reload_journal.candidate_cleanup(update.update_id)

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
async def test_restart_cleans_durable_validation_root_before_settling(
    tmp_path, monkeypatch,
):
    """模拟进程被杀：journal 义务让新 Manager 删除确切旧校验目录后才结算指针。"""
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
        validation_parent = workspace / "runtime" / "plugin-validation"
        validation_root = next(iter(validation_parent.iterdir()))
        assert validation_root.exists()
        assert validation_root in host.reload_journal.candidate_cleanup(result.update_id)
        # 模拟进程被杀：不优雅 terminate，旧 host 的内存 owner 直接废弃。
        host._stopping = True
    finally:
        monkeypatch.undo()

    recovered = PluginManager([], event_bus=EventBus(), workspace=workspace,
                              installed_cache_root=home / "cache")
    try:
        await recovered.load_all()
        assert recovered.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert validation_root.exists()
        # 内存 owner 已丢；journal 义务在清理故障未解除时仍拒绝结算。
        keep_failing = True
        monkeypatch.setattr(
            manager_module, "_remove_candidate_validation_root", fail_remove,
        )
        with pytest.raises(OSError, match="injected cleanup failure"):
            await recovered.discard_update(result.update_id)
        assert recovered.reload_journal.update(result.update_id).phase == "armed"
        assert read_pointers(plugin_base) == staged_pointers
        assert validation_root.exists()
        # 修复后重试：先删除确切旧目录，指针才允许结算。
        keep_failing = False
        await recovered.discard_update(result.update_id)
        assert recovered.reload_journal.update(result.update_id).phase == "rolled_back"
        assert read_pointers(plugin_base) == original_pointers
        assert not validation_root.exists()
        assert recovered.reload_journal.candidate_cleanup(result.update_id) == ()
    finally:
        await recovered.terminate_all()
        await host.terminate_all()
