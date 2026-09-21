"""ExecutionGrant 签发边界：spawn 只消费本授权签发的 PreparedProcess，adopt 只接管已登记 child。"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import agent.host_bridge.plugin_execution as plugin_execution
from agent.host_bridge.plugin_execution import CodeOwner, ExecutionAccess
from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.execution import EXECUTION, PreparedProcess


async def _bound_grant(tmp_path: Path, owner: str):
    """真实 CompositionRoot + ExecutionAccess.bind，返回 (root, grant, code_dir)。"""
    code_dir = tmp_path / owner
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "child.py").write_text("import time\ntime.sleep(30)\n")
    data_dir = tmp_path / (owner + "-data")
    data_dir.mkdir(exist_ok=True)
    root = CompositionRoot(f"grant-{owner}")
    execution = ExecutionAccess(
        root.instance_token,
        {
            owner: CodeOwner(
                owner + "-gen",
                code_dir,
                lambda command, cwd: (
                    sys.executable, str(code_dir / command[0]), *command[1:],
                ),
            )
        },
        candidate=True,
    )
    await root.context.provide(EXECUTION, execution)
    grants = []

    async def apply(ctx):
        grants.append(ctx.require(EXECUTION).bind(ctx))

    await root.mount(
        apply,
        name=owner,
        runtime=PluginRuntime(owner, owner + "-gen", code_dir, data_dir, tmp_path, {}),
    )
    assert grants
    return root, grants[0], code_dir


@pytest.mark.asyncio
async def test_spawn_rejects_unsigned_forged_and_cross_grant_prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_calls = []
    real_spawn = plugin_execution.spawn_process

    async def spy(*args, **kwargs):
        spawn_calls.append(args)
        return await real_spawn(*args, **kwargs)

    monkeypatch.setattr(plugin_execution, "spawn_process", spy)
    root_a, grant_a, code_a = await _bound_grant(tmp_path, "consumer")
    root_b, grant_b, code_b = await _bound_grant(tmp_path, "other")
    try:
        forged = PreparedProcess(
            object(),
            command=(sys.executable, "-c", "pass"),
            cwd=str(code_a),
            env={},
        )
        with pytest.raises(PermissionError):
            await grant_a.spawn(forged)
        # 同一授权签发的制品不能交给另一份 grant 消费。
        prepared_b = grant_b.prepare_process(("child.py",), ".", {})
        with pytest.raises(PermissionError):
            await grant_a.spawn(prepared_b)
        # 非 PreparedProcess 输入同样拒绝。
        with pytest.raises(PermissionError):
            await grant_a.spawn(("child.py",))  # type: ignore[arg-type]
        assert spawn_calls == []
    finally:
        await root_a.dispose()
        await root_b.dispose()


@pytest.mark.asyncio
async def test_signed_prepared_is_frozen_and_spawns_real_group(tmp_path: Path) -> None:
    root, grant, code_dir = await _bound_grant(tmp_path, "consumer")
    try:
        prepared = grant.prepare_process(
            ("child.py",), ".", {"IGNORED": "formal"}, {"MARKER": "yes", "PORT": "1"},
        )
        assert prepared.command == (sys.executable, str(code_dir / "child.py"))
        assert prepared.cwd == str(code_dir.resolve())
        # 候选模式只采用显式 candidate_env。
        assert prepared.env["MARKER"] == "yes"
        assert prepared.env["PORT"] == "1"
        assert "IGNORED" not in prepared.env
        # 冻结制品不可被改写，也不存在公开的派生/修改入口。
        with pytest.raises(AttributeError):
            prepared.command = ("x",)  # type: ignore[misc]
        with pytest.raises(TypeError):
            prepared.env["MARKER"] = "tampered"  # type: ignore[index]
        assert not hasattr(prepared, "derive_env")
        child, cancelled = await grant.spawn(
            prepared, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert cancelled is False
            assert isinstance(child.process.pid, int)
            # grant 不保留 registry/adopt 后门；返回句柄即唯一 owner。
            assert not hasattr(grant, "adopt")
        finally:
            await child.kill(timeout_s=5)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_saved_handle_drains_group_after_wrapper_leader_exits(
    tmp_path: Path,
) -> None:
    """wrapper leader 退出但同组后代仍活：provider 用保存句柄排空整组。"""
    import os
    import signal

    root, grant, code_dir = await _bound_grant(tmp_path, "consumer")
    try:
        pidfile = tmp_path / "grandchild.pid"
        (code_dir / "wrapper.py").write_text(
            "import subprocess, sys\n"
            "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
            "open(sys.argv[1], 'w').write(str(p.pid))\n"
        )
        prepared = grant.prepare_process(("wrapper.py", str(pidfile)), ".", {})
        # 不接管道：后代继承 stdio 会推迟 transport 的退出确认。
        child, _ = await grant.spawn(prepared)
        try:
            for _ in range(100):
                if pidfile.exists():
                    break
                await asyncio.sleep(0.05)
            grandchild_pid = int(pidfile.read_text())
            # leader 退出，PGID 内的后代仍存活。
            await asyncio.wait_for(child.process.wait(), timeout=10)
            group_id = child.group_id
            assert isinstance(group_id, int)
            os.kill(grandchild_pid, 0)
            assert os.getpgid(grandchild_pid) == group_id
            # 保存句柄仍排空整个进程组，不依赖 adopt/PID 重建。
            await child.kill(timeout_s=10)
            with pytest.raises(ProcessLookupError):
                os.kill(grandchild_pid, 0)
        finally:
            await child.kill(timeout_s=5)
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_spawn_cancellation_still_returns_owned_receipt(tmp_path: Path) -> None:
    root, grant, _ = await _bound_grant(tmp_path, "consumer")
    entered, release = asyncio.Event(), asyncio.Event()
    real_exec = asyncio.create_subprocess_exec

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await real_exec(*args, **kwargs)

    try:
        prepared = grant.prepare_process(("child.py",), ".", {})
        # spawn_process 内部走 asyncio.create_subprocess_exec；延迟它以确定性触发取消。
        import unittest.mock as mock
        with mock.patch.object(asyncio, "create_subprocess_exec", delayed):
            task = asyncio.create_task(grant.spawn(prepared))
            await entered.wait()
            task.cancel()
            release.set()
            child, cancelled = await task
        assert cancelled is True
        assert isinstance(child.process.pid, int)
        await child.kill(timeout_s=5)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_prepare_process_rejects_fixed_env_override_in_both_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """固定键在名称层永久保留：覆盖、引入、同值声明一律在签发边界被拒。"""
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("AKASHIC_BOOT_ID", raising=False)
    monkeypatch.delenv("AKASHIC_SUPERVISED", raising=False)
    root, grant, _ = await _bound_grant(tmp_path, "consumer")
    try:
        for key in ("HOME", "AKA_PLUGIN_DATA_DIR", "AKASHIC_PLUGIN_DATA_DIR",
                    "AKASHIC_WORKSPACE", "AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED",
                    "PATH", "PYTHONPATH", "LANG"):
            with pytest.raises(PermissionError):
                grant.prepare_process(("child.py",), ".", {key: "/tmp/evil"})
            with pytest.raises(PermissionError):
                grant.prepare_process(
                    ("child.py",), ".", {}, {key: "/tmp/evil"},
                )
        # 宿主未设置这些键时，调用方引入同名键同样被拒。
        for key in ("PATH", "PYTHONPATH", "AKASHIC_BOOT_ID", "AKASHIC_SUPERVISED"):
            assert key not in __import__("os").environ
            with pytest.raises(PermissionError):
                grant.prepare_process(("child.py",), ".", {}, {key: "x"})
        # 正常声明键与运行期新键不受影响。
        prepared = grant.prepare_process(
            ("child.py",), ".", {}, {"PORT": "9", "MARKER": "yes"},
        )
        assert prepared.env["PORT"] == "9"
        assert prepared.env["MARKER"] == "yes"
        # 宿主继承缺席时固定键不凭空出现。
        assert "PATH" not in prepared.env
        assert "PYTHONPATH" not in prepared.env
        # 数据根/工作区合同键仍被钉住。
        assert "HOME" in prepared.env
        assert "AKASHIC_WORKSPACE" in prepared.env
    finally:
        await root.dispose()
