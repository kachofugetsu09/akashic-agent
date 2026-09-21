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
            ("child.py",), ".", {"IGNORED": "formal"}, {"MARKER": "yes"},
            runtime_env_keys={"PORT"},
        )
        assert prepared.command == (sys.executable, str(code_dir / "child.py"))
        assert prepared.cwd == str(code_dir.resolve())
        # 候选模式只采用显式 candidate_env。
        assert prepared.env["MARKER"] == "yes"
        assert "IGNORED" not in prepared.env
        # 冻结制品不可被改写。
        with pytest.raises(AttributeError):
            prepared.command = ("x",)  # type: ignore[misc]
        with pytest.raises(TypeError):
            prepared.env["MARKER"] = "tampered"  # type: ignore[index]
        # derive_env 只允许签发时声明的运行期键；HOME/AKASHIC_WORKSPACE 等
        # 冻结键或未声明键一律拒绝（非反射公开 API 反例）。
        with pytest.raises(PermissionError):
            prepared.derive_env({"HOME": "/tmp/evil"})
        with pytest.raises(PermissionError):
            prepared.derive_env({"AKASHIC_WORKSPACE": "/tmp/evil"})
        with pytest.raises(PermissionError):
            prepared.derive_env({"PYTHONPATH": "/tmp/evil"})
        # derive_env 追加 provider 声明的运行期键，仍属同一签发者。
        derived = prepared.derive_env({"PORT": "1"})
        child, cancelled = await grant.spawn(
            derived, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert cancelled is False
            assert isinstance(child.process.pid, int)
            # adopt 只接管本授权登记的 child，返回同一句柄。
            assert grant.adopt(child.process) is child
        finally:
            await child.kill(timeout_s=5)
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_adopt_rejects_unregistered_foreign_process(tmp_path: Path) -> None:
    root, grant, _ = await _bound_grant(tmp_path, "consumer")
    foreign = await asyncio.create_subprocess_exec(
        sys.executable, "-c", "import time; time.sleep(30)",
    )
    try:
        with pytest.raises(PermissionError):
            grant.adopt(foreign)
        assert foreign.returncode is None
    finally:
        foreign.kill()
        await foreign.wait()
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
