from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Mapping

import pytest

from agent.host_bridge.client import HostBridgeShellProcessManager
from agent.host_bridge.client import HostBridgeSkillCapabilityChecker
from agent.host_bridge.factory import build_shell_process_manager
from agent.host_bridge.server import HostBridgeService, _host_environment
from agent.skills import SkillsLoader
from agent.tools.base import ToolResult


def _test_runtime_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "runtime-checkout"
    checkout.mkdir(exist_ok=True)
    (checkout / "main.py").write_text("# test runtime\n", encoding="utf-8")
    return checkout


@asynccontextmanager
async def _running_bridge(
    tmp_path: Path,
    *,
    lease_timeout_s: float = 4.0,
    env: Mapping[str, str] | None = None,
) -> AsyncIterator[Path]:
    token_file = tmp_path / "token"
    token_file.write_text("test-token\n", encoding="utf-8")
    socket_path = tmp_path / "bridge.sock"
    runtime_checkout = tmp_path / "runtime-checkout"
    runtime_checkout.mkdir(exist_ok=True)
    (runtime_checkout / "main.py").write_text("# test runtime\n", encoding="utf-8")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "agent.host_bridge.server",
        "--socket",
        str(socket_path),
        "--token-file",
        str(token_file),
        "--lease-timeout",
        str(lease_timeout_s),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--release-commit",
        "a" * 40,
        "--toolchain-digest",
        "b" * 64,
        "--runtime-checkout",
        str(runtime_checkout),
        "--bridge-python",
        sys.executable,
        env=None if env is None else dict(env),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        for _ in range(100):
            if socket_path.is_socket():
                break
            if process.returncode is not None:
                break
            await asyncio.sleep(0.05)
        if not socket_path.is_socket():
            assert process.stdout is not None
            output = (await process.stdout.read()).decode("utf-8", errors="replace")
            raise AssertionError(f"Host Bridge 未启动: {output}")
        yield socket_path
    finally:
        if process.returncode is None:
            process.terminate()
        await process.wait()


@pytest.mark.asyncio
async def test_host_bridge_preserves_execution_and_stop(tmp_path: Path) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path,
            "boot-test",
            "test-token",
            "a" * 40,
            "b" * 64,
        )
        claim = await manager.claim_boot()
        assert claim["ownerBootId"] == "boot-test"
        probe = await manager.probe()
        assert set(probe["capabilities"]) >= {"exec", "pty", "stdin", "stop"}
        assert probe["releaseCommit"] == "a" * 40
        assert probe["toolchainDigest"] == "b" * 64

        completed = await manager.exec_command(
            command="printf BRIDGE_OK",
            argv=["/usr/bin/bash", "-lc", "printf BRIDGE_OK"],
            cwd=tmp_path,
            env=os.environ.copy(),
            tty=False,
            yield_time_ms=10_000,
            max_output_tokens=1_000,
            hard_timeout_s=30,
            owner_session_key="session:test",
        )
        assert completed.output == b"BRIDGE_OK"
        assert completed.exit_code == 0

        running = await manager.exec_command(
            command="printf START; sleep 30",
            argv=["/usr/bin/bash", "-lc", "printf START; sleep 30"],
            cwd=tmp_path,
            env=os.environ.copy(),
            tty=False,
            yield_time_ms=250,
            max_output_tokens=1_000,
            hard_timeout_s=60,
            owner_session_key="session:test",
        )
        assert running.execution_id is not None
        assert await manager.terminate_execution(
            running.execution_id,
            owner_session_key="session:test",
        )
        assert await manager.active_execution_ids() == []
        assert not (await manager.shutdown()).failures


@pytest.mark.asyncio
async def test_host_bridge_runs_same_manager_sessions_concurrently(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path,
            "boot-concurrent",
            "test-token",
            "a" * 40,
            "b" * 64,
        )
        await manager.claim_boot()
        started = tmp_path / "session-a-started"
        release = tmp_path / "session-b-released"

        session_a = asyncio.create_task(
            manager.exec_command(
                command="session A waits for session B",
                argv=[
                    "/usr/bin/bash",
                    "-lc",
                    f"touch {started}; for _ in {{1..40}}; do "
                    f"test -e {release} && exit 0; sleep 0.05; done; exit 42",
                ],
                cwd=tmp_path,
                env=os.environ.copy(),
                tty=False,
                yield_time_ms=10_000,
                max_output_tokens=1_000,
                hard_timeout_s=10,
                owner_session_key="session:a",
            )
        )
        for _ in range(100):
            if started.exists():
                break
            await asyncio.sleep(0.01)
        assert started.exists()

        session_b = await manager.exec_command(
            command="session B releases session A",
            argv=["/usr/bin/bash", "-lc", f"touch {release}"],
            cwd=tmp_path,
            env=os.environ.copy(),
            tty=False,
            yield_time_ms=10_000,
            max_output_tokens=1_000,
            hard_timeout_s=10,
            owner_session_key="session:b",
        )
        session_a_result = await session_a

        assert session_b.exit_code == 0
        assert session_a_result.exit_code == 0
        assert not (await manager.shutdown()).failures


@pytest.mark.asyncio
async def test_host_bridge_rejects_wrong_token(tmp_path: Path) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path, "boot-test", "wrong", "a" * 40, "b" * 64
        )
        with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
            await manager.probe()
        await manager.close_transport()


@pytest.mark.asyncio
async def test_host_bridge_rejects_execution_before_explicit_boot_claim(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path, "boot-unclaimed", "test-token", "a" * 40, "b" * 64
        )

        with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
            await manager.probe()
        await manager.close_transport()


@pytest.mark.asyncio
async def test_host_bridge_probe_rejects_release_identity_mismatch(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path,
            "boot-test",
            "test-token",
            "c" * 40,
            "b" * 64,
        )
        with pytest.raises(RuntimeError, match="release commit"):
            await manager.probe()
        await manager.close_transport()


@pytest.mark.asyncio
async def test_host_bridge_inspection_does_not_steal_active_boot(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        owner = HostBridgeShellProcessManager(
            socket_path, "boot-owner", "test-token", "a" * 40, "b" * 64
        )
        await owner.claim_boot()
        inspector = HostBridgeShellProcessManager(
            socket_path, "doctor-probe", "test-token", "a" * 40, "b" * 64
        )

        inspected = await inspector.inspect()
        completed = await owner.exec_command(
            command="printf OWNER_STILL_ACTIVE",
            argv=["/usr/bin/bash", "-lc", "printf OWNER_STILL_ACTIVE"],
            cwd=tmp_path,
            env=os.environ.copy(),
            tty=False,
            yield_time_ms=10_000,
            max_output_tokens=1_000,
            hard_timeout_s=30,
            owner_session_key="session:owner",
        )

        assert "boot-fencing" in inspected["capabilities"]
        assert completed.output == b"OWNER_STILL_ACTIVE"
        await inspector.close_transport()
        assert not (await owner.shutdown()).failures


@pytest.mark.asyncio
async def test_host_bridge_file_tools_preserve_host_bytes(tmp_path: Path) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path, "boot-file", "test-token", "a" * 40, "b" * 64
        )
        await manager.claim_boot()
        target = tmp_path / "host-only.txt"
        written = await manager.execute_file_tool(
            "write_file",
            allowed_dir=tmp_path,
            arguments={"path": str(target), "content": "alpha\n"},
        )
        assert isinstance(written, str) and "已写入" in written
        read = await manager.execute_file_tool(
            "read_file",
            allowed_dir=tmp_path,
            arguments={"path": str(target)},
        )
        assert isinstance(read, str) and "alpha" in read
        edited = await manager.execute_file_tool(
            "edit_file",
            allowed_dir=tmp_path,
            arguments={
                "path": str(target),
                "old_text": "alpha",
                "new_text": "beta",
            },
        )
        assert isinstance(edited, str) and "已成功编辑" in edited
        assert target.read_bytes() == b"beta\n"
        await manager.shutdown()


@pytest.mark.asyncio
async def test_host_bridge_returns_image_before_core_model_projection(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(
            socket_path, "boot-image", "test-token", "a" * 40, "b" * 64
        )
        await manager.claim_boot()
        target = tmp_path / "host-only.png"
        target.write_bytes(b"\x89PNG\r\n\x1a\nbridge-image")

        result = await manager.execute_file_tool(
            "read_file",
            allowed_dir=tmp_path,
            arguments={"path": str(target)},
        )

        assert isinstance(result, ToolResult)
        assert result.content_blocks[0]["type"] == "image_url"
        assert result.content_blocks[0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        await manager.shutdown()


def test_host_environment_exposes_release_runtime_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_cli = tmp_path / "runtime-cli" / "akashic-runtime"
    runtime_cli.parent.mkdir(parents=True)
    runtime_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin")

    env = _host_environment(
        {
            "AKASHIC_RUNTIME_CHECKOUT": "/attacker/checkout",
            "AKASHIC_RUNTIME_COMMIT": "f" * 40,
        },
        "boot-runtime-cli",
        runtime_cli,
    )

    assert env["AKASHIC_RUNTIME_CLI"] == str(runtime_cli)
    assert env["PATH"].split(":", 1) == [str(runtime_cli.parent), "/usr/bin"]
    assert env.get("AKASHIC_RUNTIME_CHECKOUT") != "/attacker/checkout"
    assert env.get("AKASHIC_RUNTIME_COMMIT") != "f" * 40


def test_bridge_factory_requires_complete_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AKASHIC_HOST_BRIDGE_SOCKET", str(tmp_path / "bridge.sock"))
    monkeypatch.setenv("AKASHIC_EXECUTION_MODE", "host-bridge")
    monkeypatch.setenv("AKASHIC_RUNTIME_COMMIT", "a" * 40)
    monkeypatch.setenv("AKASHIC_HOST_TOOLCHAIN_DIGEST", "b" * 64)
    monkeypatch.delenv("AKASHIC_HOST_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("AKASHIC_BOOT_ID", raising=False)
    with pytest.raises(RuntimeError, match="必须同时提供"):
        build_shell_process_manager()


@pytest.mark.asyncio
async def test_skills_loader_checks_requirements_in_host_bridge_namespace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host_bin_dir = tmp_path / "host-bin"
    host_bin_dir.mkdir()
    executable = host_bin_dir / "host-only-cli"
    executable.symlink_to("/bin/sh")
    bridge_env = os.environ.copy()
    bridge_env["PATH"] = f"{host_bin_dir}:/usr/bin"
    bridge_env["HOST_ONLY_TOKEN"] = "never-return-this-value"

    async with _running_bridge(tmp_path, env=bridge_env) as socket_path:
        workspace = tmp_path / "workspace"
        skill_dir = workspace / "skills" / "host-capability"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: host-capability\n"
            "description: host capability test\n"
            'metadata: \'{"akashic": {"requires": {"bins": '
            '["host-only-cli", "missing-cli"], "env": '
            '["HOST_ONLY_TOKEN", "MISSING_TOKEN"]}}}\'\n'
            "---\nbody\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AKASHIC_EXECUTION_MODE", "host-bridge")
        monkeypatch.setenv("AKASHIC_HOST_BRIDGE_SOCKET", str(socket_path))
        monkeypatch.setenv("AKASHIC_HOST_BRIDGE_TOKEN", "test-token")
        monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-skills")
        monkeypatch.setenv("AKASHIC_RUNTIME_COMMIT", "a" * 40)
        monkeypatch.setenv("AKASHIC_HOST_TOOLCHAIN_DIGEST", "b" * 64)
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.delenv("HOST_ONLY_TOKEN", raising=False)
        manager = HostBridgeShellProcessManager(
            socket_path,
            "boot-skills",
            "test-token",
            "a" * 40,
            "b" * 64,
        )
        await manager.claim_boot()

        record = await asyncio.to_thread(
            lambda: SkillsLoader(
                workspace,
                builtin_skills_dir=tmp_path / "builtin",
            )
            .build_index()
            .records["host-capability"]
        )

        assert record.available is False
        assert record.missing == "CLI: missing-cli, ENV: MISSING_TOKEN"
        assert "never-return-this-value" not in repr(record)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_skill_capability_rpc_fails_loud_on_authentication_error(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        checker = HostBridgeSkillCapabilityChecker(
            socket_path,
            "boot-skills",
            "wrong-token",
            "a" * 40,
            "b" * 64,
        )

        with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
            await asyncio.to_thread(
                checker.check_skill_requirements,
                ["sh"],
                ["PATH"],
            )


@pytest.mark.asyncio
async def test_skill_capability_response_never_exposes_environment_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOST_CAPABILITY_SECRET", "never-return-this-value")
    service = HostBridgeService(
        "test-token",
        4.0,
        tmp_path / "artifacts",
        release_commit="a" * 40,
        toolchain_digest="b" * 64,
        runtime_checkout=_test_runtime_checkout(tmp_path),
        bridge_python=Path(sys.executable),
    )
    await service.claim_boot(
        {
            "token": "test-token",
            "bootId": "boot-skills",
            "managerId": "claim-skills",
            "expectedReleaseCommit": "a" * 40,
            "expectedToolchainDigest": "b" * 64,
        }
    )

    response = await service.skill_requirements(
        {
            "token": "test-token",
            "bootId": "boot-skills",
            "managerId": "manager-skills",
            "expectedReleaseCommit": "a" * 40,
            "expectedToolchainDigest": "b" * 64,
            "bins": ["definitely-missing-cli"],
            "env": ["HOST_CAPABILITY_SECRET", "MISSING_TOKEN"],
        }
    )

    assert response == {
        "available": {"bins": [], "env": ["HOST_CAPABILITY_SECRET"]},
        "missing": {
            "bins": ["definitely-missing-cli"],
            "env": ["MISSING_TOKEN"],
        },
    }
    assert "never-return-this-value" not in repr(response)


@pytest.mark.asyncio
async def test_new_boot_claim_cleans_old_boot_long_job_before_admission(
    tmp_path: Path,
) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        old_manager = HostBridgeShellProcessManager(
            socket_path, "boot-old", "test-token", "a" * 40, "b" * 64
        )
        await old_manager.claim_boot()
        pid_file = tmp_path / "old-job.pid"
        running = await old_manager.exec_command(
            command="record old boot pid and wait",
            argv=[
                "/usr/bin/bash",
                "-lc",
                f"printf %s $$ > {pid_file}; exec sleep 60",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            tty=False,
            yield_time_ms=250,
            max_output_tokens=1_000,
            hard_timeout_s=120,
            owner_session_key="session:old",
        )
        assert running.execution_id is not None
        old_pid = int(pid_file.read_text(encoding="utf-8"))
        os.kill(old_pid, 0)

        new_manager = HostBridgeShellProcessManager(
            socket_path, "boot-new", "test-token", "a" * 40, "b" * 64
        )
        claim = await new_manager.claim_boot()

        assert claim == {
            "protocolMajor": 1,
            "ok": True,
            "ownerBootId": "boot-new",
            "previousBootId": "boot-old",
            "cleanedManagerCount": 1,
            "cleanedExecutionCount": 1,
        }
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)
        with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
            await old_manager.active_execution_ids()

        completed = await new_manager.exec_command(
            command="printf NEW_BOOT_ONLY",
            argv=["/usr/bin/bash", "-lc", "printf NEW_BOOT_ONLY"],
            cwd=tmp_path,
            env=os.environ.copy(),
            tty=False,
            yield_time_ms=10_000,
            max_output_tokens=1_000,
            hard_timeout_s=30,
            owner_session_key="session:new",
        )
        assert completed.output == b"NEW_BOOT_ONLY"
        assert completed.exit_code == 0
        await old_manager.close_transport()
        assert not (await new_manager.shutdown()).failures
