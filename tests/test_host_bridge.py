from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import pytest

from agent.host_bridge.client import HostBridgeShellProcessManager
from agent.host_bridge.factory import build_shell_process_manager


@asynccontextmanager
async def _running_bridge(
    tmp_path: Path,
    *,
    lease_timeout_s: float = 4.0,
) -> AsyncIterator[Path]:
    token_file = tmp_path / "token"
    token_file.write_text("test-token\n", encoding="utf-8")
    socket_path = tmp_path / "bridge.sock"
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
        )
        probe = await manager.probe()
        assert set(probe["capabilities"]) >= {"exec", "pty", "stdin", "stop"}

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
async def test_host_bridge_rejects_wrong_token(tmp_path: Path) -> None:
    async with _running_bridge(tmp_path) as socket_path:
        manager = HostBridgeShellProcessManager(socket_path, "boot-test", "wrong")
        with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
            await manager.probe()
        await manager.close_transport()


def test_bridge_factory_requires_complete_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("AKASHIC_HOST_BRIDGE_SOCKET", str(tmp_path / "bridge.sock"))
    monkeypatch.delenv("AKASHIC_HOST_BRIDGE_TOKEN", raising=False)
    monkeypatch.delenv("AKASHIC_BOOT_ID", raising=False)
    with pytest.raises(RuntimeError, match="必须同时提供"):
        build_shell_process_manager()
