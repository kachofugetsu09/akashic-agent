"""O: process cleanup holds the actual owner's admission, not its neighbors."""
import asyncio

import pytest

from agent.plugin_composition import CompositionRoot
from agent.process_runtime import ShellProcessManager


@pytest.mark.asyncio
async def test_shell_cleanup_only_blocks_its_own_owner(tmp_path, monkeypatch):
    manager = ShellProcessManager(output_dir=tmp_path)
    root = CompositionRoot("process-isolation")
    entered, release = asyncio.Event(), asyncio.Event()
    original = manager._terminate_many

    async def cleanup(executions):
        entered.set()
        await release.wait()
        return await original(executions)

    monkeypatch.setattr(manager, "_terminate_many", cleanup)

    async def run(owner):
        async with root.context.runtime_scope():
            return await manager.exec_command(command="true", argv=["/usr/bin/true"],
                cwd=tmp_path, env={}, tty=False, yield_time_ms=250,
                max_output_tokens=100, hard_timeout_s=5, owner_session_key=owner)

    stopping = asyncio.create_task(manager.terminate_owner("a"))
    await asyncio.wait_for(entered.wait(), 2)
    same = asyncio.create_task(run("a"))
    other = asyncio.create_task(run("b"))
    try:
        result = await asyncio.wait_for(asyncio.shield(other), 2)
        assert result.exit_code == 0
        assert not same.done()
        release.set()
        assert not (await stopping).failures
        assert (await same).exit_code == 0
    finally:
        release.set()
        await asyncio.gather(stopping, same, other, return_exceptions=True)
        await manager.shutdown()
        await root.dispose()
