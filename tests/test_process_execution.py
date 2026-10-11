"""O: process cleanup holds the actual owner's admission, not its neighbors."""
import asyncio

import pytest

from agent.plugin_composition import CompositionRoot
from plugins.host_execution.process_runtime import ShellProcessManager


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


async def _exec(manager, root, tmp_path, script, *, owner="a", yield_time_ms=250):
    async with root.context.runtime_scope():
        return await manager.exec_command(command=script, argv=["/bin/sh", "-c", script],
            cwd=tmp_path, env={}, tty=False, yield_time_ms=yield_time_ms,
            max_output_tokens=100, hard_timeout_s=5, owner_session_key=owner)


@pytest.mark.asyncio
async def test_execution_states_settle_only_unobserved_exits(tmp_path):
    manager = ShellProcessManager(output_dir=tmp_path)
    root = CompositionRoot("process-states")
    try:
        # 1. 首个窗口内退出的命令直接交付终态，不留下待通知条目。
        assert (await _exec(manager, root, tmp_path, "true")).exit_code == 0
        assert await manager.execution_states("a") == []
        # 2. 后台退出后无人等待才算结算；快照不消费输出。
        running = await _exec(manager, root, tmp_path, "sleep 0.3; echo done")
        assert running.execution_id is not None
        assert [state.settled for state in await manager.execution_states("a")] == [False]
        assert await manager.execution_states("b") == []
        for _ in range(100):
            states = await manager.execution_states("a")
            if states[0].settled:
                break
            await asyncio.sleep(0.02)
        assert states[0].settled and states[0].exit_code == 0
        result = await manager.write_stdin(execution_id=running.execution_id, chars="", yield_time_ms=250,
                                           max_output_tokens=100, owner_session_key="a")
        assert result.exit_code == 0 and b"done" in result.output
        assert await manager.execution_states("a") == []
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_waiting_reader_keeps_exit_unsettled(tmp_path):
    manager = ShellProcessManager(output_dir=tmp_path)
    root = CompositionRoot("process-waiter")
    try:
        running = await _exec(manager, root, tmp_path, "sleep 0.4")
        assert running.execution_id is not None
        reader = asyncio.create_task(manager.write_stdin(
            execution_id=running.execution_id, chars="", yield_time_ms=5000,
            max_output_tokens=100, owner_session_key="a"))
        # 等待者在场时退出由它交付，状态查询从不报告结算。
        while not reader.done():
            assert not any(state.settled for state in await manager.execution_states("a"))
            await asyncio.sleep(0.01)
        assert (await reader).exit_code == 0
        assert await manager.execution_states("a") == []
    finally:
        await manager.shutdown()


def test_bridge_execution_states_round_trip():
    from plugins.host_execution.bridge import host_bridge_pb2 as pb
    from plugins.host_execution.bridge.protocol import decode_execution_states, encode_execution_states
    from plugins.host_execution.contract import ExecutionState

    states = [ExecutionState(1001, False, None, "natural"), ExecutionState(1002, True, 0, "natural"),
              ExecutionState(1003, True, -9, "timeout")]
    assert decode_execution_states(encode_execution_states(states)) == states
    broken = pb.ExecutionStatesReply(states=[pb.ExecutionState(execution_id=1004, settled=True, finish_reason="natural")])
    with pytest.raises(ValueError, match="缺少退出码"):
        decode_execution_states(broken)
