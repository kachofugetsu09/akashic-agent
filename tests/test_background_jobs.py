"""后台任务完成通知：忙时注入当前回合的下一步，闲时唤醒原会话并回传渠道。"""
import asyncio
import json
import shutil
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest
import pytest_asyncio

from agent.plugin_composition import ServiceKey
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.jobs.board import JobBoard
from plugins.jobs.contract import JOB_NOTICES, JobOutcome, JobWatch
from plugins.ledger.contract import Input, Output, ToolResult
from tests.test_default_reply import application, live_root

_FIXTURE = '''
import asyncio
from contextlib import asynccontextmanager
from agent.plugin_composition import ServiceKey
from plugins.jobs.contract import JOB_NOTICES, JobOutcome, JobWatch
from plugins.ledger.contract import ContentPart
from plugins.tools.api import Result
from plugins.tools.plugin import TOOLS
api_version = 3
name = "job_fixture"
version = "1.0.0"
inject = (TOOLS, JOB_NOTICES)
async def apply(ctx):
    gate = asyncio.Event()
    class Start:
        idempotent = False
        async def prepare(self, args, source=None):
            return {"session_id": source.messages[-1].session_id, "source": source.messages[-1].source}
        async def invoke(self, key, args):
            async def settled():
                await gate.wait()
                return JobOutcome("completed", "exit_code=0")
            ctx.require(JOB_NOTICES).watch(JobWatch(
                session_id=args["session_id"], source=args["source"], job_id="fake-1", kind="shell",
                label="fixture job", collect_hint="fixture result is ready.", settled=settled,
            ))
            return Result("success", (ContentPart("text", "started"),))
        async def query(self, key):
            return None
    @asynccontextmanager
    async def open(state):
        yield Start()
    await ctx.require(TOOLS).declare_group(ctx, always_on=True)
    await ctx.require(TOOLS).register(ctx, name="start_job", description="start a fixture job",
        parameters={"type": "object"}, open=open)
    await ctx.provide(ServiceKey("fixture.job_gate"), gate)
'''


def _job_sources(sources: Path) -> None:
    from tests.support.delivery_sources import sources as delivery_sources
    delivery_sources(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/jobs", sources / "jobs",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (sources / "job_fixture").mkdir()
    (sources / "job_fixture/plugin.py").write_text(_FIXTURE)
    provider = sources / "test_provider/plugin.py"
    provider.write_text(provider.read_text().replace(
        'ToolCall("provider-call", "write_evidence", {})', 'ToolCall("provider-call", "start_job", {})'))


async def _send(host):
    async with live_root(host) as root:
        await root.context.require(CHANNEL_INPUT)("test:room", "u1", ChannelInboundMessage(
            "test", "user", "room", "dispatch the job", datetime(2026, 10, 11, tzinfo=UTC), {},
        ))
        return root.context.require(ServiceKey("fixture.calls")), root.context.require(ServiceKey("fixture.job_gate"))


async def _rows_until(log, done):
    async def wait():
        async for _ in log.catalog().follow():
            rows = log.reader("test:room").snapshot()
            if done(rows):
                return rows
        raise AssertionError("消息订阅提前结束")
    return await asyncio.wait_for(wait(), 5)


def _finished(rows, count):
    return sum(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows) >= count


@pytest.mark.asyncio
async def test_busy_turn_sees_settled_job_on_next_step_without_extra_wake(tmp_path):
    async with application(tmp_path, replying=True, extra_sources=_job_sources) as (log, host):
        calls, gate = await _send(host)
        gate.set()
        rows = await _rows_until(log, lambda rows: _finished(rows, 1))
        # 1. 第二步请求携带结算通知；通知已被本输入领取，回合结束后不再唤醒。
        await asyncio.sleep(0.5)
        assert [type(row.body) for row in log.reader("test:room").snapshot()] == [Input, Output, ToolResult, Output]
        assert len(calls) == 2
        assert "后台任务 fake-1" not in str(calls[0].messages)
        assert "后台任务 fake-1" in str(calls[1].messages)
        assert rows[-1].source == "conversation"


@pytest.mark.asyncio
async def test_idle_session_is_woken_and_reply_is_delivered(tmp_path):
    async with application(tmp_path, replying=True, extra_sources=_job_sources) as (log, host):
        calls, gate = await _send(host)
        await _rows_until(log, lambda rows: _finished(rows, 1))
        assert len(calls) == 2
        # 1. 回合结束后才结算：唤醒一轮，回复归 job 来源并按原渠道发送。
        gate.set()
        rows = await _rows_until(log, lambda rows: _finished(rows, 2))
        assert len(calls) == 3
        assert "后台任务 fake-1" in str(calls[2].messages)
        woken = rows[-1]
        assert woken.source == "job:fake-1"
        await asyncio.wait_for(_delivered(tmp_path, woken.message_id), 5)


async def _delivered(tmp_path: Path, message_id: str) -> None:
    # 发送在唤醒回复提交之后进行；以测试 sender 的发送记录为准。
    while True:
        for sent in (tmp_path / "workspace").rglob("sent.jsonl"):
            if any(json.loads(line)[2] == message_id for line in sent.read_text().splitlines()):
                return
        await asyncio.sleep(0.05)


def _job(job_id: str, *, session_id: str = "s", source: str = "conversation") -> JobWatch:
    async def settled() -> JobOutcome | None:
        return None
    return JobWatch(session_id=session_id, source=source, job_id=job_id, kind="shell",
                    label=job_id, collect_hint="read it.", settled=settled)


def test_board_claim_is_stable_within_one_input_and_retired_by_the_next():
    board = JobBoard()
    notice = board.settle(_job("a"), JobOutcome("completed", "exit_code=0"))
    assert board.claim("s", "conversation", "input-1") == (notice,)
    assert board.claim("s", "conversation", "input-1") == (notice,)
    assert board.claim("other", "conversation", "input-1") == ()
    assert board.claim("s", "subagent", "input-1") == ()
    assert board.claim("s", "conversation", "input-2") == ()
    assert board.pending("s") == ()


def test_board_wake_only_takes_unclaimed_notices_within_budget():
    board = JobBoard(max_consecutive_wakes=1)
    first = board.settle(_job("a"), JobOutcome("completed", "exit_code=0"))
    second = board.settle(_job("b"), JobOutcome("failed", "exit_code=1"))
    # 1. 预算按用户输入连续计数；耗尽后留给下一次输入注入。
    assert board.take_wake(first, "input-1")
    assert not board.take_wake(first, "input-1")
    assert not board.take_wake(second, "input-1")
    assert board.claim("s", "conversation", "input-2") == (second,)
    # 2. 唤醒失败撤回领取，新输入恢复预算。
    board.release_wake(first)
    assert board.take_wake(first, "input-3")
    board.finish_wake(first)
    assert first not in board.pending("s")


def _shell_sources(sources: Path) -> None:
    _job_sources(sources)
    provider = sources / "test_provider/plugin.py"
    provider.write_text(provider.read_text().replace(
        'ToolCall("provider-call", "start_job", {})',
        'ToolCall("provider-call", "shell", {"command": "sleep 1; echo background-done", '
        '"description": "fixture background", "run_in_background": True, "yield_time_ms": 250})',
    ))


def _collecting_shell_sources(sources: Path, *, tool: str, chain: bool) -> None:
    """夹具模型收到通知后实际领取或停止后台进程，不只检查提醒文字。"""
    _shell_sources(sources)
    provider = sources / "test_provider/plugin.py"
    text = provider.read_text().replace('return LLMResponse("finished")', '''if len(calls) in (3, 6):
                import re
                execution_id = int(re.findall(r"后台任务 shell-(\\d+)", str(request.messages))[-1])
                return LLMResponse(None, [ToolCall("collect-call", COLLECT_TOOL, {"execution_id": execution_id})])
            return LLMResponse("finished")'''.replace("COLLECT_TOOL", repr(tool)))
    if chain:
        text = text.replace('return LLMResponse("finished")', '''if len(calls) == 4:
                return LLMResponse(None, [ToolCall("next-job", "shell", {
                    "command": "sleep 1; echo next-background-done", "description": "next background",
                    "run_in_background": True, "yield_time_ms": 250})])
            return LLMResponse("finished")''')
    provider.write_text(text)


@pytest_asyncio.fixture(params=["local", "host-bridge"])
async def shell_backend(tmp_path, monkeypatch, request):
    """真实 Socket 和服务端复跑后台 Shell，核对本地与远端的可观察语义。"""
    if request.param == "local":
        yield
        return
    from plugins.host_execution.bridge.boot import claim_host_bridge_boot
    socket = tmp_path.parent / "bridge.sock"
    for name, value in {
        "AKASHIC_EXECUTION_MODE": "host-bridge", "AKASHIC_HOST_BRIDGE_SOCKET": str(socket),
        "AKASHIC_HOST_BRIDGE_TOKEN": "jobs-fixture", "AKASHIC_BOOT_ID": "jobs-fixture",
        "AKASHIC_RUNTIME_COMMIT": "a" * 40, "AKASHIC_HOST_TOOLCHAIN_DIGEST": "b" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    token_file = tmp_path / "bridge.token"
    token_file.write_text("jobs-fixture")
    # Bridge 独立进程也能响应同步的环境探测，和正式部署保持相同边界。
    with (tmp_path / "bridge.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "plugins.host_execution.bridge.server",
            "--socket", str(socket), "--token-file", str(token_file), "--lease-timeout", "60",
            "--artifact-root", str(tmp_path / "artifacts"), "--release-commit", "a" * 40,
            "--toolchain-digest", "b" * 64, "--runtime-checkout", str(Path(__file__).parents[1]),
            "--bridge-python", sys.executable, stdout=log, stderr=log,
        )
        try:
            async with asyncio.timeout(5):
                while not socket.exists():
                    assert process.returncode is None, (tmp_path / "bridge.log").read_text()
                    await asyncio.sleep(0.01)
            assert await claim_host_bridge_boot() is not None
            yield
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,chain", [("write_stdin", False), ("task_stop", False), ("write_stdin", True)])
async def test_background_shell_outlives_turn_and_wakes_session(tmp_path, tool, chain, shell_backend):
    async with application(tmp_path, replying=True, extra_sources=partial(_collecting_shell_sources, tool=tool, chain=chain),
                           shell=True) as (log, host):
        calls, _gate = await _send(host)
        rows = await _rows_until(log, lambda rows: _finished(rows, 1))
        # 1. 首个窗口内仍在运行：工具结果给出后台句柄，回合正常结束。
        result = next(row for row in rows if isinstance(row.body, ToolResult))
        text = "\n".join(str(part.value) for part in result.body.parts)
        assert '"process_status": "running"' in text and "已转入后台" in text
        # 2. 回合清理不回收后台分区；进程自然退出后唤醒原会话。
        rows = await _rows_until(log, lambda rows: _finished(rows, 2))
        assert rows[-1].source.startswith("job:shell-")
        assert "status: completed，exit_code=0" in str(calls[2].messages)
        collected = [row for row in rows if isinstance(row.body, ToolResult)][1]
        assert collected.body.outcome == "success"
        text = "\n".join(str(part.value) for part in collected.body.parts)
        assert ('background-done' if tool == "write_stdin" else '"status": "stopped"') in text
        if chain:
            # 唤醒中追加的后台任务仍归原来源，下一次结算也能唤醒并读取。
            rows = await _rows_until(log, lambda rows: _finished(rows, 3))
            collected = [row for row in rows if isinstance(row.body, ToolResult)][-1]
            assert collected.body.outcome == "success"
            assert "next-background-done" in str(collected.body.parts)


def _gated_wake_sources(sources: Path) -> None:
    """将唤醒的模型请求停在明确边界，观察新输入和卸载的真实取消路径。"""
    _job_sources(sources)
    provider = sources / "test_provider/plugin.py"
    text = provider.read_text().replace("calls = []", """calls = []
    import asyncio
    wake_started, wake_release, wake_exited = asyncio.Event(), asyncio.Event(), asyncio.Event()""")
    text = text.replace('return LLMResponse("finished")', '''if len(calls) == 3:
                wake_started.set()
                try:
                    await wake_release.wait()
                finally:
                    wake_exited.set()
            return LLMResponse("finished")''')
    text += '\n    await ctx.provide(ServiceKey("fixture.wake_events"), (wake_started, wake_release, wake_exited))\n'
    provider.write_text(text)


@pytest.mark.asyncio
async def test_new_input_claims_notice_from_preempted_wake(tmp_path):
    async with application(tmp_path, replying=True, extra_sources=_gated_wake_sources) as (log, host):
        calls, gate = await _send(host)
        original = await _rows_until(log, lambda rows: _finished(rows, 1))
        async with live_root(host) as root:
            started, release, exited = root.context.require(ServiceKey("fixture.wake_events"))
            gate.set()
            try:
                await asyncio.wait_for(started.wait(), 5)
                await root.context.require(CHANNEL_INPUT)("test:room", "u2", ChannelInboundMessage(
                    "test", "user", "room", "new user message", datetime(2026, 10, 11, tzinfo=UTC), {},
                ))
                rows = await _rows_until(log, lambda rows: _finished(rows, 2))
                assert exited.is_set()
                assert "后台任务 fake-1" in str(calls[3].messages)
                assert rows[-1].source == "conversation"
                assert rows[:len(original)] == original
            finally:
                release.set()


@pytest.mark.asyncio
async def test_jobs_unload_cancels_active_wake(tmp_path):
    async with application(tmp_path, replying=True, extra_sources=_gated_wake_sources) as (log, host):
        _calls, gate = await _send(host)
        original = await _rows_until(log, lambda rows: _finished(rows, 1))
        async with live_root(host) as root:
            started, release, exited = root.context.require(ServiceKey("fixture.wake_events"))
            jobs = root.context.require(JOB_NOTICES)
            handle = jobs._ctx.fiber
            gate.set()
            try:
                await asyncio.wait_for(started.wait(), 5)
                # 插件卸载必须主动取消请求，不能等夹具放行后才完成。
                async with asyncio.timeout(5):
                    await handle.dispose()
                assert exited.is_set()
                assert log.reader("test:room").snapshot() == original
            finally:
                release.set()
