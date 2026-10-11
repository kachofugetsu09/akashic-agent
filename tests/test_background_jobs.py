"""后台任务完成通知：忙时注入当前回合的下一步，闲时唤醒原会话并回传渠道。"""
import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent.plugin_composition import ServiceKey
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.jobs.board import JobBoard
from plugins.jobs.contract import JobOutcome, JobWatch
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
    ).replace(_FIXTURE_CLEANUP, ""))


# 真实 standard_tools 提供回合清理；夹具 provider 的同名服务会冲突。
_FIXTURE_CLEANUP = '''    await ctx.provide(ServiceKey("tools.cleanup.v1"), partial(
        shell_cleanup, ctx, ShellOwners(ctx), ctx.require(TASKS).open(ctx),
    ))
'''


@pytest.mark.asyncio
async def test_background_shell_outlives_turn_and_wakes_session(tmp_path):
    async with application(tmp_path, replying=True, extra_sources=_shell_sources, shell=True) as (log, host):
        calls, _gate = await _send(host)
        rows = await _rows_until(log, lambda rows: _finished(rows, 1))
        # 1. 首个窗口内仍在运行：工具结果给出后台句柄，回合正常结束。
        result = next(row for row in rows if isinstance(row.body, ToolResult))
        text = "\n".join(str(part.value) for part in result.body.parts)
        assert '"process_status": "running"' in text and "已转入后台" in text
        # 2. 回合清理不回收后台分区；进程自然退出后唤醒原会话。
        rows = await _rows_until(log, lambda rows: _finished(rows, 2))
        assert rows[-1].source.startswith("job:shell-")
        assert "status: completed，exit_code=0" in str(calls[-1].messages)
