"""后台任务完成通知：忙时注入当前回合的下一步，闲时唤醒原会话并回传渠道。"""
import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path

from dataclasses import dataclass

import pytest

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.tasks import RestartRejectedError
from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, ChannelInboundMessage
from plugins.delivery.contract import FINAL_OUTPUT_DELIVERY
from plugins.jobs.board import JobBoard
from plugins.jobs.contract import JOB_NOTICES, JobOutcome, JobWatch
from plugins.ledger.contract import ContentPart, ContentReferences, Control, Input, Output
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


def _abandoning_wake_sources(sources: Path) -> None:
    """唤醒回合撞步数上限：reply program 返回 Control("abandon")，没有正文可回传。

    轮次预算由 application(max_steps=2) 走正式配置项给出：首轮两次请求（工具调用 +
    收尾）正常完成；唤醒轮两次都只产生工具调用，回执齐全，于是第三步的门槛命中
    「没有未回执调用」分支，引擎自己放弃本段。
    """
    _job_sources(sources)
    provider = sources / "test_provider/plugin.py"
    text = provider.read_text()
    # 首轮正常起任务；唤醒轮（第 3、4 次请求）只调 write_evidence，回执齐全但从不收尾，
    # 于是撞上「没有未回执调用」的门槛，由引擎自己放弃本段。
    text = text.replace("if len(calls) == 1:", "if len(calls) in (1, 3, 4):")
    text = text.replace(
        'ToolCall("provider-call", "start_job", {})',
        'ToolCall("provider-call", "start_job" if len(calls) == 1 else "write_evidence", {})',
    )
    provider.write_text(text)


@pytest.mark.asyncio
async def test_wake_that_ends_in_control_does_not_deliver_empty_message(tmp_path):
    """引擎放弃的唤醒回合没有正文：不投递空消息，结算留给下一次用户输入。"""
    async with application(tmp_path, replying=True, max_steps=2,
                           extra_sources=_abandoning_wake_sources) as (log, host):
        calls, gate = await _send(host)
        await _rows_until(log, lambda rows: _finished(rows, 1))
        # 1. 回合结束后才结算，触发唤醒；唤醒轮跑满两次请求后由引擎放弃。
        gate.set()

        def abandoned(rows):
            return any(isinstance(row.body, Control) and row.body.action == "abandon" for row in rows)

        rows = await _rows_until(log, abandoned)
        control = next(row for row in rows if isinstance(row.body, Control) and row.body.action == "abandon")
        assert control.source == "job:fake-1"
        assert control.body.reason is not None and control.body.reason.startswith("步数上限放弃")
        # 2. 控制消息不是答复：渠道不能收到它（否则只会回 rejected 空消息）。
        await asyncio.sleep(0.5)
        for sent in (tmp_path / "workspace").rglob("sent.jsonl"):
            assert control.message_id not in sent.read_text(), sent.read_text()
        # 3. 通知没有被悄悄丢掉：下一次用户输入仍能拿到这条结算。
        async with live_root(host) as root:
            await root.context.require(CHANNEL_INPUT)("test:room", "u2", ChannelInboundMessage(
                "test", "user", "room", "next message", datetime(2026, 10, 11, tzinfo=UTC), {},
            ))
        await _rows_until(log, lambda rows: _finished(rows, 2))
        assert "后台任务 fake-1" in str(calls[-1].messages)


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
    board.confirm(notice, "input-1")
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


@dataclass(frozen=True)
class _FinalTurn:
    source: str
    ending_message_id: str | None
    message_ids: tuple[str, ...] = ()


@pytest.mark.asyncio
async def test_job_final_output_wait_rejects_missing_sender_immediately(tmp_path):
    async with application(tmp_path, replying=True, extra_sources=_job_sources) as (log, host):
        session_id = "missing-channel:room"
        w_in = log.writer(
            session_id,
            author="user",
            source="conversation",
            body_types=(Input,),
            content={
                "text": lambda v: ContentReferences(),
                "channel.origin": lambda v: ContentReferences(),
            },
        )
        w_in.append(
            "u-missing",
            Input((
                ContentPart("channel.origin", {"channel": "nonexistent", "chat_id": "room", "sender": "u"}),
                ContentPart("text", "deploy"),
            )),
        )
        w_out = log.writer(
            session_id,
            author="model",
            source="job:fake-missing",
            body_types=(Output,),
            content={"text": lambda v: ContentReferences()},
        )
        w_out.append("o-missing", Output((ContentPart("text", "submitted"),), "complete"))
        async with live_root(host) as root:
            final_delivery = root.context.require(FINAL_OUTPUT_DELIVERY)
            turn = _FinalTurn(source="job:fake-missing", ending_message_id="o-missing")
            with pytest.raises(RestartRejectedError, match="没有发送能力"):
                async with asyncio.timeout(2):
                    await final_delivery.wait(log.reader(session_id), turn)
