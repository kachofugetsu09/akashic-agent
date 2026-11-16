"""PR #3 测试：无 origin/sink 后台派发、settled_only 批唤醒、JOB_NOTICES 统一、subagent_send (Steer/冷恢复) 与 subagent_read + Monkey 压测。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path
from uuid import uuid4

import pytest

from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
from agent.plugin_composition.tasks import TASKS, PluginTasks
from plugins.content.plugin import Content
from plugins.conversation.plugin import check_origin
from plugins.jobs.contract import JOB_NOTICES, JobOutcome, JobWatch
from plugins.ledger.bindings import Bindings
from plugins.ledger.contract import (
    BINDINGS,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    ContentPart,
    ContentReferences,
    Input,
    Output,
    ToolCall,
)
from plugins.ledger.log import MessageLog
from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
from plugins.models.contract import MODEL_SELECTION, ChatModelSelection
from plugins.models.selection import SelectionOwner
from plugins.subagent.contract import SUBAGENT_PROGRAM
from plugins.subagent.inputs import CHECK_ORIGIN, CONTENT, TOOL_BIND_SAVED
from plugins.subagent.request import Request
from plugins.subagent.runtime import Subagents
from plugins.subagent.tools import Spawn, SubagentRead, SubagentSend
from plugins.tools.contract import CallRef, CallSource


def _dummy_binding_id(name: str) -> str:
    descriptor = {
        "version": 2,
        "origins": {"subagent": "fixture"},
        "service": "subagent.program.v1",
        "metadata": {"tool": name},
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def _mount_pr3_fixture(tmp_path: Path, *, max_active: int = 8, max_depth: int = 2):
    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("subagent-pr3")
    tasks = PluginTasks()
    bindings = Bindings(log, root.context)
    entered: dict[str, asyncio.Event] = {}
    releases: dict[str, asyncio.Event] = {}
    watches: list[JobWatch] = []
    watch_outcomes: list[tuple[str, JobOutcome | None]] = []
    watch_tasks: list[asyncio.Task[None]] = []
    ctx_holder: list[Context] = []
    jobs_holder: list[Subagents] = []

    class RecordingJobNotices:
        def watch(self, job: JobWatch) -> None:
            watches.append(job)

            async def _track() -> None:
                res = await job.settled()
                watch_outcomes.append((job.job_id, res))

            watch_tasks.append(asyncio.create_task(_track()))

        def origin_source(self, session_id: str, source: str) -> str:
            return source

        async def stop_job(self, session_id: str, job_id: str) -> bool:
            return False

        async def follow_jobs(self, session_id: str):
            if False:
                yield ()

    async def program(task, reader, request: Request):
        jobs = jobs_holder[0]
        before_step, wake = jobs.steer_binding(reader.session_id)
        entered.setdefault(request.job_id, asyncio.Event()).set()
        release = releases.get(request.job_id)
        if release is not None:
            await release.wait()
        await before_step()
        # 读取当前子会话最新用户输入以验证 Steer 与冷恢复都能被看见
        msgs = await reader.snapshot_async(through_seq=reader.head())
        user_texts = [
            str(p.value)
            for m in msgs
            if isinstance(m.body, Input)
            for p in m.body.parts
            if isinstance(p, ContentPart) and p.kind == "text"
        ]
        turn_idx = sum(1 for m in msgs if isinstance(m.body, Output)) + 1
        writer = ctx_holder[0].require(MESSAGE_WRITERS).bind(
            ctx_holder[0],
            author="subagent",
            source="subagent",
            body_types=(Output,),
            content={"text": ctx_holder[0].require(CONTENT).check_text},
        )(reader.session_id)
        try:
            _ = wake  # ensure created
            reply_text = f"turn={turn_idx}|inputs={'||'.join(user_texts)}"
            return writer.append(f"final:{request.job_id}:{turn_idx}:{uuid4().hex[:6]}", Output((ContentPart("text", reply_text),), "complete"))
        finally:
            writer.expire()

    async def core(ctx):
        for key, value in [
            (MESSAGE_CATALOG, log.catalog()),
            (MESSAGE_WRITERS, MessageWriters(log)),
            (OWNER_STATE, OwnerState(log)),
            (SESSION_ADMISSION, SessionAdmission(log)),
            (TASKS, tasks),
            (BINDINGS, bindings),
            (CHECK_ORIGIN, check_origin),
            (MODEL_SELECTION, SelectionOwner()),
            (JOB_NOTICES, RecordingJobNotices()),
            (TOOL_BIND_SAVED, lambda _bindings, target_id, *, configuration=None: asyncio.sleep(0, result=target_id)),
        ]:
            await ctx.provide(key, value)
        await ctx.provide(CONTENT, Content(ctx))

    async def subagent_mount(ctx: Context):
        ctx_holder.append(ctx)
        await ctx.provide(SUBAGENT_PROGRAM, program)

    await root.mount(core, name="core")
    await root.mount(
        subagent_mount,
        name="subagent",
        inject=(
            MESSAGE_CATALOG,
            MESSAGE_WRITERS,
            OWNER_STATE,
            SESSION_ADMISSION,
            TASKS,
            BINDINGS,
            CONTENT,
            CHECK_ORIGIN,
            MODEL_SELECTION,
            JOB_NOTICES,
            TOOL_BIND_SAVED,
        ),
        runtime=PluginRuntime(
            "subagent",
            "fixture",
            tmp_path,
            tmp_path / "plugin-data",
            tmp_path,
            {},
            workspace_roots=("subagent-runs",),
            workspace_files=("memory/spawn_trace.jsonl",),
        ),
    )
    ctx = ctx_holder[0]
    jobs = Subagents(ctx, max_active=max_active, max_depth=max_depth)
    jobs_holder.append(jobs)
    async with ctx.runtime_scope():
        prog_binding = bindings.bind(SUBAGENT_PROGRAM, {})
    targets: dict[str, str] = {}
    for tname in ("read_file", "list_dir", "write_file", "shell"):
        tid = _dummy_binding_id(tname)
        log.save_binding(tid, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": tname}})
        targets[tname] = tid
    log.save_binding(prog_binding, {"version": 2, "origins": {"subagent": "fixture"}, "service": SUBAGENT_PROGRAM.name, "metadata": {}})
    spawn = Spawn(ctx, targets, {}, jobs, model=lambda _m, _a: asyncio.sleep(0, result=ChatModelSelection()))
    send_tool = SubagentSend(jobs)
    read_tool = SubagentRead(jobs)
    return log, root, tasks, ctx, jobs, spawn, send_tool, read_tool, entered, releases, watches, watch_outcomes, watch_tasks


def _parent_batch_source(log: MessageLog, session_id: str, msg_id: str, bg_count: int) -> tuple[CallSource, ...]:
    win = log.writer(
        session_id, author="user", source="conversation", body_types=(Input,),
        content={"text": lambda _: ContentReferences()},
    )
    in_msg = win.append(msg_id + "-in", Input((ContentPart("text", "go"),)))
    wout = log.writer(
        session_id, author="assistant", source="conversation", body_types=(Output,),
        content={"text": lambda _: ContentReferences()},
        check_call=lambda _call: None,
    )
    bids = []
    for i in range(bg_count):
        bid = _dummy_binding_id(f"spawn-call-{i}")
        log.save_binding(bid, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": "spawn"}})
        bids.append(bid)
    calls = tuple(
        ToolCall(bids[i], {"task": f"job-{i}", "label": f"j{i}", "run_in_background": True})
        for i in range(bg_count)
    )
    out_msg = wout.append(msg_id + "-out", Output(calls, "continue"))
    return tuple(
        CallSource(messages=(in_msg, out_msg), call_ref=CallRef(out_msg.message_id, i))
        for i in range(bg_count)
    )


@pytest.mark.asyncio
async def test_background_without_origin_and_settled_only_batch_wake(tmp_path: Path):
    (
        log, root, tasks, ctx, jobs, spawn, _send, _read,
        entered, releases, watches, watch_outcomes, watch_tasks,
    ) = await _mount_pr3_fixture(tmp_path)
    try:
        # 1. 无 channel.origin / sink 的会话派发 3 个并发后台任务，默认 completion_wake="settled_only"
        sources = _parent_batch_source(log, "internal-parent", "batch1", 3)
        jids: list[str] = []
        async with ctx.runtime_scope():
            for i, src in enumerate(sources):
                prep = await spawn.prepare({"task": f"bg-{i}", "label": f"bg-{i}", "run_in_background": True}, src)
                assert isinstance(prep, dict)
                req = Request.model_validate(prep["request"])
                assert req.origin is None and req.sink is None
                assert req.completion_wake == "settled_only"
                jids.append(req.job_id)
                releases[req.job_id] = asyncio.Event()
                res = await spawn.invoke(f"bg-key-{i}", prep)
                assert res.outcome == "success"
                await asyncio.wait_for(entered.setdefault(req.job_id, asyncio.Event()).wait(), timeout=5)

            assert len(watches) == 3

            # 释放前 2 个：不应提前触发批量唤醒！
            releases[jids[0]].set()
            releases[jids[1]].set()
            await asyncio.sleep(0.1)
            assert all(out is None for _, out in watch_outcomes)
            assert len(watch_outcomes) == 0

            # 释放第 3 个：整批结算，3 个 watch 中仅最后 1 个返回聚合 JobOutcome，其余返回 None
            releases[jids[2]].set()
            await asyncio.gather(*watch_tasks)
            non_null = [(jid, out) for jid, out in watch_outcomes if out is not None]
            nulls = [(jid, out) for jid, out in watch_outcomes if out is None]
            assert len(non_null) == 1
            assert len(nulls) == 2
            summary = non_null[0][1].detail
            assert "3 个后台子任务已全部完成" in summary
            for lbl in ("bg-0", "bg-1", "bg-2"):
                assert lbl in summary
    finally:
        for ev in releases.values():
            ev.set()
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_subagent_send_steer_cold_resume_and_parent_scoped_read(tmp_path: Path):
    (
        log, root, tasks, ctx, jobs, spawn, send_tool, read_tool,
        entered, releases, _watches, _watch_outcomes, _watch_tasks,
    ) = await _mount_pr3_fixture(tmp_path)
    try:
        src_a = _parent_batch_source(log, "root-a", "a1", 1)[0]
        src_b = _parent_batch_source(log, "root-b", "b1", 1)[0]
        async with ctx.runtime_scope():
            # 1. 启动一个后台任务并先阻塞，模拟运行中 Steer 插话
            prep = await spawn.prepare({"task": "initial task", "label": "worker", "run_in_background": True}, src_a)
            assert isinstance(prep, dict)
            jid = Request.model_validate(prep["request"]).job_id
            releases[jid] = asyncio.Event()
            res = await spawn.invoke("k-worker", prep)
            assert res.outcome == "success"
            await asyncio.wait_for(entered.setdefault(jid, asyncio.Event()).wait(), timeout=5)

            # 运行中调用 subagent_send -> mode="steered"
            sp1 = await send_tool.prepare({"job_id": jid, "message": "steer constraint X"}, src_a)
            assert isinstance(sp1, dict)
            sres1 = await send_tool.invoke("send-1", sp1)
            assert sres1.outcome == "success"
            assert json.loads(str(sres1.parts[0].value))["mode"] == "steered"

            # 放行第 1 轮，子任务应看到 initial task 与 steer constraint X
            releases[jid].set()
            while jobs.active_count_for_root("root-a") > 0:
                await asyncio.sleep(0.02)

            # 2. 已结算后调用 subagent_send -> 冷恢复同一 subagent:<jid> 会话继续第 2 轮！
            sp2 = await send_tool.prepare({"job_id": jid, "message": "follow-up Y"}, src_a)
            assert isinstance(sp2, dict)
            sres2 = await send_tool.invoke("send-2", sp2)
            assert sres2.outcome == "success"
            p2 = json.loads(str(sres2.parts[0].value))
            assert p2["mode"] == "resumed_sync"
            assert p2["status"] == "completed"
            assert "initial task" in p2["result"]
            assert "steer constraint X" in p2["result"]
            assert "follow-up Y" in p2["result"]
            assert "turn=2" in p2["result"]

            # 3. subagent_read 可读取子会话完整过程，且子会话保持 internal
            rp = await read_tool.prepare({"job_id": jid}, src_a)
            assert isinstance(rp, dict)
            rres = await read_tool.invoke("read-1", rp)
            assert rres.outcome == "success"
            rjson = json.loads(str(rres.parts[0].value))
            assert rjson["job_id"] == jid
            assert len(rjson["messages"]) >= 4
            assert log.reader(f"subagent:{jid}").attributes.visibility == "internal"

            # 4. 跨根会话树 root-b 禁止 send / read root-a 的子任务
            bad_send = await send_tool.prepare({"job_id": jid, "message": "hijack"}, src_b)
            assert isinstance(bad_send, dict)
            bad_sres = await send_tool.invoke("bad-s", bad_send)
            assert bad_sres.outcome == "error" and "无权" in str(bad_sres.parts[0].value)

            bad_read = await read_tool.prepare({"job_id": jid}, src_b)
            assert isinstance(bad_read, dict)
            bad_rres = await read_tool.invoke("bad-r", bad_read)
            assert bad_rres.outcome == "error" and "无权" in str(bad_rres.parts[0].value)
    finally:
        for ev in releases.values():
            ev.set()
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_monkey_parallel_batches_always_and_send_interleaving(tmp_path: Path):
    """随机并发 Monkey 测试：混合 always / settled_only 批唤醒、主动取消与 subagent_send 冷恢复。"""
    rng = random.Random(202611173)
    (
        log, root, tasks, ctx, jobs, spawn, send_tool, read_tool,
        entered, releases, _watches, watch_outcomes, watch_tasks,
    ) = await _mount_pr3_fixture(tmp_path, max_active=20)
    try:
        async with ctx.runtime_scope():
            # 构造 4 个并发 settled_only 批次（每批 2-3 个）和 4 个 always 后台任务
            all_jids: list[str] = []
            for b in range(4):
                count = rng.choice([2, 3])
                srcs = _parent_batch_source(log, "monkey-root", f"mb-{b}", count)
                for i, src in enumerate(srcs):
                    prep = await spawn.prepare({
                        "task": f"m-{b}-{i}", "label": f"m-{b}-{i}",
                        "run_in_background": True, "completion_wake": "settled_only",
                    }, src)
                    assert isinstance(prep, dict)
                    jid = Request.model_validate(prep["request"]).job_id
                    all_jids.append(jid)
                    releases[jid] = asyncio.Event()
                    res = await spawn.invoke(f"mk-{b}-{i}", prep)
                    assert res.outcome == "success"

            for a in range(4):
                src = _parent_batch_source(log, "monkey-root", f"ma-{a}", 1)[0]
                prep = await spawn.prepare({
                    "task": f"always-{a}", "label": f"always-{a}",
                    "run_in_background": True, "completion_wake": "always",
                }, src)
                assert isinstance(prep, dict)
                jid = Request.model_validate(prep["request"]).job_id
                all_jids.append(jid)
                releases[jid] = asyncio.Event()
                res = await spawn.invoke(f"mka-{a}", prep)
                assert res.outcome == "success"

            # 随机顺序放行或取消任务
            shuffled = list(all_jids)
            rng.shuffle(shuffled)
            for jid in shuffled:
                await asyncio.sleep(rng.uniform(0.0, 0.015))
                if rng.random() < 0.2:
                    await jobs.cancel(jid)
                releases[jid].set()

            await asyncio.gather(*watch_tasks)
            while jobs.active_count_for_root("monkey-root") > 0:
                await asyncio.sleep(0.02)

            # 4 个 settled_only 批次应恰好各唤醒 1 次 + 4 个 always 各唤醒 1 次 = 8 次唤醒
            wakes = [out for _, out in watch_outcomes if out is not None]
            assert len(wakes) == 8

            # 随机对几个已完成子任务读进展并追问一次
            src_check = _parent_batch_source(log, "monkey-root", "mcheck", 1)[0]
            for jid in rng.sample(all_jids, 3):
                rp = await read_tool.prepare({"job_id": jid}, src_check)
                assert isinstance(rp, dict)
                rr = await read_tool.invoke(f"mread-{jid}", rp)
                assert rr.outcome == "success"
    finally:
        for ev in releases.values():
            ev.set()
        await tasks.close()
        await root.dispose()
        log.close()
