"""Subagent 工具继承、按根会话树隔离的并发池（默认 8）与统一递归深度上限的回归及并发 Monkey 测试。"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import cast
from agent.plugin_composition import Context

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.tasks import TASKS, PluginTasks
from plugins.content.plugin import Content
from plugins.conversation.plugin import check_origin
from plugins.jobs.contract import JOB_NOTICES, JobWatch
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
)
from plugins.ledger.log import MessageLog
from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
from plugins.models.contract import MODEL_SELECTION, ChatModelSelection
from plugins.models.selection import SelectionOwner
from plugins.subagent.contract import SUBAGENT_PROGRAM, AgentResult, AgentRunner
from plugins.subagent.inputs import CHECK_ORIGIN, CONTENT, TOOL_BIND_SAVED
from plugins.subagent.request import Request
from plugins.subagent.runners import Delegate, Runners
from plugins.subagent.runtime import DEFAULT_MAX_ACTIVE, DEFAULT_MAX_DEPTH, Subagents
from plugins.subagent.tools import Spawn
from plugins.tools.contract import CallRef, CallSource, Result


def _dummy_binding_id(name: str) -> str:
    descriptor = {
        "version": 2,
        "origins": {"subagent": "fixture"},
        "service": "subagent.program.v1",
        "metadata": {"tool": name},
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def _mount_fixture(tmp_path: Path, *, max_active: int = DEFAULT_MAX_ACTIVE, max_depth: int = DEFAULT_MAX_DEPTH):
    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("subagent-pool-depth")
    tasks = PluginTasks()
    bindings = Bindings(log, root.context)
    entered: dict[str, asyncio.Event] = {}
    releases: dict[str, asyncio.Event] = {}
    seen_tools: dict[str, tuple[str, ...]] = {}
    ctx_holder: list[Context] = []

    class DummyJobNotices:
        def watch(self, job: JobWatch) -> None:
            return None

        def origin_source(self, session_id: str, source: str) -> str:
            return source

        async def stop_job(self, session_id: str, job_id: str) -> bool:
            return False

        async def follow_jobs(self, session_id: str):
            if False:
                yield ()

    async def program(task, reader, request: Request):
        seen_tools[request.job_id] = tuple(sorted(request.tools))
        entered.setdefault(request.job_id, asyncio.Event()).set()
        release = releases.get(request.job_id)
        if release is not None:
            await release.wait()
        writer = ctx_holder[0].require(MESSAGE_WRITERS).bind(  # type: ignore[attr-defined]
            ctx_holder[0],
            author="subagent",
            source="subagent",
            body_types=(Output,),
            content={"text": ctx_holder[0].require(CONTENT).check_text},  # type: ignore[attr-defined]
        )(reader.session_id)
        try:
            return writer.append("final:" + request.job_id, Output((ContentPart("text", f"ok:{request.job_id}"),), "complete"))
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
            (JOB_NOTICES, DummyJobNotices()),
            (TOOL_BIND_SAVED, lambda _bindings, target_id, *, configuration=None: asyncio.sleep(0, result=target_id)),
        ]:
            await ctx.provide(key, value)
        await ctx.provide(CONTENT, Content(ctx))

    async def subagent_mount(ctx):
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
    runners: Runners | None = None
    jobs = Subagents(
        ctx,
        max_active=max_active,
        max_depth=max_depth,
        external_active_for_root=lambda root_id: 0 if runners is None else runners.active_for_root(root_id),
    )
    async with ctx.runtime_scope():
        runners = Runners(
            ctx,
            max_active=max_active,
            result_chars=100_000,
            resolve_root=lambda sid: jobs.resolve_lineage(sid)[0],
            tree_active_count=jobs.active_count_for_root,
            check_depth=jobs.check_depth,
            ensure_index=jobs.ensure_index,
            admit_external=jobs.admit_external,
        )
        prog_binding = bindings.bind(SUBAGENT_PROGRAM, {})
    tool_names = (
        "read_file",
        "list_dir",
        "write_file",
        "edit_file",
        "shell",
        "write_stdin",
        "task_stop",
        "web_search",
        "custom_plugin_tool",
        "spawn",
        "spawn_manage",
        "delegate_codex",
        "message_push",
        "plugin_update",
    )
    targets: dict[str, str] = {}
    for tname in tool_names:
        tid = _dummy_binding_id(tname)
        log.save_binding(tid, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": tname}})
        targets[tname] = tid
    # Simulate bindings saved via Plugin.bind for subagent program
    log.save_binding(prog_binding, {"version": 2, "origins": {"subagent": "fixture"}, "service": SUBAGENT_PROGRAM.name, "metadata": {}})
    spawn = Spawn(
        ctx,
        targets,
        {"test": "sender-1"},
        jobs,
        model=lambda _msgs, _args: asyncio.sleep(0, result=ChatModelSelection()),
    )
    return log, root, tasks, ctx, jobs, runners, spawn, entered, releases, seen_tools


def _parent_source(log: MessageLog, session_id: str, msg_id: str = "parent-in") -> CallSource:
    w = log.writer(
        session_id,
        author="user",
        source="conversation",
        body_types=(Input,),
        content={"text": lambda _: ContentReferences(), "channel.origin": lambda _: ContentReferences()},
    )
    msg = w.append(msg_id, Input((ContentPart("text", "go"),)))
    return CallSource(messages=(msg,), call_ref=CallRef(msg_id + "-out", 0))


@pytest.mark.asyncio
async def test_uniform_max_depth_across_sync_and_external(tmp_path: Path):
    # max_depth = 0 完全禁止委派
    dir0 = tmp_path / "d0"
    dir0.mkdir()
    log0, root0, tasks0, ctx0, _jobs0, runners0, spawn0, *_ = await _mount_fixture(dir0, max_depth=0)
    try:
        src0 = _parent_source(log0, "root-0")
        async with ctx0.runtime_scope():
            msg = await spawn0.prepare({"task": "blocked"}, src0)
            assert isinstance(msg, str) and "max_depth=0" in msg
            runner = AgentRunner("codex", "codex", False, "read-only", lambda t, a: asyncio.sleep(0, result=AgentResult("completed", "x")))
            d0 = Delegate(runners0, ctx0, runner)
            dmsg = await d0.prepare({"task": "blocked ext"}, src0)
            assert isinstance(dmsg, str) and "max_depth=0" in dmsg
    finally:
        await tasks0.close()
        await root0.dispose()
        log0.close()

    # max_depth = 2 允许 root(0) -> child(1) -> grandchild(2)，但阻止 depth=2 再派第 3 层
    dir2 = tmp_path / "d2"
    dir2.mkdir()
    log2, root2, tasks2, ctx2, jobs2, _runners2, spawn2, *_ = await _mount_fixture(dir2, max_depth=2)
    try:
        src_root = _parent_source(log2, "root-2")
        async with ctx2.runtime_scope():
            p1 = await spawn2.prepare({"task": "level1"}, src_root)
            assert isinstance(p1, dict)
            await jobs2.accept("k1", Request.model_validate(p1["request"]), cast(str, p1["task"]))
            req1 = Request.model_validate(p1["request"])
            assert req1.depth == 1 and req1.root_session_id == "root-2"
            assert "spawn" in req1.tools

            src_child1 = _parent_source(log2, req1.session_id, "c1-in")
            p2 = await spawn2.prepare({"task": "level2"}, src_child1)
            assert isinstance(p2, dict)
            await jobs2.accept("k2", Request.model_validate(p2["request"]), cast(str, p2["task"]))
            req2 = Request.model_validate(p2["request"])
            assert req2.depth == 2 and req2.root_session_id == "root-2"
            assert "spawn" not in req2.tools and "delegate_codex" not in req2.tools

            src_child2 = _parent_source(log2, req2.session_id, "c2-in")
            p3 = await spawn2.prepare({"task": "level3-blocked"}, src_child2)
            assert isinstance(p3, str) and "上限 2 层" in p3
    finally:
        await tasks2.close()
        await root2.dispose()
        log2.close()


@pytest.mark.asyncio
async def test_concurrent_spawn_and_external_race_at_capacity_boundary(tmp_path: Path):
    """并发压测：同一根树已有 max_active-1 占槽时，spawn 与 delegate_* 同时抢最后一个名额，最多只有 1 个成功。"""
    log, root, tasks, ctx, jobs, runners, spawn, entered, releases, _seen = await _mount_fixture(
        tmp_path, max_active=3, max_depth=1
    )
    waiters: list[asyncio.Task[Result]] = []
    try:
        src = _parent_source(log, "race-root", "r-in")
        async with ctx.runtime_scope():
            # 先占满 2 个（max_active - 1）
            pre_jids = []
            for i in range(2):
                prep = await spawn.prepare({"task": f"pre-{i}", "label": f"pre-{i}"}, src)
                assert isinstance(prep, dict)
                jid = Request.model_validate(prep["request"]).job_id
                pre_jids.append(jid)
                releases[jid] = asyncio.Event()

                async def _go(k=f"pre-{i}", d=prep):
                    async with ctx.runtime_scope():
                        return await spawn.invoke(k, d)

                waiters.append(asyncio.create_task(_go()))
                await asyncio.wait_for(entered.setdefault(jid, asyncio.Event()).wait(), timeout=5)

            ext_rel = asyncio.Event()

            async def run_ext(task, activity):
                await ext_rel.wait()
                return AgentResult("completed", "ok")

            runner = AgentRunner("codex", "codex", True, "read-only", run_ext)
            delegate = Delegate(runners, ctx, runner)

            # 同时发起 2 个本地 spawn + 2 个外部 delegate_codex 抢第 3 个槽位
            race_preps = []
            for i in range(2):
                sp = await spawn.prepare({"task": f"race-s-{i}", "label": f"rs-{i}"}, src)
                assert isinstance(sp, dict)
                s_jid = Request.model_validate(sp["request"]).job_id
                releases[s_jid] = asyncio.Event()
                releases[s_jid].set()
                race_preps.append(("spawn", f"rs-{i}", sp))
                dp = await delegate.prepare({"task": f"race-d-{i}", "label": f"rd-{i}"}, src)
                assert isinstance(dp, dict)
                race_preps.append(("ext", f"rd-{i}", dp))

            async def _race(kind: str, k: str, d: dict):
                async with ctx.runtime_scope():
                    if kind == "spawn":
                        return await spawn.invoke(k, d)
                    return await delegate.invoke(k, d)

            # 保持 pre 任务不释放期间全部竞争；注意只有第 1 个赢家占槽，若赢家立刻结束可能释放，
            # 所以让 race spawn 也等待 release 直到统计完！
            for kind, k, d in race_preps:
                if kind == "spawn":
                    releases[Request.model_validate(d["request"]).job_id].clear()

            race_tasks = [asyncio.create_task(_race(knd, k, d)) for knd, k, d in race_preps]
            await asyncio.sleep(0.15)
            assert jobs.active_count_for_root("race-root") <= 3

            for ev in releases.values():
                ev.set()
            ext_rel.set()
            race_results = await asyncio.gather(*race_tasks)
            await asyncio.gather(*waiters)
            while runners.active:
                await asyncio.sleep(0.02)
            assert jobs.active_count_for_root("race-root") == 0
            # 确认确有竞争者因满容被拒绝（本地或外部）
            assert any(r.outcome == "error" for r in race_results)
    finally:
        for ev in releases.values():
            ev.set()
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
        await tasks.close()
        await root.dispose()
        log.close()
