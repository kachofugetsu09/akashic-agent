"""Subagent 工具继承、按根会话树隔离的并发池（默认 8）与统一递归深度上限的回归及并发 Monkey 测试。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from pathlib import Path
from typing import cast
from agent.plugin_composition import Context
from uuid import uuid4

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
from plugins.subagent.contract import SUBAGENT_PROGRAM, AgentActivity, AgentResult, AgentRunner
from plugins.subagent.inputs import CHECK_ORIGIN, CONTENT, TOOL_BIND_SAVED
from plugins.subagent.request import (
    EXCLUDED_ALWAYS_TOOLS,
    READ_ONLY_EXCLUDED_TOOLS,
    Request,
    check_request,
)
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
async def test_tool_inheritance_and_read_only_constraint(tmp_path: Path):
    log, root, tasks, ctx, jobs, _runners, spawn, _entered, _releases, seen_tools = await _mount_fixture(tmp_path, max_depth=2)
    try:
        src = _parent_source(log, "root-a")
        async with ctx.runtime_scope():
            # 1. 默认继承父会话可用工具（含 custom_plugin_tool，且在 depth=1 < max_depth=2 时继承 spawn/spawn_manage，排除 EXCLUDED_ALWAYS_TOOLS）
            prep_rw = await spawn.prepare({"task": "do rw work", "label": "rw"}, src)
            assert isinstance(prep_rw, dict)
            req_rw = Request.model_validate(prep_rw["request"])
            assert not req_rw.read_only
            assert req_rw.depth == 1
            assert req_rw.root_session_id == "root-a"
            assert "custom_plugin_tool" in req_rw.tools
            assert "write_file" in req_rw.tools and "shell" in req_rw.tools
            assert "spawn" in req_rw.tools and "spawn_manage" in req_rw.tools
            assert not (set(req_rw.tools) & EXCLUDED_ALWAYS_TOOLS)

            # 2. read_only=true 裁剪掉写/命令/委派类工具，但仍保留只读和自定义插件只读工具
            prep_ro = await spawn.prepare({"task": "inspect only", "label": "ro", "read_only": True}, src)
            assert isinstance(prep_ro, dict)
            req_ro = Request.model_validate(prep_ro["request"])
            assert req_ro.read_only
            assert "read_file" in req_ro.tools and "web_search" in req_ro.tools and "custom_plugin_tool" in req_ro.tools
            assert not (set(req_ro.tools) & READ_ONLY_EXCLUDED_TOOLS)
            assert "delegate_codex" not in req_ro.tools

            # 3. 兼容旧字段 profile="research" 等价于 read_only=True
            prep_legacy = await spawn.prepare({"task": "legacy research", "profile": "research"}, src)
            assert isinstance(prep_legacy, dict)
            assert Request.model_validate(prep_legacy["request"]).read_only is True

            # 4. check_request 拦截篡改：read_only=True 不能携带写入工具
            bad_req = req_ro.model_copy(update={"tools": {**req_ro.tools, "write_file": req_rw.tools["write_file"]}})
            with pytest.raises(ValueError, match="只读子任务"):
                check_request(ContentPart("subagent.request", bad_req.model_dump()), check_origin=check_origin)

            res = await spawn.invoke("k-rw", prep_rw)
            assert res.outcome == "success"
            assert "custom_plugin_tool" in seen_tools[req_rw.job_id]
    finally:
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_per_root_session_capacity_pool_default_8_and_external_sharing(tmp_path: Path):
    assert DEFAULT_MAX_ACTIVE == 8
    log, root, tasks, ctx, jobs, runners, spawn, entered, releases, _seen = await _mount_fixture(tmp_path, max_active=8, max_depth=1)
    active_waiters: list[asyncio.Task[Result]] = []
    try:
        src_a = _parent_source(log, "root-a", "a-in")
        src_b = _parent_source(log, "root-b", "b-in")

        async def launch(key: str, prep: dict):
            async with ctx.runtime_scope():
                return await spawn.invoke(key, prep)

        async with ctx.runtime_scope():
            # 在 root-a 上派发 7 个阻塞子任务
            job_ids_a: list[str] = []
            for i in range(7):
                prep = await spawn.prepare({"task": f"task-{i}", "label": f"t{i}"}, src_a)
                assert isinstance(prep, dict)
                jid = Request.model_validate(prep["request"]).job_id
                job_ids_a.append(jid)
                releases[jid] = asyncio.Event()
                active_waiters.append(asyncio.create_task(launch(f"ka-{i}", prep)))
                await asyncio.wait_for(entered.setdefault(jid, asyncio.Event()).wait(), timeout=5)

            # 再在 root-a 上占用第 8 个槽位（外部 runner）
            ext_release = asyncio.Event()
            ext_entered = asyncio.Event()

            async def run_ext(task, activity):
                await activity(AgentActivity("status", "running"))
                ext_entered.set()
                await ext_release.wait()
                return AgentResult("completed", "done")

            runner = AgentRunner("codex", "codex", True, "read-only", run_ext)
            delegate = Delegate(runners, ctx, runner)
            ext_prep = await delegate.prepare({"task": "ext-1", "label": "e1"}, src_a)
            assert isinstance(ext_prep, dict)
            ext_res = await delegate.invoke("ext-key-1", ext_prep)
            assert ext_res.outcome == "success"
            await asyncio.wait_for(ext_entered.wait(), timeout=5)

            # 第 9 个本地子任务在 root-a 上应被拒绝，第 2 个外部 runner 在 root-a 上也应被拒绝
            prep_9 = await spawn.prepare({"task": "overflow", "label": "t9"}, src_a)
            assert isinstance(prep_9, dict)
            res_9 = await spawn.invoke("ka-9", prep_9)
            assert res_9.outcome == "error" and "capacity reached" in str(res_9.parts[0].value)

            ext_prep_2 = await delegate.prepare({"task": "ext-2", "label": "e2"}, src_a)
            assert isinstance(ext_prep_2, dict)
            ext_res_2 = await delegate.invoke("ext-key-2", ext_prep_2)
            assert ext_res_2.outcome == "error" and "并发已满" in str(ext_res_2.parts[0].value)

            # root-b 完全独立，应可正常占用自己的槽位
            prep_b = await spawn.prepare({"task": "b-task", "label": "tb"}, src_b)
            assert isinstance(prep_b, dict)
            res_b = await spawn.invoke("kb-0", prep_b)
            assert res_b.outcome == "success"

            # 释放 root-a 上的 1 个任务后，root-a 又能接收新任务
            releases[job_ids_a[0]].set()
            await active_waiters[0]
            prep_retry = await spawn.prepare({"task": "after-release", "label": "tr"}, src_a)
            assert isinstance(prep_retry, dict)
            res_retry = await spawn.invoke("ka-after", prep_retry)
            assert res_retry.outcome == "success"

            # 清理剩余阻塞任务
            ext_release.set()
            for jid in job_ids_a[1:]:
                releases[jid].set()
            await asyncio.gather(*active_waiters)
            while runners.active:
                await asyncio.sleep(0.02)
            assert jobs.active_count_for_root("root-a") == 0
    finally:
        for ev in releases.values():
            ev.set()
        if active_waiters:
            await asyncio.gather(*active_waiters, return_exceptions=True)
        await tasks.close()
        await root.dispose()
        log.close()


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
async def test_monkey_concurrent_trees_capacity_and_cancellation_invariants(tmp_path: Path):
    """随机并发 Monkey 测试：多棵根会话树交织派发/阻塞/取消/完成，验证容量绝不越界且无槽位泄漏。"""
    rng = random.Random(20261117)
    log, root, tasks, ctx, jobs, _runners, spawn, entered, releases, _seen = await _mount_fixture(
        tmp_path, max_active=4, max_depth=2
    )
    roots = ["tree-alpha", "tree-beta", "tree-gamma"]
    waiters: list[asyncio.Task[Result]] = []
    try:
        sources = {r: _parent_source(log, r, f"{r}-in") for r in roots}

        async def run_one(idx: int, rname: str, ro: bool):
            async with ctx.runtime_scope():
                prep = await spawn.prepare({"task": f"m-{idx}", "label": f"m{idx}", "read_only": ro}, sources[rname])
                assert isinstance(prep, dict)
                jid = Request.model_validate(prep["request"]).job_id
                releases[jid] = asyncio.Event()
                async def invoke_scoped():
                    async with ctx.runtime_scope():
                        return await spawn.invoke(f"mkey-{idx}-{uuid4().hex[:6]}", prep)

                res_task = asyncio.create_task(invoke_scoped())

                async def driver():
                    # 等进入执行或被容量拒绝
                    await asyncio.sleep(rng.uniform(0.0, 0.03))
                    if rng.random() < 0.25:
                        async with ctx.runtime_scope():
                            await jobs.cancel(jid)
                    releases[jid].set()
                    return await res_task

                out = await driver()
                async with ctx.runtime_scope():
                    assert jobs.active_count_for_root(rname) <= 4
                return out

        for i in range(42):
            waiters.append(asyncio.create_task(run_one(i, rng.choice(roots), rng.random() < 0.4)))
            if i % 6 == 5:
                await asyncio.sleep(0.01)

        results = await asyncio.gather(*waiters)
        assert len(results) == 42
        async with ctx.runtime_scope():
            for rname in roots:
                assert jobs.active_count_for_root(rname) == 0
    finally:
        for ev in releases.values():
            ev.set()
        if waiters:
            await asyncio.gather(*waiters, return_exceptions=True)
        await tasks.close()
        await root.dispose()
        log.close()
