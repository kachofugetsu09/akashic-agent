"""PR #4 测试：cache-safe fork 前缀继承与并行占位对齐、Git worktree 隔离及按变更感知清理 + Monkey 压测。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest

from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
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
    ToolCall,
    ToolResult,
)
from plugins.ledger.log import MessageLog
from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
from plugins.models.contract import MODEL_SELECTION, ChatModelSelection
from plugins.models.selection import SelectionOwner
from plugins.subagent.contract import SUBAGENT_PROGRAM
from plugins.subagent.inputs import CHECK_ORIGIN, CONTENT, TOOL_BIND_SAVED
from plugins.subagent.request import Request
from plugins.subagent.runtime import FORK_TOOL_PLACEHOLDER, Subagents
from plugins.subagent.tools import Spawn
from plugins.tools.contract import CallRef, CallSource


def _dummy_binding_id(name: str) -> str:
    descriptor = {
        "version": 2,
        "origins": {"subagent": "fixture"},
        "service": "subagent.program.v1",
        "metadata": {"tool": name},
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _init_git_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_dir, check=True, capture_output=True)
    (repo_dir / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_dir, check=True, capture_output=True)


async def _mount_pr4_fixture(tmp_path: Path):
    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("subagent-pr4")
    tasks = PluginTasks()
    bindings = Bindings(log, root.context)
    work_actions: dict[str, str] = {}
    ctx_holder: list[Context] = []

    class DummyNotices:
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
        action = work_actions.get(request.job_id, "noop")
        if request.cwd and action == "dirty":
            (Path(request.cwd) / "dirty.txt").write_text("changed\n", encoding="utf-8")
        elif request.cwd and action == "commit":
            (Path(request.cwd) / "committed.txt").write_text("committed\n", encoding="utf-8")
            subprocess.run(["git", "add", "committed.txt"], cwd=request.cwd, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "subagent change"], cwd=request.cwd, check=True, capture_output=True)

        msgs = await reader.snapshot_async(through_seq=reader.head())
        writer = ctx_holder[0].require(MESSAGE_WRITERS).bind(
            ctx_holder[0],
            author="subagent",
            source="subagent",
            body_types=(Output,),
            content={"text": ctx_holder[0].require(CONTENT).check_text},
        )(reader.session_id)
        try:
            summary = f"msgs={len(msgs)}|cwd={request.cwd}"
            return writer.append(f"final:{request.job_id}:{uuid4().hex[:6]}", Output((ContentPart("text", summary),), "complete"))
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
            (JOB_NOTICES, DummyNotices()),
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
    jobs = Subagents(ctx, max_active=12, max_depth=2)
    async with ctx.runtime_scope():
        prog_binding = bindings.bind(SUBAGENT_PROGRAM, {})
    targets: dict[str, str] = {}
    for tname in ("read_file", "list_dir", "write_file", "shell"):
        tid = _dummy_binding_id(tname)
        log.save_binding(tid, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": tname}})
        targets[tname] = tid
    log.save_binding(prog_binding, {"version": 2, "origins": {"subagent": "fixture"}, "service": SUBAGENT_PROGRAM.name, "metadata": {}})
    spawn = Spawn(ctx, targets, {}, jobs, model=lambda _m, _a: asyncio.sleep(0, result=ChatModelSelection()))
    return log, root, tasks, ctx, jobs, spawn, work_actions


@pytest.mark.asyncio
async def test_cache_safe_fork_shares_identical_prefix_across_parallel_children(tmp_path: Path):
    log, root, tasks, ctx, _jobs, spawn, _actions = await _mount_pr4_fixture(tmp_path)
    try:
        # 构造一段带历史轮次 + 并行工具调用的父会话
        win = log.writer("parent-fork", author="user", source="conversation", body_types=(Input,),
                         content={"text": lambda _: ContentReferences()})
        wout = log.writer("parent-fork", author="assistant", source="conversation", body_types=(Output,),
                          content={"text": lambda _: ContentReferences()}, check_call=lambda _c: None)
        bid_read = _dummy_binding_id("read_file")
        bid_spawn = _dummy_binding_id("spawn")
        log.save_binding(bid_spawn, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": "spawn"}})

        m0 = win.append("p-in-0", Input((ContentPart("text", "first question"),)))
        m1 = wout.append("p-out-0", Output((ContentPart("text", "reading file"), ToolCall(bid_read, {"path": "a.py"})), "continue"))
        wres = log.writer("parent-fork", author="tool", source="conversation", body_types=(ToolResult,),
                          content={"text": lambda _: ContentReferences()}, call_ref=CallRef(m1.message_id, 1))
        m2 = wres.append("p-res-0", ToolResult(CallRef(m1.message_id, 1), "success", (ContentPart("text", "file content A"),)))
        m3 = wout.append("p-out-1", Output((ContentPart("text", "first answer"),), "complete"))

        m4 = win.append("p-in-1", Input((ContentPart("text", "fork two tasks"),)))
        m5 = wout.append("p-out-2", Output((
            ToolCall(bid_spawn, {"task": "branch A", "fork": True}),
            ToolCall(bid_spawn, {"task": "branch B", "fork": True}),
        ), "continue"))

        prefix_msgs = (m0, m1, m2, m3, m4, m5)
        src_a = CallSource(messages=prefix_msgs, call_ref=CallRef(m5.message_id, 0))
        src_b = CallSource(messages=prefix_msgs, call_ref=CallRef(m5.message_id, 1))

        async with ctx.runtime_scope():
            pa = await spawn.prepare({"task": "analyze branch A", "label": "A", "fork": True}, src_a)
            pb = await spawn.prepare({"task": "analyze branch B", "label": "B", "fork": True}, src_b)
            assert isinstance(pa, dict) and isinstance(pb, dict)
            ra = await spawn.invoke("fork-a", pa)
            rb = await spawn.invoke("fork-b", pb)
            assert ra.outcome == "success" and rb.outcome == "success"

            req_a = Request.model_validate(pa["request"])
            req_b = Request.model_validate(pb["request"])
            msgs_a = log.reader(req_a.session_id).snapshot()
            msgs_b = log.reader(req_b.session_id).snapshot()

            # 两个分叉子会话除了各自末尾的新 directive Input 与 Output 外，前置消息序列化完全一致（KV-cache 共享！）
            def norm_prefix(msgs):
                out = []
                for m in msgs[:-2]:  # 跳过末尾各自新 Input 与新 Output
                    out.append((type(m.body).__name__, [
                        (type(p).__name__, getattr(p, "kind", None), getattr(p, "value", None), getattr(p, "outcome", None))
                        for p in getattr(m.body, "parts", ())
                    ]))
                return out

            norm_a = norm_prefix(msgs_a)
            norm_b = norm_prefix(msgs_b)
            assert norm_a == norm_b
            # 确认并行未回执的工具槽位都补齐了 FORK_TOOL_PLACEHOLDER
            placeholders = [
                p.value
                for m in msgs_a[:-2]
                if isinstance(m.body, ToolResult)
                for p in m.body.parts
                if isinstance(p, ContentPart) and p.value == FORK_TOOL_PLACEHOLDER
            ]
            assert len(placeholders) == 2
    finally:
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_worktree_isolation_and_change_aware_cleanup(tmp_path: Path):
    repo_dir = tmp_path / "repo"
    _init_git_repo(repo_dir)
    non_git = tmp_path / "plain"
    non_git.mkdir()

    log, root, tasks, ctx, _jobs, spawn, work_actions = await _mount_pr4_fixture(tmp_path)
    try:
        win = log.writer("wt-parent", author="user", source="conversation", body_types=(Input,),
                         content={"text": lambda _: ContentReferences()})
        in_msg = win.append("wt-in", Input((ContentPart("text", "worktree test"),)))
        src = CallSource(messages=(in_msg,), call_ref=CallRef("wt-out", 0))

        async with ctx.runtime_scope():
            # 1. 非 Git 目录传 isolation="worktree" 应明确拒绝
            bad = await spawn.prepare({"task": "not git", "cwd": str(non_git), "isolation": "worktree"}, src)
            assert isinstance(bad, str) and "Git 仓库" in bad

            # 2. 无修改/无 commit 的 clean worktree 结束后应自动移除目录与分支
            p_clean = await spawn.prepare({"task": "inspect", "label": "clean", "cwd": str(repo_dir), "isolation": "worktree"}, src)
            assert isinstance(p_clean, dict)
            req_clean = Request.model_validate(p_clean["request"])
            assert req_clean.worktree_branch is not None and req_clean.cwd is not None
            work_actions[req_clean.job_id] = "noop"
            res_clean = await spawn.invoke("wt-clean", p_clean)
            assert res_clean.outcome == "success"
            assert not Path(req_clean.cwd).exists()
            branches = subprocess.run(["git", "-C", str(repo_dir), "branch", "--list", req_clean.worktree_branch],
                                      capture_output=True, text=True, check=True).stdout.strip()
            assert branches == ""

            # 3. 有未提交修改的 worktree 结束后保留并在结果注明
            p_dirty = await spawn.prepare({"task": "modify", "label": "dirty", "cwd": str(repo_dir), "isolation": "worktree"}, src)
            assert isinstance(p_dirty, dict)
            req_dirty = Request.model_validate(p_dirty["request"])
            assert req_dirty.cwd is not None and req_dirty.worktree_branch is not None
            work_actions[req_dirty.job_id] = "dirty"
            res_dirty = await spawn.invoke("wt-dirty", p_dirty)
            assert res_dirty.outcome == "success"
            assert Path(req_dirty.cwd).exists()
            assert "[worktree 保留]" in str(res_dirty.parts[0].value)

            # 4. 有新增 commit 的 worktree 结束后保留并注明分支
            p_comm = await spawn.prepare({"task": "commit", "label": "comm", "cwd": str(repo_dir), "isolation": "worktree"}, src)
            assert isinstance(p_comm, dict)
            req_comm = Request.model_validate(p_comm["request"])
            assert req_comm.cwd is not None and req_comm.worktree_branch is not None
            work_actions[req_comm.job_id] = "commit"
            res_comm = await spawn.invoke("wt-commit", p_comm)
            assert res_comm.outcome == "success"
            assert Path(req_comm.cwd).exists()
            assert req_comm.worktree_branch in str(res_comm.parts[0].value)
            # 确认主仓库本身工作区保持干净
            main_status = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"],
                                         capture_output=True, text=True, check=True).stdout.strip()
            assert main_status == ""
    finally:
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_monkey_concurrent_fork_and_worktrees(tmp_path: Path):
    """随机并发 Monkey 测试：混合 fork + clean/dirty/commit worktree 并发运行，验证主仓洁净与清理不互相踩踏。"""
    rng = random.Random(202611174)
    repo_dir = tmp_path / "monkey-repo"
    _init_git_repo(repo_dir)
    log, root, tasks, ctx, _jobs, spawn, work_actions = await _mount_pr4_fixture(tmp_path)
    try:
        win = log.writer("m-wt", author="user", source="conversation", body_types=(Input,),
                         content={"text": lambda _: ContentReferences()})
        in_msg = win.append("m-in", Input((ContentPart("text", "monkey"),)))
        src = CallSource(messages=(in_msg,), call_ref=CallRef("m-out", 0))

        async with ctx.runtime_scope():
            jobs_info: list[tuple[Request, str]] = []
            for i in range(8):
                mode = rng.choice(["noop", "dirty", "commit"])
                fork = rng.random() < 0.5
                prep = await spawn.prepare(
                    {"task": f"mw-{i}", "label": f"mw-{i}", "cwd": str(repo_dir), "isolation": "worktree", "fork": fork},
                    src,
                )
                assert isinstance(prep, dict)
                req = Request.model_validate(prep["request"])
                work_actions[req.job_id] = mode
                jobs_info.append((req, mode))

            async def run_one(idx: int, req: Request):
                async with ctx.runtime_scope():
                    # Re-build Prepared from req
                    return await spawn.invoke(f"mwt-{idx}", {"task": f"mw-{idx}", "request": req.model_dump()})

            results = await asyncio.gather(*(run_one(i, req) for i, (req, _) in enumerate(jobs_info)))
            assert all(r.outcome == "success" for r in results)
            for (req, mode), res in zip(jobs_info, results, strict=True):
                assert req.cwd is not None
                if mode == "noop":
                    assert not Path(req.cwd).exists()
                else:
                    assert Path(req.cwd).exists()
                    assert "[worktree 保留]" in str(res.parts[0].value)
            main_status = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain"],
                                         capture_output=True, text=True, check=True).stdout.strip()
            assert main_status == ""
    finally:
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_fork_filters_abandoned_and_foreign_sources_and_prompts_stay_identical(tmp_path: Path):
    """验证 fork 过滤异源与被 abandon 的悬空 Output、支持半途崩溃幂等续写，且 system_prompt 对并行 fork 一致。"""
    from plugins.ledger.contract import Control
    from plugins.subagent.prompts import build_spawn_subagent_prompt

    # 1. 验证真实 system_prompt 对不同 job_id / task_dir 的并行 fork=True 子任务完全字节级一致
    p1 = build_spawn_subagent_prompt(tmp_path, tmp_path / "subagent-runs" / "jobA", fork=True, cwd=str(tmp_path / "subagent-runs" / "jobA"))
    p2 = build_spawn_subagent_prompt(tmp_path, tmp_path / "subagent-runs" / "jobB", fork=True, cwd=str(tmp_path / "subagent-runs" / "jobB"))
    assert p1 == p2

    log, root, tasks, ctx, jobs, spawn, _actions = await _mount_pr4_fixture(tmp_path)
    try:
        win = log.writer("pf-ab", author="user", source="conversation", body_types=(Input,),
                         content={"text": lambda _: ContentReferences()})
        wout = log.writer("pf-ab", author="assistant", source="conversation", body_types=(Output,),
                          content={"text": lambda _: ContentReferences()}, check_call=lambda _c: None)
        wctrl = log.writer("pf-ab", author="system", source="conversation", body_types=(Control,), content={})
        wother = log.writer("pf-ab", author="system", source="other-source", body_types=(Input,),
                            content={"text": lambda _: ContentReferences()})

        bid_read = _dummy_binding_id("read_file")
        bid_spawn = _dummy_binding_id("spawn")
        log.save_binding(bid_spawn, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": "spawn"}})

        m0 = win.append("pa-in-0", Input((ContentPart("text", "first"),)))
        m_other = wother.append("pa-other", Input((ContentPart("text", "foreign noise"),)))
        # 一条后来被 Control("abandon") 丢弃的未回执 Output
        m_ab_out = wout.append("pa-ab-out", Output((ToolCall(bid_read, {"path": "abandoned.py"}),), "continue"))
        m_ctrl = wctrl.append("pa-ctrl", Control("abandon", m_ab_out.seq, "abandoned"))
        m_in1 = win.append("pa-in-1", Input((ContentPart("text", "go fork"),)))
        m_cur = wout.append("pa-cur", Output((ToolCall(bid_spawn, {"task": "sub", "fork": True}),), "continue"))

        msgs = (m0, m_other, m_ab_out, m_ctrl, m_in1, m_cur)
        src = CallSource(messages=msgs, call_ref=CallRef(m_cur.message_id, 0))

        async with ctx.runtime_scope():
            prep = await spawn.prepare({"task": "check", "label": "chk", "fork": True}, src)
            assert isinstance(prep, dict)
            req = Request.model_validate(prep["request"])

            # 模拟崩溃：ensure_session 后只写入第一条 fork-in:0，验证 _seed_fork_prefix 重试仍能补全后续消息且不报错
            from plugins.ledger.contract import SessionAttributes
            log.ensure_session(req.session_id, SessionAttributes(visibility="internal", learning="excluded"))
            w_partial = log.writer(req.session_id, author="subagent", source="subagent", body_types=(Input,),
                                   content={"text": lambda _: ContentReferences()})
            w_partial.append(f"fork:{req.job_id}:in:{m0.seq}", Input((ContentPart("text", "first"),)))
            w_partial.expire()

            res = await spawn.invoke("chk-k", prep)
            assert res.outcome == "success"
            child_msgs = log.reader(req.session_id).snapshot()
            serialized = json.dumps([[getattr(p, "value", getattr(p, "arguments", "")) for p in getattr(m.body, "parts", ())] for m in child_msgs], ensure_ascii=False)
            assert "foreign noise" not in serialized
            assert "abandoned.py" not in serialized
            assert "first" in serialized and "go fork" in serialized
            assert FORK_TOOL_PLACEHOLDER in serialized
    finally:
        await tasks.close()
        await root.dispose()
        log.close()


@pytest.mark.asyncio
async def test_worktree_unborn_head_subdir_alignment_and_send_recreate(tmp_path: Path):
    """验证 unborn HEAD 拒绝、子目录相对路径对齐，以及 clean worktree 回收后 subagent_send 自动重建。"""
    unborn = tmp_path / "unborn-repo"
    unborn.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=unborn, check=True, capture_output=True)

    repo_dir = tmp_path / "repo-sub"
    _init_git_repo(repo_dir)
    sub_pkg = repo_dir / "packages" / "app"
    sub_pkg.mkdir(parents=True)
    (sub_pkg / "index.txt").write_text("ok\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add pkg"], cwd=repo_dir, check=True, capture_output=True)

    log, root, tasks, ctx, jobs, spawn, work_actions = await _mount_pr4_fixture(tmp_path)
    try:
        win = log.writer("wt2", author="user", source="conversation", body_types=(Input,),
                         content={"text": lambda _: ContentReferences()})
        in_msg = win.append("wt2-in", Input((ContentPart("text", "wt2"),)))
        src = CallSource(messages=(in_msg,), call_ref=CallRef("wt2-out", 0))

        async with ctx.runtime_scope():
            # 1. unborn HEAD 仓库在 prepare 阶段给出明确错误
            bad = await spawn.prepare({"task": "unborn", "cwd": str(unborn), "isolation": "worktree"}, src)
            assert isinstance(bad, str) and "unborn HEAD" in bad

            # 2. 子目录 cwd 映射到 worktree 对应子目录，且 clean 后自动删除
            p_sub = await spawn.prepare({"task": "subdir", "label": "sub", "cwd": str(sub_pkg), "isolation": "worktree"}, src)
            assert isinstance(p_sub, dict)
            req = Request.model_validate(p_sub["request"])
            assert req.worktree_path is not None and req.cwd is not None
            assert Path(req.cwd) == Path(req.worktree_path) / "packages" / "app"
            work_actions[req.job_id] = "noop"
            res1 = await spawn.invoke("wt2-k", p_sub)
            assert res1.outcome == "success"
            assert not Path(req.worktree_path).exists()

            # 3. clean 回收后经 subagent_send 冷恢复，应自动重建 worktree，且本次写入文件后保留并带提示
            work_actions[req.job_id] = "dirty"
            res2 = await jobs.send("wt2", req.job_id, "now write dirty.txt", background=False)
            assert res2["mode"] == "resumed_sync"
            assert Path(req.worktree_path).exists()
            assert "[worktree 保留]" in str(res2["result"])
    finally:
        await tasks.close()
        await root.dispose()
        log.close()
