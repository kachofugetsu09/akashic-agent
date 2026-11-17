"""PR #5 测试：Workbench dashboard HTTP 接口、统一血缘 session_tree、cross-session 权限隔离及 runner_ui RPC。"""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI

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
)
from plugins.ledger.log import MessageLog
from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
from plugins.models.contract import MODEL_SELECTION, ChatModelSelection
from plugins.models.selection import SelectionOwner
from plugins.subagent.contract import SUBAGENT_PROGRAM
from plugins.subagent.dashboard import SETTINGS, SUBAGENT_UI_SERVICE, register as register_dashboard
from plugins.subagent.inputs import CHECK_ORIGIN, CONTENT, TOOL_BIND_SAVED
from plugins.subagent.request import Request
from plugins.subagent.runner_ui import register_ui
from plugins.subagent.runners import Runners
from plugins.subagent.runtime import Subagents
from plugins.subagent.tools import Spawn
from plugins.tools.contract import CallRef, CallSource
from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition


def _dummy_binding_id(name: str) -> str:
    descriptor = {
        "version": 2,
        "origins": {"subagent": "fixture"},
        "service": "subagent.program.v1",
        "metadata": {"tool": name},
    }
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@pytest.mark.asyncio
async def test_workbench_dashboard_routes_and_runner_ui_rpc(tmp_path: Path):
    log = MessageLog(tmp_path / "sessions.db")
    root = CompositionRoot("subagent-wb-test")
    tasks = PluginTasks()
    bindings = Bindings(log, root.context)
    ctx_holder: list[Context] = []
    registered_slots: list[tuple[PluginUiDefinition, object, object]] = []
    jobs_holder: list[Subagents] = []

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

    class DummyUiSlots:
        async def register_plugin_ui(self, ctx, definition, *, query, available=None, follow=None):
            registered_slots.append((definition, query, follow))

    class DummyUi:
        def register_configuration(self, app, context, key, prefix):
            return None

    class DummyConfig:
        pass

    async def program(task, reader, request: Request):
        before_step, _wake = jobs_holder[0].steer_binding(reader.session_id)
        await before_step()
        msgs = await reader.snapshot_async(through_seq=reader.head())
        inputs = [str(p.value) for m in msgs if isinstance(m.body, Input) for p in m.body.parts if isinstance(p, ContentPart) and p.kind == "text"]
        writer = ctx_holder[0].require(MESSAGE_WRITERS).bind(
            ctx_holder[0], author="subagent", source="subagent", body_types=(Output,),
            content={"text": ctx_holder[0].require(CONTENT).check_text},
        )(reader.session_id)
        try:
            return writer.append(f"wb-out:{uuid4().hex[:8]}", Output((ContentPart("text", f"seen={'|'.join(inputs)}"),), "complete"))
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
            (UI_SLOTS, DummyUiSlots()),
            (UI, DummyUi()),
            (SETTINGS, DummyConfig()),
            (TOOL_BIND_SAVED, lambda _b, tid, *, configuration=None: asyncio.sleep(0, result=tid)),
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
            MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, SESSION_ADMISSION,
            TASKS, BINDINGS, CONTENT, CHECK_ORIGIN, MODEL_SELECTION,
            JOB_NOTICES, UI_SLOTS, UI, SETTINGS, TOOL_BIND_SAVED,
        ),
        runtime=PluginRuntime(
            "subagent", "fixture", tmp_path, tmp_path / "plugin-data", tmp_path, {},
            workspace_roots=("subagent-runs",), workspace_files=("memory/spawn_trace.jsonl",),
        ),
    )
    ctx = ctx_holder[0]
    jobs = Subagents(ctx, max_active=8, max_depth=2)
    jobs_holder.append(jobs)
    async with ctx.runtime_scope():
        runners = Runners(ctx, max_active=8, result_chars=50_000, resolve_root=lambda sid: jobs.resolve_lineage(sid)[0])
        prog_binding = bindings.bind(SUBAGENT_PROGRAM, {})
        await ctx.provide(SUBAGENT_UI_SERVICE, (jobs, runners))
        await register_ui(ctx, runners, jobs)
    tid = _dummy_binding_id("read_file")
    log.save_binding(tid, {"version": 2, "origins": {"subagent": "fixture"}, "service": "tools.v1", "metadata": {"name": "read_file"}})
    log.save_binding(prog_binding, {"version": 2, "origins": {"subagent": "fixture"}, "service": SUBAGENT_PROGRAM.name, "metadata": {}})
    spawn = Spawn(ctx, {"read_file": tid}, {}, jobs, model=lambda _m, _a: asyncio.sleep(0, result=ChatModelSelection()))

    app = FastAPI()
    register_dashboard(app, ctx)  # type: ignore[arg-type]

    try:
        win = log.writer("wb-root", author="user", source="conversation", body_types=(Input,),
                         content={"text": lambda _: ContentReferences()})
        in_msg = win.append("wb-in", Input((ContentPart("text", "go"),)))
        src = CallSource(messages=(in_msg,), call_ref=CallRef("wb-out", 0))

        async with ctx.runtime_scope():
            prep = await spawn.prepare({"task": "workbench task", "label": "WB 任务"}, src)
            assert isinstance(prep, dict)
            jid = Request.model_validate(prep["request"]).job_id
            res = await spawn.invoke("wb-k1", prep)
            assert res.outcome == "success"

            # 1. 验证 runner_ui slots 包含 header.meta 和 composer.status，且 runners.list 返回 subagents 血缘
            defn, query_fn, _follow_fn = registered_slots[0]
            assert set(defn.slots) == {"header.meta", "chat.notice", "tool.result", "tool.summary"}
            listed = await query_fn("runners.list", {}, session_id="wb-root", turn_id=None)  # type: ignore[operator]
            assert len(listed["subagents"]) == 1
            assert listed["subagents"][0]["label"] == "WB 任务"

        # 2. 验证 dashboard HTTP 接口（通过 TestClient 在同一事件循环跑完异步路由需要 httpx/ASGITransport）
        import httpx
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            async with ctx.runtime_scope():
                r_tree = await client.get("/api/dashboard/subagent/tree", params={"session": "wb-root"})
                assert r_tree.status_code == 200
                assert len(r_tree.json()["subagents"]) == 1

                r_msgs = await client.get("/api/dashboard/subagent/messages", params={"session": "wb-root", "job_id": jid})
                assert r_msgs.status_code == 200
                assert any("workbench task" in json.dumps(m, ensure_ascii=False) for m in r_msgs.json()["messages"])

                # 跨会话禁止访问
                r_forbid = await client.get("/api/dashboard/subagent/messages", params={"session": "other-root", "job_id": jid})
                assert r_forbid.status_code == 403

                # 空白字符发送应返回 422
                r_blank = await client.post("/api/dashboard/subagent/send", json={
                    "session": "wb-root", "job_id": jid, "message": "   \n ",
                })
                assert r_blank.status_code == 422

                # 分页 before_seq 与 has_more 检查
                r_lim = await client.get("/api/dashboard/subagent/messages", params={"session": "wb-root", "job_id": jid, "limit": 1})
                assert r_lim.status_code == 200 and r_lim.json()["has_more"] is True
                first_seq = r_lim.json()["messages"][0]["seq"]
                r_prev = await client.get("/api/dashboard/subagent/messages", params={"session": "wb-root", "job_id": jid, "before_seq": first_seq, "limit": 10})
                assert r_prev.status_code == 200 and len(r_prev.json()["messages"]) >= 1

                # 通过 dashboard POST /send 续聊；运行期间 session_tree 仍保留完整 session_id / parent_session_id
                r_send = await client.post("/api/dashboard/subagent/send", json={
                    "session": "wb-root", "job_id": jid, "message": "dashboard follow up",
                })
                assert r_send.status_code == 200
                assert r_send.json()["mode"] == "resumed_background"
                r_tree_live = await client.get("/api/dashboard/subagent/tree", params={"session": "wb-root"})
                assert r_tree_live.status_code == 200
                live_node = r_tree_live.json()["subagents"][0]
                assert live_node["session_id"] == f"subagent:{jid}"
                assert live_node["parent_session_id"] == "wb-root"
        while jobs.active_count_for_root("wb-root") > 0:
            await asyncio.sleep(0.02)
    finally:
        await tasks.close()
        await root.dispose()
        log.close()
