from __future__ import annotations

import asyncio
from contextlib import closing
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import textwrap
from typing import cast

import httpx
import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import RuntimeSnapshot
from agent.plugins.web_ui import WebModuleDescriptor
from bootstrap.dashboard_api import create_dashboard_app
from plugins.akasha.recalls import ContextSource, Hit, Recall, RecallRecords
from agent.plugin_contracts.plugin_capabilities import DELIVERY_READ
from agent.plugin_contracts.plugin_capabilities import DRIFT_PROPOSALS
from plugins.wake.api import DRIFT_WAKE
from plugins.wake.runtime import Runtime
from plugins.wake.state import WakeState
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input, Output


_REPO_ROOT = Path(__file__).parents[1]


def _file_snapshot(path: Path) -> dict[str, tuple[int, str]]:
    """记录数据库及其协调文件的内容，不打开数据库。"""
    result: dict[str, tuple[int, str]] = {}
    for candidate in sorted(path.parent.glob(path.name + "*")):
        if candidate.is_file():
            result[candidate.name] = (
                candidate.stat().st_size,
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
    return result


def _web_headers(snapshot: RuntimeSnapshot, plugin_id: str) -> tuple[WebModuleDescriptor, dict[str, str]]:
    catalog = snapshot.web_ui_catalog
    assert catalog is not None
    module = next(item for item in catalog.modules if item.plugin_id == plugin_id)
    headers = {
        "x-akashic-web-snapshot": snapshot.snapshot_id,
        "x-akashic-web-catalog": catalog.identity,
        "x-akashic-web-module": module.plugin_id,
        "x-akashic-web-generation": module.generation_id,
    }
    return module, headers


def _render_compiled_module(
    tmp_path: Path,
    module: WebModuleDescriptor,
    payload: dict[str, object],
    marker: str,
) -> None:
    """渲染 snapshot 中的编译资源，检查实际字段是否完整。"""
    module_file = tmp_path / f"{module.plugin_id}-web_module.js"
    payload_file = tmp_path / f"{module.plugin_id}-detail.json"
    module_file.write_text(module.asset.module, encoding="utf-8")
    payload_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    script = textwrap.dedent(
        """
        import { readFileSync } from "node:fs";
        import { pathToFileURL } from "node:url";
        import { JSDOM } from "jsdom";

        const dom = new JSDOM("<!doctype html><body></body>");
        globalThis.window = dom.window;
        globalThis.document = dom.window.document;
        globalThis.HTMLElement = dom.window.HTMLElement;
        const panelModule = await import(pathToFileURL(process.env.TEST_MODULE).href + "?compiled=1");
        let panel;
        const dispose = panelModule.activate({
          http: { request: async () => { throw new Error("HTTP should not be used while rendering"); } },
          ui: { inject: (_slot, register) => {
            register({ register: (candidate) => { panel = candidate; } });
            return () => {};
          } },
        });
        if (!panel) throw new Error("compiled module did not register a panel");
        const container = document.createElement("div");
        const payload = JSON.parse(readFileSync(process.env.TEST_PAYLOAD, "utf8"));
        panel.renderDetail(payload, container, { closePane: () => {} });
        if (container.innerHTML.includes("undefined")) throw new Error(container.innerHTML);
        if (!container.textContent.includes(process.env.TEST_MARKER)) throw new Error(container.innerHTML);
        dispose();
        """
    )
    environment = {
        **os.environ,
        "TEST_MODULE": str(module_file),
        "TEST_PAYLOAD": str(payload_file),
        "TEST_MARKER": marker,
    }
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=_REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _seed_saved_recall(log: MessageLog, _host: PluginManager) -> None:
    checks = {"text": lambda _part: ContentReferences()}
    inputs = log.writer("s", author="user", source="conversation", body_types=(Input,), content=checks)
    outputs = log.writer("s", author="assistant", source="conversation", body_types=(Output,), content=checks)
    inputs.append("long-user", Input((ContentPart("text", "完整正文" * 200 + "END"),)))
    outputs.append("answer", Output((ContentPart("text", "actual answer"),), "complete"))
    RecallRecords(log.owner("plugin:akasha")).save(
        "saved-query",
        Recall(
            learning_binding="saved-binding",
            graph_version=1,
            source=ContextSource(session_id="s", source="conversation", through_seq=1),
            timestamp=datetime(2026, 9, 7, tzinfo=timezone.utc),
            limit=7,
            hits=(Hit(
                node_id=0,
                session_id="s",
                message_ids=("long-user", "answer"),
                score=0.9,
                lane="completion",
                sources=("ripple",),
                basin_ids=("basin-1",),
            ),),
            presented_message_ids=("long-user",),
            active_basin_count=4,
            pushes=9,
            residual_l1=0.3,
        ),
    )


@pytest.mark.asyncio
async def test_akasha_dashboard_reads_saved_recall_and_renders_catalog_module(tmp_path: Path) -> None:
    from tests.test_akasha_message_plugin import application

    async with application(tmp_path, embedding_available=False, before_start=_seed_saved_recall) as (log, host):
        snapshot = host.current_snapshot
        assert snapshot is not None
        module, headers = _web_headers(snapshot, "akasha")
        app = create_dashboard_app(tmp_path / "workspace", plugin_manager=host)
        sessions_db = tmp_path / "sessions.db"
        assert sessions_db.is_file()
        before_db = sessions_db.read_bytes()
        before_messages = log.reader("s").snapshot()
        with closing(sqlite3.connect(sessions_db)) as database:
            before_dump = tuple(database.iterdump())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers,
        ) as client:
            overview = await client.get("/api/dashboard/akasha-inspector/overview")
            listing = await client.get("/api/dashboard/akasha-inspector/turns?page=1&page_size=25")
            detail_response = await client.get("/api/dashboard/akasha-inspector/turns/saved-query")
        after_db = sessions_db.read_bytes()
        with closing(sqlite3.connect(sessions_db)) as database:
            after_dump = tuple(database.iterdump())

        assert [response.status_code for response in (overview, listing, detail_response)] == [200, 200, 200]
        assert overview.json() == {"available": True, "total": 1}
        row = listing.json()["items"][0]
        detail = detail_response.json()
        assert row["limit"] == detail["limit"] == 7
        assert row["hit_count"] == detail["hit_count"] == 1
        assert detail["active_basin_count"] == 4
        assert detail["pushes"] == 9
        assert detail["residual_l1"] == 0.3
        assert {"learning_binding", "max_chars", "strong", "presented_message_ids"}.isdisjoint(detail)
        messages = detail["hits"][0]["messages"]
        assert [message["message_id"] for message in messages] == ["long-user", "answer"]
        assert [message["presented"] for message in messages] == [True, False]
        assert messages[0]["text"].endswith("END")
        assert len(messages[0]["text"]) > 240
        assert before_db
        assert after_db == before_db
        assert after_dump == before_dump
        assert log.reader("s").snapshot() == before_messages
        assert not (tmp_path / "embedding-calls.txt").exists()
        _render_compiled_module(tmp_path, module, detail, "命中回忆")


@pytest.mark.asyncio
async def test_wake_dashboard_matches_target_delivery_and_compiled_module(tmp_path: Path) -> None:
    from tests.test_wake_messages import application, request

    async with application(tmp_path) as (host, log, ctx, _source, control):
        await host.start_runtime()
        runtime = cast(Runtime, control["runtime"])
        now = datetime.now(timezone.utc)
        ctx.require(DRIFT_PROPOSALS).propose("duty", "1", {"summary": "dashboard delivery"}, now)
        original = request(ctx, "drift", now, proposals=ctx.require(DRIFT_WAKE).snapshot(now)["proposals"])
        runtime.state.begin_attempt(
            attempt_id=original.flow_id,
            timer_id="timer-dashboard",
            scheduled_for=now,
            fired_at=now,
        )
        runtime.state.set_attempt_mail_watermark(attempt_id=original.flow_id, mail_watermark=0)
        runtime.source.accept(original)
        task = await runtime.source.start(original.flow_id)
        assert task is not None
        assert await asyncio.wait_for(task.join(), 10) == "shared"
        runtime.state.finish_attempt(
            attempt_id=original.flow_id,
            outcome="shared",
            owner="drift",
            detail="dashboard fixture",
            completed_at=now,
        )
        direct_receipt = ctx.require(DELIVERY_READ).status(original.notification_id, original.sink.name)
        assert direct_receipt is not None
        target_message = log.reader(original.target.session_id).get(original.notification_id)
        assert target_message is not None

        snapshot = host.current_snapshot
        assert snapshot is not None
        module, headers = _web_headers(snapshot, "wake")
        app = create_dashboard_app(tmp_path / "workspace", plugin_manager=host)
        state_path = runtime.state.path
        before_files = _file_snapshot(state_path)
        before_attempt = runtime.state.read_only().get_attempt(original.flow_id)
        before_attempts = runtime.state.read_only().list_attempts(500)
        before_runs = runtime.state.read_only().list_runs(500)
        before_target = log.reader(original.target.session_id).snapshot()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers,
        ) as client:
            listing = await client.get("/api/dashboard/wake/attempts?page=1&page_size=25")
            detail_response = await client.get(f"/api/dashboard/wake/attempts/{original.flow_id}")
        after_files = _file_snapshot(state_path)
        assert listing.status_code == detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["flow"]["request"]["notification_id"] == original.notification_id
        assert detail["flow"]["delivery"]["message_id"] == direct_receipt["message_id"]
        assert detail["flow"]["delivery"]["status"] == direct_receipt["status"] == "delivered"
        assert detail["flow"]["delivery"]["receipt"] == direct_receipt["receipt"]
        # 只读 SQLite 连接可以创建 WAL 协调文件；Wake 主库和行内容必须稳定。
        assert before_files[state_path.name] == after_files[state_path.name]
        assert runtime.state.read_only().get_attempt(original.flow_id) == before_attempt
        assert runtime.state.read_only().list_attempts(500) == before_attempts
        assert runtime.state.read_only().list_runs(500) == before_runs
        assert log.reader(original.target.session_id).snapshot() == before_target
        _render_compiled_module(tmp_path, module, detail, "delivered")


@pytest.mark.asyncio
async def test_wake_dashboard_get_missing_state_does_not_create_database(tmp_path: Path) -> None:
    from tests.test_wake_messages import application

    async with application(tmp_path) as (host, _log, _ctx, _source, control):
        await host.start_runtime()
        runtime = cast(Runtime, control["runtime"])
        state_path = runtime.state.path
        assert state_path.exists()
        state_path.unlink()
        snapshot = host.current_snapshot
        assert snapshot is not None
        _module, headers = _web_headers(snapshot, "wake")
        app = create_dashboard_app(tmp_path / "workspace", plugin_manager=host)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers,
        ) as client:
            attempts = await client.get("/api/dashboard/wake/attempts?page=1&page_size=25")
            runs = await client.get("/api/dashboard/wake/runs?page=1&page_size=25")
        assert attempts.status_code == runs.status_code == 200
        assert attempts.json()["items"] == [] and attempts.json()["total"] == 0
        assert runs.json()["items"] == [] and runs.json()["total"] == 0
        assert not state_path.exists()


@pytest.mark.parametrize("directory", ["ordinary", "资料#one", "资料?one"])
@pytest.mark.parametrize("decoy", [False, True])
def test_wake_reader_uses_exact_database_and_never_creates_another_file(
    tmp_path: Path, directory: str, decoy: bool,
) -> None:
    """合法特殊路径仍读取原库；旁边放另一份有效库，防止只检查不报错。"""
    now = datetime(2026, 9, 7, tzinfo=timezone.utc)
    state = WakeState(tmp_path / directory / "wake.sqlite3")
    state.record_screen(run_id="expected-run", owner="content", candidates_seen=0,
                        screening=(), started_at=now)
    if decoy and directory != "ordinary":
        wrong = WakeState(tmp_path / "资料")
        wrong.record_screen(run_id="wrong-database", owner="drift", candidates_seen=0,
                            screening=(), started_at=now)
    before = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*") if path.is_file()}
    with closing(sqlite3.connect(state.path)) as connection:
        original = tuple(connection.iterdump())
    reader = state.read_only()
    try:
        rows = reader.list_runs(10)
        assert [row["run_id"] for row in rows] == ["expected-run"], "只读接口打开了另一个有效数据库"
        assert reader.get_run("expected-run") is not None
        assert reader.get_run("wrong-database") is None
    finally:
        # WAL/SHM 是 SQLite 协调文件；截断路径产生的新主库不是协调文件。
        after = {str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")
                 if path.is_file() and not path.name.endswith(("-wal", "-shm"))}
        assert after <= before, "查看操作创建了非预期文件"
        with closing(sqlite3.connect(state.path)) as connection:
            assert tuple(connection.iterdump()) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("directory", ["ordinary", "资料#one"])
async def test_wake_dashboard_reads_nonempty_state_at_exact_workspace(
    tmp_path: Path, directory: str,
) -> None:
    """从真实 Dashboard 路由读取非空 Wake 记录，特殊目录不能变成 500。"""
    from tests.test_wake_messages import application

    root = tmp_path / directory
    async with application(root) as (host, _log, _ctx, _source, control):
        await host.start_runtime()
        state = control["runtime"].state
        state.record_screen(run_id="visible-run", owner="content", candidates_seen=0,
                            screening=(), started_at=datetime(2026, 9, 7, tzinfo=timezone.utc))
        snapshot = host.current_snapshot
        assert snapshot is not None
        _, headers = _web_headers(snapshot, "wake")
        app = create_dashboard_app(root / "workspace", plugin_manager=host)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test", headers=headers,
        ) as client:
            response = await client.get("/api/dashboard/wake/runs")
        assert response.status_code == 200, response.text
        assert [item["run_id"] for item in response.json()["items"]] == ["visible-run"]


def _seed_recall_pages(log: MessageLog, _host: PluginManager) -> None:
    """交错两组会话、时间和不同正文，过滤与分页不能靠单条 fixture 蒙混过关。"""
    records = RecallRecords(log.owner("plugin:akasha"))
    checks = {"text": lambda _part: ContentReferences()}
    for index in range(12):
        session = f"session-{index % 2}"
        inputs = log.writer(session, author="user", source="conversation", body_types=(Input,), content=checks)
        outputs = log.writer(session, author="assistant", source="conversation", body_types=(Output,), content=checks)
        question = inputs.append(f"question-{index}", Input((ContentPart("text", f"START-{index}" + "汉字🧪" * 200 + f"END-{index}"),)))
        answer = outputs.append(f"answer-{index}", Output((ContentPart("text", f"ANSWER-{index}"),), "complete"))
        records.save(f"recall-{index:02}", Recall(
            learning_binding="saved-binding", graph_version=1,
            source=ContextSource(session_id=session, source="conversation", through_seq=answer.seq),
            timestamp=datetime(2026, 9, 7, tzinfo=timezone.utc) + timedelta(seconds=index), limit=1,
            hits=(Hit(node_id=0, session_id=session, message_ids=(question.message_id, answer.message_id),
                      score=0.9, lane="dense", sources=("dense",)),),
            presented_message_ids=(question.message_id,), active_basin_count=0, pushes=0, residual_l1=0,
        ))


@pytest.mark.asyncio
async def test_akasha_dashboard_filters_pages_and_returns_original_detail(tmp_path: Path) -> None:
    """第二页不得混入另一会话；详情恢复列表截断的完整正文，读取不改变 SQL 事实。"""
    from tests.test_akasha_message_plugin import application

    async with application(tmp_path, embedding_available=False, before_start=_seed_recall_pages) as (log, host):
        snapshot = host.current_snapshot
        assert snapshot is not None
        _, headers = _web_headers(snapshot, "akasha")
        app = create_dashboard_app(tmp_path / "workspace", plugin_manager=host)
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as database:
            before = tuple(database.iterdump())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers) as client:
            response = await client.get("/api/dashboard/akasha-inspector/turns",
                                        params={"session_key": "session-0", "page": 2, "page_size": 2})
            assert response.status_code == 200
            page = response.json()
            assert page["total"] == 6
            assert [row["query_id"] for row in page["items"]] == ["recall-06", "recall-04"]
            for row in page["items"]:
                message = row["hits"][0]["messages"][0]
                assert message["text_truncated"] and len(message["text"]) == 240
                detail = await client.get(f"/api/dashboard/akasha-inspector/turns/{row['query_id']}")
                assert detail.status_code == 200
                original = log.reader(message["session_id"]).get(message["message_id"])
                assert detail.json()["hits"][0]["messages"][0]["text"] == original.body.parts[0].value
            # 保留原搜索合同：搜索展示正文，且与 session 过滤取交集。
            for query, session, expected in [("sTaRt-6", "session-0", ["recall-06"]),
                                             ("START-6", "session-1", []), ("END-6", "session-0", [])]:
                search = await client.get("/api/dashboard/akasha-inspector/turns",
                                          params={"q": query, "session_key": session, "page_size": 1})
                assert search.status_code == 200
                assert [row["query_id"] for row in search.json()["items"]] == expected
                assert search.json()["total"] == len(expected)
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as database:
            assert tuple(database.iterdump()) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("session", ["session-0", "missing-session"])
async def test_akasha_page_does_not_read_bodies_outside_its_results(tmp_path: Path, session: str) -> None:
    """小页和空筛选不应展开全部历史正文；观察真实 SQLite 查询而非自己造的统计。"""
    from tests.test_akasha_message_plugin import application

    async with application(tmp_path, embedding_available=False, before_start=_seed_recall_pages) as (log, host):
        snapshot = host.current_snapshot
        assert snapshot is not None
        _, headers = _web_headers(snapshot, "akasha")
        app = create_dashboard_app(tmp_path / "workspace", plugin_manager=host)
        queries: list[str] = []
        log._connection.set_trace_callback(queries.append)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers) as client:
                response = await client.get("/api/dashboard/akasha-inspector/turns",
                                            params={"session_key": session, "page_size": 1})
        finally:
            log._connection.set_trace_callback(None)
        assert response.status_code == 200
        page = response.json()
        reads = [sql for sql in queries if "FROM messages WHERE id=" in sql]
        expected = 2 if session == "session-0" else 0
        assert len(page["items"]) == (1 if expected else 0)
        assert len(reads) <= expected, f"页面最多需要 {expected} 条正文，实际读取 {len(reads)} 条"
