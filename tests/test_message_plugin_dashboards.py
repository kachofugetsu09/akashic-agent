from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import textwrap
from typing import Any

import httpx
import pytest

from bootstrap.dashboard_api import create_dashboard_app
from plugins.akasha.recalls import ContextSource, Hit, Recall, RecallRecords
from plugins.delivery.history import DELIVERY_READ
from plugins.drift.plugin import DRIFT_PROPOSALS
from plugins.wake.api import DRIFT_WAKE
from session.message import ContentPart, ContentReferences, Input, Output


_REPO_ROOT = Path(__file__).parents[1]


def _file_snapshot(path: Path) -> dict[str, tuple[int, str]]:
    """Record a small database family without opening or changing it."""
    result: dict[str, tuple[int, str]] = {}
    for candidate in sorted(path.parent.glob(path.name + "*")):
        if candidate.is_file():
            result[candidate.name] = (
                candidate.stat().st_size,
                hashlib.sha256(candidate.read_bytes()).hexdigest(),
            )
    return result


def _web_headers(snapshot: Any, plugin_id: str) -> tuple[Any, dict[str, str]]:
    catalog = getattr(snapshot, "web_ui_catalog")
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
    module: Any,
    payload: dict[str, object],
    marker: str,
) -> None:
    """Render the exact catalog asset in jsdom and reject missing fields."""
    module_file = tmp_path / f"{getattr(module, 'plugin_id')}-web_module.js"
    payload_file = tmp_path / f"{getattr(module, 'plugin_id')}-detail.json"
    module_file.write_text(getattr(module, "asset").module, encoding="utf-8")
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


def _seed_saved_recall(log: Any, _host: Any) -> None:
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
        before = _file_snapshot(tmp_path / "workspace" / "memory" / "akasha.db")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=headers,
        ) as client:
            overview = await client.get("/api/dashboard/akasha-inspector/overview")
            listing = await client.get("/api/dashboard/akasha-inspector/turns?page=1&page_size=25")
            detail_response = await client.get("/api/dashboard/akasha-inspector/turns/saved-query")
        after = _file_snapshot(tmp_path / "workspace" / "memory" / "akasha.db")

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
        assert before == after
        assert not (tmp_path / "embedding-calls.txt").exists()
        _render_compiled_module(tmp_path, module, detail, "命中回忆")


@pytest.mark.asyncio
async def test_wake_dashboard_matches_target_delivery_and_compiled_module(tmp_path: Path) -> None:
    from tests.test_wake_messages import application, request

    async with application(tmp_path) as (host, log, ctx, _source, control):
        await host.start_runtime()
        runtime = control["runtime"]
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
        # A read-only SQLite connection may create WAL coordination sidecars;
        # the Wake-owned database and its rows must remain byte-for-byte stable.
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
        runtime = control["runtime"]
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
