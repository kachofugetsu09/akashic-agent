from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
import sqlite3
import shutil
import threading
import tomllib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient as _RawTestClient

import bootstrap.dashboard_api as dashboard_api
from agent.plugins.artifacts import ArtifactPointer, write_pointers
from agent.plugins.dashboard_host import (
    PluginDashboardHost,
    SnapshotDashboardMiddleware,
)
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from bootstrap.dashboard_api import (
    _dashboard_plugin_dirs,
    create_dashboard_app as _create_dashboard_app,
)
from plugins.akasha.engine import AkashaMemoryEngine
from session.embedding_store import MessageEmbeddingStore
from session.store import SessionStore
from agent.model_runtime.context_compaction import source_plan_digest


class _TrackedTestClient(_RawTestClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._is_closed = False

    def close(self) -> None:
        if self._is_closed:
            return
        self._is_closed = True
        super().close()

    def __del__(self) -> None:
        if not self._is_closed:
            try:
                self.close()
            except Exception:
                pass


TestClient = _TrackedTestClient


@pytest.fixture(autouse=True)
def _isolate_dashboard_plugin_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """让 Dashboard 测试只观察各自声明的 HOME/manifest。"""

    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))


def _use_writable_dashboard_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin_ids: set[str],
) -> None:
    """复制指定插件源码，并为只读 Gate 合成最小测试面板产物。"""

    discovered = _dashboard_plugin_dirs(Path.cwd())
    missing = plugin_ids - discovered.keys()
    assert not missing
    writable: dict[str, Path] = {}
    for plugin_id in sorted(plugin_ids):
        target = tmp_path / "dashboard-plugins" / plugin_id
        shutil.copytree(discovered[plugin_id], target)
        for source in target.glob("dashboard_panel*.ts*"):
            source.with_suffix(".js").write_text(
                "export default {};\n",
                encoding="utf-8",
            )
        writable[plugin_id] = target
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _project_root: dict(writable),
    )


class _AkashaDashboardMemoryAdmin:
    def describe(self):
        return AkashaMemoryEngine.DESCRIPTOR

    def close(self) -> None:
        return None


def create_dashboard_app(tmp_path, **kwargs):
    return _create_dashboard_app(tmp_path, **kwargs)


def test_retired_proactive_dashboard_routes_do_not_open_legacy_database(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "proactive.db"
    legacy_bytes = b"legacy proactive database, not sqlite\x00"
    legacy.write_bytes(legacy_bytes)
    before = (legacy.stat().st_ino, hashlib.sha256(legacy_bytes).hexdigest())

    with TestClient(create_dashboard_app(tmp_path)) as client:
        for route in (
            "/api/dashboard/proactive/overview",
            "/api/dashboard/proactive/tick_logs",
            "/api/dashboard/proactive/tick_logs/legacy/steps",
        ):
            assert client.get(route).status_code == 404

    assert (
        legacy.stat().st_ino,
        hashlib.sha256(legacy.read_bytes()).hexdigest(),
    ) == before


def _seed_explicit_interaction(
    workspace: Path,
    *,
    last_consolidated: int,
) -> tuple[str, list[str]]:
    store = SessionStore(workspace / "sessions.db")
    timestamp = "2026-08-07T10:00:00+08:00"
    rows = store.persist_session(
        "mobile:review",
        created_at=timestamp,
        updated_at=timestamp,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "legacy",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "assistant",
                "content": "old",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "user",
                "content": "u1",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_input_ordinal": 0,
                },
            },
            {
                "role": "user",
                "content": "u2",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_input_ordinal": 1,
                },
            },
            {
                "role": "user",
                "content": "u3",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_input_ordinal": 2,
                },
            },
            {
                "role": "assistant",
                "content": "final",
                "timestamp": timestamp,
                "extra": {
                    "control_turn_id": "turn-review",
                    "turn_terminal": True,
                    "turn_input_count": 3,
                },
            },
            {
                "role": "user",
                "content": "later",
                "timestamp": timestamp,
                "extra": {},
            },
            {
                "role": "assistant",
                "content": "later-a",
                "timestamp": timestamp,
                "extra": {},
            },
        ],
    )
    if last_consolidated:
        legacy = rows[0]
        store.persist_compaction(
            session_key="mobile:review",
            trigger="test",
            summary="legacy checkpoint",
            source_ref="test:legacy",
            source_plan_digest=source_plan_digest(
                (
                    {
                        "id": str(legacy["id"]),
                        "seq": int(legacy["seq"]),
                        "unit_ref": "test:legacy",
                        "message": dict(legacy),
                    },
                )
            ),
            source_from_seq=int(legacy["seq"]),
            consolidated_through_seq=int(legacy["seq"]),
            source_message_ids=[str(legacy["id"])],
            retained_tail=[],
            model_runtime_id="test",
            model="test",
            context_window=100,
            threshold_tokens=80,
            hard_input_tokens=90,
            keep_recent_tokens=10,
            tokens_before=10,
            tokens_after=5,
            summary_usage={},
            generation=2,
            parent_generation=0,
        )
    if last_consolidated > 2:
        interaction = rows[2]
        store.persist_compaction(
            session_key="mobile:review",
            trigger="test",
            summary="interaction checkpoint",
            source_ref="test:interaction",
            source_plan_digest=source_plan_digest(
                (
                    {
                        "id": str(interaction["id"]),
                        "seq": int(interaction["seq"]),
                        "unit_ref": "test:interaction",
                        "message": dict(interaction),
                    },
                )
            ),
            source_from_seq=int(interaction["seq"]),
            consolidated_through_seq=int(interaction["seq"]),
            source_message_ids=[str(interaction["id"])],
            retained_tail=[],
            model_runtime_id="test",
            model="test",
            context_window=100,
            threshold_tokens=80,
            hard_input_tokens=90,
            keep_recent_tokens=10,
            tokens_before=10,
            tokens_after=5,
            summary_usage={},
            generation=last_consolidated,
            parent_generation=2,
        )
    store.close()
    return "turn-review", [str(row["id"]) for row in rows[2:6]]


@pytest.mark.asyncio
async def test_dashboard_lifespan_swallows_its_own_compile_cancellation(
    tmp_path, monkeypatch
):
    async def _pending_compile(
        _queue: dashboard_api._PluginPanelBuildQueue,
    ) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(
        dashboard_api,
        "_compile_pending_plugins_async",
        _pending_compile,
    )
    app = create_dashboard_app(tmp_path)

    async with app.router.lifespan_context(app):
        pass


@pytest.mark.asyncio
async def test_dashboard_waits_for_async_closeable() -> None:
    closed = False

    class Closeable:
        async def close(self) -> None:
            nonlocal closed
            closed = True

    await dashboard_api._close_dashboard_value(Closeable())

    assert closed


@pytest.mark.asyncio
async def test_dashboard_lifespan_exposes_unexpected_compile_failure(
    tmp_path, monkeypatch
):
    async def _failed_compile(
        _queue: dashboard_api._PluginPanelBuildQueue,
    ) -> None:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(
        dashboard_api,
        "_compile_pending_plugins_async",
        _failed_compile,
    )
    app = create_dashboard_app(tmp_path)

    with pytest.raises(RuntimeError, match="compile failed"):
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_pending_panel_probe_is_terminated_and_drained_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = dashboard_api._PluginPanelBuildQueue()
    plugin_dir = tmp_path / "plugin"
    output_dir = tmp_path / "output"
    plugin_dir.mkdir()
    queue.add(tmp_path, plugin_dir, output_dir)
    communicate_started = asyncio.Event()
    wait_started = asyncio.Event()
    release_wait = asyncio.Event()

    class Process:
        returncode: int | None = None
        terminated = False
        waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            communicate_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            raise AssertionError("graceful termination should finish")

        async def wait(self) -> int:
            self.waited = True
            wait_started.set()
            await release_wait.wait()
            self.returncode = -15
            return self.returncode

    process = Process()

    async def create_process(*_args: object, **_kwargs: object) -> Process:
        return process

    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["npx"])
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    task = asyncio.create_task(dashboard_api._compile_pending_plugins_async(queue))
    await communicate_started.wait()
    task.cancel()
    await wait_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release_wait.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.terminated is True
    assert process.waited is True


def _seed_workspace(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(
        key="telegram:100",
        metadata={"title": "alpha room"},
        last_user_at="2026-04-19T10:00:00+08:00",
    )
    store.create_session(
        key="cli:local",
        metadata={"title": "beta room"},
        last_proactive_at="2026-04-19T09:00:00+08:00",
    )
    store.insert_message(
        "telegram:100",
        role="user",
        content="你好，今晚睡觉了吗",
        ts="2026-04-19T10:01:00+08:00",
        seq=0,
        extra={"pinned": True},
    )
    store.insert_message(
        "telegram:100",
        role="assistant",
        content="还没睡呢",
        ts="2026-04-19T10:02:00+08:00",
        seq=1,
        tool_chain=[{"text": "reply", "calls": []}],
        extra={"source": "test"},
    )
    store.insert_message(
        "cli:local",
        role="user",
        content="hello from cli",
        ts="2026-04-19T09:01:00+08:00",
        seq=0,
    )
    store.close()


def _seed_pending_compaction_prepare(
    workspace: Path,
    session_key: str,
    message_ids: list[str],
) -> str:
    store = SessionStore(workspace / "sessions.db")
    meta = store.get_session_meta(session_key)
    assert meta is not None
    rows = {str(row["id"]): row for row in store.fetch_session_messages(session_key)}
    selected = [rows[message_id] for message_id in message_ids]
    assert selected
    source_ref = f"test:pending:{session_key}"
    store.prepare_compaction(
        session_key=session_key,
        session_created_at=str(meta["created_at"]),
        generation=1,
        parent_generation=0,
        source_ref=source_ref,
        source_from_seq=min(int(row["seq"]) for row in selected),
        consolidated_through_seq=max(int(row["seq"]) for row in selected),
        source_message_ids=tuple(message_ids),
        retained_tail=(),
    )
    store.close()
    return source_ref


def test_list_sessions_with_filters(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        resp = client.get(
            "/api/dashboard/sessions",
            params={"q": "alpha", "channel": "telegram"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] == 1
        assert payload["items"][0]["key"] == "telegram:100"
        assert payload["items"][0]["message_count"] == 2

        messages_resp = client.get(
            "/api/dashboard/messages",
            params={"sort_by": "seq", "sort_order": "asc"},
        )
        assert messages_resp.status_code == 200
        assert messages_resp.json()["items"][0]["seq"] == 0


def test_update_and_delete_session(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        patch_resp = client.patch(
            "/api/dashboard/sessions/telegram:100",
            json={"metadata": {"title": "patched"}},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["metadata"]["title"] == "patched"

        retired_patch = client.patch(
            "/api/dashboard/sessions/telegram:100",
            json={"last_consolidated": 9},
        )
        assert retired_patch.status_code == 422
        assert (
            client.get("/api/dashboard/sessions/telegram:100").json()[
                "last_consolidated"
            ]
            == 0
        )

        delete_resp = client.delete("/api/dashboard/sessions/telegram:100")
        assert delete_resp.status_code == 200
        delete_payload = delete_resp.json()
        assert delete_payload["action_source"] == "dashboard.session_delete"
        assert delete_payload["result"] == "committed"
        assert delete_payload["audit_id"]
        assert Path(delete_payload["backup_path"]).is_file()

        get_resp = client.get("/api/dashboard/sessions/telegram:100")
        assert get_resp.status_code == 404


def test_dashboard_update_message_returns_409_when_session_is_active(tmp_path) -> None:
    _seed_workspace(tmp_path)
    runtime_store = SessionStore(tmp_path / "sessions.db")
    message = runtime_store.get_message("telegram:100:1")
    assert message is not None
    assert runtime_store.acquire_session_admission(
        "telegram:100",
        "admission:dashboard-edit",
    )
    runtime_store.close()

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.patch(
            f"/api/dashboard/messages/{message['id']}",
            json={"content": "blocked"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "session_busy",
        "session_key": "telegram:100",
    }
    inspector = SessionStore(tmp_path / "sessions.db")
    assert inspector.get_message(str(message["id"]))["content"] == "还没睡呢"
    inspector.release_session_admission("admission:dashboard-edit")
    inspector.close()


@pytest.mark.parametrize("operation", ("edit", "delete", "batch"))
def test_dashboard_returns_distinct_409_for_pending_compaction_prepare(
    tmp_path,
    operation: str,
) -> None:
    _seed_workspace(tmp_path)
    session_key = "telegram:100"
    target_id = "telegram:100:1"
    message_ids = [target_id]
    source_ref = _seed_pending_compaction_prepare(
        tmp_path,
        session_key,
        message_ids,
    )

    with TestClient(create_dashboard_app(tmp_path)) as client:
        if operation == "edit":
            response = client.patch(
                f"/api/dashboard/messages/{target_id}",
                json={"content": "blocked"},
            )
        elif operation == "delete":
            response = client.delete(f"/api/dashboard/messages/{target_id}")
        elif operation == "batch":
            response = client.post(
                "/api/dashboard/messages/batch-delete",
                json={"ids": message_ids},
            )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "session_compaction_pending",
        "session_key": session_key,
        "source_ref": source_ref,
    }


@pytest.mark.parametrize("operation", ("single", "batch"))
def test_dashboard_rejects_session_delete_with_pending_prepare(
    tmp_path,
    operation: str,
) -> None:
    _seed_workspace(tmp_path)
    session_key = "telegram:100"
    message_id = "telegram:100:1"
    source_ref = _seed_pending_compaction_prepare(
        tmp_path,
        session_key,
        [message_id],
    )

    with TestClient(create_dashboard_app(tmp_path)) as client:
        if operation == "single":
            response = client.delete(f"/api/dashboard/sessions/{session_key}")
        else:
            response = client.post(
                "/api/dashboard/sessions/batch-delete",
                json={"keys": [session_key], "cascade": True},
            )

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "session_compaction_pending"
    assert detail["session_key"] == session_key
    assert detail["source_ref"] == source_ref
    assert detail["audit_id"]

    inspector = SessionStore(tmp_path / "sessions.db")
    assert inspector.session_exists(session_key)
    assert inspector.get_message(message_id) is not None
    prepare = inspector.get_compaction_prepare(session_key, source_ref=source_ref)
    assert prepare is not None
    audit = inspector.get_session_delete_audit(detail["audit_id"])
    assert audit is not None
    assert audit.result == "rejected"
    assert audit.backup_path is None
    inspector.close()
    assert not list((tmp_path / "backups" / "session-deletions").glob("sessions-*.db"))


def test_list_update_and_batch_delete_messages(tmp_path) -> None:
    _seed_workspace(tmp_path)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        list_resp = client.get(
            "/api/dashboard/sessions/telegram:100/messages",
            params={"q": "睡", "role": "assistant"},
        )
        assert list_resp.status_code == 200
        payload = list_resp.json()
        assert payload["total"] == 1
        message_id = payload["items"][0]["id"]
        embedding_store = MessageEmbeddingStore(tmp_path / "sessions.db")
        embedding_store.upsert(
            message_id=message_id,
            content="还没睡呢",
            model="m",
            embedding=[1.0, 0.0],
        )
        embedding_store.close()

        patch_resp = client.patch(
            f"/api/dashboard/messages/{message_id}",
            json={"content": "已经睡了", "extra": {"edited": True}},
        )
        assert patch_resp.status_code == 200
        assert patch_resp.json()["content"] == "已经睡了"
        assert patch_resp.json()["edited"] is True
        embedding_store = MessageEmbeddingStore(tmp_path / "sessions.db")
        assert (
            embedding_store.get(
                message_id=message_id,
                content="还没睡呢",
                model="m",
            )
            is None
        )
        embedding_store.upsert(
            message_id=message_id,
            content="已经睡了",
            model="m",
            embedding=[1.0, 0.0],
        )
        embedding_store.upsert(
            message_id="cli:local:0",
            content="hello from cli",
            model="m",
            embedding=[0.0, 1.0],
        )
        embedding_store.close()

        batch_resp = client.post(
            "/api/dashboard/messages/batch-delete",
            json={"ids": [message_id, "cli:local:0"]},
        )
        assert batch_resp.status_code == 200
        assert batch_resp.json()["deleted_count"] == 2
        with closing(sqlite3.connect(tmp_path / "sessions.db")) as db:
            embedding_count = db.execute(
                """
                SELECT COUNT(*)
                FROM message_embeddings
                WHERE message_id IN (?, ?)
                """,
                (message_id, "cli:local:0"),
            ).fetchone()[0]
        assert embedding_count == 0

        remain_resp = client.get(
            "/api/dashboard/messages", params={"session_key": "telegram:100"}
        )
        assert remain_resp.status_code == 200
        assert remain_resp.json()["total"] == 1


def test_explicit_interaction_rejects_generic_message_deletes(tmp_path) -> None:
    turn_id, message_ids = _seed_explicit_interaction(
        tmp_path,
        last_consolidated=6,
    )
    with TestClient(create_dashboard_app(tmp_path)) as client:
        single = client.delete(f"/api/dashboard/messages/{message_ids[1]}")
        batch = client.post(
            "/api/dashboard/messages/batch-delete",
            json={"ids": message_ids},
        )

    assert single.status_code == 409
    assert single.json()["detail"] == {
        "code": "interaction_delete_required",
        "message_id": message_ids[1],
        "control_turn_id": turn_id,
    }
    assert batch.status_code == 409
    assert batch.json()["detail"]["control_turn_id"] == turn_id
    store = SessionStore(tmp_path / "sessions.db")
    assert all(store.get_message(message_id) is not None for message_id in message_ids)
    store.close()


def test_core_dashboard_has_no_privileged_memory_routes(tmp_path) -> None:
    with TestClient(create_dashboard_app(tmp_path)) as client:
        assert client.get("/api/dashboard/memory/engine-info").status_code == 404
        assert client.get("/api/dashboard/memories").status_code == 404
        assert client.delete("/api/dashboard/interactions/turn:1").status_code == 404


def test_dashboard_lists_installed_plugin_panels(tmp_path, monkeypatch) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin" / "cache" / "github" / "status_commands"
    plugin_dir = plugin_base / ".artifacts" / "1.0.0-test"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "dashboard.py").write_text(
        "from fastapi import FastAPI\n"
        "def register(app: FastAPI, context):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.js").write_text(
        "export default {};\n", encoding="utf-8"
    )
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'status_commands'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'status_commands'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '[plugins."status_commands@github"]\nenabled = true\n', encoding="utf-8"
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))

    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
    installed = next(item for item in plugins if item["id"] == "status_commands@github")
    assert installed == {
        "id": "status_commands@github",
        "panels": [
            {
                "name": "dashboard_panel",
                "js_version": str(
                    (plugin_dir / "dashboard_panel.js").stat().st_mtime_ns
                ),
                "has_css": False,
            }
        ],
    }


def test_installed_typescript_panel_uses_runtime_cache_without_source_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin/cache/github/read_only_panel"
    plugin_dir = plugin_base / ".artifacts/1.0.0-test"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'read_only_panel'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'read_only_panel'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    (plugin_dir / "db.py").write_text(
        "def ping(): return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context):\n"
        "    from .db import ping\n"
        "    assert ping() == 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.tsx").write_text(
        "export default function Panel() { return null }\n",
        encoding="utf-8",
    )
    manifest = home / ".akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '[plugins."read_only_panel@github"]\nenabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    compiled_outputs: list[Path] = []
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"read_only_panel@github": plugin_dir},
    )

    def compile_to_cache(
        _cmd: list[str],
        _source: Path,
        output: Path,
        _name: str,
    ) -> None:
        compiled_outputs.append(output)
        output.write_text("export default {};\n", encoding="utf-8")

    monkeypatch.setattr(dashboard_api, "_run_esbuild", compile_to_cache)
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["esbuild"])
    source_before = _test_tree_digest(plugin_dir)
    cache_root = tmp_path / "runtime/dashboard-panels"

    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        installed = next(
            item for item in plugins if item["id"] == "read_only_panel@github"
        )
        assert len(installed["panels"]) == 1
        assert installed["panels"][0]["name"] == "dashboard_panel"
        assert installed["panels"][0]["has_css"] is False
        assert str(installed["panels"][0]["js_version"]).isdigit()
        response = client.get("/plugins/read_only_panel@github/dashboard_panel.js")
        assert response.status_code == 200
        assert response.text == "export default {};\n"
        assert compiled_outputs
        assert all(output.is_relative_to(cache_root) for output in compiled_outputs)
        assert not (plugin_dir / "dashboard_panel.js").exists()
        assert _test_tree_digest(plugin_dir) == source_before

    assert not cache_root.exists()
    assert _test_tree_digest(plugin_dir) == source_before


def test_installed_fresh_javascript_panel_stays_in_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin/cache/github/published_panel"
    plugin_dir = plugin_base / ".artifacts/1.0.0-test"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'published_panel'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'published_panel'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, context): return None\n",
        encoding="utf-8",
    )
    source = plugin_dir / "dashboard_panel.ts"
    source.write_text("export default 1;\n", encoding="utf-8")
    published = plugin_dir / "dashboard_panel.js"
    published.write_text("export default 1;\n", encoding="utf-8")
    published.touch()
    manifest = home / ".akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '[plugins."published_panel@github"]\nenabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"published_panel@github": plugin_dir},
    )

    def unexpected_compile(*_args: object) -> None:
        raise AssertionError("fresh published JavaScript must not compile")

    monkeypatch.setattr(dashboard_api, "_run_esbuild", unexpected_compile)
    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.get("/plugins/published_panel@github/dashboard_panel.js")
        assert response.status_code == 200
        assert response.text == "export default 1;\n"
    assert not (tmp_path / "runtime/dashboard-panels").exists()


def test_panel_cache_identity_tracks_transitive_source_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.ts").write_text(
        "import value from './fragment'; export default value;\n",
        encoding="utf-8",
    )
    fragment = plugin_dir / "fragment.ts"
    fragment.write_text("export default 'first';\n", encoding="utf-8")
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    outputs: list[Path] = []

    def compile_fragment(
        _cmd: list[str],
        _source: Path,
        output: Path,
        _name: str,
    ) -> None:
        outputs.append(output)
        output.write_text(fragment.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(dashboard_api, "_run_esbuild", compile_fragment)
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["esbuild"])

    with TestClient(create_dashboard_app(tmp_path)) as client:
        first = client.get("/plugins/panel/dashboard_panel.js")
        runtime_metadata = plugin_dir / ".venv/metadata.json"
        runtime_metadata.parent.mkdir()
        runtime_metadata.write_text('{"runtime": 1}\n', encoding="utf-8")
        unrelated = client.get("/plugins/panel/dashboard_panel.js")
        fragment.write_text("export default 'second';\n", encoding="utf-8")
        second = client.get("/plugins/panel/dashboard_panel.js")

    assert first.text == "export default 'first';\n"
    assert unrelated.text == "export default 'first';\n"
    assert second.text == "export default 'second';\n"
    assert len(outputs) == 2
    assert outputs[0].parent != outputs[1].parent


def test_stale_published_panel_is_unavailable_when_compile_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_workspace(tmp_path)
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    published = plugin_dir / "dashboard_panel.js"
    published.write_text("export default 'stale';\n", encoding="utf-8")
    source = plugin_dir / "dashboard_panel.ts"
    source.write_text("export default 'new';\n", encoding="utf-8")
    published.touch()
    source.touch()
    source_mtime = published.stat().st_mtime + 1
    os.utime(source, (source_mtime, source_mtime))
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: ["esbuild"])
    monkeypatch.setattr(dashboard_api, "_run_esbuild", lambda *_args: None)

    with TestClient(create_dashboard_app(tmp_path)) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        response = client.get("/plugins/panel/dashboard_panel.js")

    assert plugins == []
    assert response.status_code == 404
    assert not list((tmp_path / "runtime/dashboard-panels").rglob("*.js"))


def test_dashboard_rejects_symlink_panel_cache(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "runtime/dashboard-panels"
    cache.parent.mkdir(parents=True)
    cache.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="不能穿过符号链接"):
        create_dashboard_app(tmp_path)


def test_dashboard_rejects_symlink_runtime_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    runtime = tmp_path / "runtime"
    runtime.symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="不能穿过符号链接"):
        create_dashboard_app(tmp_path)


@pytest.mark.asyncio
async def test_dashboard_pending_panel_builds_are_app_lifecycle_owned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "dashboard.py").write_text(
        "def register(app, plugin_dir, workspace): return None\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard_panel.ts").write_text(
        "export default 1;\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    monkeypatch.setattr(dashboard_api, "_esbuild_command", lambda _root: None)

    started = [asyncio.Event(), asyncio.Event()]
    release_first = asyncio.Event()
    first_compiled = asyncio.Event()
    call_count = 0

    async def compile_owned_queue(
        queue: dashboard_api._PluginPanelBuildQueue,
    ) -> None:
        nonlocal call_count
        call_index = call_count
        call_count += 1
        started[call_index].set()
        if call_index == 1:
            await asyncio.Event().wait()
            return
        await release_first.wait()
        for _root, _plugin, output_dir in queue.take_all():
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "dashboard_panel.js").write_text(
                "export default 1;\n",
                encoding="utf-8",
            )
        first_compiled.set()

    monkeypatch.setattr(
        dashboard_api,
        "_compile_pending_plugins_async",
        compile_owned_queue,
    )
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    app_a = create_dashboard_app(workspace_a)
    app_b = create_dashboard_app(workspace_b)

    async with app_a.router.lifespan_context(app_a):
        await started[0].wait()
        async with app_b.router.lifespan_context(app_b):
            await started[1].wait()
        release_first.set()
        await first_compiled.wait()
        assert list((workspace_a / "runtime/dashboard-panels").rglob("*.js"))
        assert not (workspace_b / "runtime/dashboard-panels").exists()


def _test_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_standalone_dashboard_honors_builtin_plugin_manifest(
    tmp_path, monkeypatch
) -> None:
    home = tmp_path / "home"
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        "[plugins.akasha]\nenabled = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))

    plugins = _dashboard_plugin_dirs(Path.cwd())

    assert "akasha" not in plugins


def test_standalone_dashboard_rejects_invalid_manifest(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("invalid = [\n", encoding="utf-8")
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))

    with pytest.raises(tomllib.TOMLDecodeError):
        _dashboard_plugin_dirs(Path.cwd())


def test_standalone_dashboard_does_not_import_plugin_backend(
    tmp_path, monkeypatch
) -> None:
    _seed_workspace(tmp_path)
    home = tmp_path / "home"
    plugin_base = home / ".akashic-plugin" / "cache" / "github" / "observe"
    plugin_dir = plugin_base / ".artifacts" / "1.0.0-test"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "db.py").write_text(
        "def ping():\n" "    return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('backend-imported').write_text('yes')\n"
        "from .db import ping\n"
        "def register(app, context):\n"
        "    @app.get('/api/dashboard/test-relative-import')\n"
        "    def route():\n"
        "        return {'value': ping()}\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'observe'\n"
        "version = '1.0.0'\n"
        "dashboard_module = 'dashboard.py'\n"
        "async def apply(ctx, config): pass\n",
        encoding="utf-8",
    )
    (plugin_dir / "akashic.plugin.toml").write_text(
        "schema_version = 1\n"
        "name = 'observe'\n"
        "version = '1.0.0'\n"
        "api_version = 3\n"
        "entrypoint = 'plugin.py'\n",
        encoding="utf-8",
    )
    pointer = ArtifactPointer(".artifacts/1.0.0-test")
    _ = write_pointers(plugin_base, stable=pointer, latest=pointer)
    manifest_path = home / ".akashic-plugin" / "manifest.toml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        '[plugins."observe@github"]\nenabled = true\n', encoding="utf-8"
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home / ".akashic-plugin"))
    source_before = _test_tree_digest(plugin_dir)

    with TestClient(create_dashboard_app(tmp_path)) as client:
        response = client.get("/api/dashboard/test-relative-import")
    assert response.status_code == 404
    assert not (plugin_dir / "backend-imported").exists()
    assert _test_tree_digest(plugin_dir) == source_before


def test_two_standalone_dashboard_apps_do_not_import_plugin_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "db.py").write_text(
        "def ping(): return 'ok'\n",
        encoding="utf-8",
    )
    (plugin_dir / "dashboard.py").write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('backend-imported').write_text('yes')\n"
        "def register(app, context):\n"
        "    @app.get('/api/dashboard/deferred-relative-import')\n"
        "    def route():\n"
        "        from .db import ping\n"
        "        return {'value': ping()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dashboard_api,
        "_dashboard_plugin_dirs",
        lambda _root: {"panel": plugin_dir},
    )
    source_before = _test_tree_digest(plugin_dir)
    app_a = create_dashboard_app(tmp_path / "workspace-a")
    app_b = create_dashboard_app(tmp_path / "workspace-b")

    with TestClient(app_a) as client_a:
        with TestClient(app_b) as client_b:
            assert (
                client_b.get("/api/dashboard/deferred-relative-import").status_code
                == 404
            )
        assert (
            client_a.get("/api/dashboard/deferred-relative-import").status_code == 404
        )

    assert not (plugin_dir / "backend-imported").exists()
    assert _test_tree_digest(plugin_dir) == source_before


def test_plugin_asset_paths_reject_cross_platform_traversal(tmp_path) -> None:
    with TestClient(create_dashboard_app(tmp_path)) as client:
        for path in (
            "/plugins/..%5Csecret/dashboard_panel.js",
            "/plugins/C:%5Csecret/dashboard_panel.js",
            "/plugins/%5C%5Cserver%5Cshare/dashboard_panel.css",
        ):
            response = client.get(path)
            assert response.status_code == 400

        assert (
            client.get(
                "/plugins/akasha/dashboard_panel_inspector..%5Csecret.js"
            ).status_code
            == 400
        )

        assert client.get("/plugins/missing/dashboard_panel.js").status_code == 404


def test_akasha_only_exposes_read_only_inspector_panel(
    tmp_path,
    monkeypatch,
) -> None:
    _use_writable_dashboard_plugins(
        tmp_path,
        monkeypatch,
        {"akasha"},
    )
    with TestClient(
        create_dashboard_app(
            tmp_path,
        )
    ) as client:
        plugins = client.get("/api/dashboard/plugins").json()
        akasha = next(item for item in plugins if item["id"] == "akasha")
        assert [panel["name"] for panel in akasha["panels"]] == [
            "dashboard_panel_inspector"
        ]
        assert (
            client.get("/plugins/akasha/dashboard_panel_inspector.js").status_code
            == 200
        )
        assert (
            client.get("/plugins/akasha/dashboard_panel_inspector.css").status_code
            == 200
        )
        assert client.get("/api/dashboard/akasha/graph").status_code == 404
