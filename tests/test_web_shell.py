from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest
import uvicorn
from fastapi.testclient import TestClient

from agent.plugin_composition import (
    CapabilitySources,
    ConnectionDescriptor,
    ModelAvailability,
    ModelCapabilities,
    ModelCatalogSnapshot,
    ModelDescriptor,
    ModelKind,
    ModelRole,
    SetDefaultModel,
    SettingsReceipt,
)
import bootstrap.settings_api as settings_api
import bootstrap.web_shell as web_shell
from bootstrap.chat_api import create_chat_app
from bootstrap.web_runtime import chat_socket_path, prepare_runtime_socket
from bootstrap.web_shell import create_web_shell_app
from infra.channels.web_chat_channel import WebChatChannel


def test_web_shell_serves_dashboard_shell_with_embedded_chat_without_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    dashboard_static = project_root / "static" / "dashboard"
    chat_static = project_root / "static" / "chat"
    dashboard_static.mkdir(parents=True)
    chat_static.mkdir(parents=True)
    (dashboard_static / "index.html").write_text(
        "<title>Akashic Dashboard</title>", encoding="utf-8"
    )
    (chat_static / "index.html").write_text(
        "<title>Akashic Chat</title>", encoding="utf-8"
    )
    module_path = project_root / "bootstrap" / "module.py"
    monkeypatch.setattr(web_shell, "__file__", str(module_path))
    monkeypatch.setattr(settings_api, "__file__", str(module_path))

    app = create_web_shell_app(tmp_path / "config.toml", tmp_path / "workspace")

    with TestClient(app) as client:
        state = client.get("/api/shell/state")
        shell = client.get("/")
        chat = client.get("/chat")
        legacy_dashboard = client.get("/dashboard")
        settings = client.get("/settings")
        unavailable = client.get("/api/chat/sessions")
        hidden = client.post("/api/chat/model-settings/command", json={})
        rejected = client.post("/api/settings/model/command", json={})
        model_unavailable = client.post(
            "/api/settings/model/command",
            headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
            json={},
        )
        retired = client.post(
            "/api/settings/roles",
            headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
            json={},
        )
        retired_login = client.get("/api/settings/codex-login/attempt-a")

    assert state.json() == {
        "status": "needs_setup",
        "configured": False,
        "chatReady": False,
        "settingsPath": "/settings",
    }
    assert shell.status_code == 200
    assert "Akashic Dashboard" in shell.text
    assert chat.status_code == 200
    assert "Akashic Chat" in chat.text
    assert legacy_dashboard.status_code == 200
    assert settings.status_code == 200
    assert unavailable.status_code == 503
    assert unavailable.json()["code"] == "gateway_unavailable"
    assert hidden.status_code == 404
    assert rejected.status_code == 403
    assert model_unavailable.status_code == 503
    assert retired.status_code == 410
    assert retired.json()["code"] == "model_settings_moved"
    assert retired_login.status_code == 410


def test_prepare_runtime_socket_replaces_only_socket_nodes(tmp_path: Path) -> None:
    path = chat_socket_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.close()

    assert prepare_runtime_socket(path) == str(path)
    assert not path.exists()

    path.write_text("not a socket", encoding="utf-8")
    with pytest.raises(RuntimeError, match="非 socket"):
        prepare_runtime_socket(path)


def test_runtime_socket_path_stays_short_for_deep_workspace(tmp_path: Path) -> None:
    workspace = tmp_path.joinpath(*(["deep-workspace"] * 8))
    path = chat_socket_path(workspace)

    assert len(str(path).encode("utf-8")) < 100
    assert path.parent.resolve() == (workspace / "runtime").resolve()

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.close()
    assert (workspace / "runtime" / "web-chat.sock").is_socket()


def test_model_control_crosses_public_shell_and_real_chat_socket(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    socket_path = chat_socket_path(workspace)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    applied: list[object] = []
    catalog = ModelCatalogSnapshot(
        revision=4,
        connections=(
            ConnectionDescriptor(
                connection_id="account-a",
                name="Account A",
                driver_id="openai-compatible",
                auth_identity="account-a",
                availability=ModelAvailability.AVAILABLE,
            ),
        ),
        models=(
            ModelDescriptor(
                model_id="embedding-unavailable",
                connection_id="account-a",
                kind=ModelKind.EMBEDDING,
                model="wire-embedding",
                default_reasoning_effort=None,
                capabilities=ModelCapabilities(embedding_dimensions=3),
                capability_sources=CapabilitySources(embedding_dimensions="configured"),
                availability=ModelAvailability.DISABLED,
            ),
        ),
        role_bindings={},
        default_embedding_model_id=None,
    )

    class Control:
        async def catalog(self) -> ModelCatalogSnapshot:
            return catalog

        async def apply(self, command: object) -> SettingsReceipt:
            applied.append(command)
            return SettingsReceipt(revision=5, status="committed")

    chat_app = create_chat_app(
        workspace=workspace,
        channel=WebChatChannel(),
        model_control=cast(Any, Control()),
    )
    server = uvicorn.Server(
        uvicorn.Config(
            chat_app,
            uds=str(socket_path),
            log_level="critical",
            access_log=False,
            ws="none",
        )
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()),
        name="test-chat-uds",
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5
    while not socket_path.is_socket() and thread.is_alive():
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert socket_path.is_socket()

    try:
        shell = create_web_shell_app(tmp_path / "config.toml", workspace)
        with TestClient(shell) as client:
            projected = client.get("/api/settings/model/catalog")
            rejected_memory = client.post(
                "/api/settings/memory",
                headers={
                    "Origin": "http://testserver",
                    "X-Akasic-CSRF": "1",
                },
                json={
                    "enabled": True,
                    "embedding_model_id": "embedding-unavailable",
                },
            )
            changed = client.post(
                "/api/settings/model/command",
                headers={
                    "Origin": "http://testserver",
                    "X-Akasic-CSRF": "1",
                },
                json={
                    "type": "set_default",
                    "expected_revision": 4,
                    "role": "default",
                    "model_id": "chat-a",
                },
            )

        assert projected.status_code == 200
        assert projected.json()["revision"] == 4
        assert "secret" not in projected.text
        assert "hidden" not in projected.text
        assert projected.headers["cache-control"] == "no-store"
        assert rejected_memory.status_code == 404
        assert not (tmp_path / "config.toml").exists()
        assert changed.status_code == 200
        assert changed.json()["revision"] == 5
        assert applied == [SetDefaultModel(4, ModelRole.DEFAULT, "chat-a")]
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    assert not thread.is_alive()
