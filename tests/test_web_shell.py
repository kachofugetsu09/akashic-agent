from __future__ import annotations

import socket
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import bootstrap.settings_api as settings_api
import bootstrap.web_shell as web_shell
from bootstrap.web_runtime import chat_socket_path, prepare_runtime_socket
from bootstrap.web_shell import create_web_shell_app


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
