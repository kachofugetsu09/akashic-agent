from __future__ import annotations

import hashlib
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from bootstrap.settings_api import create_settings_app


_HEADERS = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}


def _config(*, enabled: bool = False, model_id: str = "") -> str:
    model_line = f'model_ref = "{model_id}"\n' if model_id else ""
    return f'''[runtime]
workspace = "workspace"

[memory]
enabled = {str(enabled).lower()}

[memory.embedding]
{model_line}'''


def _revision(config_path: Path) -> str:
    payload = config_path.read_bytes() if config_path.exists() else b""
    return hashlib.sha256(payload).hexdigest()


def test_memory_state_exposes_only_memory_owner_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(enabled=True, model_id="embedding-a"), encoding="utf-8")
    client = TestClient(create_settings_app(config_path, tmp_path / "workspace"))

    response = client.get("/api/settings/memory-state")

    assert response.status_code == 200
    assert response.json() == {
        "configured": True,
        "enabled": True,
        "embeddingModelId": "embedding-a",
        "changeLocked": False,
        "revision": _revision(config_path),
    }
    assert "models" not in response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_enabling_memory_requires_available_embedding(tmp_path: Path) -> None:
    seen: list[str] = []

    async def unavailable(model_id: str) -> bool:
        seen.append(model_id)
        return False

    client = TestClient(
        create_settings_app(
            tmp_path / "config.toml",
            tmp_path / "workspace",
            embedding_model_exists=unavailable,
        )
    )
    response = client.post(
        "/api/settings/memory",
        headers=_HEADERS,
        json={"enabled": True, "embedding_model_id": "missing", "expected_revision": ""},
    )

    assert response.status_code == 422
    assert seen == ["missing"]
    assert not (tmp_path / "config.toml").exists()


def test_enabling_memory_reports_unavailable_catalog(tmp_path: Path) -> None:
    client = TestClient(create_settings_app(tmp_path / "config.toml", tmp_path / "workspace"))

    response = client.post(
        "/api/settings/memory",
        headers=_HEADERS,
        json={"enabled": True, "embedding_model_id": "embedding-a", "expected_revision": ""},
    )

    assert response.status_code == 503


def test_memory_save_backs_up_and_publishes_atomically(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    original = _config()
    config_path.write_text(original, encoding="utf-8")
    applied: list[str] = []

    async def available(model_id: str) -> bool:
        assert model_id == "embedding-a"
        return True

    client = TestClient(
        create_settings_app(
            config_path,
            workspace,
            embedding_model_exists=available,
            on_applied=lambda: applied.append("applied"),
        )
    )
    response = client.post(
        "/api/settings/memory",
        headers=_HEADERS,
        json={
            "enabled": True,
            "embedding_model_id": "embedding-a",
            "expected_revision": _revision(config_path),
        },
    )

    assert response.status_code == 200, response.text
    operation_id = response.json()["operationId"]
    document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert document["memory"] == {
        "enabled": True,
        "embedding": {"model_ref": "embedding-a"},
    }
    backup = workspace / "backups" / "memory-settings" / operation_id / "config.before"
    assert backup.read_text(encoding="utf-8") == original
    assert backup.stat().st_mode & 0o777 == 0o600
    assert applied == ["applied"]


def test_memory_save_rejects_csrf_and_stale_revision(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    client = TestClient(create_settings_app(config_path, tmp_path / "workspace"))

    csrf = client.post("/api/settings/memory", json={"enabled": False})
    stale = client.post(
        "/api/settings/memory",
        headers=_HEADERS,
        json={"enabled": False, "embedding_model_id": "", "expected_revision": "stale"},
    )

    assert csrf.status_code == 403
    assert stale.status_code == 409


def test_existing_messages_lock_memory_change(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    config_path.write_text(_config(), encoding="utf-8")
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO messages VALUES ('message-a')")
        connection.commit()

    async def available(_model_id: str) -> bool:
        return True

    client = TestClient(
        create_settings_app(
            config_path,
            workspace,
            embedding_model_exists=available,
        )
    )
    state = client.get("/api/settings/memory-state")
    response = client.post(
        "/api/settings/memory",
        headers=_HEADERS,
        json={
            "enabled": True,
            "embedding_model_id": "embedding-a",
            "expected_revision": state.json()["revision"],
        },
    )

    assert state.json()["changeLocked"] is True
    assert response.status_code == 409
