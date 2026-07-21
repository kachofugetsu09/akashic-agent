from __future__ import annotations

import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from agent.model_runtime.auth.store import CredentialStore
from bootstrap.settings_api import create_settings_app
from bootstrap.settings_api import _new_config


def _config(secret: str = "saved-secret") -> str:
    return f'''\
[runtime]
workspace = "workspace"

[llm]
main = "deepseek_main"

[llm.runtimes.deepseek_main]
provider = "deepseek"
model = "deepseek-chat"
api_key = "{secret}"
base_url = "https://api.deepseek.com/v1"
context_window = 64000
effective_context_percent = 0.9
max_output_tokens = 8192
input_modalities = ["text"]

[agent.context]
memory_window = 40
'''


def test_state_never_returns_saved_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    store = CredentialStore(tmp_path / "auth" / "auth.json")
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=store,
    )

    response = TestClient(app).get("/api/settings/state")

    assert response.status_code == 200
    assert "saved-secret" not in response.text
    assert response.json()["runtimes"][0]["credential"] == {
        "id": "",
        "configured": True,
        "source": "inline",
    }


def test_state_marks_invalid_runtime_fields_for_repair(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _config().replace("context_window = 64000", 'context_window = "large"'),
        encoding="utf-8",
    )
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )

    response = TestClient(app).get("/api/settings/state")

    assert response.status_code == 200
    assert response.json()["mode"] == "needs_repair"
    assert response.json()["runtimes"] == []
    assert response.json()["error"] == "runtime deepseek_main 的 context_window 必须是整数"


def test_apply_writes_inline_key_and_preserves_other_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    store = CredentialStore(tmp_path / "auth" / "auth.json")

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=store,
    )
    response = TestClient(app).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "opencode-go",
            "model": "glm-5",
            "api_key": "new-secret",
            "base_url": "https://opencode.ai/zen/go/v1",
            "context_window": 128000,
            "max_output_tokens": 8192,
            "input_modalities": ["text"],
        },
    )

    assert response.status_code == 200, response.text
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["llm"]["main"] == "opencode_go_main"
    runtime = parsed["llm"]["runtimes"]["opencode_go_main"]
    assert runtime["api_key"] == "new-secret"
    assert parsed["llm"]["runtimes"]["deepseek_main"]["api_key"] == "saved-secret"
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert config_path.with_name(
        f"config.toml.{response.json()['operationId']}.bak"
    ).exists()


def test_mutation_rejects_cross_origin(tmp_path: Path) -> None:
    app = create_settings_app(
        tmp_path / "config.toml",
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )

    response = TestClient(app).post(
        "/api/settings/models",
        headers={"Origin": "https://attacker.invalid", "X-Akasic-CSRF": "1"},
        json={"provider": "codex"},
    )

    assert response.status_code == 403
    assert response.json()["code"] == "csrf_rejected"


def test_new_config_includes_web_chat_runtime_dependencies(tmp_path: Path) -> None:
    parsed = tomllib.loads(_new_config(tmp_path / "workspace"))

    assert parsed["channels"]["chat"] == {
        "enabled": True,
        "host": "127.0.0.1",
        "port": 6322,
        "channel_name": "web",
    }
    assert parsed["app_server"]["enabled"] is True


def test_failed_gateway_restart_restores_config_and_restarts_old_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    original = _config()
    config_path.write_text(original, encoding="utf-8")
    callbacks: list[str] = []

    async def validate(*_args, **_kwargs) -> None:
        return None

    def restart() -> None:
        callbacks.append("restart")
        if len(callbacks) == 1:
            raise RuntimeError("candidate failed readiness")

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
        on_applied=restart,
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "opencode-go",
            "model": "glm-5",
            "api_key": "candidate-secret",
            "base_url": "https://opencode.ai/zen/go/v1",
            "context_window": 128000,
            "max_output_tokens": 8192,
            "input_modalities": ["text"],
        },
    )

    assert response.status_code == 500
    assert config_path.read_text(encoding="utf-8") == original
    assert callbacks == ["restart", "restart"]
