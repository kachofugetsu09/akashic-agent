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


def test_settings_round_trip_preserves_explicit_legacy_output_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    legacy = _config().replace(
        "max_output_tokens = 8192\n",
        "",
    ).replace(
        "[agent.context]",
        "[agent]\nmax_tokens = 8192\n\n[agent.context]",
    )
    config_path.write_text(legacy, encoding="utf-8")

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )
    client = TestClient(app)
    state = client.get("/api/settings/state").json()

    assert state["runtimes"][0]["maxOutputTokens"] == 8192

    response = client.post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "",
            "base_url": "https://api.deepseek.com/v1",
            "context_window": 64000,
            "max_output_tokens": state["runtimes"][0]["maxOutputTokens"],
            "input_modalities": ["text"],
        },
    )

    assert response.status_code == 200, response.text
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["llm"]["runtimes"]["deepseek_main"]["max_output_tokens"] == 8192


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
            "max_output_tokens": 0,
            "input_modalities": ["text"],
        },
    )

    assert response.status_code == 200, response.text
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["llm"]["main"] == "opencode_go_main"
    runtime = parsed["llm"]["runtimes"]["opencode_go_main"]
    assert runtime["api_key"] == "new-secret"
    assert runtime["max_output_tokens"] == 0
    assert parsed["llm"]["runtimes"]["deepseek_main"]["api_key"] == "saved-secret"
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert config_path.with_name(
        f"config.toml.{response.json()['operationId']}.bak"
    ).exists()


def test_state_reports_reasoning_effort(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _config().replace(
            "input_modalities = [\"text\"]",
            "input_modalities = [\"text\"]\nreasoning_effort = \"high\"",
        ),
        encoding="utf-8",
    )
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )

    response = TestClient(app).get("/api/settings/state")

    assert response.status_code == 200
    assert response.json()["runtimes"][0]["reasoningEffort"] == "high"


def test_apply_writes_reasoning_effort(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
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
            "max_output_tokens": 0,
            "reasoning_effort": "high",
            "input_modalities": ["text"],
        },
    )

    assert response.status_code == 200, response.text
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["llm"]["runtimes"]["opencode_go_main"]["reasoning_effort"] == "high"


def test_codex_models_expose_reasoning_effort_capabilities(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")

    class FakeModel:
        slug = "o4-mini"
        capabilities = type(
            "Caps",
            (),
            {
                "context_window": 200000,
                "max_output_tokens": 100000,
                "input_modalities": ("text",),
                "supported_reasoning_efforts": ("minimal", "low", "medium", "high"),
                "default_reasoning_effort": "medium",
            },
        )()

    class FakeCatalog:
        def __init__(self, auth) -> None:
            pass

        async def list_models(self):
            return [FakeModel()]

    monkeypatch.setattr("bootstrap.settings_api.CodexModelCatalog", FakeCatalog)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )
    response = TestClient(app).post(
        "/api/settings/models",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={"provider": "codex"},
    )

    assert response.status_code == 200, response.text
    model = response.json()["models"][0]
    assert model["supportedReasoningEfforts"] == ["minimal", "low", "medium", "high"]
    assert model["defaultReasoningEffort"] == "medium"


def test_opencode_go_models_expose_reasoning_efforts(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")

    class FakeModel:
        def __init__(self, slug: str, efforts: tuple[str, ...]) -> None:
            self.slug = slug
            self.supported_reasoning_efforts = efforts

    class FakeCatalog:
        def __init__(self, api_key: str, *, base_url: str) -> None:
            pass

        async def list_models(self):
            return [
                FakeModel("deepseek-v4-pro", ("low", "medium", "high", "max")),
                FakeModel("kimi-k3", ()),
            ]

    monkeypatch.setattr("bootstrap.settings_api.OpenCodeGoModelCatalog", FakeCatalog)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )
    response = TestClient(app).post(
        "/api/settings/models",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={"provider": "opencode-go", "api_key": "secret"},
    )

    assert response.status_code == 200, response.text
    models = {item["id"]: item for item in response.json()["models"]}
    assert models["deepseek-v4-pro"]["supportedReasoningEfforts"] == [
        "low",
        "medium",
        "high",
        "max",
    ]
    assert models["kimi-k3"]["supportedReasoningEfforts"] == []


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
