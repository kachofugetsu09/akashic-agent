from __future__ import annotations

import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.store import ModelRegistryStore
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
    snapshot = ModelRegistryStore.for_workspace(tmp_path / "workspace").read_snapshot()
    assert snapshot is not None
    assert snapshot.runtimes["deepseek_main"].max_output_tokens == 8192


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


def test_apply_writes_credential_and_preserves_other_models(
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
            "input_modalities": ["text"],
        },
    )

    assert response.status_code == 200, response.text
    snapshot = ModelRegistryStore.for_workspace(tmp_path / "workspace").read_snapshot()
    assert snapshot is not None
    assert snapshot.roles["default"].runtime_id == "opencode_go_main"
    runtime = snapshot.runtimes["opencode_go_main"]
    assert runtime.max_output_tokens == 0
    assert store.api_key(runtime.auth_id) == "new-secret"
    assert snapshot.runtimes["deepseek_main"].model == "deepseek-chat"
    assert "new-secret" not in config_path.read_text(encoding="utf-8")
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert config_path.with_name(
        f"config.toml.{response.json()['operationId']}.bak"
    ).exists()


def test_first_apply_initializes_workspace_before_starting_gateway(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    calls: list[str] = []

    async def validate(*_args, **_kwargs) -> None:
        return None

    def initialize(*, config_path: Path, workspace: Path, force: bool = False):
        _ = force
        calls.append(f"init:{config_path.name}:{workspace.name}")
        return object()

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    monkeypatch.setattr("bootstrap.init_workspace.init_workspace", initialize)
    app = create_settings_app(
        config_path,
        workspace,
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
        on_applied=lambda: calls.append("start"),
    )

    response = TestClient(app).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "openai",
            "model": "custom-model",
            "api_key": "new-secret",
            "base_url": "https://example.test/v1",
        },
    )

    assert response.status_code == 200, response.text
    assert calls == ["init:config.toml:workspace", "start"]


def test_apply_derives_capabilities_when_onboarding_sends_only_connection(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            "provider": "openai",
            "model": "gpt-5.2-pro",
            "api_key": "new-secret",
            "base_url": "https://api.openai.com/v1",
        },
    )

    assert response.status_code == 200, response.text
    snapshot = ModelRegistryStore.for_workspace(tmp_path / "workspace").read_snapshot()
    assert snapshot is not None
    runtime = snapshot.runtimes["openai_main"]
    assert runtime.context_window == 400_000
    assert runtime.max_output_tokens == 128_000
    assert runtime.input_modalities == ("text", "image")
    assert runtime.capability_source == "litellm"
    assert runtime.context_window_source == "litellm"
    assert runtime.max_output_tokens_source == "litellm"
    assert runtime.input_modalities_source == "litellm"
    assert runtime.catalog_provider_id == "openai"


def test_generic_api_derives_usage_provider_from_base_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
            "provider": "openai",
            "model": "deepseek-chat",
            "api_key": "new-secret",
            "base_url": "https://api.deepseek.com/v1",
        },
    )

    assert response.status_code == 200, response.text
    snapshot = ModelRegistryStore.for_workspace(tmp_path / "workspace").read_snapshot()
    assert snapshot is not None
    runtime = snapshot.runtimes["deepseek_main"]
    assert runtime.provider == "deepseek"
    assert runtime.catalog_provider_id == "deepseek"


def test_codex_apply_uses_authoritative_catalog_without_advanced_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    store = CredentialStore(tmp_path / "auth" / "auth.json")
    store.put("codex_default", Credential(driver="codex", access_token="token"))

    class FakeModel:
        slug = "o4-mini"
        input_modalities_known = True
        capabilities = type(
            "Caps",
            (),
            {
                "context_window": 200_000,
                "max_output_tokens": 100_000,
                "input_modalities": ("text", "image"),
                "supported_reasoning_efforts": ("low", "medium", "high"),
                "supports_parallel_tool_calls": True,
            },
        )()

    class FakeCatalog:
        def __init__(self, _auth) -> None:
            pass

        async def list_models(self):
            return [FakeModel()]

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api.CodexModelCatalog", FakeCatalog)
    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    app = create_settings_app(config_path, tmp_path / "workspace", credential_store=store)
    response = TestClient(app).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "codex",
            "model": "o4-mini",
            "credential_id": "codex_default",
        },
    )

    assert response.status_code == 200, response.text
    snapshot = ModelRegistryStore.for_workspace(tmp_path / "workspace").read_snapshot()
    assert snapshot is not None
    runtime = snapshot.runtimes["codex_main"]
    assert runtime.context_window == 200_000
    assert runtime.max_output_tokens == 100_000
    assert runtime.input_modalities == ("text", "image")
    assert runtime.capability_source == "provider_catalog"
    assert runtime.context_window_source == "provider_catalog"
    assert runtime.max_output_tokens_source == "provider_catalog"
    assert runtime.input_modalities_source == "provider_catalog"


def test_apply_rejects_stale_settings_writer(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )
    client = TestClient(app)
    revision = client.get("/api/settings/state").json()["configRevision"]
    config_path.write_text(
        _config().replace("deepseek-chat", "deepseek-reasoner"),
        encoding="utf-8",
    )

    response = client.post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "expected_config_revision": revision,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "配置已经变化，请刷新后重试"


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
    snapshot = ModelRegistryStore.for_workspace(tmp_path / "workspace").read_snapshot()
    assert snapshot is not None
    assert snapshot.runtimes["opencode_go_main"].reasoning_effort == "high"


def test_role_binding_updates_database_without_rewriting_static_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    workspace = tmp_path / "workspace"
    registry = ModelRegistryStore.for_workspace(workspace)
    _ = registry.replace_from_llm_config(
        {
            "main": "main",
            "fast": "fast",
            "runtimes": {
                "main": {"provider": "openai", "model": "large"},
                "fast": {"provider": "openai", "model": "small"},
            },
        }
    )
    before = config_path.read_bytes()
    app = create_settings_app(
        config_path,
        workspace,
        credential_store=CredentialStore(tmp_path / "auth/auth.json"),
    )

    response = TestClient(app).post(
        "/api/settings/roles",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "role": "default",
            "model_id": "fast",
            "reasoning_effort": "low",
            "expected_revision": 1,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["revision"] == 2
    assert config_path.read_bytes() == before
    snapshot = registry.read_snapshot()
    assert snapshot is not None
    assert snapshot.roles["default"].runtime_id == "fast"
    assert snapshot.roles["default"].reasoning_effort == "low"


def test_two_named_sources_of_same_provider_keep_separate_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    workspace = tmp_path / "workspace"
    credentials = CredentialStore(tmp_path / "auth/auth.json")

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    client = TestClient(
        create_settings_app(
            config_path,
            workspace,
            credential_store=credentials,
        )
    )
    for source_id, source_name, secret in (
        ("deepseek_official", "DeepSeek 官方", "official-secret"),
        ("deepseek_proxy", "公司网关", "proxy-secret"),
    ):
        response = client.post(
            "/api/settings/apply",
            headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
            json={
                "provider": "openai",
                "model": "deepseek-chat",
                "source_id": source_id,
                "source_name": source_name,
                "api_key": secret,
                "base_url": f"https://{source_id}.example/v1",
            },
        )
        assert response.status_code == 200, response.text

    snapshot = ModelRegistryStore.for_workspace(workspace).read_snapshot()
    assert snapshot is not None
    named = {
        runtime.source_name: runtime
        for runtime in snapshot.runtimes.values()
        if runtime.source_name in {"DeepSeek 官方", "公司网关"}
    }
    assert set(named) == {"DeepSeek 官方", "公司网关"}
    assert credentials.api_key(named["DeepSeek 官方"].auth_id) == "official-secret"
    assert credentials.api_key(named["公司网关"].auth_id) == "proxy-secret"


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


def test_custom_api_model_uses_litellm_capability_registry(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )

    response = TestClient(app).post(
        "/api/settings/models",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "openai",
            "model": "gpt-5.2-pro",
            "base_url": "https://api.openai.com/v1",
        },
    )

    assert response.status_code == 200, response.text
    model = response.json()["models"][0]
    assert model["contextWindow"] == 400_000
    assert model["maxOutputTokens"] == 128_000
    assert model["inputModalities"] == ["text", "image"]
    assert model["supportedReasoningEfforts"] == [
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ]


def test_custom_api_discovers_models_from_openai_compatible_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    requests: list[tuple[str, dict[str, str]]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {"id": "vendor/zeta"},
                    {"id": "vendor/alpha"},
                    {"id": "vendor/alpha"},
                    {"owned_by": "missing-id"},
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == 15.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
            requests.append((url, headers))
            return FakeResponse()

    monkeypatch.setattr("bootstrap.settings_api.httpx.AsyncClient", FakeClient)
    app = create_settings_app(
        config_path,
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )

    response = TestClient(app).post(
        "/api/settings/models",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "openai",
            "model": "",
            "api_key": "catalog-secret",
            "base_url": "https://gateway.example/v1/",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "models": [{"id": "vendor/alpha"}, {"id": "vendor/zeta"}]
    }
    assert requests == [
        (
            "https://gateway.example/v1/models",
            {"Authorization": "Bearer catalog-secret"},
        )
    ]


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


def test_settings_page_allows_same_origin_and_local_dev_shell_frames(
    tmp_path: Path,
) -> None:
    app = create_settings_app(
        tmp_path / "config.toml",
        tmp_path / "workspace",
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
    )

    response = TestClient(app).get("/api/settings/state")

    assert response.status_code == 200
    assert response.headers["content-security-policy"].endswith(
        "frame-ancestors 'self' "
        "http://127.0.0.1:5173 http://localhost:5173"
    )
    assert "attacker.invalid" not in response.headers["content-security-policy"]


def test_new_config_includes_web_chat_runtime_dependencies(tmp_path: Path) -> None:
    parsed = tomllib.loads(_new_config(tmp_path / "workspace"))

    assert parsed["channels"]["chat"] == {
        "enabled": True,
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


def test_failed_first_gateway_start_removes_config_without_second_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    callbacks: list[str] = []

    async def validate(*_args, **_kwargs) -> None:
        return None

    def initialize(*, config_path: Path, workspace: Path, force: bool = False):
        _ = (config_path, workspace, force)
        return object()

    def restart() -> None:
        callbacks.append("restart")
        raise RuntimeError("candidate failed readiness")

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    monkeypatch.setattr("bootstrap.init_workspace.init_workspace", initialize)
    app = create_settings_app(
        config_path,
        workspace,
        credential_store=CredentialStore(tmp_path / "auth" / "auth.json"),
        on_applied=restart,
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "openai",
            "model": "custom-model",
            "api_key": "candidate-secret",
            "base_url": "https://example.test/v1",
        },
    )

    assert response.status_code == 500
    assert not config_path.exists()
    assert callbacks == ["restart"]
