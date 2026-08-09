from __future__ import annotations

import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent.config_models import Config
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.store import ModelRegistryStore
from bootstrap.settings_api import create_settings_app
from bootstrap.settings_api import _new_config


def _config(secret: str = "saved-secret") -> str:
    return f"""\
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
max_output_tokens = 8192
input_modalities = ["text"]

[agent.context]
[agent.context.compaction]
keep_recent_tokens = 20000
"""


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


def test_database_settings_state_redacts_credential_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    app = create_settings_app(config_path, tmp_path / "workspace")
    client = TestClient(app)
    applied = client.post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "database-secret",
            "base_url": "https://api.deepseek.com/v1",
        },
    )

    assert applied.status_code == 200, applied.text
    response = client.get("/api/settings/state")
    assert response.status_code == 200
    assert "database-secret" not in response.text
    assert "auth_payload" not in response.text
    assert response.json()["runtimes"][0]["credential"]["configured"] is True


def test_settings_round_trip_preserves_explicit_legacy_output_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    legacy = (
        _config()
        .replace(
            "max_output_tokens = 8192\n",
            "",
        )
        .replace(
            "[agent.context]",
            "[agent]\nmax_tokens = 8192\n\n[agent.context]",
        )
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
    assert (
        response.json()["error"] == "runtime deepseek_main 的 context_window 必须是整数"
    )


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
    assert (
        CredentialStore.for_workspace(tmp_path / "workspace").api_key(runtime.auth_id)
        == "new-secret"
    )
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
    app = create_settings_app(
        config_path, tmp_path / "workspace", credential_store=store
    )
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
    assert CredentialStore.for_workspace(tmp_path / "workspace").get(
        "codex_default"
    ) == Credential(driver="codex", access_token="token")


def test_codex_apply_reuses_login_connection_when_ui_supplies_source_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    workspace = tmp_path / "workspace"
    store = CredentialStore.for_workspace(workspace)
    store.provision_connection(
        "codex_default",
        name="Codex",
        provider="codex",
        base_url="https://chatgpt.com/backend-api/codex",
    )
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
    app = create_settings_app(config_path, workspace)
    response = TestClient(app).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "codex",
            "model": "o4-mini",
            "credential_id": "codex_default",
            "source_id": "ui-generated-source",
            "source_name": "Codex",
        },
    )

    assert response.status_code == 200, response.text
    with closing(sqlite3.connect(workspace / "model-registry.sqlite3")) as connection:
        rows = connection.execute(
            "SELECT id FROM model_connections WHERE auth_id = 'codex_default'"
        ).fetchall()
    assert rows == [("codex_default",)]
    snapshot = ModelRegistryStore.for_workspace(workspace).read_snapshot()
    assert snapshot is not None
    runtime = next(
        item for item in snapshot.runtimes.values() if item.provider == "codex"
    )
    assert runtime.source_id == "codex_default"


def test_opencode_connection_without_model_syncs_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    workspace = tmp_path / "workspace"

    class FakeCatalog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def list_models(self):
            return [
                SimpleNamespace(
                    slug="deepseek-v4-flash",
                    supported_reasoning_efforts=("low", "high"),
                ),
                SimpleNamespace(
                    slug="kimi-k2.5",
                    supported_reasoning_efforts=("medium", "high"),
                ),
            ]

    async def reject_validation(*_args, **_kwargs) -> None:
        raise AssertionError("目录同步不应逐个发送聊天请求")

    monkeypatch.setattr("bootstrap.settings_api.OpenCodeGoModelCatalog", FakeCatalog)
    monkeypatch.setattr(
        "bootstrap.settings_api._validate_live_candidate", reject_validation
    )
    app = create_settings_app(config_path, workspace)
    response = TestClient(app).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "opencode-go",
            "model": "",
            "source_id": "source:opencode_go_main",
            "source_name": "OpenCode Go",
            "api_key": "opencode-secret",
            "base_url": "https://opencode.ai/zen/go/v1",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["modelCount"] == 2
    snapshot = ModelRegistryStore.for_workspace(workspace).read_snapshot()
    assert snapshot is not None
    synced = {
        runtime.model: (runtime_id, runtime)
        for runtime_id, runtime in snapshot.runtimes.items()
        if runtime.source_id == "source:opencode_go_main"
    }
    assert set(synced) == {"deepseek-v4-flash", "kimi-k2.5"}
    assert synced["deepseek-v4-flash"][1].supported_reasoning_efforts == (
        "low",
        "high",
    )
    assert all(runtime.input_modalities == ("text",) for _, runtime in synced.values())
    assert snapshot.roles["default"].runtime_id == synced["deepseek-v4-flash"][0]


def test_codex_connection_without_model_syncs_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_config(), encoding="utf-8")
    workspace = tmp_path / "workspace"
    store = CredentialStore(tmp_path / "auth" / "auth.json")
    store.put("codex_default", Credential(driver="codex", access_token="token"))

    caps = SimpleNamespace(
        context_window=200_000,
        max_output_tokens=100_000,
        input_modalities=("text", "image"),
        supported_reasoning_efforts=("low", "medium", "high"),
        supports_parallel_tool_calls=True,
    )

    class FakeCatalog:
        def __init__(self, _auth) -> None:
            pass

        async def list_models(self):
            return [
                SimpleNamespace(
                    slug="gpt-5-codex",
                    input_modalities_known=True,
                    capabilities=caps,
                ),
                SimpleNamespace(
                    slug="gpt-5-codex-mini",
                    input_modalities_known=True,
                    capabilities=caps,
                ),
            ]

    monkeypatch.setattr("bootstrap.settings_api.CodexModelCatalog", FakeCatalog)
    app = create_settings_app(
        config_path,
        workspace,
        credential_store=store,
    )
    response = TestClient(app).post(
        "/api/settings/apply",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "provider": "codex",
            "model": "",
            "credential_id": "codex_default",
            "source_name": "Codex",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["modelCount"] == 2
    snapshot = ModelRegistryStore.for_workspace(workspace).read_snapshot()
    assert snapshot is not None
    synced = {
        runtime.model: runtime_id
        for runtime_id, runtime in snapshot.runtimes.items()
        if runtime.provider == "codex"
    }
    assert set(synced) == {"gpt-5-codex", "gpt-5-codex-mini"}
    assert snapshot.roles["default"].runtime_id == synced["gpt-5-codex"]


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
            'input_modalities = ["text"]',
            'input_modalities = ["text"]\nreasoning_effort = "high"',
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
    model_credentials = CredentialStore.for_workspace(workspace)
    assert (
        model_credentials.api_key(named["DeepSeek 官方"].auth_id) == "official-secret"
    )
    assert model_credentials.api_key(named["公司网关"].auth_id) == "proxy-secret"


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
        "frame-ancestors 'self' " "http://127.0.0.1:5173 http://localhost:5173"
    )
    assert "attacker.invalid" not in response.headers["content-security-policy"]


def test_new_config_includes_web_chat_runtime_dependencies(tmp_path: Path) -> None:
    parsed = tomllib.loads(_new_config(tmp_path / "workspace"))

    assert parsed["channels"]["chat"] == {
        "enabled": True,
        "channel_name": "web",
    }
    assert parsed["app_server"]["enabled"] is True


def test_first_run_defers_restart_until_memory_and_embedding_are_configured(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    reloads: list[str] = []
    restarts: list[str] = []

    async def validate_model(*_args, **_kwargs) -> None:
        return None

    async def probe_embedding(**_kwargs) -> int:
        return 1024

    monkeypatch.setattr(
        "bootstrap.settings_api._validate_live_candidate", validate_model
    )
    monkeypatch.setattr(
        "bootstrap.settings_api._probe_embedding_candidate", probe_embedding
    )
    app = create_settings_app(
        config_path,
        workspace,
        on_model_applied=lambda: reloads.append("reload"),
        on_runtime_applied=lambda: restarts.append("restart"),
    )
    client = TestClient(app)
    headers = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}

    model_response = client.post(
        "/api/settings/apply",
        headers=headers,
        json={
            "provider": "openai",
            "model": "chat-model",
            "api_key": "chat-secret",
            "base_url": "https://chat.example/v1",
            "defer_restart": True,
        },
    )
    assert model_response.status_code == 200, model_response.text
    assert reloads == []
    assert restarts == []

    state = client.get("/api/settings/state").json()
    assert state["memory"]["configured"] is False
    embedding_response = client.post(
        "/api/settings/embedding-models",
        headers=headers,
        json={
            "source_name": "向量服务",
            "provider": "openai",
            "model": "text-embedding-test",
            "api_key": "embedding-secret",
            "base_url": "https://embedding.example/v1",
            "expected_revision": state["modelRevision"],
        },
    )
    assert embedding_response.status_code == 200, embedding_response.text
    embedding = embedding_response.json()["model"]
    assert embedding["dimensions"] == 1024

    state = client.get("/api/settings/state").json()
    assert state["memory"]["embeddingModels"] == [embedding]
    memory_response = client.post(
        "/api/settings/memory",
        headers=headers,
        json={
            "enabled": True,
            "engine": "akasha",
            "embedding_model_id": embedding["id"],
            "expected_revision": state["memory"]["revision"],
        },
    )
    assert memory_response.status_code == 200, memory_response.text
    assert reloads == []
    assert restarts == ["restart"]

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["memory"] == {
        "enabled": True,
        "engine": "akasha",
        "embedding": {"model_ref": embedding["id"]},
    }
    assert "embedding-secret" not in config_path.read_text(encoding="utf-8")
    loaded = Config.load(config_path, workspace=workspace)
    assert loaded.memory.embedding.model_ref == embedding["id"]
    assert loaded.memory.embedding.model == "text-embedding-test"
    assert loaded.memory.embedding.output_dimensionality == 1024
    assert loaded.memory.embedding.api_key == "embedding-secret"


def test_onboarding_persists_skips_and_restarts_only_when_completed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    callbacks: list[str] = []

    async def validate_model(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "bootstrap.settings_api._validate_live_candidate", validate_model
    )
    app = create_settings_app(
        config_path,
        workspace,
        on_applied=lambda: callbacks.append("restart"),
    )
    client = TestClient(app)
    headers = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}

    initial = client.get("/api/settings/state").json()
    assert initial["onboarding"] == {
        "step": "welcome",
        "completed": False,
        "memoryDecision": "pending",
        "channelDecision": "pending",
    }
    started = client.post(
        "/api/settings/onboarding/start",
        headers=headers,
        json={},
    )
    assert started.status_code == 200, started.text
    assert started.json()["onboarding"]["step"] == "model"

    model = client.post(
        "/api/settings/apply",
        headers=headers,
        json={
            "provider": "openai",
            "model": "chat-model",
            "api_key": "chat-secret",
            "base_url": "https://chat.example/v1",
            "defer_restart": True,
        },
    )
    assert model.status_code == 200, model.text
    assert callbacks == []
    model_done = client.post(
        "/api/settings/onboarding/advance",
        headers=headers,
        json={"step": "model"},
    )
    assert model_done.status_code == 200, model_done.text
    config_after_model = config_path.read_bytes()

    memory_skip = client.post(
        "/api/settings/onboarding/advance",
        headers=headers,
        json={"step": "memory", "decision": "skipped"},
    )
    channel_skip = client.post(
        "/api/settings/onboarding/advance",
        headers=headers,
        json={"step": "channel", "decision": "skipped"},
    )
    assert memory_skip.status_code == 200, memory_skip.text
    assert channel_skip.status_code == 200, channel_skip.text
    assert config_path.read_bytes() == config_after_model
    pending = client.get("/api/settings/state").json()["onboarding"]
    assert pending == {
        "step": "done",
        "completed": False,
        "memoryDecision": "skipped",
        "channelDecision": "skipped",
    }

    completed = client.post(
        "/api/settings/onboarding/complete",
        headers=headers,
        json={},
    )
    repeated = client.post(
        "/api/settings/onboarding/complete",
        headers=headers,
        json={},
    )
    assert completed.status_code == 200, completed.text
    assert repeated.status_code == 200, repeated.text
    assert callbacks == ["restart"]
    assert repeated.json()["status"] == "already_completed"
    assert client.get("/api/settings/state").json()["onboarding"]["completed"] is True


def test_onboarding_optional_saves_defer_restart_until_completion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    reloads: list[str] = []
    restarts: list[str] = []

    async def validate_model(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "bootstrap.settings_api._validate_live_candidate", validate_model
    )
    app = create_settings_app(
        config_path,
        workspace,
        on_model_applied=lambda: reloads.append("reload"),
        on_runtime_applied=lambda: restarts.append("restart"),
    )
    client = TestClient(app)
    headers = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}
    started = client.post(
        "/api/settings/onboarding/start",
        headers=headers,
        json={},
    )
    assert started.status_code == 200
    assert (
        client.post(
            "/api/settings/apply",
            headers=headers,
            json={
                "provider": "openai",
                "model": "chat-model",
                "api_key": "chat-secret",
                "base_url": "https://chat.example/v1",
                "defer_restart": True,
            },
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/settings/onboarding/advance",
            headers=headers,
            json={"step": "model"},
        ).status_code
        == 200
    )

    state = client.get("/api/settings/state").json()
    memory = client.post(
        "/api/settings/memory",
        headers=headers,
        json={
            "enabled": False,
            "engine": "akasha",
            "expected_revision": state["memory"]["revision"],
            "defer_restart": True,
        },
    )
    assert memory.status_code == 200, memory.text
    assert reloads == []
    assert restarts == []
    assert (
        client.post(
            "/api/settings/onboarding/advance",
            headers=headers,
            json={"step": "memory", "decision": "configured"},
        ).status_code
        == 200
    )

    configured = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        configured + """
[mobile_realtime]
enabled = false
host = "0.0.0.0"
port = 6323
database = "data/mobile_realtime.db"
lan_hostname = "192.168.0.108"
public_url = "wss://mobile.example.com/ws"
max_attachment_mb = 50
inbox_retention_days = 7
""",
        encoding="utf-8",
    )
    state = client.get("/api/settings/state").json()
    assert state["mobileRealtime"] == {
        "enabled": False,
        "port": 6323,
        "lanHostname": "192.168.0.108",
        "publicUrl": "wss://mobile.example.com/ws",
    }
    channel = client.post(
        "/api/settings/onboarding-channel",
        headers=headers,
        json={
            "proactive_enabled": False,
            "mobile_realtime_enabled": True,
            "expected_revision": state["configRevision"],
        },
    )
    assert channel.status_code == 200, channel.text
    assert reloads == []
    assert restarts == []
    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["mobile_realtime"]["enabled"] is True
    assert parsed["mobile_realtime"]["lan_hostname"] == "192.168.0.108"
    assert parsed["mobile_realtime"]["public_url"] == "wss://mobile.example.com/ws"
    assert client.get("/api/settings/state").json()["mobileRealtime"] == {
        "enabled": True,
        "port": 6323,
        "lanHostname": "192.168.0.108",
        "publicUrl": "wss://mobile.example.com/ws",
    }
    assert (
        client.post(
            "/api/settings/onboarding/advance",
            headers=headers,
            json={"step": "channel", "decision": "configured"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/settings/onboarding/complete",
            headers=headers,
            json={},
        ).status_code
        == 200
    )
    assert reloads == []
    assert restarts == ["restart"]


def test_channels_update_preserves_saved_telegram_token(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    async def validate_model(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "bootstrap.settings_api._validate_live_candidate", validate_model
    )
    app = create_settings_app(config_path, workspace, on_applied=lambda: None)
    client = TestClient(app)
    headers = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}
    model = client.post(
        "/api/settings/apply",
        headers=headers,
        json={
            "provider": "openai",
            "model": "chat-model",
            "api_key": "chat-secret",
            "base_url": "https://chat.example/v1",
            "defer_restart": True,
        },
    )
    assert model.status_code == 200, model.text

    revision = client.get("/api/settings/state").json()["configRevision"]
    first = client.post(
        "/api/settings/channels",
        headers=headers,
        json={
            "telegram_token": "123456789:telegram-secret",
            "telegram_username": "first_user",
            "expected_revision": revision,
        },
    )
    assert first.status_code == 200, first.text

    revision = client.get("/api/settings/state").json()["configRevision"]
    update = client.post(
        "/api/settings/channels",
        headers=headers,
        json={
            "telegram_token": "",
            "telegram_username": "updated_user",
            "expected_revision": revision,
        },
    )
    assert update.status_code == 200, update.text

    parsed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert parsed["channels"]["telegram"] == {
        "token": "123456789:telegram-secret",
        "allow_from": ["updated_user"],
    }
    assert parsed["channels"]["chat"] == {
        "enabled": True,
        "channel_name": "web",
    }


def test_channel_target_discovery_uses_form_and_saved_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _config() + """
[channels.telegram]
token = "saved-token"
allow_from = ["saved_user"]
proxy = "socks5://127.0.0.1:1080"
""",
        encoding="utf-8",
    )
    telegram_calls: list[tuple[str, str, int]] = []
    qq_calls: list[tuple[str, str, int]] = []

    def fetch_telegram(
        token: str,
        username: str,
        timeout_s: int,
        _stop,
    ) -> str:
        telegram_calls.append((token, username, timeout_s))
        return "123456789"

    async def fetch_qq(
        app_id: str,
        client_secret: str,
        timeout_s: int,
        _stop,
    ) -> str:
        qq_calls.append((app_id, client_secret, timeout_s))
        return "openid-1"

    monkeypatch.setattr("bootstrap.setup_wizard._fetch_chat_id", fetch_telegram)
    monkeypatch.setattr(
        "bootstrap.setup_wizard._async_fetch_qqbot_openid",
        fetch_qq,
    )
    client = TestClient(create_settings_app(config_path, tmp_path / "workspace"))
    headers = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}

    telegram = client.post(
        "/api/settings/channels/telegram/discover-target",
        headers=headers,
        json={},
    )
    qqbot = client.post(
        "/api/settings/channels/qqbot/discover-target",
        headers=headers,
        json={"app_id": "qq-app", "client_secret": "qq-secret"},
    )

    assert telegram.status_code == 200, telegram.text
    assert telegram.json() == {"targetId": "123456789"}
    assert telegram_calls == [("saved-token", "saved_user", 45)]
    assert qqbot.status_code == 200, qqbot.text
    assert qqbot.json() == {"targetId": "c2c:openid-1"}
    assert qq_calls == [("qq-app", "qq-secret", 45)]


def test_channel_updates_preserve_fields_not_owned_by_settings_ui(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _config() + """
[channels.telegram]
token = "old-token"
allow_from = ["old-user"]
proxy = "socks5://127.0.0.1:1080"
""",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    async def validate(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr("bootstrap.settings_api._validate_live_candidate", validate)
    client = TestClient(
        create_settings_app(
            config_path,
            workspace,
            on_runtime_applied=lambda: None,
        )
    )
    headers = {"Origin": "http://testserver", "X-Akasic-CSRF": "1"}
    model = client.post(
        "/api/settings/apply",
        headers=headers,
        json={
            "provider": "openai",
            "model": "chat-model",
            "api_key": "chat-secret",
            "base_url": "https://chat.example/v1",
            "defer_restart": True,
        },
    )
    assert model.status_code == 200, model.text
    revision = client.get("/api/settings/state").json()["configRevision"]
    response = client.post(
        "/api/settings/channels",
        headers=headers,
        json={
            "telegram_token": "new-token",
            "telegram_username": "new-user",
            "expected_revision": revision,
        },
    )
    assert response.status_code == 200, response.text
    telegram = tomllib.loads(config_path.read_text(encoding="utf-8"))["channels"][
        "telegram"
    ]
    assert telegram == {
        "token": "new-token",
        "allow_from": ["new-user"],
        "proxy": "socks5://127.0.0.1:1080",
    }

    from agent.plugins.manifest import workspace_plugin_data_dir

    qqbot_path = (
        workspace_plugin_data_dir(workspace, "qqbot", "github") / "config.local.toml"
    )
    qqbot_path.parent.mkdir(parents=True, exist_ok=True)
    qqbot_path.write_text(
        'app_id = "old-id"\nclient_secret = "old-secret"\nintent = 42\n',
        encoding="utf-8",
    )
    revision = client.get("/api/settings/state").json()["configRevision"]
    response = client.post(
        "/api/settings/channels",
        headers=headers,
        json={
            "qqbot_app_id": "new-id",
            "qqbot_client_secret": "new-secret",
            "qqbot_target_id": "c2c:openid-1",
            "expected_revision": revision,
        },
    )
    assert response.status_code == 200, response.text
    assert tomllib.loads(qqbot_path.read_text(encoding="utf-8")) == {
        "app_id": "new-id",
        "client_secret": "new-secret",
        "intent": 42,
        "allow_from": ["openid-1"],
    }


def test_failed_model_connection_reload_restores_registry_and_old_generation(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    config_path.write_text(_config(), encoding="utf-8")
    registry = ModelRegistryStore.for_workspace(workspace)
    registry.replace_from_llm_config(
        {
            "main": "model-a",
            "fast": "model-b",
            "agent": "model-a",
            "vl": "model-b",
            "runtimes": {
                "model-a": {
                    "provider": "openai",
                    "model": "alpha",
                    "base_url": "https://one.example/v1",
                    "input_modalities": ["text"],
                },
                "model-b": {
                    "provider": "openai",
                    "model": "beta",
                    "base_url": "https://two.example/v1",
                    "input_modalities": ["text"],
                },
            },
        }
    )
    callbacks: list[str] = []

    def reload_models() -> None:
        callbacks.append("reload")
        if len(callbacks) == 1:
            raise RuntimeError("candidate reload failed")

    client = TestClient(
        create_settings_app(
            config_path,
            workspace,
            on_model_applied=reload_models,
        ),
        raise_server_exceptions=False,
    )
    response = client.post(
        "/api/settings/model-connections/source:model-a/remove",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={"expected_revision": 1},
    )

    assert response.status_code == 500
    snapshot = registry.read_snapshot()
    assert snapshot is not None
    assert set(snapshot.runtimes) == {"model-a", "model-b"}
    assert snapshot.roles["default"].runtime_id == "model-a"
    assert callbacks == ["reload", "reload"]


def test_memory_switch_is_rejected_after_conversation_history_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    config_path.write_text(
        _config() + '\n[memory]\nenabled = false\nengine = "default"\n',
        encoding="utf-8",
    )
    workspace.mkdir()
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        connection.execute("CREATE TABLE messages (id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO messages(id) VALUES ('message-1')")
        connection.commit()

    async def validate_model(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(
        "bootstrap.settings_api._validate_live_candidate", validate_model
    )
    app = create_settings_app(config_path, workspace)
    client = TestClient(app)
    state = client.get("/api/settings/state").json()
    response = client.post(
        "/api/settings/memory",
        headers={"Origin": "http://testserver", "X-Akasic-CSRF": "1"},
        json={
            "enabled": False,
            "engine": "akasha",
            "expected_revision": state["memory"]["revision"],
        },
    )

    assert response.status_code == 409
    assert "重建索引" in response.json()["detail"]


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
