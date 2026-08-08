from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.config_models import Config
from agent.model_runtime.errors import AuthenticationError
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.registry import ModelGeneration, ModelRegistry
from agent.model_runtime.store import ModelRegistryStore

_CONFIG = """
[runtime]
workspace = "unused"

[llm]
main = "legacy"

[llm.runtimes.legacy]
provider = "openai"
model = "legacy-model"
input_modalities = ["text"]

[agent]
system_prompt = "test"
"""


def _llm_rows() -> dict[str, object]:
    return {
        "main": "model-a",
        "fast": "model-b",
        "agent": "model-a",
        "vl": "model-b",
        "runtimes": {
            "model-a": {
                "provider": "openai",
                "model": "alpha",
                "base_url": "https://one.example/v1",
                "reasoning_effort": "medium",
                "supported_reasoning_efforts": ["low", "medium", "high"],
                "context_window": 128_000,
                "max_output_tokens": 8_192,
                "input_modalities": ["text", "image"],
                "capability_source": "litellm",
            },
            "model-b": {
                "provider": "openai",
                "model": "beta",
                "base_url": "https://two.example/v1",
                "context_window": 64_000,
                "max_output_tokens": 8_192,
                "input_modalities": ["text"],
            },
        },
    }


def test_model_store_imports_connections_models_and_roles(tmp_path: Path) -> None:
    store = ModelRegistryStore.for_workspace(tmp_path)

    revision = store.replace_from_llm_config(
        _llm_rows(),
        source_names={"model-a": "主账号", "model-b": "备用账号"},
    )
    snapshot = store.read_snapshot()

    assert revision == 1
    assert snapshot is not None
    assert snapshot.revision == 1
    assert snapshot.runtimes["model-a"].source_name == "主账号"
    assert snapshot.runtimes["model-a"].input_modalities == ("text", "image")
    assert snapshot.runtimes["model-a"].supported_reasoning_efforts == (
        "low",
        "medium",
        "high",
    )
    assert snapshot.roles["fast"].runtime_id == "model-b"
    assert snapshot.as_config_llm()["main"] == "model-a"


def test_model_store_keeps_legacy_percent_columns_inert(tmp_path: Path) -> None:
    store = ModelRegistryStore.for_workspace(tmp_path)
    rows = _llm_rows()
    runtimes = rows["runtimes"]
    assert isinstance(runtimes, dict)
    runtimes["model-a"]["effective_context_percent"] = 0.12  # type: ignore[index]
    runtimes["model-a"]["compaction_trigger_percent"] = 0.34  # type: ignore[index]

    store.replace_from_llm_config(rows)
    snapshot = store.read_snapshot()
    assert snapshot is not None
    runtime = snapshot.runtimes["model-a"]
    assert not hasattr(runtime, "effective_context_percent")
    assert not hasattr(runtime, "compaction_trigger_percent")
    projected = snapshot.as_config_llm()["runtimes"]
    assert isinstance(projected, dict)
    assert "effective_context_percent" not in projected["model-a"]
    assert "compaction_trigger_percent" not in projected["model-a"]

    with store._connect(read_only=True) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(model_definitions)"
            ).fetchall()
        }
        values = connection.execute(
            "SELECT effective_context_percent, compaction_trigger_percent "
            "FROM model_definitions WHERE id = 'model-a'"
        ).fetchone()
    assert {"effective_context_percent", "compaction_trigger_percent"} <= columns
    assert values is not None
    assert tuple(values) == (0.9, 0.74)


def test_role_update_is_revisioned_and_rejects_stale_writer(tmp_path: Path) -> None:
    store = ModelRegistryStore.for_workspace(tmp_path)
    _ = store.replace_from_llm_config(_llm_rows())

    revision = store.set_role(
        "default",
        "model-b",
        reasoning_effort="high",
        expected_revision=1,
    )

    assert revision == 2
    snapshot = store.read_snapshot()
    assert snapshot is not None
    assert snapshot.roles["default"].runtime_id == "model-b"
    assert snapshot.roles["default"].reasoning_effort == "high"
    with pytest.raises(RuntimeError, match="已经变化"):
        store.set_role("default", "model-a", expected_revision=1)


def test_config_load_prefers_workspace_model_database(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    store = ModelRegistryStore.for_workspace(tmp_path)
    _ = store.replace_from_llm_config(_llm_rows())

    config = Config.load(config_path, workspace=tmp_path)

    assert config.model == "alpha"
    assert config.runtime_id == "model-a"
    assert config.fast_runtime_id == "model-b"
    assert config.model_registry_revision == 1


def test_model_credentials_live_in_private_workspace_database(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    llm = _llm_rows()
    runtimes = llm["runtimes"]
    assert isinstance(runtimes, dict)
    runtimes["model-a"]["auth"] = "primary"  # type: ignore[index]
    store = ModelRegistryStore.for_workspace(tmp_path)

    store.replace_from_llm_config(
        llm,
        credentials={"primary": Credential(driver="api_key", access_token="secret")},
    )

    credentials = CredentialStore.for_workspace(tmp_path)
    assert credentials.api_key("primary") == "secret"
    assert Config.load(config_path, workspace=tmp_path).api_key == "secret"
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    assert not (tmp_path / "auth.json").exists()


def test_embedding_models_share_provider_connections_without_entering_chat_roles(
    tmp_path: Path,
) -> None:
    store = ModelRegistryStore.for_workspace(tmp_path)
    _ = store.replace_from_llm_config(_llm_rows())

    revision = store.upsert_embedding_model(
        model_id="embed-main",
        source_id="embedding-provider",
        source_name="向量服务",
        provider="openai",
        auth_id="embedding_default",
        base_url="https://embedding.example/v1",
        model="text-embedding-3-small",
        dimensions=1536,
        credential=Credential(driver="api_key", access_token="embedding-secret"),
        expected_revision=1,
    )

    assert revision == 2
    embedding = store.get_embedding_model("embed-main")
    assert embedding is not None
    assert embedding.model == "text-embedding-3-small"
    assert embedding.dimensions == 1536
    assert (
        CredentialStore.for_workspace(tmp_path).api_key("embedding_default")
        == "embedding-secret"
    )
    snapshot = store.read_snapshot()
    assert snapshot is not None
    assert set(snapshot.runtimes) == {"model-a", "model-b"}


def test_chat_registry_refresh_preserves_embedding_models(tmp_path: Path) -> None:
    store = ModelRegistryStore.for_workspace(tmp_path)
    _ = store.replace_from_llm_config(_llm_rows())
    _ = store.upsert_embedding_model(
        model_id="embed-main",
        source_id="embedding-provider",
        source_name="向量服务",
        provider="openai",
        auth_id="embedding_default",
        base_url="https://embedding.example/v1",
        model="text-embedding-3-small",
        dimensions=1536,
        credential=Credential(driver="api_key", access_token="embedding-secret"),
    )

    _ = store.replace_from_llm_config(_llm_rows())

    assert store.get_embedding_model("embed-main") is not None
    assert (
        CredentialStore.for_workspace(tmp_path).api_key("embedding_default")
        == "embedding-secret"
    )


def test_workspace_credential_store_rejects_broad_database_permissions(
    tmp_path: Path,
) -> None:
    store = ModelRegistryStore.for_workspace(tmp_path)
    store.replace_from_llm_config(
        _llm_rows(),
        credentials={"unused": Credential(driver="api_key", access_token="secret")},
    )
    os.chmod(store.path, 0o644)

    with pytest.raises(AuthenticationError, match="权限过宽"):
        CredentialStore.for_workspace(tmp_path).metadata()


@pytest.mark.asyncio
async def test_new_execution_reads_latest_role_while_active_scope_keeps_snapshot(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(_CONFIG, encoding="utf-8")
    store = ModelRegistryStore.for_workspace(tmp_path)
    _ = store.replace_from_llm_config(_llm_rows())

    def build(config: Config, generation_id: int) -> ModelGeneration:
        return ModelGeneration(
            generation_id=generation_id,
            config_digest=str(config.model_registry_revision),
            runtimes=dict(config.model_runtimes),
            providers={runtime_id: object() for runtime_id in config.model_runtimes},
            role_runtime_ids={
                "default": config.runtime_id,
                "fast": config.fast_runtime_id,
                "agent": config.agent_runtime_id,
                "vision": config.vl_runtime_id,
            },
            registry_revision=config.model_registry_revision,
        )

    registry = ModelRegistry(Config.load(config_path, workspace=tmp_path), build)
    async with registry.execution_scope() as active:
        assert active.describe("default")["model"] == "alpha"
        _ = store.set_role("default", "model-b", expected_revision=1)
        assert active.describe("default")["model"] == "alpha"

    async with registry.execution_scope() as next_execution:
        assert next_execution.describe("default")["model"] == "beta"
        assert next_execution.generation.registry_revision == 2
