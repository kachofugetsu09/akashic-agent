from __future__ import annotations

from dataclasses import fields, replace

import pytest

from agent.plugin_composition import (
    CHAT_MODELS,
    EMBEDDINGS,
    EmbeddingSpaceDescriptor,
    MODEL_CATALOG,
    MODEL_DRIVERS,
    MODEL_SETTINGS,
    LLMResponse,
    ModelCatalogSnapshot,
    ModelContinuation,
    ModelRole,
    ModelUnavailableError,
    RateLimitError,
    AuthenticationError,
    ModelRequest,
)
from tests.model_plugin_fakes import build_test_chat_models


def test_model_services_have_stable_distinct_keys() -> None:
    assert {
        CHAT_MODELS.name,
        EMBEDDINGS.name,
        MODEL_CATALOG.name,
        MODEL_SETTINGS.name,
        MODEL_DRIVERS.name,
    } == {
        "models.chat.v1",
        "models.embeddings.v1",
        "models.catalog.v1",
        "models.settings.v1",
        "models.drivers.v1",
    }


def test_model_request_cannot_select_provider_transport_or_secret() -> None:
    names = {item.name for item in fields(ModelRequest)}

    assert names == {
        "continuation",
        "disable_reasoning",
        "messages",
        "tools",
        "max_output_tokens",
        "system_prompt",
        "tool_choice",
        "prompt_cache_key",
        "on_delta",
    }
    assert names.isdisjoint(
        {
            "model",
            "provider",
            "base_url",
            "api_key",
            "extra_body",
            "transport",
        }
    )


def test_model_response_exposes_only_opaque_continuation() -> None:
    names = {item.name for item in fields(LLMResponse)}

    assert "continuation" in names
    assert "provider_fields" not in names


def test_continuation_and_driver_config_are_deeply_immutable() -> None:
    from agent.plugin_composition import DriverConnectionDescriptor

    source = {"nested": {"items": ["one"]}}
    descriptor = DriverConnectionDescriptor(
        connection_id="connection",
        name="Connection",
        driver_id="driver",
        endpoint="https://example.test",
        auth_identity="account",
        config=source,
    )
    continuation = ModelContinuation(binding_id="binding", payload=source)

    source["nested"]["items"].append("two")  # type: ignore[index, union-attr]
    assert descriptor.config["nested"]["items"] == ("one",)  # type: ignore[index]
    assert continuation.payload["nested"]["items"] == ("one",)  # type: ignore[index]
    with pytest.raises(TypeError):
        descriptor.config["nested"]["new"] = True  # type: ignore[index]


def test_immutable_json_rejects_non_finite_numbers_and_cycles() -> None:
    with pytest.raises(ValueError, match="有限值"):
        ModelContinuation(binding_id="binding", payload={"value": float("nan")})

    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    with pytest.raises(ValueError, match="循环引用"):
        ModelContinuation(binding_id="binding", payload=cycle)


def test_catalog_snapshot_copies_role_bindings() -> None:
    bindings = {ModelRole.DEFAULT: "chat"}
    snapshot = ModelCatalogSnapshot(
        revision=1,
        connections=(),
        models=(),
        role_bindings=bindings,
        default_embedding_model_id=None,
    )

    bindings[ModelRole.DEFAULT] = "changed"
    assert snapshot.role_bindings[ModelRole.DEFAULT] == "chat"
    with pytest.raises(TypeError):
        snapshot.role_bindings[ModelRole.DEFAULT] = "forbidden"  # type: ignore[index]


def test_model_errors_expose_retry_contract() -> None:
    assert RateLimitError.retryable is True
    assert AuthenticationError.retryable is False
    from agent.plugin_composition import RevisionConflictError

    assert RevisionConflictError.retryable is False


@pytest.mark.asyncio
async def test_chat_model_fake_preserves_role_and_rejects_foreign_continuation() -> None:
    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, **kwargs: object) -> LLMResponse:
            del kwargs
            self.calls += 1
            return LLMResponse(content="unexpected")

    provider = Provider()
    chat_models = build_test_chat_models(provider)
    async with chat_models.execution() as execution:
        assert execution.chat(ModelRole.FAST).descriptor.role is ModelRole.FAST
        assert execution.chat(ModelRole.VISION).descriptor.role is ModelRole.VISION
        model = execution.chat(ModelRole.DEFAULT)
        with pytest.raises(ModelUnavailableError):
            await model.complete(
                ModelRequest(
                    messages=(),
                    continuation=ModelContinuation(
                        binding_id="foreign-binding",
                        payload={},
                    ),
                )
            )
    assert provider.calls == 0


def test_embedding_identity_changes_with_connection_and_capabilities() -> None:
    base = EmbeddingSpaceDescriptor(
        plugin_snapshot_id="snapshot",
        model_revision=1,
        model_id="embedding",
        connection_id="connection",
        driver_id="driver",
        driver_contract_version="1",
        auth_identity="account",
        connection_fingerprint="endpoint-a",
        model="wire-model",
        dimensions=3,
        normalization="none",
        capability_digest="caps-a",
    )
    first = base

    assert first.identity != replace(
        base, connection_fingerprint="endpoint-b"
    ).identity
    assert first.identity != replace(base, capability_digest="caps-b").identity
