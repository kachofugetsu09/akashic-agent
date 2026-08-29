from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent.plugin_composition import (
    AuthenticationError,
    BoundModelDescriptor,
    CredentialHandle,
    DiscoveredModel,
    DriverConnection,
    DriverConnectionDescriptor,
    DriverEmbeddingModel,
    EmbeddingSpaceDescriptor,
    ModelDriverDefinition,
    ModelUnavailableError,
)

from .auth import finish_auth, start_auth
from .catalog import discover as discover_catalog
from .catalog import probe as probe_catalog
from .responses import CodexResponses


@dataclass(frozen=True, slots=True)
class _ConnectionConfig:
    endpoint: str
    connect_timeout: float
    read_timeout: float


def definition() -> ModelDriverDefinition:
    """Build the Codex driver contribution."""

    return ModelDriverDefinition(
        driver_id="codex",
        contract_version="1",
        open=_open,
        discover=_discover,
        probe=_probe,
        start_auth=start_auth,
        finish_auth=finish_auth,
    )


async def _open(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> DriverConnection:
    config = _connection_config(descriptor)
    if credential.connection_id != descriptor.connection_id:
        raise AuthenticationError("credential connection scope does not match")
    if credential.auth_identity != descriptor.auth_identity:
        raise AuthenticationError("credential auth identity does not match")

    def bind_chat(
        model: BoundModelDescriptor,
        raw_config: Mapping[str, Any],
    ) -> CodexResponses:
        if model.driver_id != "codex" or model.connection_id != descriptor.connection_id:
            raise ModelUnavailableError("model does not belong to this Codex connection")
        return CodexResponses(
            endpoint=config.endpoint,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            credential=credential,
            descriptor=model,
            config=_model_config(raw_config),
        )

    def bind_embedding(
        model: EmbeddingSpaceDescriptor,
        raw_config: Mapping[str, Any],
    ) -> DriverEmbeddingModel:
        _ = model, raw_config
        raise ModelUnavailableError("Codex driver does not provide embeddings")

    return DriverConnection(bind_chat=bind_chat, bind_embedding=bind_embedding)


async def _discover(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> tuple[DiscoveredModel, ...]:
    _ = _connection_config(descriptor)
    return await discover_catalog(descriptor, credential)


async def _probe(
    descriptor: DriverConnectionDescriptor,
    credential: CredentialHandle,
) -> None:
    _ = _connection_config(descriptor)
    await probe_catalog(descriptor, credential)


def _connection_config(descriptor: DriverConnectionDescriptor) -> _ConnectionConfig:
    if descriptor.driver_id != "codex":
        raise ValueError(f"unexpected driver id: {descriptor.driver_id}")
    allowed = {
        "format_version",
        "connect_timeout",
        "read_timeout",
        "catalog_timeout",
        "catalog_provider_id",
    }
    unknown = sorted(set(descriptor.config) - allowed)
    if unknown:
        raise ValueError(f"unsupported Codex connection config: {', '.join(unknown)}")
    if descriptor.config.get("format_version", 1) != 1:
        raise ValueError("unsupported Codex connection config format")
    legacy_provider = descriptor.config.get("catalog_provider_id", "codex")
    if legacy_provider not in {"", "codex"}:
        raise ValueError("Codex catalog_provider_id must be codex or empty")
    return _ConnectionConfig(
        endpoint=descriptor.endpoint.rstrip("/"),
        connect_timeout=_positive_float(
            descriptor.config.get("connect_timeout", 30.0), "connect_timeout"
        ),
        read_timeout=_positive_float(
            descriptor.config.get("read_timeout", 120.0), "read_timeout"
        ),
    )


def _model_config(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    allowed = {
        "format_version",
        "use_responses_lite",
        "reasoning_summary",
        "max_tool_schemas",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unsupported Codex model config: {', '.join(unknown)}")
    if raw.get("format_version", 1) != 1:
        raise ValueError("unsupported Codex model config format")
    lite = raw.get("use_responses_lite", False)
    if not isinstance(lite, bool):
        raise ValueError("use_responses_lite must be boolean")
    summary = raw.get("reasoning_summary", "none")
    if summary not in {"none", "auto", "concise", "detailed"}:
        raise ValueError("unsupported reasoning_summary")
    limit = raw.get("max_tool_schemas")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("max_tool_schemas must be positive or null")
    return dict(raw)


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)
