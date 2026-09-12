"""Small frozen model capability records needed by legacy migration code."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_modalities: tuple[str, ...] = ("text",)
    supports_tool_calls: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supported_reasoning_efforts: tuple[str, ...] = ()
    embedding_dimensions: int | None = None
    embedding_normalization: str | None = None


@dataclass(frozen=True, slots=True)
class CapabilitySources:
    context_window: str = "unknown"
    max_output_tokens: str = "unknown"
    input_modalities: str = "unknown"
    tool_calls: str = "unknown"
    parallel_tool_calls: str = "unknown"
    reasoning_efforts: str = "unknown"
    embedding_dimensions: str = "unknown"
    embedding_normalization: str = "unknown"


@dataclass(frozen=True, slots=True)
class EmbeddingSpaceDescriptor:
    plugin_snapshot_id: str
    model_revision: int
    model_id: str
    connection_id: str
    driver_id: str
    driver_contract_version: str
    auth_identity: str
    connection_fingerprint: str
    model: str
    dimensions: int
    normalization: str
    capability_digest: str
    schema_version: int = 1

    @property
    def identity(self) -> str:
        return ":".join((
            self.driver_id, self.driver_contract_version, self.connection_id,
            self.auth_identity, self.connection_fingerprint, self.model_id,
            str(self.dimensions), self.normalization, self.capability_digest,
            str(self.schema_version),
        ))
