from __future__ import annotations

from dataclasses import dataclass

from agent.plugin_composition.model import ServiceKey


@dataclass(frozen=True, slots=True)
class TextEmbeddingSettings:
    """Describe one configured text-embedding endpoint for plugin consumers."""

    base_url: str
    api_key: str
    model: str
    output_dimensionality: int | None


TEXT_EMBEDDING_SETTINGS = ServiceKey[TextEmbeddingSettings](
    "core.text_embedding.settings"
)
