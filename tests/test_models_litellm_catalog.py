from __future__ import annotations

from pathlib import Path

import pytest

from agent.plugin_composition.models import (
    CapabilitySources,
    DiscoveredModel,
    ModelCapabilities,
    ModelKind,
)
from plugins.models.litellm_capabilities import resolve_catalog_capabilities
from plugins.models.litellm_catalog import LiteLlmCapabilityCatalog


def test_litellm_capabilities_are_resolved_by_the_models_plugin() -> None:
    capabilities = resolve_catalog_capabilities(
        "dashscope",
        "Qwen-Max",
        models={
            "dashscope/qwen-max": {
                "max_input_tokens": 128_000,
                "max_output_tokens": 8_192,
                "supported_modalities": ["text", "image"],
                "supports_reasoning": True,
                "supports_function_calling": True,
                "supports_parallel_function_calling": False,
                "supports_low_reasoning_effort": True,
            }
        },
    )

    assert capabilities is not None
    assert capabilities.context_window == 128_000
    assert capabilities.max_output_tokens == 8_192
    assert capabilities.input_modalities == ("text", "image")
    assert capabilities.supported_reasoning_efforts == (
        "low",
        "medium",
        "high",
    )
    assert capabilities.supports_parallel_tool_calls is False


@pytest.mark.asyncio
async def test_models_catalog_enriches_provider_discovery_from_its_local_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = LiteLlmCapabilityCatalog(
        tmp_path / "litellm-capabilities.json",
        writable=False,
        bundled_models={
            "dashscope/qwen-max": {
                "max_input_tokens": 64_000,
                "max_output_tokens": 4_096,
                "supports_vision": True,
            }
        },
    )

    async def no_remote(*, etag: str, minimum_entries: int):
        del etag, minimum_entries
        return None

    monkeypatch.setattr(catalog, "_fetch_remote", no_remote)
    discovered = DiscoveredModel(
        kind=ModelKind.CHAT,
        model="Qwen-Max",
        capabilities=ModelCapabilities(),
        capability_sources=CapabilitySources(),
    )

    enriched = await catalog.enrich((discovered,), provider_id="dashscope")

    assert enriched[0].capabilities.context_window == 64_000
    assert enriched[0].capabilities.input_modalities == ("text", "image")
    assert enriched[0].capability_sources.context_window.startswith("litellm-")
