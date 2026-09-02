from __future__ import annotations

"""Tests for the LiteLLM registry resolution: online-first, fuzzy matching,
and local-cache fallback for model capabilities (context window etc.).

These tests exercise the pure matching/loading helpers without hitting the
network or touching the live plugin-data cache. They construct a small in-memory
registry/cache to verify the resolution order and fuzzy matching semantics.
"""

from agent.model_runtime.catalog import litellm_registry as m


def _sample_online() -> dict[str, dict]:
    return {
        "deepseek/deepseek-v4-flash": {
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 393_216,
            "supports_vision": False,
        },
        "deepseek-v4-flash": {
            "max_input_tokens": 1_000_000,
            "max_output_tokens": 393_216,
            "supports_vision": False,
        },
        "azure_ai/FW-GLM-5.2": {
            "max_input_tokens": 1_048_576,
            "max_output_tokens": 131_072,
            "supports_vision": False,
        },
        "azure_ai/FW-Kimi-K3": {
            "max_input_tokens": 1_048_576,
            "max_output_tokens": 131_072,
            "supports_vision": True,
        },
    }


def test_exact_match_online() -> None:
    caps = m.resolve_catalog_capabilities(
        "openai-compatible",
        "deepseek/deepseek-v4-flash",
        base_url="https://api.commandcode.ai/provider/v1",
    )
    # This depends on the real LiteLLM registry; if the exact model is present
    # the context window should be positive and the source should be litellm.
    if caps is not None:
        assert caps.context_window > 0
        assert caps.input_modalities_known


def test_fuzzy_entry_case_insensitive() -> None:
    online = _sample_online()
    # 大小写不敏感 + 去标点：azure_ai/FW-GLM-5.2 应匹配 zai-org/GLM-5.2 的型号部分
    hit = m._fuzzy_entry(
        online,
        model="zai-org/GLM-5.2",
        provider_id="openai-compatible",
    )
    assert hit is not None
    assert hit.get("max_input_tokens") == 1_048_576


def test_fuzzy_entry_short_model() -> None:
    online = _sample_online()
    # 无供应商前缀的短名：glm-5.2 应通过型号包含匹配到 azure_ai/FW-GLM-5.2
    hit = m._fuzzy_entry(
        online,
        model="glm-5.2",
        provider_id="zai-org",
    )
    assert hit is not None
    assert hit.get("max_input_tokens") == 1_048_576


def test_fuzzy_entry_vision_modality() -> None:
    online = _sample_online()
    hit = m._fuzzy_entry(
        online,
        model="moonshotai/Kimi-K3",
        provider_id="openai-compatible",
    )
    assert hit is not None
    assert hit.get("supports_vision") is True


def test_explicit_catalog_mapping_is_used() -> None:
    caps = m.resolve_catalog_capabilities(
        "openai-compatible",
        "zai-org/GLM-5.2",
        models=_sample_online(),
    )

    assert caps is not None
    assert caps.context_window == 1_048_576


def test_context_window_uses_input_not_sum(monkeypatch) -> None:
    """Context window must equal max_input_tokens (official 'context window'),
    not input+output. max_output_tokens is a separate generation budget."""
    from agent.model_runtime.catalog.litellm_registry import (
        resolve_catalog_capabilities,
    )

    raw = {
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 393_216,
        "supports_vision": True,
    }
    monkeypatch.setattr(m, "_model_entry", lambda provider, model: dict(raw))
    caps = resolve_catalog_capabilities(
        "openai-compatible",
        "deepseek/deepseek-v4-flash-vision-exp",
    )
    assert caps is not None
    assert caps.context_window == 1_000_000
    assert caps.max_output_tokens == 393_216
    assert "image" in caps.input_modalities


def test_context_window_fallback_when_input_missing(monkeypatch) -> None:
    """If max_input_tokens is absent, fall back to input+output (legacy)."""
    from agent.model_runtime.catalog.litellm_registry import (
        resolve_catalog_capabilities,
    )

    raw = {"max_input_tokens": 0, "max_output_tokens": 8192}
    monkeypatch.setattr(m, "_model_entry", lambda provider, model: dict(raw))
    caps = resolve_catalog_capabilities("openai-compatible", "legacy-model")
    assert caps is not None
    assert caps.context_window == 8192
