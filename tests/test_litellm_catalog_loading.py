"""模型目录首次加载 LiteLLM 注册表时不占用事件循环线程。"""
import asyncio
import threading
from pathlib import Path

import httpx
import pytest

from plugins.models import litellm_capabilities
from plugins.models.litellm_catalog import LiteLlmCapabilityCatalog

_BUNDLED = {"gpt-test": {"litellm_provider": "openai", "max_input_tokens": 1000, "supports_vision": True}}


def test_first_enrich_loads_registries_off_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    threads: list[str] = []

    def registry() -> dict[str, dict[str, object]]:
        threads.append(threading.current_thread().name)
        return dict(_BUNDLED)

    monkeypatch.setattr(litellm_capabilities, "_registry", registry)
    litellm_capabilities._known_provider_ids.cache_clear()
    catalog = LiteLlmCapabilityCatalog(
        tmp_path / "catalog.json", writable=False, bundled_models=_BUNDLED,
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
    )

    async def run() -> str:
        _ = await catalog.enrich((), provider_id="openai")
        return threading.current_thread().name

    loop_thread = asyncio.run(run())
    litellm_capabilities._known_provider_ids.cache_clear()
    assert threads and loop_thread not in threads


def test_failed_registry_load_is_not_cached(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    def broken() -> dict[str, dict[str, object]]:
        calls.append(1)
        raise ImportError("litellm missing")

    monkeypatch.setattr(litellm_capabilities, "_registry", broken)
    litellm_capabilities._known_provider_ids.cache_clear()
    for _ in range(2):
        with pytest.raises(ImportError, match="litellm missing"):
            litellm_capabilities.load_catalog_registries()
    litellm_capabilities._known_provider_ids.cache_clear()
    assert len(calls) == 2
