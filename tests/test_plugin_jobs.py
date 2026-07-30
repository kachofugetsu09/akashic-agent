from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.plugins.jobs import ProviderPluginLlmService


@pytest.mark.asyncio
async def test_plugin_job_distinguishes_default_and_explicit_zero_limit() -> None:
    provider = AsyncMock()
    provider.chat.return_value = SimpleNamespace(content="done")
    service = ProviderPluginLlmService(
        provider,
        model="main",
        max_tokens=256,
    )

    assert await service.generate_text(prompt="bounded") == "done"
    assert provider.chat.await_args.kwargs["max_tokens"] == 256

    assert await service.generate_text(prompt="provider", max_tokens=0) == "done"
    assert provider.chat.await_args.kwargs["max_tokens"] == 0
