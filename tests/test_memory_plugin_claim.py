from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent.plugin_composition import (
    CompositionRoot,
    EMBEDDING_MEMORY_PLUGIN,
    TextEmbeddingSettings,
)
from agent.lifecycle.types import PromptRenderCtx
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from core.memory.engine import MemoryQueryResult
from plugins.akasha.plugin import _inject_memory
from session.manager import SessionManager


class _QueryRuntimeStub:
    def __init__(self, result: MemoryQueryResult | None = None) -> None:
        self.query = AsyncMock(return_value=result)


@pytest.mark.asyncio
async def test_memory_plugin_claim_is_declarative_and_exclusive() -> None:
    root = CompositionRoot("memory-claim")

    async def first_memory(ctx) -> None:
        _ = await ctx.provide(EMBEDDING_MEMORY_PLUGIN, object())

    await root.mount(first_memory, name="first-memory")

    async def second_memory(ctx) -> None:
        _ = await ctx.provide(EMBEDDING_MEMORY_PLUGIN, object())

    await root.mount(second_memory, name="akasha")

    receipt = root.receipt()
    assert receipt.ready is False
    assert receipt.required_pending == ("akasha",)
    assert any(
        incident.owner == "akasha"
        and "DUPLICATE_SERVICE" in incident.message
        and "plugin.claim.embedding_memory" in incident.message
        for incident in receipt.incidents
    )
    await root.dispose()


def _manager(
    tmp_path: Path,
    *plugin_names: str,
) -> tuple[PluginManager, SessionManager]:
    plugin_root = Path(__file__).resolve().parents[1] / "plugins"
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace)
    return (
        PluginManager(
            [plugin_root / name for name in plugin_names],
            event_bus=EventBus(),
            workspace=workspace,
            session_manager=sessions,
            text_embedding_settings=TextEmbeddingSettings(
                base_url="http://127.0.0.1:9",
                api_key="",
                model="fixture",
                output_dimensionality=32,
            ),
            installed_cache_root=tmp_path / "plugin-home" / "cache",
        ),
        sessions,
    )


@pytest.mark.asyncio
async def test_akasha_starts_as_an_ordinary_memory_provider(tmp_path: Path) -> None:
    manager, sessions = _manager(tmp_path, "akasha")
    try:
        await manager.load_all()
        assert {item.plugin_id for item in manager.active_plugins()} == {"akasha"}
        assert manager.current_snapshot is not None
        topology = manager.current_snapshot.composition_topology
        assert topology is not None
        assert "plugin.claim.embedding_memory" in topology.services
    finally:
        await manager.terminate_all()
        sessions.close()


@pytest.mark.asyncio
async def test_akasha_injects_recall_as_an_ordinary_prompt_section() -> None:
    runtime = _QueryRuntimeStub(
        MemoryQueryResult(
            text_block="embedded recall",
            records=[],
            raw={},
        )
    )
    event = PromptRenderCtx(
        session_key="web:one",
        channel="web",
        chat_id="one",
        content="hello",
        media=None,
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        history=[],
        skill_names=[],
        disabled_sections=set(),
        turn_injection_prompt="",
    )

    await _inject_memory(event, runtime)

    assert [(item.name, item.content) for item in event.system_sections_bottom] == [
        ("memory", "embedded recall")
    ]


@pytest.mark.asyncio
async def test_akasha_prompt_section_obeys_generic_disable_switch() -> None:
    runtime = _QueryRuntimeStub()
    event = PromptRenderCtx(
        session_key="scheduler:one",
        channel="scheduler",
        chat_id="one",
        content="tick",
        media=None,
        timestamp=datetime(2026, 8, 25, tzinfo=UTC),
        history=[],
        skill_names=[],
        disabled_sections={"memory"},
        turn_injection_prompt="",
    )

    await _inject_memory(event, runtime)

    runtime.query.assert_not_awaited()
    assert event.system_sections_bottom == []
