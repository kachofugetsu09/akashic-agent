from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from agent.core.passive_support import collect_skill_mentions
from agent.core.passive_turn import DefaultReasoner
from prompts.agent import build_current_message_time_envelope
from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, MemoryServices
from agent.retrieval.protocol import MemoryRetrievalPipeline, RetrievalRequest
from bus.queue import MessageBus
from core.memory.runtime import MemoryRuntime
from tests.memory_fakes import FakeMemoryEngine


@pytest.mark.asyncio
async def test_memory_runtime_engine_is_not_masked_by_empty_services(
    tmp_path: Path,
) -> None:
    memory = FakeMemoryEngine(tmp_path)
    runtime = cast(
        MemoryRuntime,
        SimpleNamespace(
            engine=memory,
            markdown=SimpleNamespace(store=memory, maintenance=memory),
        ),
    )

    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=MagicMock(),
            tools=MagicMock(),
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_runtime=runtime,
            memory_services=MemoryServices(),
        ),
        AgentLoopConfig(),
    )

    memory.retrieve_result.text_block = "runtime memory"
    retrieval = cast(MemoryRetrievalPipeline, loop._retrieval_pipeline)
    result = await retrieval.retrieve(
        RetrievalRequest(
            message="测试",
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            history=[],
            session_metadata={},
        )
    )

    assert result.block == "runtime memory"


def test_collect_skill_mentions_returns_unique_existing_names(tmp_path):
    skills = [
        "feed-manage",
        "refactor",
    ]

    got = collect_skill_mentions(
        "请用 $feed-manage 然后 $refactor 再来一次 $feed-manage",
        skills,
    )

    assert got == ["feed-manage", "refactor"]


def test_collect_skill_mentions_ignores_unknown_skill(tmp_path):
    skills = ["known"]

    got = collect_skill_mentions("$known $unknown", skills)

    assert got == ["known"]


def test_format_request_time_anchor_contains_iso_and_label():
    text = DefaultReasoner.format_request_time_anchor(None)
    assert text.startswith("request_time=")
    assert "(" in text and ")" in text


def test_build_current_message_time_envelope_contains_today_and_tomorrow():
    message_timestamp = datetime.fromisoformat("2026-04-08T17:57:00+08:00")
    local_timestamp = message_timestamp.astimezone()

    text = build_current_message_time_envelope(message_timestamp=message_timestamp)

    assert f"当前消息时间: {local_timestamp:%Y-%m-%d %H:%M}" in text
    assert f"今天={local_timestamp:%Y-%m-%d}" in text
    assert f"明天={local_timestamp + timedelta(days=1):%Y-%m-%d}" in text
