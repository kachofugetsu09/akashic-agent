from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig, MemoryServices
from agent.config_models import ContextCompactionConfig
from agent.persona import reset_veda
from agent.provider import LLMResponse
from agent.tools.registry import ToolRegistry
from bus.events import SpawnCompletionItem
from bus.internal_events import SpawnCompletionEvent
from bus.queue import MessageBus
from tests.memory_fakes import FakeMemoryEngine
from tests.provider_fakes import ProviderContextBudgetStub
from tests.test_session_compaction_runtime import _MarkdownCompactionProbe
from session.manager import SessionManager
from core.memory.markdown import (
    MarkdownMemoryMaintenance,
    MarkdownMemoryRuntime,
    MarkdownMemoryStore,
)
from core.memory.runtime import MemoryRuntime


def _memory_runtime(
    workspace: Path,
    engine: FakeMemoryEngine,
    maintenance: _MarkdownCompactionProbe,
) -> MemoryRuntime:
    """Build the real runtime envelope around the narrow Markdown probe."""

    return MemoryRuntime(
        engine=engine,
        markdown=MarkdownMemoryRuntime(
            store=MarkdownMemoryStore(workspace),
            maintenance=cast(MarkdownMemoryMaintenance, maintenance),
        ),
    )


class _Provider(ProviderContextBudgetStub):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content="我已经整理完后台结果，结论如下。", tool_calls=[])


class _CompactingProvider(_Provider):
    context_window = 128
    runtime_id = "spawn-agent"

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        messages = kwargs.get("messages") or []
        first_content = str(messages[0].get("content", "")) if messages else ""
        if "Closed history to consolidate" in first_content:
            from agent.model_runtime.context_compaction import SUMMARY_HEADINGS

            return LLMResponse(content="\n".join(SUMMARY_HEADINGS), tool_calls=[])
        return LLMResponse(content="我已经整理完后台结果，结论如下。", tool_calls=[])

    def estimate_context_tokens(self, messages, tools):
        if any("<session-context-compaction>" in str(message.get("content", "")) for message in messages):
            return 5
        if any("Closed history to consolidate" in str(message.get("content", "")) for message in messages):
            return 100
        return 100 if messages else 0


@pytest.mark.asyncio
async def test_spawn_completion_updates_original_session_without_raw_result(tmp_path):
    _ = reset_veda(tmp_path)
    provider = _Provider()
    session_manager = SessionManager(tmp_path)
    tools = ToolRegistry()
    engine = FakeMemoryEngine(tmp_path)
    markdown = _MarkdownCompactionProbe()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=session_manager,
            workspace=tmp_path,
            memory_services=MemoryServices(engine=engine),
            memory_runtime=_memory_runtime(tmp_path, engine, markdown),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=3)),
    )

    session = session_manager.get_or_create("telegram:123")
    session.add_message("user", "帮我整理一下")
    session.add_message("assistant", "我开始处理了")
    session_manager.save(session)

    item = SpawnCompletionItem(
        channel="telegram",
        chat_id="123",
        event=SpawnCompletionEvent(
            job_id="abcd1234",
            label="整理任务",
            task="整理资料",
            status="incomplete",
            exit_reason="forced_summary",
            result="原始后台结果：文件位于 /tmp/report.md",
        ),
    )

    response = await loop._process(item)
    updated = session_manager.get_or_create("telegram:123")
    execution_context = tools.get_execution_context()

    assert response.channel == "telegram"
    assert response.chat_id == "123"
    assert execution_context is not None
    assert execution_context.turn_id.startswith("turn:")
    assert "整理" in response.content
    assert updated.messages[-1]["content"] == "我已经整理完后台结果，结论如下。"
    assert all(
        not str(m["content"]).startswith("[后台任务完成]")
        for m in updated.messages
    )
    assert all(
        m["content"] != "原始后台结果：文件位于 /tmp/report.md"
        for m in updated.messages
    )

@pytest.mark.asyncio
async def test_spawn_completion_retry_count_one_disables_retry_guidance(tmp_path):
    _ = reset_veda(tmp_path)
    provider = _Provider()
    session_manager = SessionManager(tmp_path)
    tools = ToolRegistry()
    engine = FakeMemoryEngine(tmp_path)
    markdown = _MarkdownCompactionProbe()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=session_manager,
            workspace=tmp_path,
            memory_services=MemoryServices(engine=engine),
            memory_runtime=_memory_runtime(tmp_path, engine, markdown),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=3)),
    )

    session = session_manager.get_or_create("telegram:123")
    session.add_message("user", "帮我补跑一下")
    session_manager.save(session)

    item = SpawnCompletionItem(
        channel="telegram",
        chat_id="123",
        event=SpawnCompletionEvent(
            job_id="abcd1234",
            label="补跑任务",
            task="继续整理资料",
            status="incomplete",
            exit_reason="max_iterations",
            result="还差一点",
            retry_count=1,
        ),
    )

    await loop._process(item)

    joined_messages = "\n".join(
        str(message.get("content", ""))
        for call in provider.calls
        for message in call.get("messages", [])
    )
    assert "已重试一次，不再重试" in joined_messages
    assert "调用 spawn 重试" not in joined_messages


@pytest.mark.asyncio
async def test_spawn_completion_uses_session_compaction_gate(tmp_path):
    _ = reset_veda(tmp_path)
    provider = _CompactingProvider()
    session_manager = SessionManager(tmp_path)
    tools = ToolRegistry()
    engine = FakeMemoryEngine(tmp_path)
    markdown = _MarkdownCompactionProbe()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MessageBus(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=session_manager,
            workspace=tmp_path,
            memory_services=MemoryServices(engine=engine),
            memory_runtime=_memory_runtime(tmp_path, engine, markdown),
        ),
        AgentLoopConfig(
            llm=LLMConfig(max_iterations=3),
            context_compaction=ContextCompactionConfig(
                keep_recent_tokens=1,
            ),
        ),
    )
    session = session_manager.get_or_create("telegram:123")
    session.add_message("user", "old one", control_turn_id="turn-1")
    session.add_message("assistant", "old two", control_turn_id="turn-1")
    session.add_message("user", "older three", control_turn_id="turn-2")
    session.add_message("assistant", "older four", control_turn_id="turn-2")
    session_manager.save(session)

    await loop._process(
        SpawnCompletionItem(
            channel="telegram",
            chat_id="123",
            event=SpawnCompletionEvent(
                job_id="abcd1234",
                label="整理任务",
                task="整理资料",
                status="completed",
                exit_reason="completed",
                result="后台结果",
            ),
        )
    )

    assert len(provider.calls) >= 2
    assert "Closed history to consolidate" in str(provider.calls[0]["messages"])
    assert "<session-context-compaction>" in str(provider.calls[-1]["messages"])
    assert markdown.commit_count == 1
    active = session_manager._store.get_active_compaction("telegram:123")
    assert active is not None
    assert active.generation == 1
