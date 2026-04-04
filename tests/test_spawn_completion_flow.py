from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.memory import MemoryStore
from agent.provider import LLMResponse
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage
from core.memory.port import DefaultMemoryPort
from session.manager import SessionManager


class _Provider:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content="我已经整理完后台结果，结论如下。", tool_calls=[])


@pytest.mark.asyncio
async def test_spawn_completion_updates_original_session_without_raw_result(tmp_path):
    provider = _Provider()
    session_manager = SessionManager(tmp_path)
    tools = ToolRegistry()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=session_manager,
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=3)),
    )
    loop._post_mem_worker = MagicMock()
    loop._post_mem_worker.run = AsyncMock()

    session = session_manager.get_or_create("telegram:123")
    session.add_message("user", "帮我整理一下")
    session.add_message("assistant", "我开始处理了")
    session_manager.save(session)

    msg = InboundMessage(
        channel="telegram",
        sender="spawn",
        chat_id="123",
        content="[internal spawn completed]",
        metadata={
            "internal_event": "spawn_completed",
            "spawn": {
                "job_id": "abcd1234",
                "label": "整理任务",
                "task": "整理资料",
                "status": "incomplete",
                "exit_reason": "forced_summary",
                "result": "原始后台结果：文件位于 /tmp/report.md",
            },
        },
    )

    response = await loop._process(msg)
    updated = session_manager.get_or_create("telegram:123")

    assert response.channel == "telegram"
    assert response.chat_id == "123"
    assert "整理" in response.content
    assert (
        updated.messages[-2]["content"]
        == "[后台任务完成] 整理任务 (incomplete) [forced_summary]"
    )
    assert updated.messages[-1]["content"] == "我已经整理完后台结果，结论如下。"
    assert all(
        m["content"] != "原始后台结果：文件位于 /tmp/report.md"
        for m in updated.messages
    )
    loop._post_mem_worker.run.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_completion_retry_count_one_disables_retry_guidance(tmp_path):
    provider = _Provider()
    session_manager = SessionManager(tmp_path)
    tools = ToolRegistry()
    loop = AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=session_manager,
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=3)),
    )
    loop._post_mem_worker = MagicMock()
    loop._post_mem_worker.run = AsyncMock()

    session = session_manager.get_or_create("telegram:123")
    session.add_message("user", "帮我补跑一下")
    session_manager.save(session)

    msg = InboundMessage(
        channel="telegram",
        sender="spawn",
        chat_id="123",
        content="[internal spawn completed]",
        metadata={
            "internal_event": "spawn_completed",
            "spawn": {
                "job_id": "abcd1234",
                "label": "补跑任务",
                "task": "继续整理资料",
                "status": "incomplete",
                "exit_reason": "max_iterations",
                "result": "还差一点",
                "retry_count": 1,
            },
        },
    )

    await loop._process(msg)

    joined_messages = "\n".join(
        str(message.get("content", ""))
        for call in provider.calls
        for message in call.get("messages", [])
    )
    assert "已重试一次，不再重试" in joined_messages
    assert "调用 spawn 重试" not in joined_messages
