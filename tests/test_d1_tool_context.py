from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from unittest.mock import AsyncMock

from agent.mcp.client import McpToolInfo
from agent.mcp.tool import McpToolWrapper
from agent.tools.base import Tool, get_current_tool_context
from agent.tools.memorize import MemorizeTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from core.memory.engine import (
    MemoryMutationResult,
    MemoryQueryResult,
    MemoryToolSpec,
)


class _ContextProbe(Tool):
    name = "context_probe"
    description = "测试工具上下文"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    async def execute(self, value: str) -> str:
        context = get_current_tool_context()
        origin = context.origin_channel if context else ""
        chat_id = context.origin_chat_id if context else ""
        await asyncio.sleep(0)
        return f"{value}:{origin}:{chat_id}"


@pytest.mark.asyncio
async def test_registry_rejects_model_origin_override_and_isolates_concurrent_turns():
    registry = ToolRegistry()
    registry.register(_ContextProbe(), always_on=True)

    registry.set_context(channel="mobile", chat_id="conversation-a")
    denied = await registry.execute(
        "context_probe",
        {"value": "x", "channel": "telegram"},
    )
    assert "不允许额外字段" in str(denied)

    async def run(channel: str, chat_id: str) -> str | Any:
        registry.set_context(channel=channel, chat_id=chat_id)
        return await registry.execute("context_probe", {"value": "ok"})

    assert await asyncio.gather(
        run("mobile", "conversation-a"),
        run("telegram", "conversation-b"),
    ) == ["ok:mobile:conversation-a", "ok:telegram:conversation-b"]


@pytest.mark.asyncio
async def test_message_push_uses_explicit_cross_channel_target():
    sent: list[tuple[str, str]] = []
    push = MessagePushTool()

    async def send(chat_id: str, message: str) -> None:
        sent.append((chat_id, message))

    push.register_channel("telegram", text=send)
    registry = ToolRegistry()
    registry.register(push, always_on=True, risk="external-side-effect")
    registry.set_context(channel="mobile", chat_id="conversation-a")

    result = await registry.execute(
        "message_push",
        {
            "target_channel": "telegram",
            "target_chat_id": "conversation-b",
            "message": "hello",
        },
    )
    assert result == "文本已发送"
    assert sent == [("conversation-b", "hello")]


@dataclass
class _MemoryCapture:
    mutation: Any = None

    async def mutate(self, request: Any) -> MemoryMutationResult:
        self.mutation = request
        return MemoryMutationResult(
            accepted=True,
            item_id="memory-1",
            actual_kind="event",
            status="new",
        )

    async def query(self, _request: Any) -> MemoryQueryResult:
        return MemoryQueryResult()


@pytest.mark.asyncio
async def test_memorize_reads_runtime_provenance_not_model_fields():
    memory = _MemoryCapture()
    spec = MemoryToolSpec(
        description="memorize",
        parameters={
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    )
    registry = ToolRegistry()
    registry.register(MemorizeTool(memory, spec), always_on=True, risk="write")
    registry.set_context(
        channel="mobile",
        chat_id="conversation-a",
        session_key="mobile:conversation-a",
        current_user_source_ref="mobile:conversation-a:message-1",
    )

    denied = await registry.execute(
        "memorize",
        {"summary": "fact", "chat_id": "other-chat"},
    )
    assert "不允许额外字段" in str(denied)
    await registry.execute("memorize", {"summary": "fact"})
    assert memory.mutation.scope.channel == "mobile"
    assert memory.mutation.scope.chat_id == "conversation-a"
    assert memory.mutation.source_ref == "mobile:conversation-a:message-1"


def test_registry_fork_does_not_clone_mutable_context_storage():
    registry = ToolRegistry()
    registry.set_context(channel="mobile", chat_id="conversation-a")
    fork = registry.fork()
    assert fork.get_execution_context() == registry.get_execution_context()


def test_set_context_rejects_mixed_legacy_and_origin_aliases():
    registry = ToolRegistry()

    with pytest.raises(TypeError, match="不能同时使用"):
        registry.set_context(
            channel="mobile",
            origin_channel="telegram",
        )


@pytest.mark.asyncio
async def test_execute_does_not_accept_context_override():
    registry = ToolRegistry()
    registry.register(_ContextProbe(), always_on=True)
    registry.set_context(channel="mobile", chat_id="conversation-a")

    with pytest.raises(TypeError, match="context"):
        await registry.execute(
            "context_probe",
            {"value": "x"},
            context=None,  # type: ignore[call-arg]
        )


@pytest.mark.asyncio
async def test_mcp_schema_omitted_additional_properties_remains_open():
    client = AsyncMock()
    client.name = "calendar"
    client.call.return_value = "ok"
    wrapper = McpToolWrapper(
        client,
        McpToolInfo(
            name="create_event",
            description="create event",
            input_schema={"type": "object", "properties": {"title": {}}},
        ),
    )
    registry = ToolRegistry()
    registry.register(wrapper, source_type="mcp", source_name="calendar")

    schema = registry.get_schemas()[0]["function"]["parameters"]
    assert schema["additionalProperties"] is True
    result = await registry.execute(
        "mcp_calendar__create_event",
        {"title": "demo", "timezone": "Asia/Shanghai"},
    )

    assert result == "ok"
    client.call.assert_awaited_once_with(
        "create_event",
        {"title": "demo", "timezone": "Asia/Shanghai"},
        timeout=None,
    )


@pytest.mark.asyncio
async def test_mcp_schema_explicit_false_rejects_unknown_fields():
    client = AsyncMock()
    client.name = "calendar"
    wrapper = McpToolWrapper(
        client,
        McpToolInfo(
            name="create_event",
            description="create event",
            input_schema={
                "type": "object",
                "properties": {"title": {}},
                "additionalProperties": False,
            },
        ),
    )
    registry = ToolRegistry()
    registry.register(wrapper, source_type="mcp", source_name="calendar")

    schema = registry.get_schemas()[0]["function"]["parameters"]
    assert schema["additionalProperties"] is False
    result = await registry.execute(
        "mcp_calendar__create_event",
        {"title": "demo", "timezone": "Asia/Shanghai"},
    )

    assert "不允许额外字段" in str(result)
    client.call.assert_not_awaited()
