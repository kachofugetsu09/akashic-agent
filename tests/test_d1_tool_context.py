from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent.looping.core import AgentLoop
from agent.model_runtime.types import ToolCall
from agent.mcp.client import McpToolInfo
from agent.mcp.tool import McpToolWrapper
from agent.provider import LLMResponse
from agent.subagent import SubAgent
from agent.tools.base import Tool, get_current_tool_context
from agent.tools.memorize import MemorizeTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from bus.events import ChannelMessage, DeliveryReceipt
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_fetch_spill import SpillCleanup
from core.memory.engine import (
    MemoryMutationResult,
    MemoryQueryResult,
    MemoryToolSpec,
)
from infra.channels.delivery import deliver_message_parts


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


class _ExecutionProbe(Tool):
    name = "execution_probe"
    description = "测试 execution owner"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    async def execute(self, value: str, **_: Any) -> str:
        context = get_current_tool_context()
        assert context is not None
        return f"{value}:{context.execution_id}:{context.turn_id}"


class _SubagentProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **_: Any) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[ToolCall("c1", "execution_probe", {"value": "x"})],
            )
        return LLMResponse(content="done", tool_calls=[])


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
async def test_registry_generates_execution_owner_per_call_and_ignores_overrides():
    registry = ToolRegistry()
    registry.register(_ExecutionProbe(), always_on=True)
    registry.set_context(channel="mobile", chat_id="conversation-a", turn_id="turn-1")

    first = await registry.execute(
        "execution_probe",
        {"value": "x"},
        internal_arguments={"execution_id": "model-forged"},
    )
    second = await registry.execute("execution_probe", {"value": "x"})

    first_id = str(first).split(":", 1)[1].rsplit(":", 1)[0]
    second_id = str(second).split(":", 1)[1].rsplit(":", 1)[0]
    assert first_id.startswith("operation:")
    assert second_id.startswith("operation:")
    assert first_id != second_id
    assert str(first).endswith(":turn-1")
    assert str(second).endswith(":turn-1")


@pytest.mark.asyncio
async def test_web_fetch_context_provider_rejects_untyped_result():
    from agent.tools.web_fetch import WebFetchTool

    class _Requester:
        async def get(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("context validation should happen first")

    tool = WebFetchTool(
        requester=_Requester(),  # type: ignore[arg-type]
        context_provider=lambda: {"execution_id": "e", "turn_id": "t"},  # type: ignore[return-value]
    )

    with pytest.raises(TypeError, match="ToolExecutionContext"):
        await tool.execute(url="https://example.com")


@pytest.mark.asyncio
async def test_subagent_scopes_each_tool_call_and_releases_own_turn():
    provider = _SubagentProvider()
    probe = _ExecutionProbe()
    web_fetch = WebFetchTool(requester=object())  # type: ignore[arg-type]
    web_fetch.release_turn = MagicMock(return_value=[])
    subagent = SubAgent(
        provider=provider,  # type: ignore[arg-type]
        model="m",
        tools=[probe, web_fetch],
    )

    result = await subagent.run("inspect")

    assert result == "done"
    # The generated owner is visible only during the tool call and is not a model argument.
    assert provider.calls == 2
    web_fetch.release_turn.assert_called_once()
    released_turn = web_fetch.release_turn.call_args.args[0]
    assert str(released_turn).startswith("turn:")


@pytest.mark.asyncio
async def test_subagent_spill_cleanup_failure_does_not_rewrite_success():
    provider = _SubagentProvider()
    web_fetch = WebFetchTool(requester=object())  # type: ignore[arg-type]
    web_fetch.release_turn = MagicMock(side_effect=OSError("unlink denied"))
    subagent = SubAgent(
        provider=provider,  # type: ignore[arg-type]
        model="m",
        tools=[web_fetch],
    )

    assert await subagent.run("inspect") == "done"


def test_agent_loop_releases_web_fetch_turn_and_keeps_cleanup_owner(caplog):
    web_fetch = WebFetchTool(requester=object())  # type: ignore[arg-type]
    web_fetch.release_turn = MagicMock(
        return_value=[
            SpillCleanup(
                execution_id="operation:1",
                released=False,
                status="cleanup_degraded",
                path="/tmp/response.spill",
                error="permission denied",
            )
        ]
    )
    registry = ToolRegistry()
    registry.register(web_fetch, always_on=True)
    loop = object.__new__(AgentLoop)
    loop.tools = registry

    AgentLoop._cleanup_web_fetch_owner(loop, "turn:1")

    web_fetch.release_turn.assert_called_once_with("turn:1")
    assert "cleanup_degraded" in caplog.text
    assert "operation:1" in caplog.text


@pytest.mark.asyncio
async def test_message_push_uses_explicit_cross_channel_target():
    sent: list[tuple[str, str]] = []
    push = MessagePushTool()

    async def send(chat_id: str, message: str) -> None:
        sent.append((chat_id, message))

    async def send_file(_chat_id: str, _path: str, _name: str | None) -> None:
        raise AssertionError("text-only test should not send file")

    async def send_image(_chat_id: str, _path: str) -> None:
        raise AssertionError("text-only test should not send image")

    async def deliver(message: ChannelMessage) -> DeliveryReceipt:
        return await deliver_message_parts(
            message,
            send_text=send,
            send_file=send_file,
            send_image=send_image,
        )

    push.register_channel("telegram", deliver=deliver)
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
    assert result == "消息已发送"
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

    def reinforce_items_batch(self, ids: list[str]) -> None:
        _ = ids
        return None


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
