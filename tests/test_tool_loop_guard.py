import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agent.looping.core import AgentLoop
from agent.memory import MemoryStore
from agent.provider import LLMResponse, ToolCall
from agent.procedure_hint import extract_action_tokens
from agent.subagent import SubAgent
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from core.memory.port import DefaultMemoryPort
from proactive.components import ProactiveMessageComposer


class _DummyTool(Tool):
    def __init__(self, name: str = "dummy") -> None:
        self._name = name
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "dummy tool"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
            },
            "required": ["x"],
        }

    async def execute(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return f"ok:{kwargs.get('x')}"


class _FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


def _assert_no_unresolved_tool_calls(messages: list[dict]) -> None:
    pending: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                call_id = tc.get("id")
                if call_id:
                    pending.add(call_id)
        elif m.get("role") == "tool":
            call_id = m.get("tool_call_id")
            if call_id in pending:
                pending.remove(call_id)
    if pending:
        raise AssertionError(
            f"unresolved tool_calls in message chain: {sorted(pending)}"
        )


class _StrictProvider(_FakeProvider):
    async def chat(self, **kwargs):
        messages = kwargs.get("messages") or []
        _assert_no_unresolved_tool_calls(messages)
        return await super().chat(**kwargs)


@pytest.fixture(autouse=True)
def _shared_http_resources():
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    try:
        yield
    finally:
        clear_default_shared_http_resources(resources)
        asyncio.run(resources.aclose())


class _ExitTool(Tool):
    def __init__(self, name: str = "task_note") -> None:
        self._name = name
        self.called = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "exit tool"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": [],
        }

    async def execute(self, **kwargs) -> str:
        self.called += 1
        return "noted"


class _HintMemory:
    def __init__(self, items: list[dict]) -> None:
        self._items = items
        self.calls: list[list[str]] = []

    def keyword_match_procedures(self, action_tokens: list[str]) -> list[dict]:
        self.calls.append(list(action_tokens))
        return list(self._items)


def _make_agent_loop(tmp_path: Path, provider: _FakeProvider, tool: Tool) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(tool)
    return AgentLoop(
        bus=MagicMock(),
        provider=cast(Any, provider),
        tools=tools,
        session_manager=MagicMock(),
        workspace=tmp_path,
        max_iterations=10,
        memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
    )


def test_agent_loop_breaks_on_repeated_same_signature_and_returns_summary(tmp_path):
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("c3", "dummy", {"x": 1})]),
            LLMResponse(
                content="已完成阶段A，剩余阶段B，下一步继续补齐", tool_calls=[]
            ),
        ]
    )
    loop = _make_agent_loop(tmp_path, provider, tool)

    final, tools_used, _, _vn = asyncio.run(
        loop._run_agent_loop([{"role": "user", "content": "test"}])
    )

    assert "最大迭代" not in final
    assert "下一步" in final
    # 第三次重复签名会被提前拦截，不应执行第三次工具
    assert len(tool.calls) == 2
    assert tools_used == ["dummy", "dummy"]


def test_agent_loop_does_not_false_positive_when_args_change(tmp_path):
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "dummy", {"x": 2})]),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    loop = _make_agent_loop(tmp_path, provider, tool)

    final, _, _, _vn = asyncio.run(
        loop._run_agent_loop([{"role": "user", "content": "test"}])
    )

    assert final == "done"
    assert len(tool.calls) == 2


def test_agent_loop_max_iterations_returns_progress_summary_not_template(tmp_path):
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 1})]),
            LLMResponse(
                content="目前完成数据抓取，待整理结论，下一步继续", tool_calls=[]
            ),
        ]
    )
    loop = _make_agent_loop(tmp_path, provider, tool)
    loop.max_iterations = 1

    final, _, _, _vn = asyncio.run(
        loop._run_agent_loop([{"role": "user", "content": "test"}])
    )

    assert "最大迭代" not in final
    assert "下一步" in final


def test_subagent_marks_tool_loop_and_summarizes():
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("s1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("s2", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("s3", "dummy", {"x": 1})]),
            LLMResponse(content="已完成部分，剩余部分下次继续", tool_calls=[]),
        ]
    )
    subagent = SubAgent(
        provider=cast(Any, provider), model="m", tools=[tool], max_iterations=10
    )

    result = asyncio.run(subagent.run("do work"))

    assert subagent.last_exit_reason == "tool_loop"
    assert "最大迭代" not in result
    assert len(tool.calls) == 2


def test_subagent_no_false_positive_when_same_tool_but_different_args():
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("s1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("s2", "dummy", {"x": 2})]),
            LLMResponse(content="all done", tool_calls=[]),
        ]
    )
    subagent = SubAgent(
        provider=cast(Any, provider), model="m", tools=[tool], max_iterations=10
    )

    result = asyncio.run(subagent.run("do work"))

    assert subagent.last_exit_reason == "completed"
    assert result == "all done"
    assert len(tool.calls) == 2


def test_subagent_injects_shared_procedure_hint_into_tool_result():
    tool = _DummyTool("shell")
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("s1", "shell", {"x": 1, "command": "pacman -S jq"})
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory(
        [
            {
                "id": "p1",
                "memory_type": "procedure",
                "summary": "pacman 调用时必须加 --noconfirm",
            }
        ]
    )
    subagent = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[tool],
        max_iterations=10,
        memory=cast(Any, memory),
    )

    result = asyncio.run(subagent.run("do work"))

    assert result == "done"
    assert memory.calls == [["shell", "pacman"]]
    tool_messages = [
        m for m in provider.calls[1]["messages"] if m.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert "【操作规范提醒】" in tool_messages[0]["content"]
    assert "--noconfirm" in tool_messages[0]["content"]


def test_subagent_dedupes_repeated_procedure_hint_items():
    tool = _DummyTool("shell")
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("s1", "shell", {"x": 1, "command": "pacman -S jq"})
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("s2", "shell", {"x": 2, "command": "pacman -S git"})
                ],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    memory = _HintMemory(
        [
            {
                "id": "p1",
                "memory_type": "procedure",
                "summary": "pacman 调用时必须加 --noconfirm",
            }
        ]
    )
    subagent = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[tool],
        max_iterations=10,
        memory=cast(Any, memory),
    )

    result = asyncio.run(subagent.run("do work"))

    assert result == "done"
    second_round_tool_messages = [
        m for m in provider.calls[2]["messages"] if m.get("role") == "tool"
    ]
    assert len(second_round_tool_messages) == 2
    assert "【操作规范提醒】" in second_round_tool_messages[0]["content"]
    assert "【操作规范提醒】" not in second_round_tool_messages[1]["content"]


def test_extract_action_tokens_includes_web_fetch_host_and_path_tokens():
    tokens = extract_action_tokens(
        "web_fetch",
        {"url": "https://www.bilibili.com/video/BV1xx?p=1"},
    )

    assert tokens == [
        "web_fetch",
        "www.bilibili.com",
        "bilibili.com",
        "video",
        "bv1xx",
    ]


def test_composer_no_false_positive_when_args_change():
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("p1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("p2", "dummy", {"x": 2})]),
            LLMResponse(content="给用户的最终消息", tool_calls=[]),
        ]
    )
    composer = ProactiveMessageComposer(
        provider=cast(Any, provider),
        model="m",
        max_tokens=256,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        collect_global_memory=lambda: "",
        max_tool_iterations=10,
    )
    composer._tools = [tool]
    composer._tool_map = cast(dict[str, Tool], {tool.name: tool})
    composer._tool_schemas = [tool.to_schema()]

    result = asyncio.run(
        composer.compose_message(items=[], recent=[], decision_signals={})
    )

    assert result == "给用户的最终消息"
    assert len(tool.calls) == 2


def test_agent_loop_does_not_trigger_on_two_repeats_only(tmp_path):
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "dummy", {"x": 1})]),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    loop = _make_agent_loop(tmp_path, provider, tool)

    final, _, _, _vn = asyncio.run(
        loop._run_agent_loop([{"role": "user", "content": "t"}])
    )

    assert final == "final"
    assert len(tool.calls) == 2


def test_agent_loop_does_not_false_positive_when_tool_order_changes(tmp_path):
    t1 = _DummyTool("a")
    t2 = _DummyTool("b")
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("r1-1", "a", {"x": 1}),
                    ToolCall("r1-2", "b", {"x": 1}),
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall("r2-1", "b", {"x": 1}),
                    ToolCall("r2-2", "a", {"x": 1}),
                ],
            ),
            LLMResponse(content="ok", tool_calls=[]),
        ]
    )

    tools = ToolRegistry()
    tools.register(t1)
    tools.register(t2)
    loop = AgentLoop(
        bus=MagicMock(),
        provider=cast(Any, provider),
        tools=tools,
        session_manager=MagicMock(),
        workspace=tmp_path,
        max_iterations=10,
        memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
    )

    final, _, _, _vn = asyncio.run(
        loop._run_agent_loop([{"role": "user", "content": "t"}])
    )

    assert final == "ok"
    assert len(t1.calls) == 2
    assert len(t2.calls) == 2


def test_subagent_max_iterations_returns_summary_and_reason():
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("s1", "dummy", {"x": 1})]),
            LLMResponse(content="已完成检索，剩余整理，下一步继续", tool_calls=[]),
        ]
    )
    subagent = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[tool],
        max_iterations=1,
    )

    result = asyncio.run(subagent.run("do work"))

    assert subagent.last_exit_reason == "forced_summary"
    assert "最大迭代" not in result
    assert "下一步" in result
    assert provider.calls[-1]["tools"] == []


def test_subagent_max_iterations_summary_failure_uses_fallback():
    tool = _DummyTool("dummy")

    class _SummaryFailProvider(_FakeProvider):
        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall("s1", "dummy", {"x": 1})],
                )
            raise RuntimeError("summary failed")

    provider = _SummaryFailProvider([])
    subagent = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[tool],
        max_iterations=1,
    )

    result = asyncio.run(subagent.run("do work"))

    assert subagent.last_exit_reason == "forced_summary_fallback"
    assert "当前进度" in result or "关键步骤" in result


def test_composer_breaks_on_repeated_same_signature_and_summarizes():
    tool = _DummyTool("dummy")
    provider = _FakeProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("p1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("p2", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("p3", "dummy", {"x": 1})]),
            LLMResponse(content="已核对部分信息，后续继续补齐", tool_calls=[]),
        ]
    )
    composer = ProactiveMessageComposer(
        provider=cast(Any, provider),
        model="m",
        max_tokens=256,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        collect_global_memory=lambda: "",
        max_tool_iterations=10,
    )
    composer._tools = [tool]
    composer._tool_map = cast(dict[str, Tool], {tool.name: tool})
    composer._tool_schemas = [tool.to_schema()]

    result = asyncio.run(
        composer.compose_message(items=[], recent=[], decision_signals={})
    )

    assert "最大迭代" not in result
    assert len(tool.calls) == 2


def test_composer_prompt_allows_standalone_interest_based_opening():
    provider = _FakeProvider([LLMResponse(content="给用户的最终消息", tool_calls=[])])
    composer = ProactiveMessageComposer(
        provider=cast(Any, provider),
        model="m",
        max_tokens=256,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        collect_global_memory=lambda: "",
        max_tool_iterations=2,
    )

    result = asyncio.run(
        composer.compose_message(items=[], recent=[], decision_signals={})
    )

    assert result == "给用户的最终消息"
    system_prompt = provider.calls[0]["messages"][0]["content"]
    user_prompt = provider.calls[0]["messages"][1]["content"]
    assert "不必强行承接近期对话" in system_prompt
    assert "也可以直接从最相关的那条信息流自然起题" in user_prompt


def test_composer_prompt_requires_source_name_and_clickable_url_when_citing():
    provider = _FakeProvider([LLMResponse(content="给用户的最终消息", tool_calls=[])])
    composer = ProactiveMessageComposer(
        provider=cast(Any, provider),
        model="m",
        max_tokens=256,
        format_items=lambda _: "1. 测试标题 [TestFeed]\n测试内容\n原文链接: https://example.com/post",
        format_recent=lambda _: "",
        collect_global_memory=lambda: "",
        max_tool_iterations=2,
    )

    result = asyncio.run(
        composer.compose_message(items=[], recent=[], decision_signals={})
    )

    assert result == "给用户的最终消息"
    system_prompt = provider.calls[0]["messages"][0]["content"]
    user_prompt = provider.calls[0]["messages"][1]["content"]
    preflight_prompt = provider.calls[0]["messages"][2]["content"]
    assert "附上可点击的原文 URL" in system_prompt
    assert "每个被你明确提到的具体进展，都应附上对应链接" in system_prompt
    assert "系统不会替你自动补来源" in system_prompt
    assert "每条被提到的更新都应带上对应链接" in user_prompt
    assert "优先在正文自然带上“来源名 + URL”" in preflight_prompt


def test_composer_prompt_allows_longer_aggregated_message():
    provider = _FakeProvider([LLMResponse(content="给用户的最终消息", tool_calls=[])])
    composer = ProactiveMessageComposer(
        provider=cast(Any, provider),
        model="m",
        max_tokens=256,
        format_items=lambda _: "",
        format_recent=lambda _: "",
        collect_global_memory=lambda: "",
        max_tool_iterations=2,
    )

    result = asyncio.run(
        composer.compose_message(items=[], recent=[], decision_signals={})
    )

    assert result == "给用户的最终消息"
    system_prompt = provider.calls[0]["messages"][0]["content"]
    user_prompt = provider.calls[0]["messages"][1]["content"]
    assert "不超过400字" in system_prompt
    assert "优先合并成一条更完整的主动消息" in system_prompt
    assert "不超过400字" in user_prompt
    assert "应优先把 2-3 条自然整合进一条消息里" in user_prompt


def test_agent_loop_summary_path_keeps_tool_chain_closed(tmp_path):
    tool = _DummyTool("dummy")
    provider = _StrictProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("c1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("c2", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("c3", "dummy", {"x": 1})]),
            LLMResponse(content="已总结当前进度", tool_calls=[]),
        ]
    )
    loop = _make_agent_loop(tmp_path, provider, tool)

    final, _, _, _vn = asyncio.run(
        loop._run_agent_loop([{"role": "user", "content": "t"}])
    )

    assert "已总结" in final
    assert len(tool.calls) == 2


def test_subagent_loop_path_runs_mandatory_exit_with_closed_chain():
    tool = _DummyTool("dummy")
    exit_tool = _ExitTool("task_note")
    provider = _StrictProvider(
        [
            LLMResponse(content="", tool_calls=[ToolCall("s1", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("s2", "dummy", {"x": 1})]),
            LLMResponse(content="", tool_calls=[ToolCall("s3", "dummy", {"x": 1})]),
            LLMResponse(
                content="",
                tool_calls=[ToolCall("e1", "task_note", {"note": "checkpoint"})],
            ),
            LLMResponse(content="当前进度已记录", tool_calls=[]),
        ]
    )
    subagent = SubAgent(
        provider=cast(Any, provider),
        model="m",
        tools=[tool, exit_tool],
        max_iterations=10,
        mandatory_exit_tools=["task_note"],
    )

    result = asyncio.run(subagent.run("do work"))

    assert subagent.last_exit_reason == "tool_loop"
    assert "记录" in result
    assert len(tool.calls) == 2
    assert exit_tool.called == 1
