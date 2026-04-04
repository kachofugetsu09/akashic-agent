from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.looping.core import AgentLoop
from agent.looping.ports import AgentLoopConfig, AgentLoopDeps, LLMConfig
from agent.memory import MemoryStore
from agent.provider import LLMResponse, ToolCall
from agent.tools.filesystem import ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.spawn import SpawnTool
from core.memory.port import DefaultMemoryPort


class _FakeProvider:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("provider.chat called more than expected")
        return self._responses.pop(0)


def _make_loop(
    tmp_path: Path,
    *,
    provider: _FakeProvider,
    manager: AsyncMock,
) -> AgentLoop:
    tools = ToolRegistry()
    tools.register(SpawnTool(manager, tools), risk="external-side-effect")
    tools.register(ReadFileTool(allowed_dir=tmp_path))
    return AgentLoop(
        AgentLoopDeps(
            bus=MagicMock(),
            provider=cast(Any, provider),
            tools=tools,
            session_manager=MagicMock(),
            workspace=tmp_path,
            memory_port=DefaultMemoryPort(MemoryStore(tmp_path)),
        ),
        AgentLoopConfig(llm=LLMConfig(max_iterations=4)),
    )


@pytest.mark.asyncio
async def test_baseline_multistep_research_prefers_spawn(tmp_path: Path):
    manager = AsyncMock()
    manager.spawn = AsyncMock(return_value="已创建后台任务")
    manager.spawn_sync = AsyncMock(return_value="[子任务结果]\n结论：三个文件差异已整理")
    manager.get_running_count = MagicMock(return_value=0)
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "call_spawn",
                        "spawn",
                        {
                            "task": (
                                "任务目标：比较 spawn/profile 相关 3 个文件的实现差异。\n"
                                "关键约束：只做只读调研，不修改文件。\n"
                                "关键上下文：用户要一份简短结论，说明 research/scripting/general 的区别。\n"
                                "期望输出格式：中文要点总结。"
                            ),
                            "label": "调研",
                            "profile": "research",
                            "run_in_background": False,
                        },
                    )
                ],
            ),
            LLMResponse(content="已整理成结论，下面直接汇报。", tool_calls=[]),
        ]
    )
    loop = _make_loop(tmp_path, provider=provider, manager=manager)

    final, tools_used, _, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "请调查 3 个文件的实现差异并汇总结论"}]
    )

    assert final == "已整理成结论，下面直接汇报。"
    assert tools_used == ["spawn"]
    manager.spawn_sync.assert_awaited_once()
    manager.spawn.assert_not_called()
    spawn_kwargs = manager.spawn_sync.await_args.kwargs
    assert spawn_kwargs["profile"] == "research"
    assert "任务目标" in spawn_kwargs["task"]
    assert "关键约束" in spawn_kwargs["task"]
    assert "期望输出格式" in spawn_kwargs["task"]


@pytest.mark.asyncio
async def test_baseline_simple_file_question_stays_inline(tmp_path: Path):
    manager = AsyncMock()
    manager.spawn = AsyncMock(return_value="unexpected")
    manager.spawn_sync = AsyncMock(return_value="unexpected")
    manager.get_running_count = MagicMock(return_value=0)
    target = tmp_path / "subagent_profiles.py"
    target.write_text('PROFILE_RESEARCH = "research"\n', encoding="utf-8")
    provider = _FakeProvider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "call_read",
                        "read_file",
                        {"path": str(target)},
                    )
                ],
            ),
            LLMResponse(content="默认 profile 是 research。", tool_calls=[]),
        ]
    )
    loop = _make_loop(tmp_path, provider=provider, manager=manager)

    final, tools_used, _, _, _ = await loop._run_agent_loop(
        [{"role": "user", "content": "读取这个文件，告诉我默认 profile 是什么"}]
    )

    assert final == "默认 profile 是 research。"
    assert tools_used == ["read_file"]
    manager.spawn.assert_not_called()
    manager.spawn_sync.assert_not_called()
    first_call_tools = [
        schema["function"]["name"] for schema in provider.calls[0].get("tools", [])
    ]
    assert "spawn" in first_call_tools
    assert "read_file" in first_call_tools
