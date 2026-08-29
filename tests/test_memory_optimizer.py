"""覆盖当前 proactive memory optimizer 行为。"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime

import pytest

from core.memory.optimizer import (
    MemoryOptimizerBusy,
    MemoryOptimizer,
    MemoryOptimizerOutputError,
    MemoryOptimizerLoop,
)
from core.memory.markdown import MarkdownMemoryStore
from agent.plugin_composition import BoundModelDescriptor, LLMResponse, ModelRequest
from tests.model_plugin_fakes import BoundChatModelFake, build_test_model_store


class _MemoryProvider:
    """Provide deterministic driver responses behind the shared chat-model fake."""

    def __init__(
        self,
        *responses: LLMResponse | BaseException,
        side_effect: Callable[..., Awaitable[LLMResponse]] | None = None,
    ) -> None:
        self.context_window = 0
        self.max_output_tokens: int | None = None
        self.requests: list[dict[str, object]] = []
        self._responses = list(responses)
        self._side_effect = side_effect

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] = (),
    ) -> int:
        del messages, tools
        return 0

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> int:
        del messages
        return 0

    async def chat(self, **kwargs: object) -> LLMResponse:
        self.requests.append(kwargs)
        if self._side_effect is not None:
            return await self._side_effect(**kwargs)
        if not self._responses:
            return LLMResponse(content="")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class _RequestRecorder:
    """Record the public ModelRequest passed to one bound chat model."""

    def __init__(self, content: str) -> None:
        self.requests: list[ModelRequest] = []
        self._response = LLMResponse(content=content)

    @property
    def descriptor(self) -> BoundModelDescriptor:
        return BoundChatModelFake(self).descriptor

    @property
    def max_tool_schemas(self) -> int | None:
        return None

    def estimate_context_tokens(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] = (),
    ) -> int:
        del messages, tools
        return 0

    def estimate_appended_message_tokens(
        self,
        messages: Sequence[Mapping[str, object]],
    ) -> int:
        del messages
        return 0

    async def complete(self, request: ModelRequest) -> LLMResponse:
        self.requests.append(request)
        return self._response


_VALID_MEMORY = """# 用户长期记忆

## 用户事实
- 新版事实

## 用户偏好
- 新版偏好

## 用户明确要求长期记住的关键内容
- 新版要求
"""

_VALID_SELF = """# Akashic 的自我认知

## 人格与形象
- 新版人格

## 我对当前用户的理解
- 新版理解

## 我们关系的定义
- 新版关系
"""


def _provider_with_responses(*responses: str) -> _MemoryProvider:
    return _MemoryProvider(*(LLMResponse(content=response) for response in responses))


def test_optimize_skips_when_memory_pending_history_all_empty(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    provider = _MemoryProvider()

    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert provider.requests == []


def test_optimize_commits_marker_only_pending_snapshot(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    _ = memory.pending_file.write_text(
        "<!-- consolidation:test:pending -->\n",
        encoding="utf-8",
    )
    provider = _MemoryProvider()
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    asyncio.run(optimizer.optimize())

    assert provider.requests == []
    assert not memory._snapshot_path.exists()
    assert memory.pending_file.exists()
    assert memory.read_pending() == ""


def test_optimize_rewrites_memory_from_first_llm_call(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")

    provider = _provider_with_responses(_VALID_MEMORY, "")
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert memory.read_long_term().strip() == _VALID_MEMORY.strip()
    assert (memory.memory_dir / "MEMORY.bak.md").read_text(encoding="utf-8") == (
        "old profile"
    )
    history = list((memory.memory_dir / "backups").glob("MEMORY.*.bak.md"))
    assert len(history) == 1
    assert history[0].read_text(encoding="utf-8") == "old profile"


def test_optimize_rejects_invalid_memory_and_restores_pending(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term(_VALID_MEMORY)
    memory.append_pending("- [identity] 新身份")
    provider = _provider_with_responses("好的，已经整理完成。")
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(MemoryOptimizerOutputError, match="MEMORY.md"):
        asyncio.run(optimizer.optimize())

    assert memory.read_long_term() == _VALID_MEMORY
    assert "新身份" in memory.read_pending()
    assert not (memory.memory_dir / "MEMORY.bak.md").exists()


def test_optimize_rolls_back_snapshot_when_merge_returns_empty(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")
    memory.append_pending("- pending fact")

    provider = _provider_with_responses("", "")
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()


def test_optimize_rolls_back_snapshot_and_propagates_merge_failure(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")
    memory.append_pending("- pending fact")
    provider = _MemoryProvider(RuntimeError("merge failed"))
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(RuntimeError, match="merge failed"):
        asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()
    assert memory.read_long_term().strip() == "old profile"


def test_optimize_rolls_back_snapshot_when_memory_write_fails(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.append_pending("- pending fact")

    async def break_memory_file(**kwargs: object) -> LLMResponse:
        del kwargs
        memory.memory_file.mkdir()
        return LLMResponse(content=_VALID_MEMORY)

    provider = _MemoryProvider(side_effect=break_memory_file)
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(IsADirectoryError):
        asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()


def test_optimize_propagates_cancellation_and_restores_pending(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.append_pending("- pending fact")
    provider = _MemoryProvider(asyncio.CancelledError())
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(optimizer.optimize())

    assert "pending fact" in memory.read_pending()
    assert not memory._snapshot_path.exists()


def test_optimize_updates_self_using_pending_only(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old")
    memory.write_self("原 SELF")
    memory.append_pending("- [preference] 回复保持简洁。")

    provider = _provider_with_responses(
        _VALID_MEMORY,
        _VALID_SELF,
    )
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    assert memory.read_self().strip().startswith("# Akashic 的自我认知")
    assert "新版理解" in memory.read_self()
    assert (memory.memory_dir / "SELF.bak.md").read_text(encoding="utf-8") == (
        "原 SELF"
    )
    history = list((memory.memory_dir / "backups").glob("SELF.*.bak.md"))
    assert len(history) == 1
    assert history[0].read_text(encoding="utf-8") == "原 SELF"

    messages = provider.requests[1]["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[1], Mapping)
    self_prompt = messages[1]["content"]
    assert "- [preference] 回复保持简洁。" in self_prompt


def test_optimize_propagates_self_update_failure(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old")
    memory.write_self("## 原 SELF")
    memory.append_pending("- [preference] 回复保持简洁。")
    provider = _MemoryProvider(
        LLMResponse(content=_VALID_MEMORY),
        RuntimeError("self update failed"),
    )
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(RuntimeError, match="self update failed"):
        asyncio.run(optimizer.optimize())

    assert memory.read_long_term().strip() == _VALID_MEMORY.strip()
    assert memory.read_self().strip() == "## 原 SELF"


def test_optimize_rejects_invalid_self_without_overwriting_file(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old")
    memory.write_self(_VALID_SELF)
    memory.append_pending("- [preference] 回复保持简洁。")
    provider = _provider_with_responses(_VALID_MEMORY, "# Akashic 的自我认知")
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0

    with pytest.raises(MemoryOptimizerOutputError, match="SELF.md"):
        asyncio.run(optimizer.optimize())

    assert memory.read_self() == _VALID_SELF
    assert not (memory.memory_dir / "SELF.bak.md").exists()


def test_merge_memory_ignores_history_and_only_uses_pending(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    memory.write_long_term("old profile")
    memory.append_pending("- [identity] 新身份")

    provider = _provider_with_responses(_VALID_MEMORY, "")
    optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
    optimizer._STEP_DELAY_SECONDS = 0
    asyncio.run(optimizer.optimize())

    messages = provider.requests[0]["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[1], Mapping)
    prompt = messages[1]["content"]

    assert "近期历史摘要" not in prompt
    assert "- [identity] 新身份" in prompt


def test_request_text_response_uses_expected_chat_kwargs(tmp_path):
    memory = MarkdownMemoryStore(tmp_path)
    provider = _RequestRecorder("merged")
    optimizer = MemoryOptimizer(memory, build_test_model_store(_MemoryProvider()))

    result = asyncio.run(
        optimizer._request_text_response(
            provider,
            system_content="system",
            user_content="user",
            max_tokens=123,
        )
    )

    assert result == "merged"
    request = provider.requests[0]
    assert request.tools == ()
    assert request.max_output_tokens == 123


def test_optimize_reports_busy_instead_of_waiting(tmp_path):
    async def run_case() -> None:
        memory = MarkdownMemoryStore(tmp_path)
        provider = _MemoryProvider()
        optimizer = MemoryOptimizer(memory, build_test_model_store(provider))
        started = asyncio.Event()
        release = asyncio.Event()

        async def blocked_optimize(_provider: object) -> None:
            started.set()
            await release.wait()

        optimizer._optimize = blocked_optimize  # type: ignore[method-assign]
        running = asyncio.create_task(optimizer.optimize())
        await started.wait()

        assert optimizer.is_running
        with pytest.raises(MemoryOptimizerBusy):
            await optimizer.optimize()

        release.set()
        await running

    asyncio.run(run_case())


def test_seconds_until_next_tick_aligns_to_interval_boundary():
    now = datetime(2026, 2, 23, 12, 34, 56)
    loop = MemoryOptimizerLoop(None, interval_seconds=3600, _now_fn=lambda: now)

    secs = loop._seconds_until_next_tick()

    assert abs(secs - (25 * 60 + 4)) < 0.001


def test_seconds_until_next_tick_always_positive():
    for h in range(24):
        now = datetime(2026, 2, 23, h, 59, 59)
        loop = MemoryOptimizerLoop(None, interval_seconds=300, _now_fn=lambda n=now: n)
        assert loop._seconds_until_next_tick() > 0
