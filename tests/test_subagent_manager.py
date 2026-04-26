import asyncio
from typing import Any, cast

import pytest

from agent.background.subagent_manager import SubagentManager
from agent.policies.delegation import SpawnDecision, SpawnDecisionMeta
from agent.provider import LLMResponse
from bus.events import SpawnCompletionItem
from bus.queue import MessageBus


class _Provider:
    async def chat(self, **kwargs: Any) -> LLMResponse:
        raise AssertionError("provider.chat should not be called in this test")


@pytest.mark.asyncio
async def test_subagent_manager_spawn_is_non_blocking(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=cast(Any, _Provider()),
        workspace=tmp_path,
        bus=bus,
        model="m",
        max_tokens=256,
        fetch_requester=object(),  # type: ignore[arg-type]
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_run_subagent(**kwargs):
        started.set()
        await release.wait()

    manager._run_subagent = _fake_run_subagent  # type: ignore[assignment]

    text = await manager.spawn(
        task="do work",
        label="job",
        origin_channel="telegram",
        origin_chat_id="123",
        decision=SpawnDecision(
            should_spawn=True,
            label="job",
            meta=SpawnDecisionMeta(
                source="heuristic",
                confidence="high",
                reason_code="long_running",
            ),
        ),
    )

    assert "已创建后台任务" in text
    await asyncio.wait_for(started.wait(), timeout=0.2)
    assert manager.get_running_count() == 1

    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert manager.get_running_count() == 0


@pytest.mark.asyncio
async def test_subagent_manager_announces_completion_to_origin_session(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=cast(Any, _Provider()),
        workspace=tmp_path,
        bus=bus,
        model="m",
        max_tokens=256,
        fetch_requester=object(),  # type: ignore[arg-type]
    )

    class _FakeSubAgent:
        last_exit_reason = "forced_summary"

        async def run(self, task: str) -> str:
            assert task == "research this"
            return "已完成检索，剩余整理，下一步继续"

    manager._build_subagent = (
        lambda *, task_dir, profile="research": _FakeSubAgent()
    )  # type: ignore[assignment]

    await manager.spawn(
        task="research this",
        label="research",
        origin_channel="telegram",
        origin_chat_id="42",
        decision=SpawnDecision(
            should_spawn=True,
            label="research",
            meta=SpawnDecisionMeta(
                source="heuristic",
                confidence="medium",
                reason_code="context_isolation_needed",
            ),
        ),
    )

    item = await asyncio.wait_for(bus.consume_inbound(), timeout=0.2)

    assert isinstance(item, SpawnCompletionItem)
    assert item.channel == "telegram"
    assert item.chat_id == "42"
    assert item.event.status == "incomplete"
    assert item.event.exit_reason == "forced_summary"
    assert item.decision is not None
    assert item.decision.meta.reason_code == "context_isolation_needed"

    trace_path = tmp_path / "memory" / "spawn_trace.jsonl"
    lines = [
        line for line in trace_path.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 2
    started = __import__("json").loads(lines[0])
    completed = __import__("json").loads(lines[1])
    assert started["trace_type"] == "spawn"
    assert started["subject"]["kind"] == "job"
    assert completed["payload"]["status"] == "incomplete"


@pytest.mark.asyncio
async def test_subagent_manager_lists_and_cancels_running_job(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=cast(Any, _Provider()),
        workspace=tmp_path,
        bus=bus,
        model="m",
        max_tokens=256,
        fetch_requester=object(),  # type: ignore[arg-type]
    )

    class _WaitingSubAgent:
        last_exit_reason = "running"

        async def run(self, task: str) -> str:
            await asyncio.Future()
            return "never"

    manager._build_subagent = (
        lambda *, task_dir, profile="research": _WaitingSubAgent()
    )  # type: ignore[assignment]

    await manager.spawn(
        task="long task",
        label="long",
        origin_channel="telegram",
        origin_chat_id="42",
    )

    jobs = manager.list_running_jobs()
    assert len(jobs) == 1
    job_id = str(jobs[0]["job_id"])
    assert jobs[0]["label"] == "long"
    assert await manager.cancel(job_id) is True

    item = await asyncio.wait_for(bus.consume_inbound(), timeout=0.2)
    assert isinstance(item, SpawnCompletionItem)
    assert item.event.status == "cancelled"
    assert item.event.exit_reason == "cancelled"
    await asyncio.sleep(0)
    assert manager.get_running_count() == 0


@pytest.mark.asyncio
async def test_spawn_sync_uses_shorter_iteration_budget(tmp_path):
    bus = MessageBus()
    manager = SubagentManager(
        provider=cast(Any, _Provider()),
        workspace=tmp_path,
        bus=bus,
        model="m",
        max_tokens=256,
        fetch_requester=object(),  # type: ignore[arg-type]
    )
    observed: dict[str, object] = {}

    class _FakeSubAgent:
        last_exit_reason = "completed"

        async def run(self, task: str) -> str:
            return "ok"

    def _fake_build_subagent(*, task_dir, profile="research", max_iterations=50):
        observed["task_dir"] = task_dir
        observed["profile"] = profile
        observed["max_iterations"] = max_iterations
        return _FakeSubAgent()

    manager._build_subagent = _fake_build_subagent  # type: ignore[assignment]

    result = await manager.spawn_sync(task="research this", label="job")

    assert "退出原因: completed" in result
    assert observed["profile"] == "research"
    assert observed["max_iterations"] == 10
