from typing import cast, Any

import pytest

from agent.background.runtime import (
    AgentBackgroundJobRunner,
    AgentBackgroundJobSpec,
)


class _FakeAgent:
    def __init__(self, *, exit_reason: str, result: str) -> None:
        self.last_exit_reason = exit_reason
        self._result = result

    async def run(self, task: str) -> str:
        assert task
        return self._result


class _ErrorAgent:
    async def run(self, task: str) -> str:
        assert task
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_background_job_runner_marks_completed():
    runner = AgentBackgroundJobRunner(
        lambda: cast(Any, _FakeAgent(exit_reason="completed", result="done"))
    )

    result = await runner.run(
        AgentBackgroundJobSpec(
            job_id="j1",
            job_kind="conversation_spawn",
            label="job",
            task="do work",
            max_iterations=20,
            completion_mode="message_bus",
            persistence_mode="ephemeral",
        )
    )

    assert result.status == "completed"
    assert result.exit_reason == "completed"
    assert result.result_summary == "done"


@pytest.mark.asyncio
async def test_background_job_runner_marks_incomplete():
    runner = AgentBackgroundJobRunner(
        lambda: cast(Any, _FakeAgent(exit_reason="forced_summary", result="partial"))
    )

    result = await runner.run(
        AgentBackgroundJobSpec(
            job_id="j2",
            job_kind="conversation_spawn",
            label="job",
            task="do work",
            max_iterations=40,
            completion_mode="direct_notify",
            persistence_mode="task_dir",
        )
    )

    assert result.status == "incomplete"
    assert result.exit_reason == "forced_summary"
    assert result.result_summary == "partial"


@pytest.mark.asyncio
async def test_background_job_runner_can_preserve_caller_error_contract():
    seen: list[str] = []
    runner = AgentBackgroundJobRunner(lambda: cast(Any, _ErrorAgent()))

    result = await runner.run(
        AgentBackgroundJobSpec(
            job_id="j3",
            job_kind="conversation_spawn",
            label="job",
            task="do work",
            max_iterations=40,
            completion_mode="direct_notify",
            persistence_mode="task_dir",
        ),
        on_exception=lambda e: seen.append(str(e)),
        error_result_summary="",
    )

    assert seen == ["boom"]
    assert result.status == "error"
    assert result.exit_reason == "error"
    assert result.result_summary == ""
