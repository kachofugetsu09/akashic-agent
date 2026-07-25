from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.task_plan.service import TaskPlanService
from agent.task_plan.store import TaskPlanStore
from agent.tools.base import normalize_tool_result
from agent.tools.registry import ToolRegistry
from agent.tools.task_plan import (
    CreateTaskPlanTool,
    InspectTaskPlanTool,
    UpdateTaskStepTool,
)


def test_create_task_plan_description_says_not_to_execute_or_spawn() -> None:
    description = CreateTaskPlanTool.description

    assert "不执行步骤" in description
    assert "不启动后台任务" in description
    assert "三步计划" in description


def test_inspect_task_plan_description_distinguishes_spawn_jobs() -> None:
    description = InspectTaskPlanTool.description

    assert "TaskPlan" in description
    assert "后台任务" in description
    assert "spawn" in description


def _service(tmp_path: Path) -> TaskPlanService:
    return TaskPlanService(TaskPlanStore(tmp_path / "task_plans.db"))


def test_task_tool_schemas_do_not_expose_session_keys(tmp_path: Path) -> None:
    service = _service(tmp_path)

    for tool in (
        CreateTaskPlanTool(service),
        UpdateTaskStepTool(service),
        InspectTaskPlanTool(service),
    ):
        properties = tool.to_schema()["function"]["parameters"]["properties"]
        assert "session_key" not in properties
        assert "_session_key" not in properties


@pytest.mark.asyncio
async def test_create_task_plan_requires_session_context(tmp_path: Path) -> None:
    tool = CreateTaskPlanTool(_service(tmp_path))

    raw = await tool.execute(title="Fix RAG", steps=["Read logs"])
    payload = json.loads(normalize_tool_result(raw).preview())

    assert payload["ok"] is False
    assert payload["error_code"] == "missing_session_context"


@pytest.mark.asyncio
async def test_create_and_inspect_task_plan_use_protected_session(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    create = CreateTaskPlanTool(service)
    inspect = InspectTaskPlanTool(service)

    raw = await create.execute(
        _session_key="cli:s1",
        title="Fix RAG",
        steps=["Read logs", "Update docs"],
    )
    created = json.loads(raw)
    assert created["ok"] is True

    raw = await inspect.execute(_session_key="cli:s1")
    inspected = json.loads(raw)
    assert inspected["ok"] is True
    assert inspected["task"]["title"] == "Fix RAG"


@pytest.mark.asyncio
async def test_update_task_step_accepts_step_index(tmp_path: Path) -> None:
    service = _service(tmp_path)
    create = CreateTaskPlanTool(service)
    update = UpdateTaskStepTool(service)

    created = json.loads(
        await create.execute(
            _session_key="cli:s1",
            title="Fix RAG",
            steps=["Read logs"],
        )
    )
    task_id = created["task"]["task_id"]

    raw = await update.execute(
        _session_key="cli:s1",
        task_id=task_id,
        step_index=1,
        status="completed",
        result_summary="Logs reviewed",
        tool_names=["inspect_turn_trace"],
    )
    payload = json.loads(normalize_tool_result(raw).preview())

    assert payload["ok"] is True
    assert payload["task"]["steps"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_registry_protected_context_overrides_model_session(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    registry = ToolRegistry()
    registry.register(CreateTaskPlanTool(service))
    registry.set_context(_session_key="cli:real")

    raw = await registry.execute(
        "create_task_plan",
        {
            "_session_key": "cli:fake",
            "title": "Fix RAG",
            "steps": ["Read logs"],
        },
    )
    payload = json.loads(normalize_tool_result(raw).preview())

    assert payload["ok"] is True
    assert service.get_active_task_plan(session_key="cli:real") is not None
    assert service.get_active_task_plan(session_key="cli:fake") is None
