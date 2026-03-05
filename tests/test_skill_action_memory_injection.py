import asyncio
from pathlib import Path
from typing import Any, cast

from proactive.skill_action import SkillActionDef, SkillActionRunner


class _RegistryStub:
    def list_enabled(self):
        return []


class _FakeSubAgent:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.last_exit_reason = "completed"

    async def run(self, prompt: str) -> str:
        return "done"


def test_skill_action_injects_only_procedure_preference_rules():
    called: dict = {}
    captured: dict = {}

    async def _retrieve(query: str, memory_types: list[str] | None):
        called["query"] = query
        called["types"] = memory_types
        return [
            {"memory_type": "procedure", "summary": "先写 task_note"},
            {"memory_type": "preference", "summary": "汇报用简洁中文"},
        ]

    def _format(items: list[dict]) -> str:
        return "## 【流程规范】用户偏好与规则\n- 先写 task_note\n- 汇报用简洁中文"

    def _factory(action_id: str, system_prompt_override: str):
        captured["action_id"] = action_id
        captured["system_prompt"] = system_prompt_override
        return _FakeSubAgent(system_prompt_override)

    runner = SkillActionRunner(
        cast(Any, _RegistryStub()),
        subagent_factory=cast(Any, _factory),
        memory_retrieve_fn=_retrieve,
        memory_format_fn=_format,
    )
    action = SkillActionDef(
        id="a1",
        name="test action",
        action_type="agent",
        task_prompt="调研并整理结论",
    )

    ok, out = asyncio.run(runner.run(action))

    assert ok is True
    assert out == "done"
    assert called["types"] == ["procedure", "preference"]
    assert called["query"] == "调研并整理结论"
    assert captured["action_id"] == "a1"
    assert "你是 Akashic" in captured["system_prompt"]
    assert "【流程规范】用户偏好与规则" in captured["system_prompt"]
    assert "多轮持久任务机制" in captured["system_prompt"]


def test_skill_action_falls_back_when_memory_retrieve_fails():
    captured: dict = {}

    async def _retrieve(query: str, memory_types: list[str] | None):
        raise RuntimeError("boom")

    def _format(items: list[dict]) -> str:
        return ""

    def _factory(action_id: str, system_prompt_override: str):
        captured["system_prompt"] = system_prompt_override
        return _FakeSubAgent(system_prompt_override)

    runner = SkillActionRunner(
        cast(Any, _RegistryStub()),
        subagent_factory=cast(Any, _factory),
        memory_retrieve_fn=_retrieve,
        memory_format_fn=_format,
    )
    action = SkillActionDef(
        id="a2",
        name="fallback action",
        action_type="agent",
        task_prompt="执行既定流程",
    )

    ok, out = asyncio.run(runner.run(action))

    assert ok is True
    assert out == "done"
    # 回退到默认系统提示
    assert "你是 Akashic" in captured["system_prompt"]
    assert "多轮持久任务机制" in captured["system_prompt"]


def test_skill_action_empty_task_prompt_uses_action_name_and_keeps_query_clean(
    tmp_path: Path,
):
    called: dict = {}

    async def _retrieve(query: str, memory_types: list[str] | None):
        called["query"] = query
        called["types"] = memory_types
        return []

    def _format(items: list[dict]) -> str:
        return ""

    def _factory(action_id: str, system_prompt_override: str):
        return _FakeSubAgent(system_prompt_override)

    action_id = "a3"
    task_dir = tmp_path / action_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "TASK.md").write_text("# task\n", encoding="utf-8")

    runner = SkillActionRunner(
        cast(Any, _RegistryStub()),
        subagent_factory=cast(Any, _factory),
        memory_retrieve_fn=_retrieve,
        memory_format_fn=_format,
        agent_tasks_dir=tmp_path,
    )
    action = SkillActionDef(
        id=action_id,
        name="name fallback",
        action_type="agent",
        task_prompt="",
    )

    ok, out = asyncio.run(runner.run(action))

    assert ok is True
    assert out == "done"
    assert called["types"] == ["procedure", "preference"]
    assert called["query"] == "name fallback"
    assert "任务存在 TASK.md" not in called["query"]
