from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proactive.skill_action import (
    SkillActionDef,
    SkillActionRegistry,
    SkillActionRunner,
)


def _write_actions(path: Path, actions: list[dict]) -> None:
    path.write_text(json.dumps({"version": 1, "actions": actions}), encoding="utf-8")


def test_skill_action_registry_load_reload_and_get(tmp_path: Path):
    path = tmp_path / "skill_actions.json"
    _write_actions(
        path,
        [
            {"id": "a1", "name": "shell", "command": "echo ok", "enabled": True},
            {"id": "a2", "name": "agent", "action_type": "agent", "enabled": True},
            {"id": "", "name": "bad", "command": "x"},
        ],
    )
    registry = SkillActionRegistry(path)
    assert [item.id for item in registry.list_enabled()] == ["a1", "a2"]
    assert registry.get("a1").name == "shell"
    assert registry.get("missing") is None

    _write_actions(path, [{"id": "a3", "name": "later", "command": "echo hi", "enabled": False}])
    assert registry.list_enabled() == []

    empty_registry = SkillActionRegistry(tmp_path / "missing.json")
    assert empty_registry.list_enabled() == []


def test_skill_action_runner_pick_availability_and_state(tmp_path: Path):
    class _Registry:
        def __init__(self, actions):
            self._actions = actions

        def list_enabled(self):
            return self._actions

    action1 = SkillActionDef(id="a1", name="one", weight=2, daily_max=1)
    action2 = SkillActionDef(id="a2", name="two", weight=1, min_interval_minutes=10)
    done_dir = tmp_path / "tasks" / "a2"
    done_dir.mkdir(parents=True)
    (done_dir / ".done").write_text("done", encoding="utf-8")
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "a1": {
                    "last_run_at": datetime.now(timezone.utc).isoformat(),
                    "runs_today": 1,
                    "window_key": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d"),
                }
            }
        ),
        encoding="utf-8",
    )
    runner = SkillActionRunner(
        _Registry([action1, action2]),
        rng=SimpleNamespace(choices=lambda seq, weights, k: [seq[0]]),
        state_path=state_path,
        agent_tasks_dir=tmp_path / "tasks",
    )

    assert runner.pick() is None
    assert runner.available_count() == 0

    now = datetime.now(timezone.utc)
    rec = runner._get_record("a3", now - timedelta(days=1))
    assert rec.runs_today == 0
    runner._record_run("a3", now, success=True)
    assert runner._records["a3"].runs_today == 1
    runner._save_state()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert "a3" in saved


@pytest.mark.asyncio
async def test_skill_action_runner_agent_guard_paths(tmp_path: Path):
    class _Registry:
        def list_enabled(self):
            return []

    runner = SkillActionRunner(_Registry(), state_path=tmp_path / "state.json")
    ok, out = await runner.run(SkillActionDef(id="a1", name="agent", action_type="agent", task_prompt="x"))
    assert (ok, out) == (False, "")

    empty_runner = SkillActionRunner(_Registry(), subagent_factory=lambda *args: None)
    ok, out = await empty_runner.run(SkillActionDef(id="a2", name="empty", action_type="agent", task_prompt=""))
    assert (ok, out) == (False, "")


@pytest.mark.asyncio
async def test_skill_action_runner_shell_success_failure_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    class _Registry:
        def list_enabled(self):
            return []

    runner = SkillActionRunner(_Registry(), state_path=tmp_path / "state.json")

    class _Proc:
        def __init__(self, rc: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
            self.returncode = rc
            self._stdout = stdout
            self._stderr = stderr
            self.killed = False

        async def communicate(self):
            return self._stdout, self._stderr

        def kill(self):
            self.killed = True

    async def _create_ok(*args, **kwargs):
        return _Proc(0, b"done", b"")

    monkeypatch.setattr("proactive.skill_action.asyncio.create_subprocess_shell", _create_ok)
    ok, out = await runner.run(SkillActionDef(id="s1", name="ok", command="echo ok"))
    assert (ok, out) == (True, "done")

    async def _create_fail(*args, **kwargs):
        return _Proc(1, b"", b"bad")

    monkeypatch.setattr("proactive.skill_action.asyncio.create_subprocess_shell", _create_fail)
    ok, out = await runner.run(SkillActionDef(id="s2", name="bad", command="exit 1"))
    assert (ok, out) == (False, "")

    proc = _Proc(0)

    async def _slow_communicate():
        await asyncio.sleep(10)

    proc.communicate = _slow_communicate

    async def _create_timeout(*args, **kwargs):
        return proc

    async def _raise_timeout(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr("proactive.skill_action.asyncio.create_subprocess_shell", _create_timeout)
    monkeypatch.setattr("proactive.skill_action.asyncio.wait_for", _raise_timeout)
    ok, out = await runner.run(
        SkillActionDef(id="s3", name="timeout", command="sleep 1", timeout_seconds=1)
    )
    assert (ok, out) == (False, "")
    assert proc.killed is True


@pytest.mark.asyncio
async def test_skill_action_runner_agent_background_exception(monkeypatch: pytest.MonkeyPatch):
    class _Registry:
        def list_enabled(self):
            return []

    runner = SkillActionRunner(
        _Registry(),
        subagent_factory=lambda *args: object(),
    )

    class _FakeRunner:
        def __init__(self, factory):
            self.factory = factory

        async def run(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr("proactive.skill_action.AgentBackgroundJobRunner", _FakeRunner)
    ok, out = await runner.run(
        SkillActionDef(id="a5", name="agent", action_type="agent", task_prompt="调研")
    )
    assert (ok, out) == (False, "")
