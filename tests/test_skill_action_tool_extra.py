from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent.tools.skill_action_tool import (
    SkillActionListTool,
    SkillActionRegisterTool,
    SkillActionResetTool,
    SkillActionRestartTool,
    SkillActionRewriteTool,
    SkillActionStatusTool,
    SkillActionUnregisterTool,
    SkillActionUpdateTool,
    _clear_task_notes,
    _load,
)


def _read_actions(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("actions", [])


@pytest.mark.asyncio
async def test_skill_action_register_unregister_and_list(tmp_path: Path):
    skill_path = tmp_path / "skill_actions.json"
    task_dir = tmp_path / "agent-tasks"

    register = SkillActionRegisterTool(skill_path, task_dir)
    assert await register.execute(id=" ", name="x", command="echo") == "错误：id 不能为空。"
    assert "shell 类型必须提供 command" in await register.execute(id="s1", name="shell")
    assert "agent 类型必须提供 task_prompt" in await register.execute(
        id="a1",
        name="agent",
        action_type="agent",
    )

    result = await register.execute(
        id="a1",
        name="agent",
        action_type="agent",
        task_prompt="调研最新动态",
        daily_max=2,
        min_interval_minutes=30,
    )
    assert "已注册 skill action" in result
    assert (task_dir / "a1" / "TASK.md").exists()

    result = await register.execute(id="s1", name="shell", command="echo hi")
    assert "已注册 skill action" in result
    assert len(_read_actions(skill_path)) == 2

    list_tool = SkillActionListTool(skill_path, task_dir)
    text = await list_tool.execute()
    assert "[启用] a1" in text
    assert "[有TASK.md]" in text

    (task_dir / "a1" / ".done").write_text("done", encoding="utf-8")
    text = await list_tool.execute()
    assert "[已完成] a1" in text

    unregister = SkillActionUnregisterTool(skill_path)
    assert "未找到" in await unregister.execute(id="missing")
    assert "已停用" in await unregister.execute(id="a1")
    text = await list_tool.execute()
    assert "[停用] a1" in text
    assert "已删除" in await unregister.execute(id="s1", delete=True)
    assert len(_read_actions(skill_path)) == 1


@pytest.mark.asyncio
async def test_skill_action_status_update_restart_rewrite_and_reset(tmp_path: Path):
    task_dir = tmp_path / "agent-tasks"
    action_dir = task_dir / "a1"
    action_dir.mkdir(parents=True)
    task_md = action_dir / "TASK.md"
    task_md.write_text(
        "# title\n\n## 用户补充说明\n<!-- note -->\n\n## 运行历史\nold",
        encoding="utf-8",
    )
    (action_dir / ".done").write_text("完成摘要", encoding="utf-8")
    (action_dir / "result.txt").write_text("x", encoding="utf-8")
    db_path = tmp_path / "notes.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE task_notes(namespace TEXT)")
    conn.execute("INSERT INTO task_notes(namespace) VALUES (?)", ("a1",))
    conn.commit()
    conn.close()

    status = SkillActionStatusTool(task_dir)
    assert "目录不存在" in await status.execute(id="missing")
    text = await status.execute(id="a1")
    assert "=== TASK.md ===" in text
    assert "=== 状态: 已完成 ===" in text
    assert "result.txt" in text

    update = SkillActionUpdateTool(task_dir)
    assert "note 不能为空" in await update.execute(id="a1", note=" ")
    assert "找不到 agent-tasks/missing/TASK.md" in await update.execute(id="missing", note="x")
    text = await update.execute(id="a1", note="以后先整理来源")
    assert "下次执行时 subagent 会读取并遵循" in text
    assert "以后先整理来源" in task_md.read_text(encoding="utf-8")

    restart = SkillActionRestartTool(task_dir, db_path)
    text = await restart.execute(id="a1", clear_outputs=True)
    assert "已重置 skill action id=a1" in text
    assert not (action_dir / ".done").exists()
    assert not (action_dir / "result.txt").exists()
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM task_notes WHERE namespace='a1'").fetchone()[0]
    conn.close()
    assert count == 0

    rewrite = SkillActionRewriteTool(task_dir, db_path)
    assert "content 不能为空" in await rewrite.execute(id="a1", content=" ")
    assert "找不到 agent-tasks/missing/" in await rewrite.execute(id="missing", content="# x")
    text = await rewrite.execute(id="a1", content="# new", restart=False)
    assert "执行状态未变更" in text
    assert task_md.read_text(encoding="utf-8") == "# new"

    (action_dir / ".done").write_text("done", encoding="utf-8")
    text = await rewrite.execute(id="a1", content="# again")
    assert "重新执行" in text
    assert not (action_dir / ".done").exists()

    reset = SkillActionResetTool(task_dir)
    assert "无需重置" in await reset.execute(id="a1")
    (action_dir / ".done").write_text("done", encoding="utf-8")
    assert "已重置 skill action id=a1" in await reset.execute(id="a1")


def test_skill_action_tool_helpers(tmp_path: Path):
    path = tmp_path / "skill_actions.json"
    assert _load(path) == {"version": 1, "actions": []}

    db_path = tmp_path / "notes.db"
    _clear_task_notes(db_path, "x")

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE task_notes(namespace TEXT)")
    conn.execute("INSERT INTO task_notes(namespace) VALUES ('x')")
    conn.commit()
    conn.close()
    _clear_task_notes(db_path, "x")

    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM task_notes WHERE namespace='x'").fetchone()[0]
    conn.close()
    assert count == 0
