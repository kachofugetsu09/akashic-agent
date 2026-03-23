from __future__ import annotations

from pathlib import Path

from agent.tools.registry import ToolRegistry
from agent.tools.skill_action_tool import (
    SkillActionListTool,
    SkillActionRegisterTool,
    SkillActionResetTool,
    SkillActionRestartTool,
    SkillActionRewriteTool,
    SkillActionStatusTool,
    SkillActionUnregisterTool,
    SkillActionUpdateTool,
)


def register_skill_action_tools(tools: ToolRegistry, workspace: Path) -> None:
    skill_actions_path = workspace / "skill_actions.json"
    agent_tasks_dir = workspace / "agent-tasks"
    db_path = agent_tasks_dir / "task_notes.db"
    tools.register(SkillActionRegisterTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir), tags=["skill", "task"], risk="write", search_keywords=["注册技能", "创建技能", "添加skill", "新建技能"])
    tools.register(SkillActionUnregisterTool(skill_actions_path), tags=["skill", "task"], risk="write", search_keywords=["删除技能", "注销技能", "移除skill"])
    tools.register(SkillActionListTool(skill_actions_path, agent_tasks_dir=agent_tasks_dir), tags=["skill", "task"], risk="read-only", search_keywords=["技能列表", "查看技能", "skill列表", "有哪些技能"])
    tools.register(SkillActionStatusTool(agent_tasks_dir), tags=["skill", "task"], risk="read-only", search_keywords=["技能状态", "任务进度", "skill状态", "任务运行情况"])
    tools.register(SkillActionUpdateTool(agent_tasks_dir), tags=["skill", "task"], risk="write", search_keywords=["更新技能", "修改技能", "skill更新"])
    tools.register(SkillActionRestartTool(agent_tasks_dir, db_path=db_path), tags=["skill", "task"], risk="write", search_keywords=["重启技能", "重新运行skill", "skill重启"])
    tools.register(SkillActionResetTool(agent_tasks_dir), tags=["skill", "task"], risk="write", search_keywords=["重置技能", "清空技能状态", "skill重置"])
    tools.register(SkillActionRewriteTool(agent_tasks_dir, db_path=db_path), tags=["skill", "task"], risk="write", search_keywords=["重写技能", "重构skill", "skill重写"])
