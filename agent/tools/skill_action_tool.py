"""
skill_action_tool.py — LLM 管理后台 skill actions 的工具

提供两个工具：
  skill_action_register   — 注册或更新一个后台定期执行的 skill action
  skill_action_unregister — 停用（或删除）一个已注册的 skill action
  skill_action_list       — 列出当前所有注册的 skill actions

写入 skill_actions.json 后，ProactiveLoop 在下次 tick 时自动热重载，无需重启。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.tools.base import Tool
from infra.persistence.json_store import atomic_save_json, load_json

logger = logging.getLogger(__name__)

_DOMAIN = "skill_action_tool"


def _load(path: Path) -> dict:
    data = load_json(path, default=None, domain=_DOMAIN)
    if not isinstance(data, dict):
        return {"version": 1, "actions": []}
    return data


def _save(path: Path, data: dict) -> None:
    atomic_save_json(path, data, domain=_DOMAIN)


class SkillActionRegisterTool(Tool):
    """注册或更新一个后台 skill action。"""

    name = "skill_action_register"
    description = (
        "注册一个后台定期执行的 skill action。"
        "支持两种类型：\n"
        "  shell — 执行固定 shell 命令（如运行脚本），需提供 command\n"
        "  agent — 用自然语言描述任务，由 AI 自主完成（如调研、分析），需提供 task_prompt\n"
        "注册后，当我空闲时会从已注册的 actions 中随机抽一个执行。\n"
        "用户说'有空帮我做某事'时调用此工具；智能任务用 agent 类型，脚本任务用 shell 类型。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "唯一 ID，建议格式：{类型}-{标识}，如 research-agent-papers",
            },
            "name": {
                "type": "string",
                "description": "人类可读的任务名称，如 '调研最新 Agent 论文'",
            },
            "action_type": {
                "type": "string",
                "enum": ["shell", "agent"],
                "description": (
                    "任务类型：shell=执行固定命令；agent=AI 自主完成自然语言任务。"
                    "默认 shell。"
                ),
                "default": "shell",
            },
            "command": {
                "type": "string",
                "description": "shell 类型必填：要执行的完整 shell 命令（含所有参数）",
            },
            "task_prompt": {
                "type": "string",
                "description": (
                    "agent 类型必填：自然语言任务描述，说清楚要做什么、结果怎么处理。"
                    "例如：'搜索最近3个月的 multi-agent 相关论文，整理5篇最值得读的，"
                    "保存到 workspace/research/agent-papers.md，然后把摘要发给我'"
                ),
            },
            "daily_max": {
                "type": "integer",
                "description": "每日最多执行次数，默认 3。0 表示不限制。",
                "minimum": 0,
                "default": 3,
            },
            "min_interval_minutes": {
                "type": "integer",
                "description": "两次执行之间的最小间隔（分钟），默认 90。",
                "minimum": 0,
                "default": 90,
            },
            "weight": {
                "type": "number",
                "description": "随机抽取时的权重，越大越容易被选中，默认 1.0。",
                "minimum": 0.01,
                "default": 1.0,
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "shell 类型执行超时时间（秒），默认 300。",
                "minimum": 10,
                "default": 300,
            },
        },
        "required": ["id", "name"],
    }

    def __init__(self, skill_actions_path: Path) -> None:
        self._path = skill_actions_path

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        name = kwargs["name"].strip()
        action_type = str(kwargs.get("action_type", "shell")).strip()
        command = str(kwargs.get("command", "")).strip()
        task_prompt = str(kwargs.get("task_prompt", "")).strip()

        if not action_id:
            return "错误：id 不能为空。"
        if action_type not in ("shell", "agent"):
            return f"错误：action_type 须为 shell 或 agent，收到 {action_type!r}"
        if action_type == "shell" and not command:
            return "错误：shell 类型必须提供 command。"
        if action_type == "agent" and not task_prompt:
            return "错误：agent 类型必须提供 task_prompt。"

        entry = {
            "id": action_id,
            "name": name,
            "action_type": action_type,
            "command": command,
            "task_prompt": task_prompt,
            "enabled": True,
            "daily_max": int(kwargs.get("daily_max", 3)),
            "min_interval_minutes": int(kwargs.get("min_interval_minutes", 90)),
            "weight": float(kwargs.get("weight", 1.0)),
            "timeout_seconds": int(kwargs.get("timeout_seconds", 300)),
        }

        data = _load(self._path)
        actions: list[dict] = data.get("actions", [])

        # 更新已有 / 追加新条目
        updated = False
        for i, a in enumerate(actions):
            if a.get("id") == action_id:
                actions[i] = entry
                updated = True
                break
        if not updated:
            actions.append(entry)

        data["actions"] = actions
        _save(self._path, data)

        verb = "更新" if updated else "注册"
        logger.info("[skill_action_tool] %s action id=%s type=%s", verb, action_id, action_type)
        detail = f"  任务: {task_prompt[:80]}" if action_type == "agent" else f"  命令: {command}"
        return (
            f"已{verb} skill action：{name}\n"
            f"  id: {action_id}\n"
            f"  类型: {action_type}\n"
            f"{detail}\n"
            f"  每日上限: {entry['daily_max']} 次\n"
            f"  最小间隔: {entry['min_interval_minutes']} 分钟\n"
            "下次空闲时我会自动执行它（无需重启）。"
        )


class SkillActionUnregisterTool(Tool):
    """停用或删除一个已注册的 skill action。"""

    name = "skill_action_unregister"
    description = "停用或删除一个后台 skill action。停用后我就不再在空闲时执行该任务。"
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "要停用的 action ID",
            },
            "delete": {
                "type": "boolean",
                "description": "true=从列表中彻底删除；false=仅设 enabled=false（默认 false）",
                "default": False,
            },
        },
        "required": ["id"],
    }

    def __init__(self, skill_actions_path: Path) -> None:
        self._path = skill_actions_path

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        do_delete = bool(kwargs.get("delete", False))

        data = _load(self._path)
        actions: list[dict] = data.get("actions", [])

        found = False
        new_actions = []
        for a in actions:
            if a.get("id") == action_id:
                found = True
                if not do_delete:
                    a["enabled"] = False
                    new_actions.append(a)
                # do_delete=True 时直接跳过，不加入 new_actions
            else:
                new_actions.append(a)

        if not found:
            return f"未找到 id={action_id!r} 的 skill action。"

        data["actions"] = new_actions
        _save(self._path, data)

        verb = "删除" if do_delete else "停用"
        logger.info("[skill_action_tool] %s action id=%s", verb, action_id)
        return f"已{verb} skill action id={action_id}。"


class SkillActionListTool(Tool):
    """列出当前所有注册的 skill actions。"""

    name = "skill_action_list"
    description = "列出当前所有已注册的后台 skill actions（包括已停用的）。"
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, skill_actions_path: Path, agent_tasks_dir: Path | None = None) -> None:
        self._path = skill_actions_path
        self._agent_tasks_dir = agent_tasks_dir

    async def execute(self, **kwargs: Any) -> str:
        data = _load(self._path)
        actions: list[dict] = data.get("actions", [])
        if not actions:
            return "当前没有注册任何 skill action。"

        lines = []
        for a in actions:
            action_id = a.get("id", "")
            if not a.get("enabled", True):
                status = "停用"
            elif self._agent_tasks_dir and (self._agent_tasks_dir / action_id / ".done").exists():
                status = "已完成"
            else:
                status = "启用"
            lines.append(
                f"[{status}] {action_id} — {a.get('name', '')}\n"
                f"  每日上限: {a.get('daily_max', '?')} 次 | "
                f"最小间隔: {a.get('min_interval_minutes', '?')} 分钟 | "
                f"权重: {a.get('weight', 1.0)}"
            )
        return "\n".join(lines)


class SkillActionResetTool(Tool):
    """清除 skill action 的已完成标记，使其重新参与调度。"""

    name = "skill_action_reset"
    description = (
        "清除 skill action 的已完成（DONE）标记，使该任务重新参与后台调度。"
        "当任务被误判为完成、或需要重新执行时使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "要重置的 action ID",
            },
        },
        "required": ["id"],
    }

    def __init__(self, agent_tasks_dir: Path) -> None:
        self._agent_tasks_dir = agent_tasks_dir

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        done_file = self._agent_tasks_dir / action_id / ".done"
        if not done_file.exists():
            return f"skill action id={action_id!r} 当前没有已完成标记，无需重置。"
        try:
            done_file.unlink()
            logger.info("[skill_action_tool] 重置完成标记 id=%s", action_id)
            return f"已重置 skill action id={action_id}，下次空闲时将重新执行。"
        except Exception as e:
            return f"重置失败：{e}"
