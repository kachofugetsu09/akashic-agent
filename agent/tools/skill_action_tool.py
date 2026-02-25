"""
skill_action_tool.py — LLM 管理后台 skill actions 的工具

提供两个工具：
  skill_action_register   — 注册或更新一个后台定期执行的 skill action
  skill_action_unregister — 停用（或删除）一个已注册的 skill action
  skill_action_list       — 列出当前所有注册的 skill actions

写入 skill_actions.json 后，ProactiveLoop 在下次 tick 时自动热重载，无需重启。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)


def _load(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"version": 1, "actions": []}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class SkillActionRegisterTool(Tool):
    """注册或更新一个后台 skill action。"""

    name = "skill_action_register"
    description = (
        "注册一个后台定期执行的 skill action（如：推进小说阅读进度、整理知识库等）。"
        "注册后，当我决定不主动发消息时，会从已注册的 actions 中随机抽一个执行。"
        "如果用户希望我'空闲时自动做某件事'，就调用这个工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "唯一 ID，建议格式：{类型}-{标识}，如 novel-read-2236",
            },
            "name": {
                "type": "string",
                "description": "人类可读的任务名称，如 '小说2236增量阅读'",
            },
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令（完整命令，含所有参数）",
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
                "description": "执行超时时间（秒），默认 300。",
                "minimum": 10,
                "default": 300,
            },
        },
        "required": ["id", "name", "command"],
    }

    def __init__(self, skill_actions_path: Path) -> None:
        self._path = skill_actions_path

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        name = kwargs["name"].strip()
        command = kwargs["command"].strip()
        if not action_id or not command:
            return "错误：id 和 command 不能为空。"

        entry = {
            "id": action_id,
            "name": name,
            "command": command,
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
        logger.info("[skill_action_tool] %s action id=%s", verb, action_id)
        return (
            f"已{verb} skill action：{name}\n"
            f"  id: {action_id}\n"
            f"  每日上限: {entry['daily_max']} 次\n"
            f"  最小间隔: {entry['min_interval_minutes']} 分钟\n"
            f"  命令: {command}\n"
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

    def __init__(self, skill_actions_path: Path) -> None:
        self._path = skill_actions_path

    async def execute(self, **kwargs: Any) -> str:
        data = _load(self._path)
        actions: list[dict] = data.get("actions", [])
        if not actions:
            return "当前没有注册任何 skill action。"

        lines = []
        for a in actions:
            status = "启用" if a.get("enabled", True) else "停用"
            lines.append(
                f"[{status}] {a.get('id')} — {a.get('name', '')}\n"
                f"  每日上限: {a.get('daily_max', '?')} 次 | "
                f"最小间隔: {a.get('min_interval_minutes', '?')} 分钟 | "
                f"权重: {a.get('weight', 1.0)}"
            )
        return "\n".join(lines)
