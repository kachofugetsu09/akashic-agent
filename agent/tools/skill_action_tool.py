"""
skill_action_tool.py — LLM 管理后台 skill actions 的工具

提供以下工具：
  skill_action_register   — 注册或更新一个后台定期执行的 skill action
  skill_action_unregister — 停用（或删除）一个已注册的 skill action
  skill_action_list       — 列出当前所有注册的 skill actions（含状态）
  skill_action_status     — 查看某个 action 的详细状态（TASK.md + 产出文件）
  skill_action_update     — 追加用户指令到 TASK.md，让下次执行时遵循
  skill_action_restart    — 重置任务状态，让任务从头开始（可选清空产出）
  skill_action_reset      — 仅清除 .done 完成标记

写入 skill_actions.json 后，ProactiveLoop 在下次 tick 时自动热重载，无需重启。
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.tools.base import Tool
from infra.persistence.json_store import atomic_save_json, load_json

logger = logging.getLogger(__name__)

_DOMAIN = "skill_action_tool"

# TASK.md 模板
_TASK_MD_TEMPLATE = """\
# {name}

## 原始任务描述
{task_prompt}

## 用户补充说明
<!-- 被动 agent 根据用户对话自动追加，每条带日期 -->

## 运行历史
<!-- subagent 每次运行结束后追加记录 -->
"""


def _load(path: Path) -> dict:
    data = load_json(path, default=None, domain=_DOMAIN)
    if not isinstance(data, dict):
        return {"version": 1, "actions": []}
    return data


def _save(path: Path, data: dict) -> None:
    atomic_save_json(path, data, domain=_DOMAIN)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _clear_task_notes(db_path: Path, namespace: str) -> None:
    """删除某个 namespace 下的所有 task_note 记录。"""
    if not db_path.exists():
        return
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("DELETE FROM task_notes WHERE namespace=?", (namespace,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("[skill_action_tool] 清除 task_notes 失败 namespace=%s: %s", namespace, e)


class SkillActionRegisterTool(Tool):
    """注册或更新一个后台 skill action，agent 类型会自动创建 TASK.md 活文档。"""

    name = "skill_action_register"
    description = (
        "注册一个后台定期执行的 skill action。\n"
        "支持两种类型：\n"
        "  shell — 执行固定 shell 命令（如运行脚本），需提供 command\n"
        "  agent — 用自然语言描述任务，由 AI 自主完成（如调研、分析），需提供 task_prompt\n"
        "agent 类型会自动创建 TASK.md 活文档，用户后续可通过 skill_action_update 追加指令。\n"
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
                "description": "任务类型：shell=执行固定命令；agent=AI 自主完成自然语言任务。默认 shell。",
                "default": "shell",
            },
            "command": {
                "type": "string",
                "description": "shell 类型必填：要执行的完整 shell 命令（含所有参数）",
            },
            "task_prompt": {
                "type": "string",
                "description": (
                    "agent 类型必填：详细的任务描述，说清楚：目标是什么、用哪些资源（文件路径）、"
                    "有哪些约束、成功的标准是什么。这会写入 TASK.md，用户随后可以追加补充说明。"
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

    def __init__(self, skill_actions_path: Path, agent_tasks_dir: Path | None = None) -> None:
        self._path = skill_actions_path
        self._agent_tasks_dir = agent_tasks_dir

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
            "enabled": True,
            "daily_max": int(kwargs.get("daily_max", 3)),
            "min_interval_minutes": int(kwargs.get("min_interval_minutes", 90)),
            "weight": float(kwargs.get("weight", 1.0)),
            "timeout_seconds": int(kwargs.get("timeout_seconds", 300)),
        }

        data = _load(self._path)
        actions: list[dict] = data.get("actions", [])

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

        # agent 类型：创建/更新 TASK.md
        task_md_note = ""
        if action_type == "agent" and self._agent_tasks_dir:
            action_dir = self._agent_tasks_dir / action_id
            action_dir.mkdir(parents=True, exist_ok=True)
            task_md_path = action_dir / "TASK.md"
            if not task_md_path.exists():
                task_md_path.write_text(
                    _TASK_MD_TEMPLATE.format(name=name, task_prompt=task_prompt),
                    encoding="utf-8",
                )
                task_md_note = f"\n已创建任务文档：agent-tasks/{action_id}/TASK.md"
            else:
                task_md_note = f"\n（TASK.md 已存在，未覆盖。如需修改请用 skill_action_update。）"

        verb = "更新" if updated else "注册"
        logger.info("[skill_action_tool] %s action id=%s type=%s", verb, action_id, action_type)
        detail = f"  命令: {command}" if action_type == "shell" else f"  任务文档: agent-tasks/{action_id}/TASK.md"
        return (
            f"已{verb} skill action：{name}\n"
            f"  id: {action_id}\n"
            f"  类型: {action_type}\n"
            f"{detail}\n"
            f"  每日上限: {entry['daily_max']} 次\n"
            f"  最小间隔: {entry['min_interval_minutes']} 分钟\n"
            f"{task_md_note}\n"
            "下次空闲时我会自动执行它（无需重启）。"
        )


class SkillActionUnregisterTool(Tool):
    """停用或删除一个已注册的 skill action。"""

    name = "skill_action_unregister"
    description = "停用或删除一个后台 skill action。停用后我就不再在空闲时执行该任务。"
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "要停用的 action ID"},
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
    parameters = {"type": "object", "properties": {}}

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
            has_task_md = (
                self._agent_tasks_dir and (self._agent_tasks_dir / action_id / "TASK.md").exists()
            )
            task_md_hint = " [有TASK.md]" if has_task_md else ""
            lines.append(
                f"[{status}] {action_id} — {a.get('name', '')}{task_md_hint}\n"
                f"  每日上限: {a.get('daily_max', '?')} 次 | "
                f"最小间隔: {a.get('min_interval_minutes', '?')} 分钟 | "
                f"权重: {a.get('weight', 1.0)}"
            )
        return "\n".join(lines)


class SkillActionStatusTool(Tool):
    """查看某个 action 的详细状态：TASK.md 内容 + 产出文件列表。"""

    name = "skill_action_status"
    description = (
        "查看某个后台 action 的详细状态，包括：\n"
        "- TASK.md 任务文档（目标、用户补充说明、运行历史）\n"
        "- 任务目录下已产出的文件列表\n"
        "用于了解任务做到哪、有什么结果、用户之前补充了什么。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "action ID"},
        },
        "required": ["id"],
    }

    def __init__(self, agent_tasks_dir: Path) -> None:
        self._agent_tasks_dir = agent_tasks_dir

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        action_dir = self._agent_tasks_dir / action_id

        if not action_dir.exists():
            return f"action {action_id!r} 还没有任何产出（目录不存在）。"

        parts: list[str] = []

        # TASK.md 内容
        task_md = action_dir / "TASK.md"
        if task_md.exists():
            content = task_md.read_text(encoding="utf-8")
            # 超长截断
            if len(content) > 3000:
                content = content[:3000] + "\n\n[...已截断，完整内容见 TASK.md]"
            parts.append(f"=== TASK.md ===\n{content}")
        else:
            parts.append("（无 TASK.md）")

        # .done 状态
        done_file = action_dir / ".done"
        if done_file.exists():
            done_summary = done_file.read_text(encoding="utf-8").strip()
            parts.append(f"\n=== 状态: 已完成 ===\n{done_summary[:300]}")
        else:
            parts.append("\n=== 状态: 进行中 ===")

        # 产出文件列表
        files = [
            p.relative_to(action_dir)
            for p in sorted(action_dir.rglob("*"))
            if p.is_file() and p.name not in (".done",)
        ]
        if files:
            file_list = "\n".join(f"  {f}" for f in files[:30])
            parts.append(f"\n=== 产出文件 ({len(files)} 个) ===\n{file_list}")
        else:
            parts.append("\n=== 产出文件: 无 ===")

        return "\n".join(parts)


class SkillActionUpdateTool(Tool):
    """向 TASK.md 追加用户指令，让下次执行时遵循新方向。"""

    name = "skill_action_update"
    description = (
        "向某个 agent action 的 TASK.md 追加用户补充说明。\n"
        "当用户对任务给出新方向、补充约束、纠正错误时使用。\n"
        "追加内容会在下次 subagent 执行时被读取。\n"
        "如果同时需要重跑，配合 skill_action_restart 使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "action ID"},
            "note": {
                "type": "string",
                "description": "要追加的用户补充说明，用自然语言写清楚方向或纠正",
            },
        },
        "required": ["id", "note"],
    }

    def __init__(self, agent_tasks_dir: Path) -> None:
        self._agent_tasks_dir = agent_tasks_dir

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        note = kwargs["note"].strip()

        if not note:
            return "错误：note 不能为空。"

        task_md = self._agent_tasks_dir / action_id / "TASK.md"
        if not task_md.exists():
            return f"错误：找不到 agent-tasks/{action_id}/TASK.md，请先注册该任务。"

        content = task_md.read_text(encoding="utf-8")
        dated_note = f"\n[{_today()}] {note}"

        # 追加到"用户补充说明"区块
        marker = "## 用户补充说明"
        if marker in content:
            # 找到区块后紧跟的注释行，在其后插入
            idx = content.index(marker) + len(marker)
            # 跳过注释行
            rest = content[idx:]
            comment_end = rest.find("\n\n")
            if comment_end != -1:
                insert_pos = idx + comment_end
            else:
                insert_pos = idx + len(rest)
            content = content[:insert_pos] + dated_note + content[insert_pos:]
        else:
            # 没有该区块，直接追加
            content += f"\n\n{marker}{dated_note}\n"

        task_md.write_text(content, encoding="utf-8")
        logger.info("[skill_action_tool] update TASK.md id=%s", action_id)
        return (
            f"已追加到 agent-tasks/{action_id}/TASK.md 的「用户补充说明」区块：\n"
            f"{dated_note}\n\n"
            "下次执行时 subagent 会读取并遵循。"
            "如需立即重跑，可调用 skill_action_restart。"
        )


class SkillActionRestartTool(Tool):
    """重置任务状态，让任务从头开始（清除 .done 标记、task_note、可选清空产出文件）。"""

    name = "skill_action_restart"
    description = (
        "重置某个 agent action 的执行状态，让它在下次空闲时重新运行。\n"
        "会清除：.done 完成标记、SQLite task_note 进度记录。\n"
        "可选清空产出文件（TASK.md 会保留）。\n"
        "当任务做得不好、需要重来，或更新了 TASK.md 指令后使用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "action ID"},
            "clear_outputs": {
                "type": "boolean",
                "description": "是否删除任务目录下的产出文件（TASK.md 始终保留）。默认 false。",
                "default": False,
            },
        },
        "required": ["id"],
    }

    def __init__(self, agent_tasks_dir: Path, db_path: Path) -> None:
        self._agent_tasks_dir = agent_tasks_dir
        self._db_path = db_path

    async def execute(self, **kwargs: Any) -> str:
        action_id = kwargs["id"].strip()
        clear_outputs = bool(kwargs.get("clear_outputs", False))

        action_dir = self._agent_tasks_dir / action_id
        cleared: list[str] = []

        # 1. 清除 .done
        done_file = action_dir / ".done"
        if done_file.exists():
            done_file.unlink()
            cleared.append(".done 完成标记")

        # 2. 清除 task_note
        _clear_task_notes(self._db_path, action_id)
        cleared.append("task_note 进度记录")

        # 3. 可选清空产出文件（保留 TASK.md）
        if clear_outputs and action_dir.exists():
            deleted_files = []
            for f in action_dir.rglob("*"):
                if f.is_file() and f.name != "TASK.md":
                    f.unlink()
                    deleted_files.append(str(f.relative_to(action_dir)))
            # 删除空子目录
            for d in sorted(action_dir.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
            cleared.append(f"产出文件 ({len(deleted_files)} 个)")

        logger.info("[skill_action_tool] restart id=%s clear_outputs=%s", action_id, clear_outputs)
        cleared_str = "、".join(cleared) if cleared else "无需清理"
        return (
            f"已重置 skill action id={action_id}：\n"
            f"  清除内容：{cleared_str}\n"
            "下次空闲时将重新执行（TASK.md 中的用户补充说明会被遵循）。"
        )


class SkillActionResetTool(Tool):
    """仅清除 skill action 的已完成标记，使其重新参与调度。"""

    name = "skill_action_reset"
    description = (
        "清除 skill action 的已完成（DONE）标记，使该任务重新参与后台调度。"
        "当任务被误判为完成、或需要重新执行时使用。"
        "如需完整重置（含进度记录），请用 skill_action_restart。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "要重置的 action ID"},
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
