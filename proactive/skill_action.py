"""
skill_action.py — Skill Action 注册表与执行器

在 ProactiveLoop 的 tick 中，当 LLM 决策为 idle（不发 chat 消息）时，
从注册的 skill actions 中随机抽取一个并执行，作为有意义的后台行动替代 idle。

配置文件：~/.akasic/workspace/skill_actions.json（或通过 path 指定）
"""

from __future__ import annotations

import asyncio
import logging
import random as _random_module
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from core.common.timekit import parse_iso as _parse_iso, utcnow as _utcnow
from infra.persistence.json_store import atomic_save_json, load_json

if TYPE_CHECKING:
    from agent.provider import LLMProvider
    from agent.subagent import SubAgent

logger = logging.getLogger(__name__)


@dataclass
class SkillActionDef:
    """单个 skill action 的定义。"""

    id: str
    name: str
    action_type: str = "shell"  # "shell" | "agent"
    command: str = ""           # shell 类型：shell 命令，支持 $VAR 展开
    task_prompt: str = ""       # agent 类型：自然语言任务描述
    enabled: bool = True
    one_shot: bool = False  # True=成功执行一次后自动标记完成，不再触发
    weight: float = 1.0  # 随机抽取权重（越大越容易被选中）
    daily_max: int = 5  # 每日最多执行次数（0 = 不限）
    min_interval_minutes: int = 60  # 同一 action 两次执行的最小间隔（分钟）
    timeout_seconds: int = 300  # 执行超时时间（shell 类型用）
    cwd: Optional[str] = None  # 工作目录（shell 类型，None 则继承进程目录）

    @classmethod
    def from_dict(cls, d: dict) -> "SkillActionDef":
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            action_type=str(d.get("action_type", "shell")),
            command=str(d.get("command", "")),
            task_prompt=str(d.get("task_prompt", "")),
            enabled=bool(d.get("enabled", True)),
            one_shot=bool(d.get("one_shot", False)),
            weight=float(d.get("weight", 1.0)),
            daily_max=int(d.get("daily_max", 5)),
            min_interval_minutes=int(d.get("min_interval_minutes", 60)),
            timeout_seconds=int(d.get("timeout_seconds", 300)),
            cwd=d.get("cwd") or None,
        )


@dataclass
class _ActionRecord:
    """单个 action 的运行记录（内存 + 持久化）。"""

    last_run_at: Optional[datetime] = None
    runs_today: int = 0
    window_key: str = ""  # 日期窗口 key（YYYY-MM-DD）


class SkillActionRegistry:
    """
    从 JSON 配置文件加载 skill actions，支持热重载。
    文件格式：
    {
      "version": 1,
      "actions": [
        {
          "id": "novel-read-once",
          "name": "小说推进阅读",
          "command": "python3 /path/to/reader_task.py read-once --kb /path/to/kb/shenming",
          "enabled": true,
          "weight": 1.0,
          "daily_max": 3,
          "min_interval_minutes": 90,
          "timeout_seconds": 300
        }
      ]
    }
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float = 0.0
        self._actions: list[SkillActionDef] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._actions = []
            return

        # 1. 检查文件是否变更（mtime 缓存）
        mtime = self._path.stat().st_mtime
        if mtime == self._mtime:
            return

        # 2. 读取并解析
        raw = load_json(self._path, default=None, domain="skill_action.registry")
        if raw is None:
            return
        self._actions = [
            SkillActionDef.from_dict(a)
            for a in raw.get("actions", [])
            if a.get("id") and (a.get("command") or a.get("task_prompt"))
        ]
        self._mtime = mtime
        logger.info(
            "[skill_action] 已加载 %d 个 skill actions from %s",
            len(self._actions),
            self._path,
        )

    def list_enabled(self) -> list[SkillActionDef]:
        """返回所有已启用的 action（每次调用会检查文件是否变更）。"""
        self._load()
        return [a for a in self._actions if a.enabled]

    def get(self, action_id: str) -> Optional[SkillActionDef]:
        self._load()
        for a in self._actions:
            if a.id == action_id:
                return a
        return None


_AGENT_SYSTEM_PROMPT = (
    "你是一个自主后台 Agent，在用户空闲时执行预先设定的任务。\n"
    "你有固定的工具集，专注完成分配的任务。\n"
    "\n"
    "## 进度管理\n"
    "每次开始任务前，先调用 task_recall(namespace=任务ID) 查看上次进度。\n"
    "完成重要步骤时，用 task_note(namespace=任务ID, key=..., value=...) 记录检查点，\n"
    "例如：已找到的资料列表、已完成的阶段、中间结果路径等。\n"
    "这样下次运行可以从断点继续，而不是重头再来。\n"
    "任务彻底完成后，调用 task_done(summary=...) 标记完成，之后该任务将不再自动触发。\n"
    "未完成时不要调用 task_done，让任务下次继续跑。\n"
    "\n"
    "## 其他规则\n"
    "任务完成后，必须调用 notify_owner 发送消息，否则视为未完成。\n"
    "消息中须简要说明：①做了哪些步骤 ②得到了什么结果。\n"
    "禁止在没有实际执行步骤的情况下声称任务完成。\n"
    "不要执行任务描述范围之外的操作。\n"
    "遇到工具调用失败时，换个方式继续，不要在最终回复中提及失败细节。"
)


class SkillActionRunner:
    """
    执行 skill actions，管理每日配额与最小间隔。
    支持两种类型：
      shell — 直接执行 shell 命令（原有行为）
      agent — 用 SubAgent + 受限工具集执行自然语言任务
    """

    def __init__(
        self,
        registry: SkillActionRegistry,
        *,
        rng: _random_module.Random | None = None,
        state_path: Optional[Path] = None,
        subagent_factory: Optional[Callable[[str], "SubAgent"]] = None,
        agent_tasks_dir: Optional[Path] = None,  # 用于检查 .done 文件
    ) -> None:
        self._registry = registry
        self._rng = rng or _random_module.Random()
        self._records: dict[str, _ActionRecord] = {}
        self._state_path = state_path
        self._subagent_factory = subagent_factory
        self._agent_tasks_dir = agent_tasks_dir
        self._load_state()

    # ── 公开接口 ──────────────────────────────────────────────────

    def pick(self) -> Optional[SkillActionDef]:
        """
        从可用（enabled + 未超配额 + 过了最小间隔）的 actions 中随机抽取一个。
        按 weight 加权随机。返回 None 表示当前没有可用 action。
        """
        now = datetime.now(timezone.utc)
        candidates: list[SkillActionDef] = []
        weights: list[float] = []
        for action in self._registry.list_enabled():
            rec = self._get_record(action.id, now)
            if not self._is_available(action, rec, now):
                continue
            candidates.append(action)
            weights.append(max(0.001, action.weight))
        if not candidates:
            return None
        chosen = self._rng.choices(candidates, weights=weights, k=1)[0]
        return chosen

    async def run(self, action: SkillActionDef) -> tuple[bool, str]:
        """
        异步执行 action。
        返回 (success, output_str)。执行后无论成功失败都更新配额记录。
        """
        now = datetime.now(timezone.utc)
        logger.info(
            "[skill_action] 开始执行 id=%s name=%r type=%s",
            action.id,
            action.name,
            action.action_type,
        )
        if action.action_type == "agent":
            return await self._run_agent_action(action, now)
        return await self._run_shell_action(action, now)

    async def _run_agent_action(
        self, action: SkillActionDef, now: datetime
    ) -> tuple[bool, str]:
        """用 SubAgent 执行自然语言任务。"""
        if not self._subagent_factory:
            logger.warning(
                "[skill_action] agent 类型任务需要 subagent_factory，但未配置 id=%s", action.id
            )
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""
        if not action.task_prompt.strip():
            logger.warning("[skill_action] agent 任务 task_prompt 为空 id=%s", action.id)
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""
        try:
            subagent = self._subagent_factory(action.id)
            augmented_prompt = (
                f"[任务ID: {action.id}]\n"
                f"[工作目录: agent-tasks/{action.id}/]\n"
                f"[共享配置目录: agent-tasks/shared/ — 内含 API keys 等公共配置，可用 read_file 读取]\n\n"
                f"{action.task_prompt}"
            )
            result = await subagent.run(augmented_prompt)
            success = bool(result)
            logger.info(
                "[skill_action] agent 任务完成 id=%s success=%s result_len=%d",
                action.id,
                success,
                len(result),
            )
            self._record_run(action.id, now, success=success)
            self._save_state()
            return success, result
        except Exception as e:
            logger.exception("[skill_action] agent 任务异常 id=%s error=%s", action.id, e)
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""

    async def _run_shell_action(
        self, action: SkillActionDef, now: datetime
    ) -> tuple[bool, str]:
        """执行 shell 命令（原有逻辑）。"""
        logger.info("[skill_action] shell cmd=%r", action.command)
        try:
            proc = await asyncio.create_subprocess_shell(
                action.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=action.cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=action.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                logger.warning(
                    "[skill_action] 执行超时 id=%s timeout=%ds",
                    action.id,
                    action.timeout_seconds,
                )
                self._record_run(action.id, now, success=False)
                self._save_state()
                return False, ""

            rc = proc.returncode
            stdout_str = (stdout or b"").decode("utf-8", errors="replace").strip()
            stderr_str = (stderr or b"").decode("utf-8", errors="replace").strip()
            if rc == 0:
                logger.info(
                    "[skill_action] 执行成功 id=%s rc=%d stdout_len=%d",
                    action.id,
                    rc,
                    len(stdout_str),
                )
                if stdout_str:
                    logger.debug("[skill_action] stdout: %s", stdout_str[:500])
                self._record_run(action.id, now, success=True)
                self._save_state()
                return True, stdout_str
            else:
                logger.warning(
                    "[skill_action] 执行失败 id=%s rc=%d stderr=%r",
                    action.id,
                    rc,
                    stderr_str[:200],
                )
                self._record_run(action.id, now, success=False)
                self._save_state()
                return False, ""
        except Exception as e:
            logger.exception("[skill_action] 执行异常 id=%s error=%s", action.id, e)
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""

    def available_count(self) -> int:
        """返回当前有多少个 action 处于可用状态（供调试日志）。"""
        now = datetime.now(timezone.utc)
        return sum(
            1
            for a in self._registry.list_enabled()
            if self._is_available(a, self._get_record(a.id, now), now)
        )

    # ── 内部状态管理 ──────────────────────────────────────────────

    def _get_record(self, action_id: str, now: datetime) -> _ActionRecord:
        rec = self._records.get(action_id) or _ActionRecord()
        # 滚动日期窗口
        today_key = now.astimezone().strftime("%Y-%m-%d")
        if rec.window_key != today_key:
            rec.runs_today = 0
            rec.window_key = today_key
        self._records[action_id] = rec
        return rec

    def _is_done(self, action_id: str) -> bool:
        """检查 agent-tasks/{action_id}/.done 是否存在。"""
        if not self._agent_tasks_dir:
            return False
        return (self._agent_tasks_dir / action_id / ".done").exists()

    def _is_available(
        self, action: SkillActionDef, rec: _ActionRecord, now: datetime
    ) -> bool:
        if self._is_done(action.id):
            logger.debug("[skill_action] id=%s 已标记完成，跳过", action.id)
            return False
        if action.daily_max > 0 and rec.runs_today >= action.daily_max:
            logger.debug(
                "[skill_action] id=%s 已达今日配额 runs_today=%d daily_max=%d",
                action.id,
                rec.runs_today,
                action.daily_max,
            )
            return False
        if rec.last_run_at is not None and action.min_interval_minutes > 0:
            elapsed_minutes = (now - rec.last_run_at).total_seconds() / 60
            if elapsed_minutes < action.min_interval_minutes:
                logger.debug(
                    "[skill_action] id=%s 最小间隔未满 elapsed=%.1fmin min=%.0fmin",
                    action.id,
                    elapsed_minutes,
                    action.min_interval_minutes,
                )
                return False
        return True

    def _record_run(self, action_id: str, now: datetime, *, success: bool) -> None:
        rec = self._records.get(action_id) or _ActionRecord()
        today_key = now.astimezone().strftime("%Y-%m-%d")
        if rec.window_key != today_key:
            rec.runs_today = 0
            rec.window_key = today_key
        rec.last_run_at = now
        rec.runs_today += 1
        self._records[action_id] = rec
        logger.info(
            "[skill_action] 记录运行 id=%s success=%s runs_today=%d window=%s",
            action_id,
            success,
            rec.runs_today,
            today_key,
        )

    # ── 持久化（可选）───────────────────────────────────────────

    def _load_state(self) -> None:
        if not self._state_path:
            return

        # 1. 从磁盘读取
        raw = load_json(self._state_path, default=None, domain="skill_action.runner")
        if raw is None:
            return

        # 2. 解析运行记录
        for action_id, entry in raw.items():
            last_run = entry.get("last_run_at")
            rec = _ActionRecord(
                last_run_at=_parse_iso(last_run),
                runs_today=int(entry.get("runs_today", 0)),
                window_key=str(entry.get("window_key", "")),
            )
            self._records[action_id] = rec
        logger.info("[skill_action] 已加载运行状态 from %s", self._state_path)

    def _save_state(self) -> None:
        if not self._state_path:
            return

        # 1. 序列化运行记录
        data = {}
        for action_id, rec in self._records.items():
            data[action_id] = {
                "last_run_at": rec.last_run_at.isoformat() if rec.last_run_at else None,
                "runs_today": rec.runs_today,
                "window_key": rec.window_key,
            }

        # 2. 原子写入
        atomic_save_json(self._state_path, data, domain="skill_action.runner")
