"""
skill_action.py — Skill Action 注册表与执行器

在 ProactiveLoop 的 tick 中，当 LLM 决策为 idle（不发 chat 消息）时，
从注册的 skill actions 中随机抽取一个并执行，作为有意义的后台行动替代 idle。

配置文件：~/.akasic/workspace/skill_actions.json（或通过 path 指定）
"""

from __future__ import annotations

import asyncio
import json
import logging
import random as _random_module
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SkillActionDef:
    """单个 skill action 的定义。"""

    id: str
    name: str
    command: str  # shell 命令，支持 $VAR 展开
    enabled: bool = True
    weight: float = 1.0  # 随机抽取权重（越大越容易被选中）
    daily_max: int = 5  # 每日最多执行次数（0 = 不限）
    min_interval_minutes: int = 60  # 同一 action 两次执行的最小间隔（分钟）
    timeout_seconds: int = 300  # 执行超时时间
    cwd: Optional[str] = None  # 工作目录（None 则继承进程目录）

    @classmethod
    def from_dict(cls, d: dict) -> "SkillActionDef":
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            command=str(d["command"]),
            enabled=bool(d.get("enabled", True)),
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
        try:
            mtime = self._path.stat().st_mtime
            if mtime == self._mtime:
                return
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._actions = [
                SkillActionDef.from_dict(a)
                for a in raw.get("actions", [])
                if a.get("id") and a.get("command")
            ]
            self._mtime = mtime
            logger.info(
                "[skill_action] 已加载 %d 个 skill actions from %s",
                len(self._actions),
                self._path,
            )
        except Exception as e:
            logger.warning("[skill_action] 加载 skill_actions.json 失败: %s", e)

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


class SkillActionRunner:
    """
    执行 skill actions，管理每日配额与最小间隔。
    配额记录存储在内存中（进程重启后重置），适合低频场景。
    如需跨重启持久化，可将 state_path 设为 JSON 文件。
    """

    def __init__(
        self,
        registry: SkillActionRegistry,
        *,
        rng: _random_module.Random | None = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._registry = registry
        self._rng = rng or _random_module.Random()
        self._records: dict[str, _ActionRecord] = {}
        self._state_path = state_path
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
        异步执行 action 的 shell 命令。
        返回 (True, stdout_str) 表示成功（退出码 0），(False, "") 表示失败或超时。
        执行后无论成功失败都更新配额记录。
        """
        now = datetime.now(timezone.utc)
        logger.info(
            "[skill_action] 开始执行 id=%s name=%r cmd=%r",
            action.id,
            action.name,
            action.command,
        )
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

    def _is_available(
        self, action: SkillActionDef, rec: _ActionRecord, now: datetime
    ) -> bool:
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
        if not self._state_path or not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            for action_id, entry in raw.items():
                last_run = entry.get("last_run_at")
                rec = _ActionRecord(
                    last_run_at=datetime.fromisoformat(last_run).replace(
                        tzinfo=timezone.utc
                    )
                    if last_run
                    else None,
                    runs_today=int(entry.get("runs_today", 0)),
                    window_key=str(entry.get("window_key", "")),
                )
                self._records[action_id] = rec
            logger.info("[skill_action] 已加载运行状态 from %s", self._state_path)
        except Exception as e:
            logger.warning("[skill_action] 加载运行状态失败: %s", e)

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for action_id, rec in self._records.items():
                data[action_id] = {
                    "last_run_at": rec.last_run_at.isoformat()
                    if rec.last_run_at
                    else None,
                    "runs_today": rec.runs_today,
                    "window_key": rec.window_key,
                }
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self._state_path)
        except Exception as e:
            logger.warning("[skill_action] 保存运行状态失败: %s", e)
