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
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from agent.persona import AKASHIC_IDENTITY
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
    command: str = ""  # shell 类型：shell 命令，支持 $VAR 展开
    task_prompt: str = ""  # agent 类型：自然语言任务描述
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
            if a.get("id")
            and (
                a.get("command")
                or a.get("task_prompt")
                or a.get("action_type")
                == "agent"  # agent 类型可用 TASK.md 代替 task_prompt
            )
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
    "你是 Akashic，正在用户空闲时执行预先设定的后台任务。\n"
    f"身份基线：{AKASHIC_IDENTITY}\n"
    "你有固定的工具集，专注完成分配的任务；通过 notify_owner 对用户汇报时，保持这个身份语气。\n"
    "\n"
    "## 多轮持久任务机制（最重要，必须理解）\n"
    "你的任务是**跨多次运行**逐步完成的，每次运行有步骤预算上限（约 40 步）。\n"
    "这意味着：\n"
    "- 一次运行不需要、也不应该试图完成所有阶段\n"
    "- 步骤预算不够时，做完当前阶段后立即用 task_note 记录检查点，然后正常结束\n"
    "- 系统会在你空闲时再次触发你，你从 task_recall 读取上次的检查点继续\n"
    "- **绝对禁止**因为「步骤快用完了」就跳过阶段、压缩步骤、或伪造完成\n"
    "- 做完一个完整的阶段比草草完成所有阶段更有价值\n"
    "\n"
    "## 任务文档（TASK.md）\n"
    "TASK.md 是用户写给你的任务书，内容由用户维护，你只能读取，不能修改其中的：\n"
    "  - 任务目标\n"
    "  - 约束条件\n"
    "  - 用户补充说明\n"
    "每次开始前用 read_file 读取 TASK.md，了解任务目标、约束和用户最新补充说明。\n"
    "\n"
    "任务结束时（无论完成与否），只允许在 TASK.md 末尾的「## 运行历史」区块用 edit_file 追加本次记录，\n"
    "格式如下（禁止修改该区块以外的任何内容）：\n"
    "  ### 第N次 (YYYY-MM-DD) — [完成/未完成]\n"
    "  - 本次完成的步骤\n"
    "  - 产出文件路径\n"
    "  - 下次需要继续的事项\n"
    "\n"
    "## 进度管理（task_note / task_recall）\n"
    "task_note 和 task_recall 是你跨次运行的核心记忆机制，是给下一次运行的你自己看的，与 TASK.md 无关。\n"
    "每次任务开始时，第一步必须调用 task_recall(namespace=任务ID) 查询所有已记录的检查点，\n"
    "根据检查点判断当前处于哪个阶段、上次做到哪一步、有哪些中间结果，再决定从哪里继续。\n"
    "不要依赖 TASK.md 的运行历史来判断进度——那是给用户看的摘要，不是可靠的状态机。\n"
    "\n"
    "每完成一个关键步骤，立即调用 task_note 记录检查点，粒度要足够细，例如：\n"
    "  task_note(namespace=任务ID, key='phase', value='研究完成，结论：用GraphRAG方案')\n"
    "  task_note(namespace=任务ID, key='novel_total_lines', value='21375')\n"
    "  task_note(namespace=任务ID, key='processed_up_to_line', value='500')\n"
    "  task_note(namespace=任务ID, key='demo_status', value='已写完，路径：demo/rag_demo.py')\n"
    "下次运行时 task_recall 能直接拿到这些值，不需要重新推断。\n"
    "\n"
    "任务彻底完成后，调用 task_done(summary=...) 标记完成，之后该任务将不再自动触发。\n"
    "未完成时不要调用 task_done，让任务下次继续跑。\n"
    "\n"
    "## 文件路径规则\n"
    "文件工具（read_file / write_file / list_dir / edit_file）支持两种路径写法：\n"
    "  - 相对路径：相对于 agent-tasks/ 目录，例如 `rag-novel-eva-research/TASK.md`\n"
    "  - 绝对路径：完整路径，例如 `/home/user/.akasic/workspace/agent-tasks/rag-novel-eva-research/TASK.md`\n"
    "相对路径推荐写法：`<任务ID>/文件名`，读 TASK.md 时用 `<任务ID>/TASK.md`。\n"
    "禁止写出 `agent-tasks/` 前缀的相对路径（因工作目录已是 agent-tasks/，会导致路径错误）。\n"
    "\n"
    "## 执行纪律\n"
    "1. **严格按 TASK.md 规定的阶段顺序执行**，禁止跳过任何阶段，哪怕剩余 iterations 不多。\n"
    "   宁可本次只完成阶段 1-2，下次继续，也不能跳到后面的阶段。\n"
    "2. **产出文件名必须与 TASK.md 规定完全一致**，禁止自行更改文件名。\n"
    "   例如 TASK.md 写 `survey.md`，就必须写 `survey.md`，不能写成 `rag_evaluation_design.md`。\n"
    "3. 每完成一个阶段，立即用 task_note 记录阶段检查点，然后再继续下一阶段。\n"
    "4. 任务完成后，必须调用 notify_owner 发送消息，否则视为未完成。\n"
    "   消息中须简要说明：①做了哪些步骤 ②得到了什么结果。\n"
    "5. 禁止在没有实际执行步骤的情况下声称任务完成。\n"
    "6. 不要执行任务描述范围之外的操作。\n"
    "7. 遇到工具调用失败时，换个方式继续，不要在最终回复中提及失败细节。"
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
        subagent_factory: Optional[Callable[..., "SubAgent"]] = None,
        agent_tasks_dir: Optional[Path] = None,  # 用于检查 .done 文件
        memory_retrieve_fn: Optional[
            Callable[[str, list[str] | None], Awaitable[list[dict]]]
        ] = None,
        memory_format_fn: Optional[Callable[[list[dict]], str]] = None,
    ) -> None:
        self._registry = registry
        self._rng = rng or _random_module.Random()
        self._records: dict[str, _ActionRecord] = {}
        self._state_path = state_path
        self._subagent_factory = subagent_factory
        self._agent_tasks_dir = agent_tasks_dir
        self._memory_retrieve_fn = memory_retrieve_fn
        self._memory_format_fn = memory_format_fn
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
                "[skill_action] agent 类型任务需要 subagent_factory，但未配置 id=%s",
                action.id,
            )
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""

        # 决定任务入口：优先读 TASK.md，回退到 task_prompt
        task_md_path = (
            self._agent_tasks_dir / action.id / "TASK.md"
            if self._agent_tasks_dir
            else None
        )
        has_task_md = task_md_path is not None and task_md_path.exists()

        if not has_task_md and not action.task_prompt.strip():
            logger.warning(
                "[skill_action] agent 任务无 TASK.md 且 task_prompt 为空 id=%s",
                action.id,
            )
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""

        try:
            system_prompt = await self._build_system_prompt(action, has_task_md)
            subagent = self._subagent_factory(action.id, system_prompt)
            if has_task_md:
                augmented_prompt = (
                    f"[任务ID: {action.id}]\n"
                    f"[工作目录: agent-tasks/{action.id}/]\n"
                    f"[共享配置目录: agent-tasks/shared/ — 内含 API keys 等公共配置]\n\n"
                    f"请按以下顺序开始：\n"
                    f'1. 调用 task_recall(namespace="{action.id}") 查询上次进度检查点\n'
                    f"2. 用 read_file 读取 agent-tasks/{action.id}/TASK.md，"
                    f"了解任务目标、用户最新补充说明和历史运行记录\n"
                    f"3. 结合进度检查点和 TASK.md，决定从哪里继续执行"
                )
            else:
                augmented_prompt = (
                    f"[任务ID: {action.id}]\n"
                    f"[工作目录: agent-tasks/{action.id}/]\n"
                    f"[共享配置目录: agent-tasks/shared/ — 内含 API keys 等公共配置]\n\n"
                    f"请按以下顺序开始：\n"
                    f'1. 调用 task_recall(namespace="{action.id}") 查询上次进度检查点\n'
                    f"2. 结合进度检查点，执行以下任务：\n\n"
                    f"{action.task_prompt}"
                )
            result = await subagent.run(augmented_prompt)
            exit_reason = getattr(subagent, "last_exit_reason", "completed")
            success = bool(result) and exit_reason == "completed"
            logger.info(
                "[skill_action] agent 任务完成 id=%s success=%s exit_reason=%s result_len=%d",
                action.id,
                success,
                exit_reason,
                len(result),
            )
            self._record_run(action.id, now, success=success)
            self._save_state()
            return success, result
        except Exception as e:
            logger.exception(
                "[skill_action] agent 任务异常 id=%s error=%s", action.id, e
            )
            self._record_run(action.id, now, success=False)
            self._save_state()
            return False, ""

    async def _build_system_prompt(
        self,
        action: SkillActionDef,
        has_task_md: bool,
    ) -> str:
        """按 action 相关性注入 procedure/preference 规则。"""
        if not self._memory_retrieve_fn or not self._memory_format_fn:
            return self._compose_system_prompt("", has_task_md)
        query = (action.task_prompt or "").strip()
        if not query:
            query = action.name.strip() or action.id

        try:
            items = await self._memory_retrieve_fn(
                query,
                ["procedure", "preference"],
            )
            block = (self._memory_format_fn(items) or "").strip()
            if block:
                logger.info(
                    "[skill_action] 注入 memory2 规则 id=%s hits=%d",
                    action.id,
                    len(items),
                )
            return self._compose_system_prompt(block, has_task_md)
        except Exception as e:
            logger.warning(
                "[skill_action] 注入 memory2 规则失败 id=%s err=%s", action.id, e
            )
            return self._compose_system_prompt("", has_task_md)

    @staticmethod
    def _compose_system_prompt(memory_block: str, has_task_md: bool) -> str:
        parts: list[str] = []
        block = (memory_block or "").strip()
        if block:
            parts.append(block)
        if has_task_md:
            parts.append(
                "## 任务入口约束\n"
                "本次任务目录存在 TASK.md。你必须先读 TASK.md，再按其中阶段顺序推进并在运行历史追加记录。"
            )
        parts.append(_AGENT_SYSTEM_PROMPT)
        return "\n\n".join(parts)

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
