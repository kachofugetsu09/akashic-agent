from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent.plugin_composition import (
    CONTINUATIONS,
    SCOPED_TURNS,
    TOOL_CATALOG,
    Context,
    PluginContinuations,
    PluginScopedTurns,
    PluginToolDefinition,
    ServiceView,
    ToolGrant,
    TurnExecutionScope,
)
from agent.prompting.section_names import RETRIEVED_MEMORY_SECTION
from agent.control.scoped_turn import ScopedTurnHandle
from agent.tools.base import Tool, ToolExecutionContext
from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.shell import ShellTaskStopTool, ShellTool, ShellWriteStdinTool
from .delegation import DelegationPolicy
from .prompts import build_spawn_subagent_prompt
from agent.turn_effects import PostCommitEffect, TurnStorage

api_version = 3
name = "subagent"
version = "3.0.0"
desc = "Recursive scoped Turn implementation for subagent work"
author = "Akashic Core"
inject = (SCOPED_TURNS, CONTINUATIONS, TOOL_CATALOG)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ("subagent-runs",)
workspace_files = ("memory/SELF.md", "memory/spawn_trace.jsonl")

logger = logging.getLogger(__name__)
_MAX_ACTIVE = 3
_RESULT_MAX_CHARS = 12_000
_SYNC_RESULT_MAX_CHARS = 100_000
_PROFILE_TOOLS: dict[str, tuple[str, ...]] = {
    "research": ("read_file", "list_dir", "web_fetch", "web_search"),
    "scripting": (
        "read_file",
        "list_dir",
        "write_file",
        "edit_file",
        "shell",
        "write_stdin",
        "task_stop",
    ),
    "general": (
        "read_file",
        "list_dir",
        "web_fetch",
        "web_search",
        "write_file",
        "edit_file",
        "shell",
        "write_stdin",
        "task_stop",
    ),
}


@dataclass(frozen=True, slots=True)
class ShadowSubagentResult:
    job_id: str
    session_id: str
    turn_id: str
    status: str
    response: str
    task_dir: str


@dataclass(frozen=True, slots=True)
class RunningJob:
    job_id: str
    label: str
    task: str
    profile: str
    origin_channel: str
    origin_chat_id: str
    task_dir: str
    retry_count: int
    started_at: str
    status: str = "running"


@dataclass(slots=True)
class _Child:
    handle: ScopedTurnHandle
    shell: ShellTool | None

    async def result(self) -> tuple[str, str, str]:
        result = await self.handle.result()
        response = result.final_response or ""
        status = result.status.value
        exit_reason = "completed" if status == "completed" else status
        return response, status, exit_reason

    async def interrupt(self) -> None:
        _ = await self.handle.interrupt()

    async def cleanup(self) -> None:
        try:
            await self.handle.cleanup()
        finally:
            if self.shell is not None:
                _ = await self.shell.shutdown()


@dataclass(slots=True)
class _Background:
    job: RunningJob
    child: _Child
    task: asyncio.Task[None]
    announced: bool = False


class _SubagentRuntime:
    """Own spawn admission, profiles, completion, traces, and cleanup."""

    def __init__(
        self,
        turns: PluginScopedTurns,
        continuations: PluginContinuations,
        task_root: Path,
        trace_path: Path,
    ) -> None:
        self._turns = turns
        self._continuations = continuations
        self._task_root = task_root
        self._trace_path = trace_path
        self._background: dict[str, _Background] = {}
        self._tokens: set[str] = set()
        self._closed = False

    @property
    def running_count(self) -> int:
        return len(self._tokens)

    async def shadow_run(
        self,
        context: ToolExecutionContext,
        arguments: Mapping[str, object],
    ) -> ShadowSubagentResult:
        """Run the S1-compatible private entry through the production child path."""

        task, profile = _task_and_profile(arguments)
        token = self._admit("shadow")
        try:
            job_id, task_dir = self._start_job("shadow", context, profile)
            child = await self._start_child(job_id, task, profile, task_dir, context)
            try:
                response, status, _ = await child.result()
            finally:
                await child.cleanup()
            self._trace(job_id, status, context.turn_id, profile, task_dir)
            return ShadowSubagentResult(
                job_id,
                child.handle.thread_id,
                child.handle.id,
                status,
                response,
                str(task_dir),
            )
        finally:
            self._tokens.remove(token)

    async def spawn_sync(
        self,
        context: ToolExecutionContext,
        *,
        task: str,
        label: str | None,
        profile: str,
    ) -> str:
        """Run one child and return its terminal result to the parent Tool call."""

        token = self._admit("sync")
        display_label = (label or task[:30]).strip()
        job_id, task_dir = self._start_job("sync", context, profile)
        child: _Child | None = None
        try:
            child = await self._start_child(job_id, task, profile, task_dir, context)
            response, status, exit_reason = await child.result()
            if status not in {"completed", "interrupted", "cancelled"}:
                response = response or "执行出错：child Turn failed"
                exit_reason = "error"
        except asyncio.CancelledError:
            if child is not None:
                await child.interrupt()
            raise
        except Exception as exc:
            logger.exception("[spawn_sync] child failed job_id=%s", job_id)
            response = f"执行出错：{exc}"
            exit_reason = "error"
        finally:
            if child is not None:
                await child.cleanup()
            self._tokens.remove(token)
        if len(response) > _SYNC_RESULT_MAX_CHARS:
            original_len = len(response)
            response = response[:_SYNC_RESULT_MAX_CHARS] + (
                f"\n...[结果已截断，原始长度 {original_len}]"
            )
        self._trace(job_id, "completed", context.turn_id, profile, task_dir)
        return f"[子任务「{display_label}」结果]\n退出原因: {exit_reason}\n\n{response}"

    async def spawn_background(
        self,
        context: ToolExecutionContext,
        *,
        task: str,
        label: str | None,
        profile: str,
        retry_count: int,
    ) -> str:
        """Accept one child Turn before returning the existing background receipt."""

        token = self._admit("background")
        job_id, task_dir = self._start_job("background", context, profile)
        display_label = (label or task[:30] or job_id).strip()
        try:
            child = await self._start_child(job_id, task, profile, task_dir, context)
            job = RunningJob(
                job_id,
                display_label,
                task,
                profile,
                context.origin_channel,
                context.origin_chat_id,
                str(task_dir),
                retry_count,
                datetime.now(UTC).isoformat(),
            )
            worker = asyncio.create_task(
                self._settle_background(job_id), name=f"subagent-plugin:{job_id}"
            )
            self._background[job_id] = _Background(job, child, worker)
            worker.add_done_callback(lambda done: self._finish(job_id, token, done))
        except BaseException:
            self._tokens.remove(token)
            raise
        return (
            f"已创建后台任务「{display_label}」（job_id={job_id}）。"
            "不要等待其完成；请直接向用户说明你已开始处理，完成后会继续回复。"
        )

    def list_jobs(self) -> list[dict[str, object]]:
        return [asdict(state.job) for state in self._background.values()]

    async def cancel(self, job_id: str) -> bool:
        state = self._background.get(job_id)
        if state is None or state.task.done():
            return False
        await self._announce(state, "cancelled", "后台任务已按请求取消。")
        await state.child.interrupt()
        return True

    async def close(self) -> None:
        self._closed = True
        for job_id in tuple(self._background):
            _ = await self.cancel(job_id)
        tasks = tuple(state.task for state in self._background.values())
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    def _admit(self, kind: str) -> str:
        if self._closed:
            raise RuntimeError("subagent plugin generation 已清理")
        if self.running_count >= _MAX_ACTIVE:
            raise RuntimeError(
                "subagent capacity reached: "
                f"active={self.running_count}, max={_MAX_ACTIVE}; current spawn rejected"
            )
        token = f"{kind}:{secrets.token_hex(8)}"
        self._tokens.add(token)
        return token

    def _start_job(
        self, kind: str, context: ToolExecutionContext, profile: str
    ) -> tuple[str, Path]:
        job_id = secrets.token_hex(4)
        task_dir = self._task_root / job_id
        task_dir.mkdir(parents=True, exist_ok=False)
        self._trace(job_id, "started", context.turn_id, profile, task_dir, kind=kind)
        return job_id, task_dir

    async def _start_child(
        self,
        job_id: str,
        task: str,
        profile: str,
        task_dir: Path,
        context: ToolExecutionContext,
    ) -> _Child:
        overrides, shell = _profile_overrides(profile, task_dir)
        session_id = await self._turns.create_session(
            metadata={
                "programmatic": True,
                "source": "subagent",
                "parent_turn_id": context.turn_id,
                "job_id": job_id,
                "profile": profile,
                "ephemeral": True,
            }
        )
        scope = TurnExecutionScope(
            prompt_hints=(
                build_spawn_subagent_prompt(self._task_root.parent, task_dir, profile),
            ),
            tool_grant=ToolGrant.only(_PROFILE_TOOLS[profile]),
            tool_overrides=overrides,
            disabled_prompt_sections=frozenset({RETRIEVED_MEMORY_SECTION}),
            storage=TurnStorage.IN_MEMORY,
            post_commit_effect=PostCommitEffect.SUPPRESS,
            tool_source="subagent",
        )
        handle = await self._turns.start(session_id, task, scope=scope)
        return _Child(handle, shell)

    async def _settle_background(self, job_id: str) -> None:
        state = self._background[job_id]
        try:
            response, status, exit_reason = await state.child.result()
            if not state.announced:
                await self._announce(state, exit_reason, response)
            self._trace(
                job_id,
                "cancelled" if status in {"cancelled", "interrupted"} else "completed",
                state.child.handle.id,
                state.job.profile,
                Path(state.job.task_dir),
                status=status,
                exit_reason=exit_reason,
            )
        finally:
            await state.child.cleanup()

    async def _announce(
        self, state: _Background, exit_reason: str, result: str
    ) -> None:
        if state.announced:
            return
        state.announced = True
        if len(result) > _RESULT_MAX_CHARS:
            original_len = len(result)
            result = result[:_RESULT_MAX_CHARS] + (
                f"\n...[结果已截断，原始长度 {original_len}]"
            )
        await self._continuations.submit(
            channel=state.job.origin_channel,
            chat_id=state.job.origin_chat_id,
            sender="spawn",
            content=_completion_message(state.job, exit_reason, result),
        )

    def _finish(self, job_id: str, token: str, task: asyncio.Task[None]) -> None:
        self._background.pop(job_id, None)
        self._tokens.remove(token)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "[spawn] background settlement failed job_id=%s error=%s",
                job_id,
                task.exception(),
            )

    def _trace(
        self,
        job_id: str,
        phase: str,
        parent_turn_id: str,
        profile: str,
        task_dir: Path,
        **extra: object,
    ) -> None:
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "job_id": job_id,
            "phase": phase,
            "parent_turn_id": parent_turn_id,
            "profile": profile,
            "task_dir": str(task_dir),
            "timestamp": datetime.now(UTC).isoformat(),
            **extra,
        }
        with self._trace_path.open("a", encoding="utf-8") as stream:
            _ = stream.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _spawn(
    bound: _SubagentRuntime,
    context: ToolExecutionContext,
    arguments: Mapping[str, object],
) -> str:
    task, profile = _task_and_profile(arguments)
    label = arguments.get("label")
    if label is not None and not isinstance(label, str):
        raise TypeError("label 必须是字符串")
    raw_retry_count = arguments.get("retry_count", 0)
    if isinstance(raw_retry_count, bool) or not isinstance(raw_retry_count, int):
        raise TypeError("retry_count 必须是整数")
    retry_count = max(0, raw_retry_count)
    decision = DelegationPolicy().decide(
        task=task, label=label, running_count=bound.running_count
    )
    if not decision.should_spawn:
        return f"任务被拦截：{decision.block_reason}"
    if bool(arguments.get("run_in_background", False)):
        if not context.origin_channel.strip() or not context.origin_chat_id.strip():
            return "错误：当前会话上下文缺失，无法创建后台任务"
        try:
            return await bound.spawn_background(
                context,
                task=task,
                label=label,
                profile=profile,
                retry_count=retry_count,
            )
        except RuntimeError as exc:
            return f"错误：{exc}"
    try:
        return await bound.spawn_sync(context, task=task, label=label, profile=profile)
    except RuntimeError as exc:
        return f"错误：{exc}"


async def _spawn_manage(
    bound: _SubagentRuntime,
    context: ToolExecutionContext,
    arguments: Mapping[str, object],
) -> str:
    _ = context
    action = arguments.get("action")
    if action == "list":
        return json.dumps(
            {"running_count": bound.running_count, "jobs": bound.list_jobs()},
            ensure_ascii=False,
        )
    if action == "cancel":
        job_id = str(arguments.get("job_id") or "").strip()
        if not job_id:
            return json.dumps({"error": "缺少 job_id"}, ensure_ascii=False)
        cancelled = await bound.cancel(job_id)
        return json.dumps(
            {
                "job_id": job_id,
                "status": "cancel_requested" if cancelled else "not_found",
            },
            ensure_ascii=False,
        )
    return json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False)


async def apply(ctx: Context, config: object) -> None:
    """Mount exact-generation production Tools and private runtime."""

    _ = config
    turns = ctx.require(SCOPED_TURNS)
    continuations = ctx.require(CONTINUATIONS)
    bound = _SubagentRuntime(
        turns,
        continuations,
        ctx.workspace_root("subagent-runs"),
        ctx.workspace_file("memory/spawn_trace.jsonl"),
    )

    async def spawn_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        return await _spawn(bound, context, arguments)

    async def manage_handler(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        return await _spawn_manage(bound, context, arguments)

    tools = ctx.require(TOOL_CATALOG)
    await tools.register(ctx, _spawn_definition(), spawn_handler)
    await tools.register(ctx, _manage_definition(), manage_handler)

    def setup() -> object:
        async def cleanup() -> None:
            await bound.close()

        return cleanup

    _ = await ctx.effect(setup, label="subagent-runtime")


def is_active(services: ServiceView) -> bool:
    turns = services.get(SCOPED_TURNS)
    continuations = services.get(CONTINUATIONS)
    return bool(
        turns is not None
        and turns.formal
        and continuations is not None
        and continuations.formal
    )


def _task_and_profile(arguments: Mapping[str, object]) -> tuple[str, str]:
    task = arguments.get("task")
    profile = arguments.get("profile", "research")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("subagent task 必须是非空字符串")
    if not isinstance(profile, str) or profile not in _PROFILE_TOOLS:
        raise ValueError(f"未知 subagent profile: {profile!r}")
    return task, profile


def _profile_overrides(
    profile: str, task_dir: Path
) -> tuple[dict[str, Tool], ShellTool | None]:
    if profile == "research":
        return {}, None
    shell = ShellTool(allow_network=profile == "general", working_dir=task_dir)
    tools: list[Tool] = [
        WriteFileTool(allowed_dir=task_dir),
        EditFileTool(allowed_dir=task_dir),
        shell,
        ShellWriteStdinTool(shell.manager),
        ShellTaskStopTool(shell.manager),
    ]
    return {tool.name: tool for tool in tools}, shell


def _completion_message(job: RunningJob, exit_reason: str, result: str) -> str:
    labels = {
        "completed": "正常完成",
        "failed": "执行出错",
        "error": "执行出错",
        "cancelled": "已取消",
        "interrupted": "已取消",
    }
    guidance = (
        "⚠️ 已重试一次，不再重试。\n" "请直接将已获得的结果汇报给用户。"
        if job.retry_count >= 1
        else (
            "**处理指引（按顺序判断，选其一执行）**\n"
            "1. 结果完整 → 直接向用户汇报，不提及内部机制\n"
            "2. 结果不完整 → 最多调用 spawn 重试一次\n"
            "3. 结果为空或出错 → 告知用户失败并询问是否重试。"
        )
    )
    return (
        "[后台任务回传]\n"
        f"任务标签: {job.label or '后台任务'}\n"
        f"原始任务: {job.task.strip() or '（未提供）'}\n"
        f"退出原因: {labels.get(exit_reason, exit_reason or '未知')}\n"
        f"执行结果:\n{result.strip() or '（无结果）'}\n\n{guidance}\n\n"
        "禁止在回复中提及 subagent、spawn、job_id、内部事件等内部概念。"
    )


def _spawn_definition() -> PluginToolDefinition:
    return PluginToolDefinition(
        name="spawn",
        description="""\
把一个有界的多步任务交给独立 subagent 执行，主 agent 专注决策和用户沟通。

何时使用 spawn（同时满足所有条件）：
- 预计需要 4 步以上工具调用
- 可以完全独立完成，中途不需要用户确认
- 产出是报告 / 文件 / 分析结论，而非"立刻执行的行动"

何时不用 spawn：
- 只需 1–3 次工具调用 → 直接调用工具，更快
- 直接回答问题（查询 / 解释 / 计算）→ 直接回答
- 任务需要修改当前会话状态（写 session memory）→ 主 agent 自己做
- 任务需要和用户来回确认才能推进
- 用户说"发送/告诉/立即执行"——需要立即生效的行动

执行模式（run_in_background）：
- false（默认）：同步执行，主会话等待结果后直接回复用户；适合研究后需要立即回答的任务，预计 ≤ 10 次工具调用
- true：后台执行，主会话立即继续，结果完成后系统带回；适合独立长任务，预计 > 60 秒或 > 15 次工具调用

工具权限 profile：
- research（默认）：只读调研，可搜索 / 读文件 / 抓网页，无法执行命令或写文件；大多数场景选此
- scripting：执行型，可运行 shell 命令 / 在任务目录写文件，无法访问网络
- general：两者兼有，仅在任务明确需要"边调研边执行"时使用

如何写好 task 参数：
subagent 没有看过当前会话。像给刚进房间的同事写交接文档：
- 任务目标：一句话说清楚产出物是什么
- 关键约束：格式 / 范围 / 截止 / 不能做什么
- 关键上下文：用户相关偏好、当前状态摘要、已经试过什么
- 期望输出格式：文本报告 / Markdown / JSON / 写入文件

同步模式调用后主 agent 等待结果再回复用户。
后台模式调用后本轮只做简短确认，结果完成后系统会带回当前会话继续处理。\
""",
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "交给 subagent 的完整任务描述。必须包含：\n"
                        "1. 任务目标（一句话，说清楚产出物）\n"
                        "2. 关键约束（格式 / 范围 / 截止）\n"
                        "3. 关键上下文（用户偏好、当前状态、已试过什么）\n"
                        "4. 期望输出格式"
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "3–5 字的任务短标签，用于状态显示",
                },
                "profile": {
                    "type": "string",
                    "enum": ["research", "scripting", "general"],
                    "description": (
                        "subagent 的工具权限配置：\n"
                        "- research（默认）：只读调研，可搜索 / 读文件 / 抓网页\n"
                        "- scripting：执行型，可运行 shell 命令 / 在任务目录写文件\n"
                        "- general：两者兼有，仅在明确需要时使用"
                    ),
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "false（默认）：同步执行，主会话等待结果后直接回复用户。\n"
                        "true：后台执行，主会话立即继续，结果完成后系统带回。"
                    ),
                },
                "retry_count": {
                    "type": "integer",
                    "description": "当前后台任务已重试次数。首次调用为 0，重试时传 1。",
                    "minimum": 0,
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        handler_export="spawn",
        risk="read-write",
        always_on=True,
        search_hint="后台执行 子任务 多步调研 独立任务",
    )


def _manage_definition() -> PluginToolDefinition:
    return PluginToolDefinition(
        name="spawn_manage",
        description="""\
管理当前运行中的后台 subagent。

可用 action：
- list：列出正在运行的后台任务，包含 job_id、label、profile、task_dir、任务摘要和启动时间
- cancel：按 job_id 取消后台任务；取消后系统会把“已取消”作为后台任务完成事件回灌当前会话

只在用户询问后台任务状态、要求查看 job_id、或明确要求停止某个后台任务时使用。\
""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "cancel"],
                    "description": "list 查看运行中任务；cancel 取消指定 job_id",
                },
                "job_id": {
                    "type": "string",
                    "description": "action=cancel 时要取消的后台任务 job_id",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        handler_export="spawn_manage",
        risk="external-side-effect",
        always_on=True,
        search_hint="查看 取消 后台任务 subagent job_id",
    )
