from __future__ import annotations

import json
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent.plugin_composition import (
    Context,
    PluginScopedTurns,
    SCOPED_TURNS,
    ServiceView,
    ToolGrant,
    TurnExecutionScope,
)
from agent.tools.base import ToolExecutionContext

api_version = 3
name = "subagent"
version = "3.0.0"
desc = "Recursive scoped Turn implementation for subagent shadow validation"
author = "Akashic Core"
inject = (SCOPED_TURNS,)
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ("subagent-runs",)

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


class _SubagentRuntime:
    """Own private profile, task directory, and trace composition."""

    def __init__(
        self,
        turns: PluginScopedTurns,
        task_root: Path,
        data_root: Path,
    ) -> None:
        self._turns = turns
        self._task_root = task_root
        self._data_root = data_root
        self._closed = False

    async def run(
        self,
        context: ToolExecutionContext,
        arguments: Mapping[str, object],
    ) -> ShadowSubagentResult:
        """Admit one shadow child and settle its Core-owned handle."""

        # 1. Plugin owns profile admission and provisional filesystem state.
        if self._closed:
            raise RuntimeError("subagent plugin generation 已清理")
        task = arguments.get("task")
        profile = arguments.get("profile", "research")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("subagent task 必须是非空字符串")
        if not isinstance(profile, str) or profile not in _PROFILE_TOOLS:
            raise ValueError(f"未知 subagent profile: {profile!r}")
        job_id = secrets.token_hex(4)
        task_dir = self._task_root / job_id
        task_dir.mkdir(parents=True, exist_ok=False)
        self._append_trace(job_id, "started", context.turn_id, profile)

        # 2. Core owns the recursive Turn from accepted receipt through cleanup.
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
            prompt_hints=(_profile_prompt(profile, task_dir),),
            tool_grant=ToolGrant.only(_PROFILE_TOOLS[profile]),
            memory_read=False,
            memory_write=False,
            stateless=True,
            tool_source="subagent",
        )
        handle = await self._turns.start(session_id, task, scope=scope)
        result = await handle.result()

        # 3. Plugin projects the terminal fact without changing Core settlement.
        self._append_trace(job_id, result.status.value, context.turn_id, profile)
        return ShadowSubagentResult(
            job_id=job_id,
            session_id=session_id,
            turn_id=handle.accepted.turn_id,
            status=result.status.value,
            response=result.final_response or "",
            task_dir=str(task_dir),
        )

    def close(self) -> None:
        self._closed = True

    def _append_trace(
        self,
        job_id: str,
        phase: str,
        parent_turn_id: str,
        profile: str,
    ) -> None:
        self._data_root.mkdir(parents=True, exist_ok=True)
        record = {
            "job_id": job_id,
            "phase": phase,
            "parent_turn_id": parent_turn_id,
            "profile": profile,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        with (self._data_root / "shadow_trace.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            _ = stream.write(json.dumps(record, ensure_ascii=False) + "\n")


runtime: _SubagentRuntime | None = None


async def shadow_run(
    context: ToolExecutionContext,
    arguments: Mapping[str, object],
) -> ShadowSubagentResult:
    """Run the non-published S1 entrypoint for differential fixtures."""

    if runtime is None:
        raise RuntimeError("subagent plugin runtime 未激活")
    return await runtime.run(context, arguments)


async def apply(ctx: Context, config: object) -> None:
    """Bind the public scoped Turn service without publishing a production Tool."""

    _ = config

    def setup() -> object:
        global runtime
        if runtime is not None:
            raise RuntimeError("subagent plugin runtime 重复激活")
        bound = _SubagentRuntime(
            ctx.require(SCOPED_TURNS),
            ctx.workspace_root("subagent-runs"),
            ctx.data_root,
        )
        runtime = bound

        def cleanup() -> None:
            global runtime
            bound.close()
            if runtime is bound:
                runtime = None

        return cleanup

    _ = await ctx.effect(setup, label="subagent-shadow-runtime")


def is_active(services: ServiceView) -> bool:
    turns = services.get(SCOPED_TURNS)
    return turns is not None and turns.formal


def _profile_prompt(profile: str, task_dir: Path) -> str:
    if profile == "research":
        return (
            "你是只读调研型子 agent。禁止 shell 和文件写入；直接返回调查报告。"
            f"任务目录仅用于运行身份：{task_dir}"
        )
    if profile == "scripting":
        return (
            "你是执行型子 agent。禁止网络，写入只允许当前任务目录："
            f"{task_dir}。完成后直接返回结果。"
        )
    return (
        "你是通用型子 agent。不得再次 spawn，写入只允许当前任务目录："
        f"{task_dir}。完成后直接返回结果。"
    )
