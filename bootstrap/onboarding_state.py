from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

OnboardingStep = Literal["welcome", "model", "memory", "channel", "done"]
OnboardingDecision = Literal["pending", "configured", "skipped"]

_VERSION = 1
_STEPS = {"welcome", "model", "memory", "channel", "done"}
_DECISIONS = {"pending", "configured", "skipped"}


@dataclass(frozen=True)
class OnboardingState:
    """记录 onboarding 进度，不复制任何运行配置或凭据。"""

    step: OnboardingStep
    completed: bool
    memory_decision: OnboardingDecision
    channel_decision: OnboardingDecision
    updated_at: str

    def public(self) -> dict[str, object]:
        return {
            "step": self.step,
            "completed": self.completed,
            "memoryDecision": self.memory_decision,
            "channelDecision": self.channel_decision,
        }


def read_onboarding_state(
    workspace: Path,
    *,
    model_configured: bool,
    memory_configured: bool,
    channel_configured: bool,
) -> OnboardingState:
    """读取显式进度；没有状态文件时只为旧 workspace 推导兼容视图。"""
    path = onboarding_state_path(workspace)
    if path.is_file():
        return _decode_state(path.read_text(encoding="utf-8"))
    if not model_configured:
        return _new_state("welcome")
    if memory_configured and channel_configured:
        return _new_state(
            "done",
            completed=True,
            memory_decision="configured",
            channel_decision="configured",
        )
    if memory_configured:
        return _new_state("channel", memory_decision="configured")
    return _new_state("memory")


def start_onboarding(workspace: Path, *, model_configured: bool) -> OnboardingState:
    """开始或重新开始完整引导，从模型确认步骤进入。"""
    _ = model_configured
    state = _new_state("model")
    _write_state(workspace, state)
    return state


def advance_onboarding(
    workspace: Path,
    *,
    expected_step: Literal["model", "memory", "channel"],
    decision: Literal["configured", "skipped"] | None = None,
) -> OnboardingState:
    """校验当前步骤并记录一次不可跳跃的用户决定。"""
    state = _read_explicit_state(workspace)
    if state.step != expected_step or state.completed:
        raise ValueError(f"onboarding 当前位于 {state.step}，不能提交 {expected_step}")

    if expected_step == "model":
        if decision is not None:
            raise ValueError("模型步骤不接受可选决定")
        next_state = replace(state, step="memory", updated_at=_now())
    elif expected_step == "memory":
        if decision is None:
            raise ValueError("记忆步骤必须选择已配置或稍后设置")
        next_state = replace(
            state,
            step="channel",
            memory_decision=decision,
            updated_at=_now(),
        )
    else:
        if decision is None:
            raise ValueError("联系方式步骤必须选择已配置或稍后设置")
        next_state = replace(
            state,
            step="done",
            channel_decision=decision,
            updated_at=_now(),
        )
    _write_state(workspace, next_state)
    return next_state


def go_back_onboarding(workspace: Path) -> OnboardingState:
    """退回上一步，并清除从该步之后产生的选择。"""

    state = _read_explicit_state(workspace)
    if state.completed:
        raise ValueError("已完成的 onboarding 需要重新开始，不能直接返回")
    if state.step == "memory":
        next_state = replace(
            state,
            step="model",
            memory_decision="pending",
            channel_decision="pending",
            updated_at=_now(),
        )
    elif state.step == "channel":
        next_state = replace(
            state,
            step="memory",
            memory_decision="pending",
            channel_decision="pending",
            updated_at=_now(),
        )
    elif state.step == "done":
        next_state = replace(
            state,
            step="channel",
            completed=False,
            channel_decision="pending",
            updated_at=_now(),
        )
    else:
        raise ValueError(f"onboarding 当前位于 {state.step}，没有可返回的步骤")
    _write_state(workspace, next_state)
    return next_state


def complete_onboarding(workspace: Path) -> OnboardingState:
    """在所有选择已确认后持久化 onboarding 完成状态。"""
    state = _read_explicit_state(workspace)
    if state.completed:
        return state
    if state.step != "done":
        raise ValueError(f"onboarding 当前位于 {state.step}，尚不能完成")
    next_state = replace(state, completed=True, updated_at=_now())
    _write_state(workspace, next_state)
    return next_state


def onboarding_state_path(workspace: Path) -> Path:
    return workspace / "onboarding-state.json"


def _read_explicit_state(workspace: Path) -> OnboardingState:
    path = onboarding_state_path(workspace)
    if not path.is_file():
        raise ValueError("onboarding 尚未开始")
    return _decode_state(path.read_text(encoding="utf-8"))


def _new_state(
    step: OnboardingStep,
    *,
    completed: bool = False,
    memory_decision: OnboardingDecision = "pending",
    channel_decision: OnboardingDecision = "pending",
) -> OnboardingState:
    return OnboardingState(
        step=step,
        completed=completed,
        memory_decision=memory_decision,
        channel_decision=channel_decision,
        updated_at=_now(),
    )


def _decode_state(source: str) -> OnboardingState:
    """严格解析状态文件，让损坏和不可能状态在 owner 边界暴露。"""
    payload = json.loads(source)
    if not isinstance(payload, dict) or payload.get("version") != _VERSION:
        raise ValueError("onboarding 状态版本无效")
    step = payload.get("step")
    completed = payload.get("completed")
    memory_decision = payload.get("memory_decision")
    channel_decision = payload.get("channel_decision")
    updated_at = payload.get("updated_at")
    if step not in _STEPS or type(completed) is not bool:
        raise ValueError("onboarding 状态字段无效")
    if memory_decision not in _DECISIONS or channel_decision not in _DECISIONS:
        raise ValueError("onboarding 选择字段无效")
    if not isinstance(updated_at, str) or not updated_at:
        raise ValueError("onboarding 更新时间无效")
    state = OnboardingState(
        step=cast(OnboardingStep, step),
        completed=completed,
        memory_decision=cast(OnboardingDecision, memory_decision),
        channel_decision=cast(OnboardingDecision, channel_decision),
        updated_at=updated_at,
    )
    _validate_state(state)
    return state


def _validate_state(state: OnboardingState) -> None:
    if state.completed and state.step != "done":
        raise ValueError("已完成的 onboarding 必须停留在完成步骤")
    if state.step in {"welcome", "model", "memory"} and (
        state.memory_decision != "pending" or state.channel_decision != "pending"
    ):
        raise ValueError("onboarding 选择顺序无效")
    if state.step == "channel" and (
        state.memory_decision == "pending" or state.channel_decision != "pending"
    ):
        raise ValueError("onboarding 记忆选择尚未完成")
    if state.step == "done" and (
        state.memory_decision == "pending" or state.channel_decision == "pending"
    ):
        raise ValueError("onboarding 可选步骤尚未完成")


def _write_state(workspace: Path, state: OnboardingState) -> None:
    """以 0600 权限原子替换状态文件，避免半写状态被当成有效进度。"""
    _validate_state(state)
    workspace.mkdir(parents=True, exist_ok=True)
    path = onboarding_state_path(workspace)
    payload = json.dumps(
        {
            "version": _VERSION,
            "step": state.step,
            "completed": state.completed,
            "memory_decision": state.memory_decision,
            "channel_decision": state.channel_decision,
            "updated_at": state.updated_at,
        },
        ensure_ascii=False,
        indent=2,
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=workspace)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _now() -> str:
    return datetime.now(UTC).isoformat()
