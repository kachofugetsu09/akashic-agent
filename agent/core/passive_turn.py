from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal, cast

import agent.core.passive_support as support
from agent.control.context import running_turn_id
from core.common.diagnostic_log import (
    diagnostic_context,
    diagnostic_line,
    turn_milestone,
)
from core.error_context import (
    current_client_message_id,
    current_provider_attempt,
    current_provider_call_id,
    current_provider_operation,
    current_session_key,
)
from agent.control.ports import InputLock, TurnUserInput
from agent.core.runtime_support import ToolDiscoveryState
from agent.core.types import (
    ContextBundle,
    ReasonerResult,
)
from agent.prompting import is_context_frame
from agent.plugin_composition import (
    BoundChatModel,
    ChatModels,
    ContentSafetyError,
    ContextLengthError,
    LLMResponse,
    ModelContinuation,
    ModelRequest,
    ModelRole,
    ModelUsage,
    PreparedProviderRequest,
    PROVIDER_REQUEST_PROJECTION,
    ProviderRequestBinding,
    ProviderRequestGate,
    ProviderRequestProjection,
    ProviderProjectionError,
    ProviderTurnInput,
    ProviderTurnProjection,
    RequestHistoryUnit,
    ToolCall,
    UsageCoverage,
)
from agent.tool_runtime import (
    append_assistant_tool_calls,
    append_tool_result,
    tool_call_batch_snapshot,
)
from agent.tools.base import normalize_tool_result
from agent.tools.events import ToolExecutionRequest, ToolExecutionResult, ToolGrant
from agent.control.turn_scope import get_current_turn_scope
from agent.tools.executor import ToolExecutor
from agent.tools.registry import begin_turn_search_scope, end_turn_search_scope
from agent.turns.outbound import OutboundDispatch, OutboundPort
from agent.plugin_composition.channels import ChannelDeliveryReceipt
from bus.event_bus import EventBus
from bus.events import (
    InboundMessage,
    OutboundMessage,
    TurnDisposition,
    TurnTerminalStatus,
)
from bus.events_lifecycle import (
    ToolCallCompleted,
    ToolCallStarted,
    TurnOutputCompleted,
)
from agent.lifecycle.phase import Phase
from agent.lifecycle.phases.after_reasoning import (
    AfterReasoningFrame,
    default_after_reasoning_modules,
)
from agent.lifecycle.phases.after_step import AfterStepFrame, default_after_step_modules
from agent.lifecycle.phases.after_turn import AfterTurnFrame, default_after_turn_modules
from agent.lifecycle.phases.before_reasoning import (
    BeforeReasoningFrame,
    default_before_reasoning_modules,
)
from agent.lifecycle.phases.before_step import (
    BeforeStepFrame,
    default_before_step_modules,
)
from agent.lifecycle.phases.before_turn import (
    BeforeTurnFrame,
    default_before_turn_modules,
)
from agent.lifecycle.phases.prompt_render import (
    PromptRenderFrame,
    default_prompt_render_modules,
)
from agent.lifecycle.types import (
    AfterReasoningInput,
    AfterStepCtx,
    AfterToolResultCtx,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeToolCallCtx,
    BeforeTurnCtx,
    PromptRenderInput,
    PromptRenderResult,
    TurnSnapshot,
    TurnState,
    TurnPersistencePolicy,
)
from agent.plugins.snapshot import get_current_runtime_snapshot

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.runtime_support import SessionLike, TurnRunResult
    from agent.looping.ports import LLMConfig, SessionServices
    from agent.tools.registry import ToolRegistry

# 1. 统一通过模块 logger 记录关键分支，供排障和回归测试抓取。
logger = logging.getLogger(__name__)


def _host_runtime_execution_hint() -> str:
    if os.environ.get("AKASHIC_EXECUTION_MODE", "local") != "host-bridge":
        return ""
    commit = os.environ.get("AKASHIC_RUNTIME_COMMIT", "")
    checkout = os.environ.get("AKASHIC_RUNTIME_CHECKOUT", "")
    if not commit or not checkout:
        raise RuntimeError("host-bridge 模式缺少运行时 commit/checkout")
    return (
        "【容器运行时】当前 Core commit="
        f"{commit}，宿主只读参考源码={checkout}。Shell 在宿主执行；"
        "调用当前运行时控制命令必须使用 akashic-runtime（或 $AKASHIC_RUNTIME_CLI），"
        "不要用 host 的 python 直接运行 checkout/main.py。调试修复请从该 commit 新建 worktree。"
    )


def _persistence_from_metadata(
    metadata: dict[str, Any] | None,
) -> TurnPersistencePolicy:
    return TurnPersistencePolicy(
        persist_user=not bool((metadata or {}).get("omit_user_turn")),
        persist_assistant=not bool((metadata or {}).get("omit_assistant_turn")),
    )


# 被动链路核心入口，负责串起 lifecycle 模块链与 reasoner。
#
# ┌─ 输入
# │  └─ AgentLoop._react
# │     └─ PassiveTurnPipeline.run
# │        ├─ BeforeTurn
# │        │  └─ 获取 session + ContextStore.prepare + EventBus.emit
# │        ├─ BeforeReasoning
# │        │  └─ 同步工具上下文 + EventBus.emit + prompt 预热
# │        ├─ Reasoner.run_turn
# │        │  ├─ PromptRender
# │        │  │  └─ ContextBuilder.render + plugin prompt 模块
# │        │  └─ Reasoner.run
# │        │     ├─ BeforeStep
# │        │     │  └─ token 估算 + EventBus.emit + 注入提示
# │        │     └─ AfterStep
# │        │        └─ EventBus.fanout
# │        ├─ AfterReasoning
# │        │  └─ 解析 + EventBus.emit + 持久化 + 构建出站消息
# │        └─ AfterTurn
# │           └─ 广播 TurnCommitted + 广播 AfterTurn + dispatch
# └─ 完成

# ── 被动 turn 内联常量 ──────────────────────────────────────────
_SUMMARY_MAX_TOKENS = 512
_INCOMPLETE_SUMMARY_PROMPT = """当前任务需要先暂停继续调用工具，请直接输出给用户看的中文阶段性回复。
必须基于已有上下文，不要编造结果。
必须包含四点：
1) 已经使用了哪些工具或操作，以及拿到了什么关键信息；
2) 当前已经做到哪一步；
3) 还缺什么信息或步骤；
4) 如果继续，下一步会怎么做。
可以提到工具名称和关键结果，但不要暴露 tool_call_id、schema、内部 prompt 或原始参数 JSON。
禁止输出"已达到最大迭代次数"这类模板句；不要输出 JSON。"""


@dataclass(frozen=True)
class _ProviderCallResult:
    response: LLMResponse
    prepared: PreparedProviderRequest | None
    auxiliary_usages: tuple[ModelUsage, ...] = ()


@dataclass(frozen=True)
class _ProviderAttemptIdentity:
    """一次逻辑 provider 调用 + 其中某次 attempt 的统一身份（1-based）。

    通过 contextvar 共享给 request projection gate、provider attempt 里程碑和
    first-delta 观测，保证同 call/attempt 的所有事件可以互相 join。
    """

    call_ordinal: int
    provider_attempt: int = 0


_provider_call_identity: ContextVar[_ProviderAttemptIdentity | None] = ContextVar(
    "provider_call_identity",
    default=None,
)


@dataclass
class _TurnRequestState:
    gate: ProviderRequestGate
    agent_model: BoundChatModel
    # 本 turn 内 provider 调用序号与里程碑状态（随 turn state 自然释放）；
    # call_started_at 是当前 provider attempt 的启动时刻（attempt 2 会重置）。
    provider_call_ordinal: int = 0
    call_started_at: float = 0.0
    first_any_logged: bool = False
    first_thinking_logged: bool = False
    first_answer_logged: bool = False
    continuation: ModelContinuation | None = None


class _PassThroughGate:
    """Preserve the provider request unchanged when no projector is active."""

    def __init__(self, binding: ProviderRequestBinding) -> None:
        self._model = cast(BoundChatModel, binding.agent_model)
        self._pending_start = 1 + binding.history_count

    @property
    def pending_start(self) -> int:
        return self._pending_start

    async def prepare(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        max_output_tokens: int | None,
        trigger: str,
        force: bool,
    ) -> PreparedProviderRequest:
        _ = max_output_tokens, trigger, force
        return PreparedProviderRequest(
            pending_start=self._pending_start,
            estimated_tokens=self._model.estimate_context_tokens(messages, tools),
            token_quality="estimated",
            changed=False,
        )

    def can_retry_context_error(self, *, context_window: int) -> bool:
        _ = context_window
        return False

    def record_completed_batch(
        self,
        messages: list[dict[str, Any]],
        *,
        batch_start: int,
    ) -> None:
        _ = messages, batch_start

    async def record_response(self, **kwargs: Any) -> None:
        _ = kwargs


class _PassThroughTurn:
    """Default history projection used when the optional plugin is absent."""

    def __init__(self, history: list[dict[str, Any]]) -> None:
        self._history = tuple(dict(message) for message in history)

    @property
    def history(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(message) for message in self._history)

    def bind(self, binding: ProviderRequestBinding) -> ProviderRequestGate:
        return _PassThroughGate(binding)


def _turn_log_id(key: str, msg: InboundMessage) -> str:
    persisted = msg.metadata.get("turnId")
    if isinstance(persisted, str) and persisted:
        return persisted
    raw = f"{key}|{msg.timestamp.isoformat()}|{msg.content[:80]}"
    return f"local-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _phase_error_reason(phase: str) -> str:
    return {
        "command": "command_error",
        "before_turn": "before_turn_error",
        "before_reasoning": "before_reasoning_error",
        "reasoner": "provider_error",
    }[phase]


def _disabled_tools_from_msg(msg: object) -> set[str]:
    metadata: object = getattr(msg, "metadata", None)
    if not isinstance(metadata, dict):
        return set()
    raw = metadata.get("disabled_tools")
    if isinstance(raw, str):
        return {raw} if raw else set()
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw if str(item)}
    return set()


class _NoopOutboundPort:
    async def dispatch(self, outbound: OutboundDispatch) -> ChannelDeliveryReceipt:
        _ = outbound
        raise RuntimeError("PassiveTurnPipeline committed Channel outbound port 未绑定")


@dataclass
class PassiveTurnDeps:
    session: "SessionServices"
    context_store: "ContextStore"
    context: "ContextBuilder"
    tools: "ToolRegistry"
    reasoner: "Reasoner"
    event_bus: "EventBus | None" = None
    outbound_port: "OutboundPort | None" = None


@dataclass
class _PassivePhaseBundle:
    before_turn: Phase[TurnState, BeforeTurnCtx, BeforeTurnFrame]
    before_reasoning: Phase[
        BeforeReasoningInput,
        BeforeReasoningCtx,
        BeforeReasoningFrame,
    ]
    after_reasoning: Phase[
        AfterReasoningInput,
        TurnSnapshot,
        AfterReasoningFrame,
    ]
    after_turn: Phase[TurnSnapshot, OutboundMessage, AfterTurnFrame]


class PassiveTurnPipeline:
    """
    ┌──────────────────────────────────────┐
    │ PassiveTurnPipeline                  │
    ├──────────────────────────────────────┤
    │ 1. BeforeTurn（会话准备）             │
    │ 2. BeforeReasoning                   │
    │ 3. 执行 reasoner（含 BeforeStep/AfterStep）│
    │ 4. AfterReasoning（parse + 持久化 + 构建出站消息）│
    │ 5. AfterTurn（TurnCommitted + dispatch） │
    │ 6. 返回出站消息                      │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: PassiveTurnDeps) -> None:
        self._session = deps.session
        self._context_store = deps.context_store
        self._context = deps.context
        self._tools = deps.tools
        self._reasoner = deps.reasoner
        self._outbound_port = deps.outbound_port or _NoopOutboundPort()
        bus = deps.event_bus or EventBus()
        self._bus = bus

        self._rebuild_phases()

    def _rebuild_phases(self) -> None:
        self._phases = _PassivePhaseBundle(
            before_turn=self._build_before_turn_phase(),
            before_reasoning=self._build_before_reasoning_phase(),
            after_reasoning=self._build_after_reasoning_phase(),
            after_turn=self._build_after_turn_phase(),
        )

    def _build_before_turn_phase(
        self,
    ) -> Phase[TurnState, BeforeTurnCtx, BeforeTurnFrame]:
        return Phase(
            default_before_turn_modules(
                self._bus,
                self._session.session_manager,
                self._context_store,
            ),
            frame_factory=BeforeTurnFrame,
        )

    def _build_before_reasoning_phase(
        self,
    ) -> Phase[BeforeReasoningInput, BeforeReasoningCtx, BeforeReasoningFrame]:
        return Phase(
            default_before_reasoning_modules(
                self._bus,
                self._tools,
                self._session.session_manager,
                self._context,
            ),
            frame_factory=BeforeReasoningFrame,
        )

    def _build_after_reasoning_phase(
        self,
    ) -> Phase[AfterReasoningInput, TurnSnapshot, AfterReasoningFrame]:
        return Phase(
            default_after_reasoning_modules(
                self._bus,
                self._session,
            ),
            frame_factory=AfterReasoningFrame,
        )

    def _build_after_turn_phase(
        self,
    ) -> Phase[TurnSnapshot, OutboundMessage, AfterTurnFrame]:
        return Phase(
            default_after_turn_modules(
                self._bus,
                self._outbound_port,
                self._context,
            ),
            frame_factory=AfterTurnFrame,
        )

    def _runtime_phases(self) -> _PassivePhaseBundle:
        return self._phases

    async def run_command(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage | None:
        """Execute one committed command before Session and model admission."""

        # 1. Only the command catalog frozen into this turn's snapshot may handle it.
        snapshot = get_current_runtime_snapshot()
        command_registry = snapshot.command_registry if snapshot is not None else None
        if command_registry is None:
            return None
        started = time.perf_counter()
        turn_id = _turn_log_id(key, msg)
        state = TurnState(
            msg=msg,
            session_key=key,
            dispatch_outbound=dispatch_outbound,
            persistence=_persistence_from_metadata(msg.metadata),
        )
        try:
            execution = await command_registry.execute(
                msg.content,
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                sender=msg.sender,
            )
        except Exception as exc:
            logger.exception(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="phase_error",
                    flow="passive",
                    phase="command",
                    session=key,
                    turn=turn_id,
                    action="fail",
                    reason=_phase_error_reason("command"),
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    error_type=type(exc).__name__,
                    note=str(exc)[:160],
                )
            )
            if not dispatch_outbound:
                raise
            return await self._control_outbound(
                state,
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="处理消息时出错，请稍后再试。",
                ),
            )
        if execution is None:
            return None

        # 2. A known command settles through the normal outbound owner without a Turn.
        logger.info(
            diagnostic_line(
                "PassiveTurnPipeline.run",
                event="gate_exit",
                flow="passive",
                phase="command",
                session=key,
                turn=turn_id,
                action="short_circuit",
                reason=execution.name,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
        )
        return await self._control_outbound(
            state,
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=execution.result.text,
                turn_disposition=TurnDisposition.SHORT_CIRCUITED,
            ),
        )

    # 核心方法：处理一条普通被动消息，并提交最终出站结果。
    async def run(
        self,
        msg: InboundMessage,
        key: str,
        *,
        chat_models: ChatModels | None = None,
        model_id: str | None = None,
        reasoning_effort: str | None = None,
        dispatch_outbound: bool = True,
        command_admitted: bool = False,
    ) -> OutboundMessage:
        started = time.perf_counter()
        phase_started = started
        active_phase = "before_turn"
        turn_id = _turn_log_id(key, msg)
        state = TurnState(
            msg=msg,
            session_key=key,
            dispatch_outbound=dispatch_outbound,
            persistence=_persistence_from_metadata(msg.metadata),
        )
        with diagnostic_context(session=key, flow="passive", turn=turn_id):
            logger.info(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="start",
                    flow="passive",
                    phase="before_turn",
                    session=key,
                    turn=turn_id,
                    action="run",
                )
            )
            # try/except 只包前置模块链和 reasoning：在派发前兜底并返回错误提示。
            try:
                # Phase 0: stable command catalog 在 Session 与模型调用前短路。
                if not command_admitted:
                    active_phase = "command"
                    command_result = await self.run_command(
                        msg,
                        key,
                        dispatch_outbound=dispatch_outbound,
                    )
                    if command_result is not None:
                        return command_result

                # Phase 1: BeforeTurn 模块链（会话、上下文、BeforeTurn 事件）。
                active_phase = "before_turn"
                phase_started = time.perf_counter()
                with diagnostic_context(phase="before_turn"):
                    before_turn = await self._runtime_phases().before_turn.run(state)
                # TurnState 存内部默认 metadata；BeforeTurnCtx 存插件导出，同名 key 以后者覆盖。
                state.extra_metadata.update(before_turn.extra_metadata)
                if before_turn.abort:
                    logger.info(
                        diagnostic_line(
                            "PassiveTurnPipeline.run",
                            event="gate_exit",
                            flow="passive",
                            phase="before_turn",
                            session=key,
                            turn=turn_id,
                            action="abort",
                            reason="before_turn_abort",
                            duration_ms=int(
                                (time.perf_counter() - phase_started) * 1000
                            ),
                        )
                    )
                    return await self._control_outbound(
                        state,
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=before_turn.abort_reply,
                            turn_disposition=TurnDisposition.SHORT_CIRCUITED,
                        ),
                    )
                logger.info(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="end",
                        flow="passive",
                        phase="before_turn",
                        session=key,
                        turn=turn_id,
                        action="continue",
                        duration_ms=int((time.perf_counter() - phase_started) * 1000),
                    )
                )

                # Phase 2: BeforeReasoning 模块链（工具上下文、BeforeReasoning 事件、prompt warmup）。
                active_phase = "before_reasoning"
                phase_started = time.perf_counter()
                with diagnostic_context(phase="before_reasoning"):
                    before_reasoning = (
                        await self._runtime_phases().before_reasoning.run(
                            BeforeReasoningInput(state=state, before_turn=before_turn)
                        )
                    )
                if before_reasoning.abort:
                    logger.info(
                        diagnostic_line(
                            "PassiveTurnPipeline.run",
                            event="gate_exit",
                            flow="passive",
                            phase="before_reasoning",
                            session=key,
                            turn=turn_id,
                            action="abort",
                            reason="before_reasoning_abort",
                            duration_ms=int(
                                (time.perf_counter() - phase_started) * 1000
                            ),
                        )
                    )
                    return await self._control_outbound(
                        state,
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=before_reasoning.abort_reply,
                            turn_disposition=TurnDisposition.SHORT_CIRCUITED,
                        ),
                    )
                if msg.metadata.get("_pluginCandidateValidation") is True:
                    state.extra_metadata["_activeSkillNames"] = list(
                        before_reasoning.skill_names
                    )
                reasoning_hints = list(before_reasoning.extra_hints)
                runtime_hint = _host_runtime_execution_hint()
                if runtime_hint:
                    reasoning_hints.append(runtime_hint)
                logger.info(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="end",
                        flow="passive",
                        phase="before_reasoning",
                        session=key,
                        turn=turn_id,
                        action="continue",
                        counts=f"skills:{len(before_reasoning.skill_names)},hints:{len(reasoning_hints)}",
                        duration_ms=int((time.perf_counter() - phase_started) * 1000),
                    )
                )

                # Phase 3-4: Reasoning（BeforeStep/AfterStep 模块链在 Reasoner 内部执行）。
                active_phase = "reasoner"
                phase_started = time.perf_counter()
                session = state.session
                if session is None:
                    raise RuntimeError("Passive turn requires TurnState.session")

                async def run_reasoner(
                    current_agent_model: BoundChatModel,
                    current_fallback_model: BoundChatModel,
                ) -> TurnRunResult:
                    if isinstance(msg, InboundMessage):
                        msg.metadata["model_binding"] = _model_binding_payload(
                            current_agent_model
                        )
                    with diagnostic_context(phase="reasoner"):
                        return await self._reasoner.run_turn(
                            msg=msg,
                            agent_model=current_agent_model,
                            fallback_model=current_fallback_model,
                            skill_names=list(before_reasoning.skill_names) or None,
                            session=session,
                            base_history=None,
                            extra_hints=reasoning_hints or None,
                        )

                if chat_models is None:
                    raise RuntimeError("Passive turn reasoning requires CHAT_MODELS")
                async with chat_models.execution(
                    model_id=model_id,
                    reasoning_effort=reasoning_effort,
                ) as execution:
                    turn_result = await run_reasoner(
                        execution.chat(ModelRole.AGENT),
                        execution.chat(ModelRole.DEFAULT),
                    )
                state.extra_metadata["turn_duration_ms"] = int(
                    (time.perf_counter() - started) * 1000
                )
                logger.info(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="end",
                        flow="passive",
                        phase="reasoner",
                        session=key,
                        turn=turn_id,
                        action="continue",
                        duration_ms=int((time.perf_counter() - phase_started) * 1000),
                    )
                )
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="phase_error",
                        flow="passive",
                        phase=active_phase,
                        session=key,
                        turn=turn_id,
                        action="fail",
                        reason=_phase_error_reason(active_phase),
                        duration_ms=int((time.perf_counter() - phase_started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                if not dispatch_outbound:
                    raise
                return await self._control_outbound(
                    state,
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="处理消息时出错，请稍后再试。",
                        terminal_status=TurnTerminalStatus.FAILED,
                    ),
                )

            phase_started = time.perf_counter()
            try:
                # Phase 5: AfterReasoning 模块链（parse、AfterReasoning 事件、持久化、出站消息）。
                with diagnostic_context(phase="after_reasoning"):
                    after_reasoning = await self._runtime_phases().after_reasoning.run(
                        AfterReasoningInput(state=state, turn_result=turn_result)
                    )
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="phase_error",
                        flow="passive",
                        phase="after_reasoning",
                        session=key,
                        turn=turn_id,
                        action="fail",
                        reason="invalid_output",
                        duration_ms=int((time.perf_counter() - phase_started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                raise
            logger.info(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="end",
                    flow="passive",
                    phase="after_reasoning",
                    session=key,
                    turn=turn_id,
                    action="continue",
                    duration_ms=int((time.perf_counter() - phase_started) * 1000),
                )
            )

            phase_started = time.perf_counter()
            try:
                # Phase 6: AfterTurn 模块链（TurnCommitted fanout、AfterTurn fanout、dispatch）。
                with diagnostic_context(phase="after_turn"):
                    outbound = await self._runtime_phases().after_turn.run(
                        after_reasoning
                    )
            except Exception as exc:
                logger.exception(
                    diagnostic_line(
                        "PassiveTurnPipeline.run",
                        event="phase_error",
                        flow="passive",
                        phase="after_turn",
                        session=key,
                        turn=turn_id,
                        action="fail",
                        reason="write_error",
                        duration_ms=int((time.perf_counter() - phase_started) * 1000),
                        error_type=type(exc).__name__,
                        note=str(exc)[:160],
                    )
                )
                raise
            logger.info(
                diagnostic_line(
                    "PassiveTurnPipeline.run",
                    event="end",
                    flow="passive",
                    phase="after_turn",
                    session=key,
                    turn=turn_id,
                    action="done",
                    duration_ms=int((time.perf_counter() - phase_started) * 1000),
                )
            )
            return outbound

    # 供已准入的外部消息复用 AfterReasoning + dispatch 流程。
    async def post_reasoning(
        self,
        msg: InboundMessage,
        session_key: str,
        turn_result: "TurnRunResult",
        *,
        dispatch_outbound: bool = True,
        persistence: TurnPersistencePolicy | None = None,
    ) -> OutboundMessage:
        state = TurnState(
            msg=msg,
            session_key=session_key,
            dispatch_outbound=dispatch_outbound,
            session=self._session.session_manager.get_or_create(session_key),
            persistence=persistence or _persistence_from_metadata(msg.metadata),
        )
        after_reasoning = await self._runtime_phases().after_reasoning.run(
            AfterReasoningInput(state=state, turn_result=turn_result)
        )
        return await self._runtime_phases().after_turn.run(after_reasoning)

    # abort / 错误路径的统一 dispatch helper，只有 dispatch_outbound=True 时才发送。
    async def _control_outbound(
        self,
        state: TurnState,
        outbound: OutboundMessage,
    ) -> OutboundMessage:
        if state.dispatch_outbound:
            _ = await self._outbound_port.dispatch(
                OutboundDispatch(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=outbound.content,
                    thinking=outbound.thinking,
                    reply_to=outbound.reply_to,
                    metadata=outbound.metadata,
                    media=outbound.media,
                    attachment_refs=outbound.attachment_refs,
                    session_message_id=outbound.session_message_id,
                    control_turn_id=(
                        outbound.control_turn_id or running_turn_id.get() or None
                    ),
                    execution_attempt_id=(
                        outbound.execution_attempt_id or running_turn_id.get() or None
                    ),
                    terminal_status=outbound.terminal_status,
                )
            )
        return outbound


class ContextStore(ABC):
    """
    ┌──────────────────────────────────────┐
    │ ContextStore                         │
    ├──────────────────────────────────────┤
    │ 1. 读取 session history              │
    │ 2. 收 skill mentions                 │
    │ 3. 输出 ContextBundle                │
    └──────────────────────────────────────┘
    """

    @abstractmethod
    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        """准备本轮对话需要的上下文。"""


class DefaultContextStore(ContextStore):
    def __init__(
        self,
        *,
        context: "ContextBuilder",
    ) -> None:
        self._context = context

    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        # 1. 先读取 session history，并转换成 retrieval pipeline 需要的结构。
        raw_history = (
            []
            if bool((msg.metadata or {}).get("skip_session_history"))
            else list(session.get_history())
        )
        history_messages = support.to_history_messages(raw_history)

        # 2. 最后补齐 ContextBundle；动态上下文由普通 Prompt lifecycle 插件追加。
        skill_names = [
            record.name
            for record in self._context.skills.list_skill_records(
                filter_unavailable=False
            )
        ]
        skill_mentions = support.collect_skill_mentions(
            msg.content,
            skill_names,
        )
        return ContextBundle(
            skill_mentions=skill_mentions,
            history_messages=history_messages,
        )


class Reasoner(ABC):

    @abstractmethod
    async def run(
        self,
        initial_messages: list[dict],
        *,
        agent_model: BoundChatModel,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        disabled_tools: set[str] | None = None,
    ) -> ReasonerResult:
        """执行多轮 tool loop，并返回本轮结果。"""

    @abstractmethod
    async def run_turn(
        self,
        *,
        msg,
        session: "SessionLike",
        agent_model: BoundChatModel,
        fallback_model: BoundChatModel,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        extra_hints: list[str] | None = None,
    ) -> "TurnRunResult":
        """执行完整被动 turn，包括 retry / trim / tool loop。"""

    async def render_prompt(
        self,
        input: PromptRenderInput,
    ) -> PromptRenderResult:
        raise NotImplementedError


class DefaultReasoner(Reasoner):
    def __init__(
        self,
        llm_config: "LLMConfig",
        tools: "ToolRegistry",
        discovery: ToolDiscoveryState,
        *,
        tool_search_enabled: bool,
        context: "ContextBuilder | None" = None,
        event_bus: "EventBus | None" = None,
        non_preloadable_names: Callable[[], set[str]] | None = None,
    ) -> None:
        self._llm_config = llm_config
        self._tools = tools
        self._discovery = discovery
        self._tool_search_enabled = tool_search_enabled
        self._context = context
        self._event_bus = event_bus
        self._non_preloadable_names = non_preloadable_names or set
        self._tool_executor = ToolExecutor()
        self._stream_sink_factory: (
            Callable[[object], Callable[[dict[str, str] | str], Awaitable[None]] | None]
            | None
        ) = None
        bus = event_bus or EventBus()
        self._bus = bus
        self._before_step = self._build_before_step_phase()
        self._after_step = self._build_after_step_phase()
        self._prompt_render: (
            Phase[
                PromptRenderInput,
                PromptRenderResult,
                PromptRenderFrame,
            ]
            | None
        ) = (
            self._build_prompt_render_phase(context) if context is not None else None
        )

    @staticmethod
    def _provider_request_projection() -> ProviderRequestProjection | None:
        """Resolve an optional projector from this Turn's frozen plugin Root."""

        from agent.plugins.snapshot import get_lifecycle_runtime_snapshot

        snapshot = get_lifecycle_runtime_snapshot()
        if snapshot is None or snapshot.composition_root is None:
            return None
        return snapshot.composition_root.context.get(PROVIDER_REQUEST_PROJECTION)

    def _initial_visible_tools(
        self,
        *,
        model: BoundChatModel,
        preloaded_tools: set[str] | None,
        preloaded_tool_order: list[str] | None,
        disabled: set[str],
    ) -> tuple[set[str], list[str]]:
        always_on = self._tools.get_always_on_names() - disabled
        always_on_order = self._tools.get_registered_order(always_on)
        preload_order = [
            name
            for name in preloaded_tool_order or sorted(preloaded_tools or set())
            if name not in disabled
        ]
        normal_order = list(dict.fromkeys([*always_on_order, *preload_order]))
        max_schemas = _provider_max_tool_schemas(model)
        if max_schemas <= 0 or len(normal_order) <= max_schemas:
            return set(normal_order), normal_order

        if len(always_on_order) > max_schemas:
            projected = _project_tool_order(
                [*reversed(preload_order), *always_on_order],
                max_schemas,
            )
            return set(projected), projected

        # always_on 是运行时合同；预加载工具只能占用剩余槽位。
        projected = _project_tool_order(
            [*always_on_order, *reversed(preload_order)],
            max_schemas,
        )
        return set(projected), projected

    def _build_before_step_phase(
        self,
    ) -> Phase[BeforeStepInput, BeforeStepCtx, BeforeStepFrame]:
        return Phase(
            default_before_step_modules(self._bus),
            frame_factory=BeforeStepFrame,
        )

    def _build_after_step_phase(
        self,
    ) -> Phase[AfterStepCtx, AfterStepCtx, AfterStepFrame]:
        return Phase(
            default_after_step_modules(self._bus),
            frame_factory=AfterStepFrame,
        )

    def _build_prompt_render_phase(
        self,
        context: "ContextBuilder",
    ) -> Phase[PromptRenderInput, PromptRenderResult, PromptRenderFrame]:
        return Phase(
            default_prompt_render_modules(self._bus, context),
            frame_factory=PromptRenderFrame,
        )

    async def render_prompt(
        self,
        input: PromptRenderInput,
    ) -> PromptRenderResult:
        if self._context is None:
            raise RuntimeError("DefaultReasoner.render_prompt requires context")
        if self._prompt_render is None:
            self._prompt_render = self._build_prompt_render_phase(self._context)
        return await self._prompt_render.run(input)

    def _runtime_step_phases(
        self,
    ) -> tuple[
        Phase[BeforeStepInput, BeforeStepCtx, BeforeStepFrame],
        Phase[AfterStepCtx, AfterStepCtx, AfterStepFrame],
    ]:
        return self._before_step, self._after_step

    def _build_request_state(
        self,
        *,
        projection: ProviderTurnProjection,
        initial_messages: list[dict],
        history_count: int,
        attempt_replay: list[dict[str, Any]],
        prior_tool_groups: int,
        channel: str,
        chat_id: str,
        agent_model: BoundChatModel,
        fallback_model: BoundChatModel,
    ) -> _TurnRequestState:
        """Bind ordinary plugin state to Core's provider-call bookkeeping."""

        gate = projection.bind(
            ProviderRequestBinding(
                initial_messages=initial_messages,
                history_count=history_count,
                attempt_replay=attempt_replay,
                prior_tool_groups=prior_tool_groups,
                channel=channel,
                chat_id=chat_id,
                agent_model=agent_model,
                fallback_model=fallback_model,
                max_output_tokens=self._llm_config.max_tokens,
            )
        )
        return _TurnRequestState(
            gate=gate,
            agent_model=agent_model,
            continuation=_load_model_continuation(initial_messages, agent_model),
        )

    def set_stream_sink_factory(
        self,
        factory: (
            Callable[[object], Callable[[dict[str, str] | str], Awaitable[None]] | None]
            | None
        ),
    ) -> None:
        self._stream_sink_factory = factory

    async def run_turn(
        self,
        *,
        msg,
        session: "SessionLike",
        agent_model: BoundChatModel,
        fallback_model: BoundChatModel,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        extra_hints: list[str] | None = None,
    ) -> "TurnRunResult":
        """Run one Turn and revoke every Session projection lease on all exits."""

        grants: list[object] = []
        try:
            return await self._run_turn_with_projection(
                msg=msg,
                session=session,
                agent_model=agent_model,
                fallback_model=fallback_model,
                skill_names=skill_names,
                base_history=base_history,
                extra_hints=extra_hints,
                projection_grants=grants,
            )
        finally:
            for grant in grants:
                session.revoke_projection_grant(grant)

    async def _run_turn_with_projection(
        self,
        *,
        msg,
        session: "SessionLike",
        agent_model: BoundChatModel,
        fallback_model: BoundChatModel,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        extra_hints: list[str] | None = None,
        projection_grants: list[object],
    ) -> "TurnRunResult":
        from agent.core.runtime_support import TurnRunResult

        if self._context is None:
            raise RuntimeError("DefaultReasoner.run_turn requires context")
        if self._prompt_render is None:
            self._prompt_render = self._build_prompt_render_phase(self._context)

        # 1. 先准备 retry trace、history 和 preload 工具集合。
        retry_attempts: list[dict[str, object]] = []
        retry_trace: dict[str, object] = {
            "attempts": retry_attempts,
            "selected_plan": None,
            "trimmed_sections": [],
        }
        runtime = self._provider_request_projection()
        projection_grant: object | None = None
        if runtime is None:
            source = (
                list(base_history)
                if base_history is not None
                else list(session.get_history(max_messages=500))
            )
            projection: ProviderTurnProjection = _PassThroughTurn(
                [dict(message) for message in source]
            )
        else:
            units = tuple(
                RequestHistoryUnit(
                    source_from_seq=unit.source_from_seq,
                    consolidated_through_seq=unit.consolidated_through_seq,
                    source_message_ids=unit.source_message_ids,
                    messages_json=json.dumps(
                        unit.messages,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    message_refs=unit.message_refs,
                )
                for unit in session.history_units(after_seq=-1)
            )
            projection_grant = session.issue_projection_grant(running_turn_id.get())
            projection_grants.append(projection_grant)
            projection = await runtime.open_turn(
                ProviderTurnInput(
                    session_key=session.key,
                    session_created_at=(
                        session.created_at.isoformat()
                        if isinstance(session.created_at, datetime)
                        else str(session.created_at)
                    ),
                    history_units=units,
                    access_grant=projection_grant,
                )
            )
        metadata = getattr(msg, "metadata", None) or {}
        if bool(metadata.get("skip_session_history")):
            projection = _PassThroughTurn([])
        source_history = [dict(message) for message in projection.history]
        raw_attempt_replay = metadata.get("_control_attempt_replay", [])
        if not isinstance(raw_attempt_replay, list) or not all(
            isinstance(item, dict) for item in raw_attempt_replay
        ):
            raise RuntimeError("control attempt replay 契约无效")
        attempt_replay = [
            dict(cast(dict[str, Any], item)) for item in raw_attempt_replay
        ]
        raw_prior_input_count = metadata.get("_control_prior_input_count", 0)
        if (
            not isinstance(raw_prior_input_count, int)
            or isinstance(raw_prior_input_count, bool)
            or raw_prior_input_count < 0
        ):
            raise RuntimeError("control prior input count 契约无效")
        prior_input_count = raw_prior_input_count
        raw_prior_tool_chain = metadata.get("_control_prior_tool_chain", [])
        if not isinstance(raw_prior_tool_chain, list) or not all(
            isinstance(item, dict) for item in raw_prior_tool_chain
        ):
            raise RuntimeError("control prior tool chain 契约无效")
        prior_tool_chain = [
            dict(cast(dict[str, Any], item)) for item in raw_prior_tool_chain
        ]
        total_history = len(source_history)
        preloaded: set[str] | None = None
        preloaded_order: list[str] = []
        if self._tool_search_enabled:
            preloaded_order = self._discovery.get_preloaded_ordered(session.key)
            preloaded = set(preloaded_order)
            logger.info(
                "[tool_search] LRU preloaded=%s",
                preloaded_order if preloaded_order else "[]",
            )
        stream_sink = (
            self._stream_sink_factory(msg)
            if self._stream_sink_factory is not None
            else None
        )
        disabled_tools = _disabled_tools_from_msg(msg)
        turn_scope = get_current_turn_scope()
        raw_disabled_sections = metadata.get("disabled_prompt_sections", [])
        if not isinstance(raw_disabled_sections, list) or not all(
            isinstance(section, str) and section.strip() == section and section
            for section in raw_disabled_sections
        ):
            raise ValueError("disabled_prompt_sections 必须是非空字符串数组")
        disabled_prompt_sections = set(raw_disabled_sections)
        if turn_scope is not None:
            disabled_prompt_sections.update(turn_scope.disabled_prompt_sections)
        if turn_scope is not None:
            registered_tools = set(self._tools.get_registered_names())
            missing_preloads = set(turn_scope.preloaded_tools) - registered_tools
            if missing_preloads:
                raise RuntimeError(
                    "Turn scope preload Tool 未注册: "
                    + ", ".join(sorted(missing_preloads))
                )
            if preloaded is None:
                preloaded = set()
            for name in turn_scope.preloaded_tools:
                if name not in preloaded:
                    preloaded.add(name)
                    preloaded_order.append(name)
            disabled_tools |= {
                name
                for name in registered_tools
                if not turn_scope.tool_grant.allows(name)
            }
        rollout_fact = str(
            (getattr(msg, "metadata", None) or {}).get("_plugin_rollout_fact", "")
        )
        if rollout_fact:
            extra_hints = [
                *(extra_hints or []),
                "【插件运行时事实】"
                + rollout_fact
                + " 这是 Core 已核实的上一轮结果；请用自然语言告诉用户，"
                "不要要求用户查询状态。",
            ]
        if turn_scope is not None and turn_scope.prompt_hints:
            extra_hints = [*(extra_hints or []), *turn_scope.prompt_hints]
        raw_turn_input_source = (getattr(msg, "metadata", None) or {}).get(
            "_control_turn_input_source"
        )
        turn_input_source: InputLock | None = None
        if raw_turn_input_source is not None:
            if not all(
                callable(getattr(raw_turn_input_source, name, None))
                for name in ("lock", "used_inputs")
            ):
                raise RuntimeError("control turn input source 契约无效")
            turn_input_source = cast(InputLock, raw_turn_input_source)
        # 2. 单 plan 执行完整 payload；安全错误不再切换窗口，直接返回用户可读错误。
        retry_attempts.append(
            {
                "name": "full_context",
                "history_window": total_history,
                "disabled_sections": sorted(disabled_prompt_sections),
            }
        )
        history_for_attempt = list(source_history)
        history_for_attempt.extend(attempt_replay)
        turn_injection_prompt = build_turn_injection_prompt(
            tools=self._tools,
            tool_search_enabled=self._tool_search_enabled,
            visible_names=(
                self._initial_visible_tools(
                    model=agent_model,
                    preloaded_tools=preloaded,
                    preloaded_tool_order=preloaded_order,
                    disabled=disabled_tools,
                )[0]
                | disabled_tools
                if self._tool_search_enabled
                else None
            ),
        )
        prompt_render = await self.render_prompt(
            PromptRenderInput(
                session_key=session.key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=msg.content,
                multimodal=(
                    "image" in agent_model.descriptor.capabilities.input_modalities
                ),
                media=msg.media if msg.media else None,
                timestamp=msg.timestamp,
                history=history_for_attempt,
                skill_names=skill_names,
                disabled_sections=disabled_prompt_sections,
                turn_injection_prompt=turn_injection_prompt,
                extra_hints=extra_hints,
            )
        )
        initial_messages = prompt_render.messages
        if turn_input_source is not None:
            used_inputs = turn_input_source.used_inputs()
            if prior_input_count >= len(used_inputs):
                raise RuntimeError("control attempt 缺少当前用户输入")
            self._append_turn_inputs(
                initial_messages,
                used_inputs[prior_input_count + 1 :],
                multimodal=(
                    "image" in agent_model.descriptor.capabilities.input_modalities
                ),
            )
        request_state = self._build_request_state(
            projection=projection,
            initial_messages=initial_messages,
            history_count=len(source_history),
            attempt_replay=attempt_replay,
            prior_tool_groups=len(prior_tool_chain),
            channel=msg.channel,
            chat_id=msg.chat_id,
            agent_model=agent_model,
            fallback_model=fallback_model,
        )
        llm_user_content, llm_context_frame = extract_model_facing_turn(
            initial_messages
        )
        try:
            search_scope = begin_turn_search_scope(
                turn_id=running_turn_id.get(),
                session_key=session.key,
                attempt=0,
            )
            try:
                result = await self.run(
                    initial_messages,
                    agent_model=agent_model,
                    request_time=msg.timestamp,
                    preloaded_tools=preloaded,
                    preloaded_tool_order=preloaded_order,
                    preflight_injected=True,
                    on_content_delta=stream_sink,
                    tool_event_session_key=session.key,
                    tool_event_channel=msg.channel,
                    tool_event_chat_id=msg.chat_id,
                    disabled_tools=disabled_tools,
                    turn_input_source=turn_input_source,
                    initial_attempt_replay=attempt_replay,
                    initial_prior_tool_groups=len(prior_tool_chain),
                    request_state=request_state,
                )
            finally:
                end_turn_search_scope(search_scope)
            tools_used = result.tools_used
            tools_unlocked = result.tools_unlocked
            tool_chain = result.tool_chain
            if prior_tool_chain:
                tool_chain = [*prior_tool_chain, *tool_chain]
                tools_used = [
                    *[
                        str(call["name"])
                        for group in prior_tool_chain
                        for call in cast(list[dict[str, object]], group["calls"])
                    ],
                    *tools_used,
                ]
            media = result.media

            if self._tool_search_enabled and (tools_used or tools_unlocked):
                self._discovery.update(
                    session.key,
                    [*tools_unlocked, *tools_used],
                    self._tools.get_always_on_names(),
                    self._non_preloadable_names(),
                )
            retry_trace["selected_plan"] = "full_context"
            retry_trace["trimmed_sections"] = []
            if prior_input_count == 0 and isinstance(llm_user_content, (str, list)):
                retry_trace["llm_user_content"] = llm_user_content
            if (
                prior_input_count == 0
                and isinstance(llm_context_frame, str)
                and llm_context_frame.strip()
            ):
                retry_trace["llm_context_frame"] = llm_context_frame
            retry_trace["react_stats"] = dict(result.react_stats)
            raw_model_state = result.model_state
            raw_mobile_attention = result.mobile_attention
            if raw_mobile_attention not in (None, "confirmation"):
                raise RuntimeError("reasoner 返回了无效 mobile_attention")
            return TurnRunResult(
                reply=result.reply,
                tools_used=tools_used,
                tool_chain=tool_chain,
                media=[str(item) for item in media if str(item).strip()],
                thinking=result.thinking,
                streamed=result.streamed,
                context_retry=retry_trace,
                model_state=(
                    cast(dict[str, object], raw_model_state)
                    if isinstance(raw_model_state, dict)
                    else None
                ),
                mobile_attention=cast(
                    Literal["confirmation"] | None,
                    raw_mobile_attention,
                ),
            )
        except ContentSafetyError:
            logger.warning("安全拦截：当前消息本身可能违规")
            await self._observe_output_completed(
                session_key=session.key,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            return TurnRunResult(
                reply="你的消息触发了安全审查，无法处理。",
                context_retry=retry_trace,
            )
        except ContextLengthError:
            if _context_window(agent_model) <= 0:
                raise
            logger.warning("上下文超长：当前完整 payload 超过模型输入边界")
            await self._observe_output_completed(
                session_key=session.key,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            return TurnRunResult(
                reply="上下文过长无法处理，请尝试新建对话。",
                context_retry=retry_trace,
            )
        except asyncio.TimeoutError:
            logger.warning("LLM 流响应超时，远端连接中断")
            await self._observe_output_completed(
                session_key=session.key,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )
            return TurnRunResult(
                reply="模型流响应中断，请刷新对话重试。",
                context_retry=retry_trace,
            )
    async def run(
        self,
        initial_messages: list[dict],
        *,
        agent_model: BoundChatModel,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preloaded_tool_order: list[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
        disabled_tools: set[str] | None = None,
        turn_input_source: InputLock | None = None,
        initial_attempt_replay: list[dict[str, Any]] | None = None,
        initial_prior_tool_groups: int = 0,
        request_state: _TurnRequestState | None = None,
    ) -> ReasonerResult:
        # 1. 初始化消息上下文、本轮工具轨迹。
        messages = list(initial_messages)
        tools_used: list[str] = []
        tools_unlocked: list[str] = []
        tool_chain: list[dict[str, Any]] = []
        outbound_media: list[str] = []
        mobile_attention: Literal["confirmation"] | None = None
        # 2. 初始化本轮可见工具集合。
        visible_names: set[str] | None = None
        visible_order: list[str] | None = None
        streamed = False
        react_input_samples: list[int] = []
        react_usages: list[ModelUsage] = []
        react_call_usages: list[ModelUsage] = []
        react_finish_reasons: list[str | None] = []
        disabled = set(disabled_tools or set())
        turn_scope = get_current_turn_scope()
        if request_state is None:
            raise RuntimeError("provider request gate required")
        gate = request_state.gate
        if on_content_delta is not None:
            on_content_delta = self._wrap_turn_first_delta(
                request_state,
                on_content_delta,
            )
        pending_start_override: int | None = gate.pending_start
        before_step_phase, after_step_phase = self._runtime_step_phases()
        if self._tool_search_enabled:
            always_on = self._tools.get_always_on_names()
            visible_names, visible_order = self._initial_visible_tools(
                model=request_state.agent_model,
                preloaded_tools=preloaded_tools,
                preloaded_tool_order=preloaded_tool_order,
                disabled=disabled,
            )
            logger.info(
                "[tool_search] visible=%d 个工具 always_on=%d preloaded=%d "
                "provider_limit=%s need_search=%s",
                len(visible_names),
                len(always_on),
                len(preloaded_tools or set()),
                _provider_max_tool_schemas(request_state.agent_model) or "unlimited",
                "yes" if len(visible_names) == len(always_on) else "maybe",
            )

        iteration = -1
        terminal_deadline = False
        while True:
            iteration += 1
            max_iterations = (
                turn_scope.max_iterations
                if turn_scope is not None and turn_scope.max_iterations is not None
                else self._llm_config.max_iterations
            )
            if max_iterations > 0 and iteration >= max_iterations:
                terminal_tools = (
                    turn_scope.terminal_tools if turn_scope is not None else ()
                )
                if terminal_tools and not terminal_deadline:
                    terminal_deadline = True
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "调查预算已经用完。不要继续调查或解释；现在必须且只能"
                                "调用以下一个工具提交最终决定："
                                + "、".join(terminal_tools)
                                + "。"
                            ),
                        }
                    )
                else:
                    summary, summary_usages = await self._summarize_incomplete_progress(
                        messages,
                        reason="max_iterations",
                        iteration=iteration,
                        tools_used=tools_used,
                        request_state=request_state,
                    )
                    react_usages.extend(summary_usages)
                    result = self._build_result(
                        reply=summary,
                        tools_used=tools_used,
                        tool_chain=tool_chain,
                        media=outbound_media,
                        visible_names=visible_names,
                        thinking=None,
                        streamed=False,
                        react_input_samples=react_input_samples,
                        cache_usages=react_call_usages,
                        tools_unlocked=tools_unlocked,
                        model_usages=react_usages,
                        finish_reasons=react_finish_reasons,
                        mobile_attention=mobile_attention,
                        model_state=_model_continuation_state(
                            request_state.continuation
                        ),
                    )
                    await self._lock_turn_input_source(turn_input_source)
                    await self._observe_output_completed(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                    )
                    return result
            batch_start = (
                pending_start_override
                if pending_start_override is not None
                else len(messages)
            )
            pending_start_override = None
            # 3. BeforeStep 模块链：token 估算、BeforeStep 事件、提示注入。
            step_ctx = await before_step_phase.run(
                BeforeStepInput(
                    session_key=tool_event_session_key,
                    channel=tool_event_channel,
                    chat_id=tool_event_chat_id,
                    iteration=iteration,
                    messages=messages,
                    visible_names=visible_names,
                )
            )
            if step_ctx.early_stop:
                summary, summary_usages = await self._summarize_incomplete_progress(
                    messages,
                    reason="early_stop",
                    iteration=iteration + 1,
                    tools_used=tools_used,
                    request_state=request_state,
                )
                react_usages.extend(summary_usages)
                result = self._build_result(
                    reply=step_ctx.early_stop_reply or summary,
                    tools_used=tools_used,
                    tool_chain=tool_chain,
                    media=outbound_media,
                    visible_names=visible_names,
                    thinking=None,
                    streamed=False,
                    react_input_samples=react_input_samples,
                    cache_usages=react_call_usages,
                    tools_unlocked=tools_unlocked,
                    model_usages=react_usages,
                    finish_reasons=react_finish_reasons,
                    mobile_attention=mobile_attention,
                    model_state=_model_continuation_state(
                        request_state.continuation
                    ),
                )
                await self._lock_turn_input_source(turn_input_source)
                await self._observe_output_completed(
                    session_key=tool_event_session_key,
                    channel=tool_event_channel,
                    chat_id=tool_event_chat_id,
                )
                return result
            # 4. 构造本轮工具 schema，并按完整 provider input 判断压缩水位。
            schema_names: list[str] | set[str] | None = (
                list(visible_order) if visible_order is not None else None
            )
            if schema_names is None and disabled:
                schema_names = self._tools.get_registered_names() - disabled
            elif schema_names is not None:
                schema_names = [name for name in schema_names if name not in disabled]
            tool_schemas = self._tools.get_schemas(names=schema_names)
            execution_grant = (
                turn_scope.tool_grant if turn_scope is not None else ToolGrant()
            )
            if terminal_deadline and turn_scope is not None:
                tool_schemas = self._tools.get_schemas(
                    names=set(turn_scope.terminal_tools)
                )
                execution_grant = ToolGrant.only(turn_scope.terminal_tools)
            max_tool_schemas = _provider_max_tool_schemas(request_state.agent_model)
            if (
                max_tool_schemas > 0
                and len(tool_schemas) > max_tool_schemas
                and not self._tool_search_enabled
            ):
                raise RuntimeError(
                    "当前模型 endpoint 最多接受 "
                    f"{max_tool_schemas} 个工具 schema；请开启 tool_search 后重试"
                )
            call_result = await self._call_provider(
                request_state,
                messages,
                tools=tool_schemas,
                max_tokens=self._llm_config.max_tokens,
                on_content_delta=on_content_delta,
                cache_namespace=tool_event_session_key,
            )
            response = call_result.response
            prepared = call_result.prepared
            react_usages.extend(call_result.auxiliary_usages)
            if prepared is None:
                raise RuntimeError("provider request gate 未返回 prepared context")
            batch_start = prepared.pending_start
            react_input_samples.append(prepared.estimated_tokens)
            logger.info(
                "[LLM调用] 第%d轮，可见工具=%s input_tokens~=%d quality=%s compacted=%s",
                iteration + 1,
                (
                    f"{len(visible_names)}个"
                    if visible_names is not None
                    else "全部（tool_search未开启）"
                ),
                prepared.estimated_tokens,
                prepared.token_quality,
                prepared.changed,
            )
            response_usage = response.usage or ModelUsage()
            react_usages.append(response_usage)
            react_call_usages.append(response_usage)
            react_finish_reasons.append(response.finish_reason)
            if on_content_delta is not None and response.content:
                streamed = True
            terminal_tools = turn_scope.terminal_tools if turn_scope is not None else ()
            at_terminal_budget = bool(
                terminal_tools
                and max_iterations > 0
                and iteration + 1 >= max_iterations
            )
            # 5. 空 thinking 响应关闭 thinking 修复一次，再进入正常分支。
            if (
                not response.content
                and not response.tool_calls
                and response.thinking
                and not at_terminal_budget
            ):
                logger.warning(
                    "[空回复重试] 第%d轮，content为空但thinking非空，"
                    "finish_reason=%s，关闭thinking修复一次",
                    iteration + 1,
                    response.finish_reason,
                )
                retry_assistant: dict[str, Any] = {"role": "assistant", "content": ""}
                model_state = _model_continuation_state(response.continuation)
                if model_state is not None:
                    retry_assistant["model_state"] = model_state
                messages.append(retry_assistant)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你刚才只输出了思考过程，没有给出正式回复或工具调用。"
                            "请继续：需要操作时通过已提供的结构化工具调用，"
                            "不要把工具调用协议写入文本；否则直接回复用户。"
                        ),
                    }
                )
                retry_result = await self._call_provider(
                    request_state,
                    messages,
                    tools=tool_schemas,
                    max_tokens=self._llm_config.max_tokens,
                    disable_thinking=True,
                    on_content_delta=on_content_delta,
                    cache_namespace=tool_event_session_key,
                )
                retry_response = retry_result.response
                react_usages.extend(retry_result.auxiliary_usages)
                retry_prepared = retry_result.prepared
                if retry_prepared is None:
                    raise RuntimeError(
                        "provider request gate 未返回 retry prepared context"
                    )
                batch_start = retry_prepared.pending_start
                retry_usage = retry_response.usage or ModelUsage()
                react_usages.append(retry_usage)
                react_call_usages.append(retry_usage)
                react_finish_reasons.append(retry_response.finish_reason)
                if retry_response.content or retry_response.tool_calls:
                    response = retry_response
                    if on_content_delta is not None and response.content:
                        streamed = True
                    logger.info(
                        "[空回复重试] 修复成功，finish_reason=%s content=%s "
                        "tool_calls=%d",
                        response.finish_reason,
                        bool(response.content),
                        len(response.tool_calls),
                    )
                else:
                    logger.warning(
                        "[空回复重试] 重试仍为空，finish_reason=%s，使用fallback",
                        retry_response.finish_reason,
                    )

            # 5a. Scoped Turn 要求结构化终态时，在同一 Turn 内纠正一次。
            if (
                terminal_tools
                and not response.tool_calls
                and not terminal_deadline
                and (max_iterations <= 0 or iteration + 1 < max_iterations)
            ):
                terminal_retry_assistant: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content or "",
                }
                model_state = _model_continuation_state(response.continuation)
                if model_state is not None:
                    terminal_retry_assistant["model_state"] = model_state
                messages.append(terminal_retry_assistant)
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "你刚才没有提交本 Turn 要求的结构化终态。"
                            "不要继续解释；现在必须且只能调用以下一个工具："
                            + "、".join(terminal_tools)
                            + "。"
                        ),
                    }
                )
                tool_schemas = self._tools.get_schemas(names=set(terminal_tools))
                execution_grant = ToolGrant.only(terminal_tools)
                terminal_deadline = True
                terminal_retry = await self._call_provider(
                    request_state,
                    messages,
                    tools=tool_schemas,
                    max_tokens=self._llm_config.max_tokens,
                    disable_thinking=True,
                    on_content_delta=None,
                    cache_namespace=tool_event_session_key,
                )
                response = terminal_retry.response
                react_usages.extend(terminal_retry.auxiliary_usages)
                retry_prepared = terminal_retry.prepared
                if retry_prepared is None:
                    raise RuntimeError(
                        "provider request gate 未返回 terminal retry prepared context"
                    )
                batch_start = retry_prepared.pending_start
                response_usage = response.usage or ModelUsage()
                react_usages.append(response_usage)
                react_call_usages.append(response_usage)
                react_finish_reasons.append(response.finish_reason)
                logger.info(
                    "[结构化终态重试] 第%d轮，tool_calls=%d",
                    iteration + 1,
                    len(response.tool_calls),
                )

            if (
                terminal_tools
                and not response.tool_calls
                and not terminal_deadline
                and max_iterations > 0
                and iteration + 1 >= max_iterations
            ):
                messages.append(
                    {"role": "assistant", "content": response.content or ""}
                )
                continue

            # 6. 模型返回 tool_calls 时，进入工具执行分支。
            if response.tool_calls:
                logger.info(
                    "[LLM决策→工具] 第%d轮，调用: %s",
                    iteration + 1,
                    [tc.name for tc in response.tool_calls],
                )
                model_state = _model_continuation_state(response.continuation)
                append_assistant_tool_calls(
                    messages,
                    content=response.content,
                    tool_calls=response.tool_calls,
                    provider_fields=(
                        {"model_state": model_state}
                        if model_state is not None
                        else None
                    ),
                )
                tool_batch = tool_call_batch_snapshot(response.tool_calls)

                # 7. 逐个执行本轮工具调用。
                iter_calls: list[dict[str, Any]] = []
                terminal_completed = False
                for tool_batch_index, tool_call in enumerate(response.tool_calls):
                    arguments = cast(dict[str, Any], tool_call.arguments)
                    if terminal_completed:
                        await self._observe_tool_call_started(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                        )
                        result = "同一批次已有终态决定；此后的工具调用不再执行。"
                        append_tool_result(
                            messages,
                            tool_call_id=tool_call.id,
                            content=result,
                            tool_name=tool_call.name,
                            execution_status="blocked",
                        )
                        await self._observe_tool_call_completed(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                            final_arguments=arguments,
                            status="blocked",
                            result_preview=result,
                        )
                        iter_calls.append(
                            {
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "status": "blocked",
                                "arguments": arguments,
                                "result": result,
                            }
                        )
                        continue
                    if tool_call.name in disabled:
                        await self._observe_tool_call_started(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                        )
                        result = (
                            f"工具 '{tool_call.name}' 在当前后台任务中不可用。"
                            "请直接返回要发送的最终内容，不要主动推送。"
                        )
                        append_tool_result(
                            messages,
                            tool_call_id=tool_call.id,
                            content=result,
                            tool_name=tool_call.name,
                            execution_status="blocked",
                        )
                        await self._observe_tool_call_completed(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                            final_arguments=arguments,
                            status="blocked",
                            result_preview=support.log_preview(result),
                        )
                        iter_calls.append(
                            {
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "status": "blocked",
                                "arguments": arguments,
                                "result": result,
                            }
                        )
                        continue
                    # 6.1 deferred 工具未解锁时，先回填 select: 引导错误。
                    if (
                        visible_names is not None
                        and tool_call.name not in visible_names
                    ):
                        exec_result = await self._tool_executor.preflight(
                            ToolExecutionRequest(
                                call_id=tool_call.id,
                                tool_name=tool_call.name,
                                arguments=arguments,
                                source=(
                                    turn_scope.tool_source
                                    if turn_scope is not None
                                    else "passive"
                                ),
                                session_key=tool_event_session_key,
                                channel=tool_event_channel,
                                chat_id=tool_event_chat_id,
                                tool_batch=tool_batch,
                                tool_batch_index=tool_batch_index,
                                grant=execution_grant,
                            )
                        )
                        await self._observe_tool_call_started(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                        )
                        logger.warning(
                            "[工具未解锁] LLM 尝试调用 '%s'，但该工具 schema 不可见，引导模型先 tool_search",
                            tool_call.name,
                        )
                        result = (
                            f"工具 '{tool_call.name}' 当前未加载（schema 不可见）。"
                            f'请先调用 tool_search(query="select:{tool_call.name}") 加载，'
                            "然后再调用该工具。不要放弃当前任务。"
                        )
                        append_tool_result(
                            messages,
                            tool_call_id=tool_call.id,
                            content=result,
                            execution_status="blocked",
                        )
                        await self._observe_tool_call_completed(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                            final_arguments=arguments,
                            status="blocked",
                            result_preview=support.log_preview(result),
                        )
                        iter_calls.append(
                            {
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": arguments,
                                "result": result,
                            }
                        )
                        continue

                    # 6.2 通过统一执行器跑 typed admission + 真实工具。
                    async def _execute_tool(
                        name: str,
                        arguments: dict[str, Any],
                    ) -> Any:
                        internal_arguments: dict[str, Any] = {}
                        if name == "tool_search" and visible_names is not None:
                            internal_arguments["excluded_names"] = (
                                visible_names | disabled
                            )
                            max_schemas = _provider_max_tool_schemas(
                                request_state.agent_model
                            )
                            if max_schemas > 0:
                                internal_arguments["max_unlocked"] = max_schemas - 1
                        if name == "message_push":
                            internal_arguments["_commit_role"] = "passive"
                        return await self._tools.execute(
                            name,
                            arguments,
                            tool_override=(
                                turn_scope.tool_overrides.get(name)
                                if turn_scope is not None
                                else None
                            ),
                            internal_arguments=internal_arguments,
                            raise_errors=True,
                        )

                    _args_preview = support.log_preview(arguments, 120)
                    logger.info(
                        "[工具执行→] %s  args=%s", tool_call.name, _args_preview
                    )
                    await self._observe_tool_call_started(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        iteration=iteration + 1,
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=arguments,
                    )
                    # 工具调用统一先过 ToolExecutor，完成 typed prepare/authorize。
                    await self._bus.fanout(
                        BeforeToolCallCtx(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                        )
                    )
                    exec_result = await self._tool_executor.execute(
                        ToolExecutionRequest(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=arguments,
                            source=(
                                turn_scope.tool_source
                                if turn_scope is not None
                                else "passive"
                            ),
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            tool_batch=tool_batch,
                            tool_batch_index=tool_batch_index,
                            grant=execution_grant,
                        ),
                        _execute_tool,
                    )
                    if exec_result.status == "success":
                        tools_used.append(tool_call.name)
                        terminal_completed = (
                            terminal_completed or tool_call.name in terminal_tools
                        )
                    result = exec_result.output
                    await self._bus.fanout(
                        AfterToolResultCtx(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            tool_name=tool_call.name,
                            arguments=dict(exec_result.final_arguments),
                            result=str(result),
                            status=exec_result.status,
                        )
                    )
                    normalized = normalize_tool_result(result)
                    if normalized.mobile_attention is not None:
                        if exec_result.status != "success":
                            raise RuntimeError("失败工具不能声明 mobile_attention")
                        mobile_attention = normalized.mobile_attention
                    _result_preview = support.log_preview(normalized.preview())
                    _result_len = len(normalized.preview() or "")
                    await self._observe_tool_call_completed(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        iteration=iteration + 1,
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=arguments,
                        final_arguments=exec_result.final_arguments,
                        status=exec_result.status,
                        result_preview=normalized.preview(),
                        runtime_provenance=normalized.runtime_provenance,
                    )
                    logger.info(
                        "[工具结果←] %s  结果预览=%s  result_len=%d",
                        tool_call.name,
                        _result_preview,
                        _result_len,
                    )
                    append_tool_result(
                        messages,
                        tool_call_id=tool_call.id,
                        content=result,
                        tool_name=tool_call.name,
                        execution_status=exec_result.status,
                    )
                    if (
                        exec_result.status == "success"
                        and tool_call.name == "message_push"
                    ):
                        _collect_current_akashic_push_media(
                            outbound_media,
                            exec_result.final_arguments,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                        )

                    # 6.3 tool_search 的结果会扩展下一轮可见工具。
                    if (
                        exec_result.status == "success"
                        and tool_call.name == "tool_search"
                        and visible_names is not None
                    ):
                        _newly_unlocked = [
                            name
                            for name in self._discovery.unlock_names_from_result(
                                normalized.text
                            )
                            if name not in visible_names and name not in disabled
                        ]
                        if _newly_unlocked:
                            tools_unlocked.extend(_newly_unlocked)
                            if visible_order is not None:
                                previous_visible = set(visible_order)
                                always_on_order = self._tools.get_registered_order(
                                    self._tools.get_always_on_names() - disabled
                                )
                                max_schemas = _provider_max_tool_schemas(
                                    request_state.agent_model
                                )
                                retained = _project_tool_order(
                                    [*always_on_order, *visible_order],
                                    (
                                        max(1, max_schemas - len(_newly_unlocked))
                                        if max_schemas > 0
                                        else 0
                                    ),
                                )
                                visible_order = _project_tool_order(
                                    [
                                        *retained,
                                        *_newly_unlocked,
                                        *always_on_order,
                                        *visible_order,
                                    ],
                                    max_schemas,
                                )
                                visible_names = set(visible_order)
                                dropped = previous_visible - visible_names
                                if dropped:
                                    logger.info(
                                        "[工具投影] 为新解锁工具释放 schema 槽位: %s",
                                        sorted(dropped),
                                    )
                            logger.info(
                                "[工具解锁] tool_search 新解锁: %s",
                                sorted(_newly_unlocked),
                            )
                        else:
                            logger.info("[工具解锁] tool_search 未解锁新工具")
                    # tool_chain 持久化的是“执行后的事实”：
                    # 最终参数、结果状态与预览，供后续回放与 session 复原。
                    iter_calls.append(
                        {
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "status": exec_result.status,
                            "arguments": arguments,
                            "final_arguments": exec_result.final_arguments,
                            "result": normalized.preview(),
                        }
                    )
                # 7. 本轮工具执行完后，记录 tool_chain。
                tool_chain_group = {"text": response.content, "calls": iter_calls}
                if response.thinking is not None:
                    tool_chain_group["reasoning_content"] = response.thinking
                model_state = _model_continuation_state(response.continuation)
                if model_state is not None:
                    tool_chain_group["model_state"] = model_state
                tool_chain.append(tool_chain_group)
                gate.record_completed_batch(
                    messages,
                    batch_start=batch_start,
                )
                if terminal_completed:
                    result = self._build_result(
                        reply=response.content or "",
                        tools_used=tools_used,
                        tool_chain=tool_chain,
                        media=outbound_media,
                        visible_names=visible_names,
                        thinking=response.thinking,
                        streamed=streamed,
                        react_input_samples=react_input_samples,
                        cache_usages=react_call_usages,
                        tools_unlocked=tools_unlocked,
                        model_usages=react_usages,
                        finish_reasons=react_finish_reasons,
                        mobile_attention=mobile_attention,
                        model_state=_model_continuation_state(
                            request_state.continuation
                        ),
                    )
                    await self._lock_turn_input_source(turn_input_source)
                    await self._observe_output_completed(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                    )
                    return result
                pressure_tokens = request_state.agent_model.estimate_context_tokens(
                    messages,
                    tool_schemas,
                )
                # 7a. AfterStep 模块链（工具分支）：通知观察者本轮工具执行完毕。
                after_step = await after_step_phase.run(
                    AfterStepCtx(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        iteration=iteration,
                        context_tokens_estimate=pressure_tokens,
                        tools_called=tuple(tc.name for tc in response.tool_calls),
                        partial_reply=response.content or "",
                        tools_used_so_far=tuple(tools_used),
                        tool_chain_partial=tuple(tool_chain),
                        partial_thinking=response.thinking,
                        has_more=True,
                    )
                )
                if after_step.early_stop:
                    reason = after_step.early_stop_reason or "after_step"
                    logger.warning(
                        "[插件收尾] reason=%s tokens~=%d，停止继续调用工具并收尾",
                        reason,
                        pressure_tokens,
                    )
                    summary, summary_usages = await self._summarize_incomplete_progress(
                        messages,
                        reason=reason,
                        iteration=iteration + 1,
                        tools_used=tools_used,
                        request_state=request_state,
                    )
                    react_usages.extend(summary_usages)
                    result = self._build_result(
                        reply=summary,
                        tools_used=tools_used,
                        tool_chain=tool_chain,
                        media=outbound_media,
                        visible_names=visible_names,
                        thinking=None,
                        streamed=False,
                        react_input_samples=react_input_samples,
                        cache_usages=react_call_usages,
                        tools_unlocked=tools_unlocked,
                        model_usages=react_usages,
                        finish_reasons=react_finish_reasons,
                        mobile_attention=mobile_attention,
                        model_state=_model_continuation_state(
                            request_state.continuation
                        ),
                    )
                    await self._lock_turn_input_source(turn_input_source)
                    await self._observe_output_completed(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                    )
                    return result
                continue

            # 8. 没有 tool_calls 时，说明本轮得到最终回复。
            logger.info(
                "[LLM决策→回复] 第%d轮，共调用工具%d次: %s",
                iteration + 1,
                len(tools_used),
                tools_used if tools_used else "无",
            )
            result = self._build_result(
                reply=response.content or "模型未返回可用回复，请重试。",
                tools_used=tools_used,
                tool_chain=tool_chain,
                media=outbound_media,
                visible_names=visible_names,
                thinking=response.thinking,
                streamed=streamed,
                react_input_samples=react_input_samples,
                cache_usages=react_call_usages,
                tools_unlocked=tools_unlocked,
                model_usages=react_usages,
                finish_reasons=react_finish_reasons,
                mobile_attention=mobile_attention,
                model_state=_model_continuation_state(request_state.continuation),
            )
            await self._lock_turn_input_source(turn_input_source)
            # 输出完成信号：最终回复的最后一个 delta 已交付、input source 已锁，
            # 在 AfterStep 收尾之前立即发出，慢插件不得推迟 composer 解锁。
            await self._observe_output_completed(
                session_key=tool_event_session_key,
                channel=tool_event_channel,
                chat_id=tool_event_chat_id,
            )
            messages.append({"role": "assistant", "content": response.content})
            # 8b. AfterStep 模块链（最终回复分支）：通知观察者本轮推理结束。
            _ = await after_step_phase.run(
                AfterStepCtx(
                    session_key=tool_event_session_key,
                    channel=tool_event_channel,
                    chat_id=tool_event_chat_id,
                    iteration=iteration,
                    context_tokens_estimate=support.estimate_messages_tokens(messages),
                    tools_called=(),
                    partial_reply=response.content or "",
                    tools_used_so_far=tuple(tools_used),
                    tool_chain_partial=tuple(tool_chain),
                    partial_thinking=response.thinking,
                    has_more=False,
                )
            )
            return result

    @staticmethod
    async def _lock_turn_input_source(source: InputLock | None) -> None:
        """在提交最终候选前封口 active attempt。"""

        if source is not None:
            await source.lock()

    def _append_turn_inputs(
        self,
        messages: list[dict[str, Any]],
        inputs: tuple[TurnUserInput, ...],
        *,
        multimodal: bool,
    ) -> None:
        """使用首条消息相同的 envelope 追加有序用户输入。"""

        if not inputs:
            return
        if self._context is None:
            raise RuntimeError("logical interaction input requires ContextBuilder")
        for item in inputs:
            messages.append(
                {
                    "role": "user",
                    "content": self._context.build_user_message_content(
                        item.content,
                        list(item.media) if item.media else None,
                        multimodal=multimodal,
                        message_timestamp=item.timestamp,
                    ),
                }
            )

    async def _observe_tool_call_started(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        iteration: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> None:
        if self._event_bus is None or not session_key:
            return
        await self._event_bus.observe(
            ToolCallStarted(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                iteration=iteration,
                call_id=call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                turn_id=running_turn_id.get(),
            )
        )

    async def _observe_tool_call_completed(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
        iteration: int,
        call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        final_arguments: dict[str, Any],
        status: str,
        result_preview: str,
        runtime_provenance: dict[str, str] | None = None,
    ) -> None:
        if self._event_bus is None or not session_key:
            return
        await self._event_bus.observe(
            ToolCallCompleted(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                iteration=iteration,
                call_id=call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                final_arguments=dict(final_arguments),
                status=status,
                result_preview=result_preview,
                runtime_provenance=dict(runtime_provenance or {}),
                turn_id=running_turn_id.get(),
            )
        )

    async def _observe_output_completed(
        self,
        *,
        session_key: str,
        channel: str,
        chat_id: str,
    ) -> None:
        """在最后可见输出交付后、AfterStep 收尾前发出展示层 output.completed。"""

        if self._event_bus is None or not session_key:
            return
        await self._event_bus.observe(
            TurnOutputCompleted(
                session_key=session_key,
                channel=channel,
                chat_id=chat_id,
                turn_id=running_turn_id.get(),
                client_message_id=current_client_message_id.get(),
            )
        )

    async def _prepare_provider_gate(
        self,
        state: _TurnRequestState,
        messages: list[dict],
        *,
        tools: list[dict],
        max_output_tokens: int | None = None,
        trigger: Literal["soft_limit", "context_overflow"] = "soft_limit",
        force: bool = False,
    ) -> PreparedProviderRequest:
        # 真实路径里程碑：初始 gate 在 provider start 之前的耗时（含首字前的
        # projection 停顿单独记录，与 provider TTFT 分离。
        identity = _provider_call_identity.get()
        call_ordinal = identity.call_ordinal if identity is not None else 0
        gate_started = time.monotonic()
        self._milestone_request_prepare(
            "tl:request_projection.prepare.start",
            call_ordinal=call_ordinal,
            trigger=trigger,
            force=force,
        )
        try:
            prepared = await state.gate.prepare(
                messages,
                tools=tools,
                trigger=trigger,
                force=force,
                max_output_tokens=max_output_tokens,
            )
        except asyncio.CancelledError:
            self._milestone_request_prepare(
                "tl:request_projection.prepare.cancelled",
                call_ordinal=call_ordinal,
                trigger=trigger,
                force=force,
                outcome="cancelled",
                duration_ms=(time.monotonic() - gate_started) * 1_000,
            )
            raise
        except Exception:
            self._milestone_request_prepare(
                "tl:request_projection.prepare.error",
                call_ordinal=call_ordinal,
                trigger=trigger,
                force=force,
                outcome="error",
                duration_ms=(time.monotonic() - gate_started) * 1_000,
            )
            raise
        self._milestone_request_prepare(
            "tl:request_projection.prepare.done",
            call_ordinal=call_ordinal,
            trigger=trigger,
            force=force,
            outcome="done",
            duration_ms=(time.monotonic() - gate_started) * 1_000,
            compacted=prepared.changed,
        )
        return prepared

    def _milestone_request_prepare(
        self,
        event: str,
        *,
        call_ordinal: int,
        trigger: str,
        force: bool,
        outcome: str = "",
        duration_ms: float | None = None,
        compacted: bool | None = None,
    ) -> None:
        counts = (
            f"call_ordinal={call_ordinal} "
            f"provider_call_id={current_provider_call_id.get() or '-'} "
            f"trigger={trigger} force={str(force).lower()}"
        )
        if compacted is not None:
            counts += f" compacted={str(compacted).lower()}"
        turn_milestone(
            logger,
            event,
            session_id=current_session_key.get() or "",
            turn_id=running_turn_id.get(),
            client_message_id=current_client_message_id.get(),
            duration_ms=duration_ms,
            outcome=outcome,
            counts=counts,
        )

    def _milestone_provider_attempt(
        self,
        event: str,
        identity: _ProviderAttemptIdentity,
        *,
        outcome: str = "",
        duration_ms: float | None = None,
        kind: str = "",
    ) -> None:
        counts = (
            f"call_ordinal={identity.call_ordinal} "
            f"provider_attempt={identity.provider_attempt} "
            f"provider_call_id={current_provider_call_id.get() or '-'}"
        )
        if kind:
            counts += f" kind={kind}"
        turn_milestone(
            logger,
            event,
            session_id=current_session_key.get() or "",
            turn_id=running_turn_id.get(),
            client_message_id=current_client_message_id.get(),
            duration_ms=duration_ms,
            outcome=outcome,
            counts=counts,
        )

    def _wrap_turn_first_delta(
        self,
        state: _TurnRequestState,
        inner: Callable[[dict[str, str]], Awaitable[None]],
    ) -> Callable[[dict[str, str]], Awaitable[None]]:
        """记录真正首非空 thinking/answer 回调的 TTFT（从当前 provider attempt start 起算）。

        每个 turn 各类型只打一次，多个 tool round 不重复；delta 原样透传，不延迟、不合并。
        采样发生在下游回调之前，下游消费慢不会污染首字时长。
        """

        def emit(event: str, *, kind: str = "") -> None:
            identity = _provider_call_identity.get()
            call_id = current_provider_call_id.get()
            counts = " ".join(
                [
                    *(
                        [
                            f"call_ordinal={identity.call_ordinal}",
                            f"provider_attempt={identity.provider_attempt}",
                        ]
                        if identity is not None
                        else []
                    ),
                    *([f"provider_call_id={call_id}"] if call_id else []),
                    *([f"kind={kind}"] if kind else []),
                ]
            )
            turn_milestone(
                logger,
                event,
                session_id=current_session_key.get() or "",
                turn_id=running_turn_id.get(),
                client_message_id=current_client_message_id.get(),
                duration_ms=(
                    (time.monotonic() - state.call_started_at) * 1_000
                    if state.call_started_at > 0
                    else None
                ),
                counts=counts,
            )

        async def wrapped(delta: dict[str, str]) -> None:
            thinking = delta.get("thinking_delta")
            if isinstance(thinking, str) and thinking:
                if not state.first_any_logged:
                    state.first_any_logged = True
                    emit("tl:turn.first_any", kind="thinking")
                if not state.first_thinking_logged:
                    state.first_thinking_logged = True
                    emit("tl:turn.first_thinking")
            answer = delta.get("content_delta")
            if isinstance(answer, str) and answer:
                if not state.first_any_logged:
                    state.first_any_logged = True
                    emit("tl:turn.first_any", kind="answer")
                if not state.first_answer_logged:
                    state.first_answer_logged = True
                    emit("tl:turn.first_answer")
            await inner(delta)

        return wrapped

    async def _call_provider(
        self,
        state: _TurnRequestState,
        messages: list[dict],
        *,
        tools: list[dict],
        max_tokens: int,
        disable_thinking: bool = False,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        cache_namespace: str = "",
    ) -> _ProviderCallResult:
        """Gate one full business payload and allow one forced retry on overflow."""

        # 每个逻辑 provider 调用先分配 1-based call_ordinal，作为本调用所有
        # 里程碑（projection gate / attempt / first delta）的统一身份。
        state.provider_call_ordinal += 1
        call_ordinal = state.provider_call_ordinal
        identity_token = _provider_call_identity.set(
            _ProviderAttemptIdentity(call_ordinal=call_ordinal)
        )
        # 中性逻辑调用身份：从 projection gate 开始到整个 call 终态全程保持，
        # finally 精确 reset；attempt=2 复用同一 call_id，仅替换 provider_attempt。
        provider_call_id = uuid.uuid4().hex
        call_id_token = current_provider_call_id.set(provider_call_id)
        attempt_token = current_provider_attempt.set(1)
        operation_token = current_provider_operation.set("business")
        attempt_two_token: Token[int] | None = None
        try:
            prepared = await self._prepare_provider_gate(
                state,
                messages,
                tools=tools,
                max_output_tokens=max_tokens,
            )
            auxiliary_usages: list[ModelUsage] = []
            if prepared.changed:
                state.continuation = None
            auxiliary_usages.extend(prepared.auxiliary_usages)
            request_message_count = len(messages)

            def build_request() -> ModelRequest:
                return ModelRequest(
                    messages=messages,
                    tools=tools,
                    max_output_tokens=max_tokens,
                    tool_choice="auto",
                    disable_reasoning=disable_thinking,
                    on_delta=on_content_delta,
                    prompt_cache_key=_prompt_cache_key(
                        state.agent_model,
                        cache_namespace,
                    ),
                    continuation=state.continuation,
                )

            # 时间链：attempt 1（provider.call.start 在 projection gate 之后）。
            identity = _ProviderAttemptIdentity(
                call_ordinal=call_ordinal,
                provider_attempt=1,
            )
            _ = _provider_call_identity.set(identity)
            state.call_started_at = time.monotonic()
            self._milestone_provider_attempt("tl:provider.call.start", identity)
            try:
                response = await state.agent_model.complete(build_request())
            except asyncio.CancelledError:
                self._milestone_provider_attempt(
                    "tl:provider.call.cancelled",
                    identity,
                    outcome="cancelled",
                    duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                )
                raise
            except ContextLengthError:
                if not state.gate.can_retry_context_error(
                    context_window=_context_window(state.agent_model)
                ):
                    self._milestone_provider_attempt(
                        "tl:provider.call.error",
                        identity,
                        outcome="error",
                        duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                    )
                    raise
                self._milestone_provider_attempt(
                    "tl:provider.call.retry",
                    identity,
                    outcome="context_overflow",
                    duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                )
                forced = await self._prepare_provider_gate(
                    state,
                    messages,
                    tools=tools,
                    max_output_tokens=max_tokens,
                    trigger="context_overflow",
                    force=True,
                )
                auxiliary_usages.extend(forced.auxiliary_usages)
                if forced.changed:
                    state.continuation = None
                prepared = forced
                request_message_count = len(messages)
                # 强制压缩重试是同一个 call，attempt=2。
                identity = _ProviderAttemptIdentity(
                    call_ordinal=call_ordinal,
                    provider_attempt=2,
                )
                _ = _provider_call_identity.set(identity)
                attempt_two_token = current_provider_attempt.set(2)
                state.call_started_at = time.monotonic()
                self._milestone_provider_attempt("tl:provider.call.start", identity)
                try:
                    response = await state.agent_model.complete(build_request())
                except asyncio.CancelledError:
                    self._milestone_provider_attempt(
                        "tl:provider.call.cancelled",
                        identity,
                        outcome="cancelled",
                        duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                    )
                    raise
                except Exception:
                    self._milestone_provider_attempt(
                        "tl:provider.call.error",
                        identity,
                        outcome="error",
                        duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                    )
                    raise
            except Exception:
                self._milestone_provider_attempt(
                    "tl:provider.call.error",
                    identity,
                    outcome="error",
                    duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                )
                raise
            response.tool_calls = [
                _tool_call_with_plain_arguments(call) for call in response.tool_calls
            ]
            # tool-first fallback：chat 刚返回判定 tool_calls 时立刻记录 duration。
            if response.tool_calls and not state.first_any_logged:
                state.first_any_logged = True
                self._milestone_provider_attempt(
                    "tl:turn.first_any",
                    identity,
                    duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
                    kind="tool",
                )
            self._milestone_provider_attempt(
                "tl:provider.call.done",
                identity,
                outcome="done",
                duration_ms=(time.monotonic() - state.call_started_at) * 1_000,
            )
            await state.gate.record_response(
                message_count=request_message_count,
                tools=tools,
                usage=response.usage,
            )
            state.continuation = response.continuation
            return _ProviderCallResult(
                response=response,
                prepared=prepared,
                auxiliary_usages=tuple(auxiliary_usages),
            )
        finally:
            if attempt_two_token is not None:
                current_provider_attempt.reset(attempt_two_token)
            current_provider_operation.reset(operation_token)
            current_provider_attempt.reset(attempt_token)
            current_provider_call_id.reset(call_id_token)
            _provider_call_identity.reset(identity_token)

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
        request_state: _TurnRequestState | None = None,
    ) -> tuple[str, tuple[ModelUsage, ...]]:
        # 1. 先构造收尾总结 prompt。
        summary_prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            f"[已调用工具] {', '.join(tools_used[-8:]) if tools_used else '无'}\n\n"
            + _INCOMPLETE_SUMMARY_PROMPT
        )

        # 2. 先尝试让模型给一段中文收尾总结。
        provider_usages: tuple[ModelUsage, ...] = ()
        try:
            summary_messages = messages + [
                support.build_context_hint_message(
                    "summary_request",
                    summary_prompt,
                )
            ]
            summary_max_tokens = (
                min(_SUMMARY_MAX_TOKENS, self._llm_config.max_tokens)
                if self._llm_config.max_tokens > 0
                else _SUMMARY_MAX_TOKENS
            )
            if request_state is None:
                raise RuntimeError("provider request gate required")
            response_result = await self._call_provider(
                request_state,
                summary_messages,
                tools=[],
                max_tokens=summary_max_tokens,
                disable_thinking=True,
            )
            response = response_result.response
            usages = [*response_result.auxiliary_usages]
            if response.usage is not None:
                usages.append(response.usage)
            provider_usages = tuple(usages)
            text = (response.content or "").strip()
            if text:
                return text, provider_usages
        except ProviderProjectionError:
            raise
        except Exception as exc:
            logger.warning("生成预算收尾总结失败: %s", exc)

        # 3. 模型收尾失败时，返回固定兜底文案。
        tool_text = "、".join(tools_used[-8:]) if tools_used else "无"
        done = f"已尝试 {iteration} 轮，调用工具 {len(tools_used)} 次（{tool_text}）。"
        return (
            f"这次任务还没完全收束。{done}"
            "我先停在当前进度，后续会继续基于已有工具结果补齐缺失信息并给你最终结论。",
            provider_usages,
        )

    def _build_result(
        self,
        *,
        reply: str,
        tools_used: list[str],
        tool_chain: list[dict[str, Any]],
        media: list[str],
        visible_names: set[str] | None,
        thinking: str | None,
        streamed: bool,
        react_input_samples: list[int],
        cache_usages: list[ModelUsage],
        tools_unlocked: list[str] | None = None,
        model_state: dict[str, object] | None = None,
        model_usages: list[ModelUsage] | None = None,
        finish_reasons: list[str | None] | None = None,
        mobile_attention: Literal["confirmation"] | None = None,
    ) -> ReasonerResult:
        # 1. 汇总运行时元数据。
        react_stats: dict[str, object] = {
            "iteration_count": len(react_input_samples),
            "turn_input_sum_tokens": sum(react_input_samples),
            "turn_input_peak_tokens": max(react_input_samples, default=0),
            "final_call_input_tokens": (
                react_input_samples[-1] if react_input_samples else 0
            ),
        }
        known_cache = [
            item
            for item in cache_usages
            if item.input_tokens is not None and item.cached_input_tokens is not None
        ]
        if known_cache:
            cache_prompt_tokens = sum(item.input_tokens or 0 for item in known_cache)
            cache_hit_tokens = sum(
                item.cached_input_tokens or 0 for item in known_cache
            )
            react_stats["cache_prompt_tokens"] = cache_prompt_tokens
            react_stats["cache_hit_tokens"] = cache_hit_tokens
            hit_rate = (
                cache_hit_tokens / cache_prompt_tokens
                if cache_prompt_tokens > 0
                else 0.0
            )
            logger.info(
                "[KV缓存] 本轮 prompt_tokens=%d hit_tokens=%d hit_rate=%.2f%%",
                cache_prompt_tokens,
                cache_hit_tokens,
                hit_rate * 100,
            )
        usage = _aggregate_usage(model_usages or [])
        react_stats["model_usage"] = {
            "input_tokens": usage.input_tokens,
            "cache_write_input_tokens": usage.cache_write_input_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "output_tokens": usage.output_tokens,
            "reasoning_output_tokens": usage.reasoning_output_tokens,
            "request_count": usage.request_count,
            "covered_request_count": usage.covered_request_count,
            "coverage": usage.coverage.value,
        }
        react_stats["finish_reasons"] = list(finish_reasons or [])

        # 2. 最后返回标准 ReasonerResult。
        return ReasonerResult(
            reply=reply,
            thinking=thinking,
            streamed=streamed,
            tools_used=list(tools_used),
            tools_unlocked=list(tools_unlocked or []),
            tool_chain=list(tool_chain),
            media=list(media),
            visible_names=set(visible_names) if visible_names is not None else None,
            react_stats=react_stats,
            model_state=model_state,
            mobile_attention=mobile_attention,
        )

    @staticmethod
    def format_request_time_anchor(ts: datetime | None) -> str:
        # 1. 空时间戳时，使用当前本地时间。
        if ts is None:
            ts = datetime.now().astimezone()
        elif ts.tzinfo is None:
            ts = ts.astimezone()

        # 2. 输出稳定的 request_time 锚点字符串。
        return f"request_time={ts.isoformat()} ({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})"


# ── 模块级辅助函数 ──────────────────────────────────────────────


def extract_model_facing_turn(
    messages: list[dict],
) -> tuple[object | None, str | None]:
    if not messages:
        return None, None
    user_content = (
        messages[-1].get("content") if messages[-1].get("role") == "user" else None
    )
    if len(messages) < 2:
        return user_content, None
    frame = messages[-2]
    frame_content = frame.get("content")
    if isinstance(frame_content, str) and is_context_frame(frame_content):
        return user_content, frame_content
    return user_content, None


def _collect_current_akashic_push_media(
    target: list[str],
    arguments: dict[str, Any],
    *,
    channel: str,
    chat_id: str,
) -> None:
    if channel != "akashic":
        return
    if str(arguments.get("target_channel") or "").strip() != channel:
        return
    if str(arguments.get("target_chat_id") or "").strip() != chat_id:
        return
    for key in ("image", "file"):
        value = str(arguments.get(key) or "").strip()
        if value and value not in target:
            target.append(value)


def build_turn_injection_prompt(
    *,
    tools: "ToolRegistry",
    tool_search_enabled: bool,
    visible_names: set[str] | None,
) -> str:
    if not tool_search_enabled:
        return ""
    return build_deferred_tools_hint(tools, visible=visible_names)


def _provider_max_tool_schemas(provider: object) -> int:
    raw_limit = getattr(provider, "max_tool_schemas", None)
    if raw_limit is None:
        return 0
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
        raise TypeError("provider.max_tool_schemas 必须是整数")
    if raw_limit < 0:
        raise ValueError("provider.max_tool_schemas 不能为负数")
    return raw_limit


def _context_window(model: BoundChatModel) -> int:
    return model.descriptor.capabilities.context_window or 0


def _prompt_cache_key(model: BoundChatModel, namespace: str) -> str | None:
    if not namespace:
        return None
    payload = f"{model.descriptor.binding_id}\0{namespace}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_continuation_state(
    continuation: ModelContinuation | None,
) -> dict[str, object] | None:
    if continuation is None:
        return None
    return {
        "schema_version": 2,
        "binding_id": continuation.binding_id,
        "payload": _plain_json(continuation.payload),
    }


def _model_binding_payload(model: BoundChatModel) -> dict[str, object]:
    """Project one exact public model binding into Turn observability data."""

    descriptor = model.descriptor
    return {
        "binding_id": descriptor.binding_id,
        "plugin_snapshot_id": descriptor.plugin_snapshot_id,
        "model_revision": descriptor.model_revision,
        "model_id": descriptor.model_id,
        "connection_id": descriptor.connection_id,
        "driver_id": descriptor.driver_id,
        "driver_contract_version": descriptor.driver_contract_version,
        "model": descriptor.model,
        "role": descriptor.role.value,
        "reasoning_effort": descriptor.reasoning_effort or "",
        "capability_digest": descriptor.capability_digest,
    }


def _load_model_continuation(
    messages: list[dict],
    model: BoundChatModel,
) -> ModelContinuation | None:
    """Resume schema v2 only from the latest assistant and exact binding."""

    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        state = message.get("model_state")
        if not isinstance(state, dict):
            return None
        if state.get("schema_version") == 2:
            if state.get("binding_id") != model.descriptor.binding_id:
                return None
            payload = state.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("message model_state.payload 必须是对象")
            return ModelContinuation(
                binding_id=model.descriptor.binding_id,
                payload=payload,
            )
        return None
    return None


def _plain_json(value: object, *, _seen: set[int] | None = None) -> object:
    """Copy one JSON value without coercing invalid plugin data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("model JSON number 必须是有限值")
        return value
    if isinstance(value, (Mapping, list, tuple)):
        seen = set() if _seen is None else _seen
        identity = id(value)
        if identity in seen:
            raise ValueError("model JSON 不允许循环引用")
        seen.add(identity)
        try:
            if isinstance(value, Mapping):
                copied: dict[str, object] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ValueError("model JSON object key 必须是字符串")
                    copied[key] = _plain_json(item, _seen=seen)
                return copied
            return [_plain_json(item, _seen=seen) for item in value]
        finally:
            seen.remove(identity)
    raise ValueError(f"model JSON 不支持 {type(value).__name__}")


def _tool_call_with_plain_arguments(call: ToolCall) -> ToolCall:
    """Copy one plugin tool call into the runtime-owned JSON representation."""

    arguments = _plain_json(call.arguments)
    if not isinstance(arguments, dict):
        raise TypeError("model tool call arguments 必须是对象")
    return ToolCall(
        id=call.id,
        name=call.name,
        arguments=cast(dict[str, Any], arguments),
    )


def _aggregate_usage(items: list[ModelUsage]) -> ModelUsage:
    def total(field: str) -> int | None:
        known = [value for item in items if (value := getattr(item, field)) is not None]
        return sum(known) if known else None

    request_count = sum(item.request_count for item in items)
    covered = sum(item.covered_request_count for item in items)
    coverage = (
        UsageCoverage.UNAVAILABLE
        if not items
        or all(item.coverage is UsageCoverage.UNAVAILABLE for item in items)
        else (
            UsageCoverage.EXACT
            if covered == request_count
            and all(item.coverage is UsageCoverage.EXACT for item in items)
            else UsageCoverage.PARTIAL
        )
    )
    return ModelUsage(
        input_tokens=total("input_tokens"),
        cache_write_input_tokens=total("cache_write_input_tokens"),
        cached_input_tokens=total("cached_input_tokens"),
        output_tokens=total("output_tokens"),
        reasoning_output_tokens=total("reasoning_output_tokens"),
        request_count=request_count,
        covered_request_count=covered,
        coverage=coverage,
    )


def _project_tool_order(candidates: list[str], limit: int) -> list[str]:
    ordered = list(dict.fromkeys(name for name in candidates if name))
    if "tool_search" in ordered:
        ordered.remove("tool_search")
        ordered.insert(0, "tool_search")
    if limit <= 0:
        return ordered
    return ordered[:limit]


def build_deferred_tools_hint(
    tools: "ToolRegistry",
    visible: set[str] | None = None,
) -> str:
    deferred = tools.get_deferred_names(visible=visible)
    builtin = deferred["builtin"]
    mcp = deferred["mcp"]

    if not builtin and not mcp:
        return ""

    lines: list[str] = ["【未加载工具目录（知道名字但 schema 未暴露）】"]
    if builtin:
        lines.append(f"内置: {', '.join(builtin)}")
    for server, names in mcp.items():
        lines.append(f"MCP ({server}): {', '.join(names)}")

    total = len(builtin) + sum(len(v) for v in mcp.values())
    lines.append(
        f"\n共 {total} 个。加载方式：\n"
        '- 已知工具名 → tool_search(query="select:工具名")，支持逗号分隔多个\n'
        '- 描述功能   → tool_search(query="关键词") 搜索匹配'
    )
    return "\n".join(lines) + "\n\n"
