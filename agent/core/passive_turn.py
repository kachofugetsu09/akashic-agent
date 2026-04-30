from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

import agent.core.passive_support as support
from agent.core.runtime_support import ToolDiscoveryState
from agent.core.types import (
    ContextBundle,
    ContextRequest,
    LLMToolCall,
    ReasonerResult,
)
from agent.prompting import DEFAULT_CONTEXT_TRIM_PLANS, is_context_frame
from agent.provider import ContentSafetyError, ContextLengthError
from agent.retrieval.protocol import RetrievalRequest
from agent.tool_hooks import ToolExecutionRequest, ToolExecutor
from agent.tool_runtime import append_assistant_tool_calls, append_tool_result, tool_call_signature
from agent.tools.base import normalize_tool_result
from agent.tools.tool_search import ToolSearchTool
from agent.turns.outbound import OutboundDispatch, OutboundPort
from bus.event_bus import EventBus
from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    ToolCallCompleted,
    ToolCallStarted,
)
from agent.lifecycle.phase import Phase
from agent.lifecycle.phases.after_reasoning import (
    AfterReasoningFrame,
    default_after_reasoning_modules,
)
from agent.lifecycle.phases.after_turn import AfterTurnFrame, default_after_turn_modules
from agent.lifecycle.phases.before_reasoning import (
    BeforeReasoningFrame,
    default_before_reasoning_modules,
)
from agent.lifecycle.phases.before_turn import BeforeTurnFrame, default_before_turn_modules
from agent.lifecycle.types import (
    AfterReasoningInput,
    AfterReasoningResult,
    AfterStepCtx,
    AfterToolResultCtx,
    BeforeReasoningCtx,
    BeforeReasoningInput,
    BeforeStepCtx,
    BeforeStepInput,
    BeforeToolCallCtx,
    BeforeTurnCtx,
    TurnSnapshot,
    TurnState,
)

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.runtime_support import SessionLike, TurnRunResult
    from agent.looping.ports import LLMConfig, LLMServices, SessionServices
    from agent.retrieval.protocol import MemoryRetrievalPipeline
    from agent.tool_hooks.base import ToolHook
    from session.manager import SessionManager
    from agent.tools.registry import ToolRegistry

# 1. 统一通过模块 logger 记录关键分支，供排障和回归测试抓取。
logger = logging.getLogger(__name__)

# 被动链路核心入口，负责串起 lifecycle 模块链与 reasoner。
#
# ┌─ inbound
# │  └─ AgentCore.process
# │     └─ PassiveTurnPipeline.run
# │        ├─ BeforeTurn
# │        │  └─ session acquire + ContextStore.prepare + EventBus.emit
# │        ├─ BeforeReasoning
# │        │  └─ tool context sync + EventBus.emit + prompt warmup
# │        ├─ Reasoner.run_turn
# │        │  └─ Reasoner.run
# │        │     ├─ BeforeStep
# │        │     │  └─ token estimate + EventBus.emit + hint injection
# │        │     └─ AfterStep
# │        │        └─ EventBus.fanout
# │        ├─ AfterReasoning
# │        │  └─ parse + EventBus.emit + persist + outbound build
# │        └─ AfterTurn
# │           └─ TurnCommitted fanout + AfterTurn fanout + dispatch
# └─ done

# ── 被动 turn 内联常量 ──────────────────────────────────────────
_SAFETY_RETRY_RATIOS = (1.0, 0.5, 0.0)
_TOOL_LOOP_REPEAT_LIMIT = 3
_SUMMARY_MAX_TOKENS = 512
_INCOMPLETE_SUMMARY_PROMPT = """当前任务未在预算内完成，请直接输出给用户的中文收尾说明（不要提及系统/工具内部细节）。
必须包含三点：
1) 已完成到哪一步（基于当前上下文的事实）；
2) 目前还缺什么信息或步骤；
3) 下一步你会怎么继续。
禁止输出"已达到最大迭代次数"这类模板句；不要输出 JSON。"""

class _NoopOutboundPort:
    async def dispatch(self, outbound: OutboundDispatch) -> bool:
        return False


@dataclass
class AgentCoreDeps:
    session: "SessionServices"
    context_store: "ContextStore"
    context: "ContextBuilder"
    tools: "ToolRegistry"
    reasoner: "Reasoner"
    event_bus: "EventBus | None" = None
    outbound_port: "OutboundPort | None" = None
    history_window: int = 500
    before_turn_plugin_modules_early: list[object] | None = None
    before_turn_plugin_modules_late: list[object] | None = None


class AgentCore:
    """
    ┌──────────────────────────────────────┐
    │ AgentCore                            │
    ├──────────────────────────────────────┤
    │ 1. 持有 PassiveTurnPipeline          │
    │ 2. 委托 pipeline 处理被动消息        │
    └──────────────────────────────────────┘
    """

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._passive_pipeline = PassiveTurnPipeline(deps)

    @property
    def pipeline(self) -> "PassiveTurnPipeline":
        return self._passive_pipeline

    def add_before_turn_plugin_modules(
        self,
        early: list[object],
        late: list[object],
    ) -> None:
        self._passive_pipeline.add_before_turn_plugin_modules(early, late)

    async def process(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        return await self._passive_pipeline.run(
            msg,
            key,
            dispatch_outbound=dispatch_outbound,
        )


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

    def __init__(self, deps: AgentCoreDeps) -> None:
        self._session = deps.session
        self._context_store = deps.context_store
        self._context = deps.context
        self._tools = deps.tools
        self._reasoner = deps.reasoner
        self._outbound_port = deps.outbound_port
        self._before_turn_plugin_modules_early = list(deps.before_turn_plugin_modules_early or [])
        self._before_turn_plugin_modules_late = list(deps.before_turn_plugin_modules_late or [])
        bus = deps.event_bus or EventBus()
        self._bus = bus

        self._before_turn = self._build_before_turn_phase()
        self._before_reasoning: Phase[
            BeforeReasoningInput,
            BeforeReasoningCtx,
            BeforeReasoningFrame,
        ] = Phase(
            default_before_reasoning_modules(
                bus,
                deps.tools,
                self._session.session_manager,
                deps.context,
            ),
            frame_factory=BeforeReasoningFrame,
        )
        self._after_reasoning: Phase[
            AfterReasoningInput,
            AfterReasoningResult,
            AfterReasoningFrame,
        ] = Phase(
            default_after_reasoning_modules(bus, self._session),
            frame_factory=AfterReasoningFrame,
        )
        outbound_port = deps.outbound_port or _NoopOutboundPort()
        self._after_turn: Phase[TurnSnapshot, OutboundMessage, AfterTurnFrame] = Phase(
            default_after_turn_modules(
                bus,
                outbound_port,
                deps.context,
                deps.history_window,
            ),
            frame_factory=AfterTurnFrame,
        )

    def add_before_turn_plugin_modules(
        self,
        early: list[object],
        late: list[object],
    ) -> None:
        self._before_turn_plugin_modules_early.extend(early)
        self._before_turn_plugin_modules_late.extend(late)
        self._before_turn = self._build_before_turn_phase()

    def _build_before_turn_phase(self) -> Phase[TurnState, BeforeTurnCtx, BeforeTurnFrame]:
        return Phase(
            default_before_turn_modules(
                self._bus,
                self._session.session_manager,
                self._context_store,
                plugin_modules_early=cast("list[Any]", self._before_turn_plugin_modules_early),
                plugin_modules_late=cast("list[Any]", self._before_turn_plugin_modules_late),
            ),
            frame_factory=BeforeTurnFrame,
        )

    # 核心方法：处理一条普通被动消息，并提交最终出站结果。
    async def run(
        self,
        msg: InboundMessage,
        key: str,
        *,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        state = TurnState(
            msg=msg,
            session_key=key,
            dispatch_outbound=dispatch_outbound,
        )
        # try/except 只包前置模块链和 reasoning：在派发前兜底并返回错误提示。
        try:
            # Phase 1: BeforeTurn 模块链（会话、上下文、BeforeTurn 事件）。
            before_turn = await self._before_turn.run(state)
            if before_turn.abort:
                return await self._control_outbound(
                    state,
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=before_turn.abort_reply,
                    ),
                )

            # Phase 2: BeforeReasoning 模块链（工具上下文、BeforeReasoning 事件、prompt warmup）。
            before_reasoning = await self._before_reasoning.run(
                BeforeReasoningInput(state=state, before_turn=before_turn)
            )
            if before_reasoning.abort:
                return await self._control_outbound(
                    state,
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=before_reasoning.abort_reply,
                    ),
                )

            # Phase 3-4: Reasoning（BeforeStep/AfterStep 模块链在 Reasoner 内部执行）。
            session = state.session
            if session is None:
                raise RuntimeError("Passive turn requires TurnState.session")
            turn_result = await self._reasoner.run_turn(
                msg=msg,
                skill_names=list(before_reasoning.skill_names) or None,
                session=session,
                base_history=None,
                retrieved_memory_block=before_reasoning.retrieved_memory_block,
                extra_hints=list(before_reasoning.extra_hints) or None,
            )
        except Exception:
            logger.exception("PassiveTurnPipeline.run failed before dispatch session=%s", key)
            return await self._control_outbound(
                state,
                OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content="处理消息时出错，请稍后再试。",
                ),
            )

        # Phase 5: AfterReasoning 模块链（parse、AfterReasoning 事件、持久化、出站消息）。
        after_reasoning = await self._after_reasoning.run(
            AfterReasoningInput(state=state, turn_result=turn_result)
        )

        # Phase 6: AfterTurn 模块链（TurnCommitted fanout、AfterTurn fanout、dispatch）。
        return await self._after_turn.run(
            TurnSnapshot(
                state=state,
                outbound=after_reasoning.outbound,
                ctx=after_reasoning.ctx,
            )
        )

    # 供外部调用方（如 spawn completion）复用 AfterReasoning + dispatch 流程。
    async def post_reasoning(
        self,
        msg: InboundMessage,
        session_key: str,
        turn_result: "TurnRunResult",
        *,
        dispatch_outbound: bool = True,
        retrieval_raw: object | None = None,
    ) -> OutboundMessage:
        state = TurnState(
            msg=msg,
            session_key=session_key,
            dispatch_outbound=dispatch_outbound,
            session=self._session.session_manager.get_or_create(session_key),
            retrieval_raw=retrieval_raw,
        )
        after_reasoning = await self._after_reasoning.run(
            AfterReasoningInput(state=state, turn_result=turn_result)
        )
        return await self._after_turn.run(
            TurnSnapshot(
                state=state,
                outbound=after_reasoning.outbound,
                ctx=after_reasoning.ctx,
            )
        )

    # abort / 错误路径的统一 dispatch helper，只有 dispatch_outbound=True 时才发送。
    async def _control_outbound(
        self,
        state: TurnState,
        outbound: OutboundMessage,
    ) -> OutboundMessage:
        if state.dispatch_outbound and self._outbound_port is not None:
            _ = await self._outbound_port.dispatch(
                OutboundDispatch(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=outbound.content,
                    thinking=outbound.thinking,
                    metadata=outbound.metadata,
                    media=outbound.media,
                )
            )
        return outbound


class ContextStore(ABC):
    """
    ┌──────────────────────────────────────┐
    │ ContextStore                         │
    ├──────────────────────────────────────┤
    │ 1. 读取 session history              │
    │ 2. 调 retrieval pipeline             │
    │ 3. 收 skill mentions                 │
    │ 4. 输出 ContextBundle                │
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
        retrieval: "MemoryRetrievalPipeline",
        context: "ContextBuilder",
        history_window: int = 500,
    ) -> None:
        self._retrieval = retrieval
        self._context = context
        self._history_window = max(1, int(history_window))

    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        # 1. 先读取 session history，并转换成 retrieval pipeline 需要的结构。
        raw_history = list(session.get_history())
        history_messages = support.to_history_messages(raw_history)

        # 2. 再执行 retrieval，保持当前 pipeline 行为不变。
        retrieval_result = await self._retrieval.retrieve(
            RetrievalRequest(
                message=msg.content,
                session_key=session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                history=history_messages,
                session_metadata=(
                    session.metadata if isinstance(session.metadata, dict) else {}
                ),
                timestamp=msg.timestamp,
            )
        )

        # 3. 最后补齐 ContextBundle，把主链正式字段直接收进显式合同。
        skill_mentions = support.collect_skill_mentions(
            msg.content,
            self._context.skills.list_skills(filter_unavailable=False),
        )
        return ContextBundle(
            history=support.to_chat_messages(raw_history),
            memory_blocks=[retrieval_result.block] if retrieval_result.block else [],
            skill_mentions=skill_mentions,
            retrieved_memory_block=retrieval_result.block or "",
            retrieval_trace_raw=(
                retrieval_result.trace.raw
                if retrieval_result.trace is not None
                else None
            ),
            retrieval_metadata=dict(retrieval_result.metadata or {}),
            history_messages=history_messages,
        )

class Reasoner(ABC):

    @abstractmethod
    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
    ) -> ReasonerResult:
        """执行多轮 tool loop，并返回本轮结果。"""

    @abstractmethod
    async def run_turn(
        self,
        *,
        msg,
        session: "SessionLike",
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
        extra_hints: list[str] | None = None,
    ) -> "TurnRunResult":
        """执行完整被动 turn，包括 retry / trim / tool loop。"""

    def add_tool_hooks(self, hooks: list["ToolHook"]) -> None:
        """子类可重写以注入 tool hooks。默认 no-op。"""


class DefaultReasoner(Reasoner):
    def __init__(
        self,
        llm: "LLMServices",
        llm_config: "LLMConfig",
        tools: "ToolRegistry",
        discovery: ToolDiscoveryState,
        *,
        tool_search_enabled: bool,
        memory_window: int,
        context: "ContextBuilder | None" = None,
        session_manager: "SessionManager | None" = None,
        event_bus: "EventBus | None" = None,
    ) -> None:
        self._llm = llm
        self._llm_config = llm_config
        self._tools = tools
        self._discovery = discovery
        self._tool_search_enabled = tool_search_enabled
        self._memory_window = memory_window
        self._context = context
        self._session_manager = session_manager
        self._event_bus = event_bus
        # Direct reference to ToolSearchTool so we can pass excluded_names
        # explicitly instead of routing through the ContextVar side-channel.
        _ts = tools.get_tool("tool_search")
        self._tool_search_tool: ToolSearchTool | None = (
            _ts if isinstance(_ts, ToolSearchTool) else None
        )
        self._tool_executor = ToolExecutor([])
        self._stream_sink_factory: Callable[
            [object], Callable[[dict[str, str] | str], Awaitable[None]] | None
        ] | None = None
        bus = event_bus or EventBus()
        self._bus = bus
        from agent.lifecycle.phases.before_step import (
            BeforeStepFrame,
            default_before_step_modules,
        )
        from agent.lifecycle.phases.after_step import (
            AfterStepFrame,
            default_after_step_modules,
        )

        self._before_step: Phase[BeforeStepInput, BeforeStepCtx, BeforeStepFrame] = Phase(
            default_before_step_modules(bus),
            frame_factory=BeforeStepFrame,
        )
        self._after_step: Phase[AfterStepCtx, AfterStepCtx, AfterStepFrame] = Phase(
            default_after_step_modules(bus),
            frame_factory=AfterStepFrame,
        )

    def add_tool_hooks(self, hooks: list["ToolHook"]) -> None:
        self._tool_executor.add_hooks(hooks)

    def set_stream_sink_factory(
        self,
        factory: Callable[
            [object], Callable[[dict[str, str] | str], Awaitable[None]] | None
        ]
        | None,
    ) -> None:
        self._stream_sink_factory = factory

    async def run_turn(
        self,
        *,
        msg,
        session: "SessionLike",
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
        extra_hints: list[str] | None = None,
    ) -> "TurnRunResult":
        from agent.core.runtime_support import TurnRunResult

        if self._context is None or self._session_manager is None:
            raise RuntimeError("DefaultReasoner.run_turn requires context and session_manager")

        # 1. 先准备 retry trace、history 和 preload 工具集合。
        retry_attempts: list[dict[str, object]] = []
        retry_trace: dict[str, object] = {
            "attempts": retry_attempts,
            "selected_plan": None,
            "trimmed_sections": [],
        }
        source_history = (
            base_history
            if base_history is not None
            else get_history_since_consolidated(session, self._memory_window)
        )
        total_history = len(source_history)
        preloaded: set[str] | None = None
        if self._tool_search_enabled:
            preloaded = self._discovery.get_preloaded(session.key)
            logger.info(
                "[tool_search] LRU preloaded=%s",
                sorted(preloaded) if preloaded else "[]",
            )
        stream_sink = (
            self._stream_sink_factory(msg) if self._stream_sink_factory is not None else None
        )

        # 2. 再按 trim plan + history window 顺序逐轮尝试。
        attempts = self._build_attempt_plans(total_history)
        for attempt, plan in enumerate(attempts):
            retry_attempts.append(
                {
                    "name": plan["name"],
                    "history_window": plan["history_window"],
                    "disabled_sections": sorted(plan["disabled_sections"]),
                }
            )
            history_for_attempt = self._slice_history(
                source_history,
                plan["history_window"],
            )
            turn_injection_prompt = build_turn_injection_prompt(
                tools=self._tools,
                tool_search_enabled=self._tool_search_enabled,
                visible_names=preloaded if self._tool_search_enabled else None,
            )
            initial_messages = self._context.render(
                ContextRequest(
                    history=history_for_attempt,
                    current_message=msg.content,
                    media=msg.media if msg.media else None,
                    skill_names=skill_names,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    message_timestamp=msg.timestamp,
                    retrieved_memory_block=retrieved_memory_block,
                    disabled_sections=plan["disabled_sections"],
                    turn_injection_prompt=turn_injection_prompt,
                )
            ).messages
            # 将 BeforeReasoning 产生的额外提示注入每次 retry 的 prompt。
            if extra_hints:
                initial_messages.append(
                    support.build_context_hint_message(
                        "plugin_hints",
                        "\n".join(extra_hints),
                    )
                )
            llm_user_content, llm_context_frame = extract_model_facing_turn(
                initial_messages
            )
            try:
                result = await self.run(
                    initial_messages,
                    request_time=msg.timestamp,
                    preloaded_tools=preloaded,
                    preflight_injected=True,
                    on_content_delta=stream_sink,
                    tool_event_session_key=session.key,
                    tool_event_channel=msg.channel,
                    tool_event_chat_id=msg.chat_id,
                )
                tools_used = list(result.metadata.get("tools_used") or [])
                tool_chain = list(result.metadata.get("tool_chain") or [])
                if attempt > 0:
                    window = plan["history_window"]
                    retry_trace["selected_plan"] = plan["name"]
                    retry_trace["trimmed_sections"] = sorted(plan["disabled_sections"])
                    logger.warning(
                        "重试成功 plan=%s window=%d disabled=%s，修剪 session 历史",
                        plan["name"],
                        window,
                        sorted(plan["disabled_sections"]),
                    )
                    if window == 0:
                        session.messages.clear()
                    else:
                        session.messages = session.messages[-window:]
                    session.last_consolidated = 0
                    await self._session_manager.save_async(cast(Any, session))

                if self._tool_search_enabled and tools_used:
                    self._discovery.update(
                        session.key,
                        tools_used,
                        self._tools.get_always_on_names(),
                    )
                if attempt == 0:
                    retry_trace["selected_plan"] = plan["name"]
                    retry_trace["trimmed_sections"] = sorted(plan["disabled_sections"])
                if isinstance(llm_user_content, (str, list)):
                    retry_trace["llm_user_content"] = llm_user_content
                if isinstance(llm_context_frame, str) and llm_context_frame.strip():
                    retry_trace["llm_context_frame"] = llm_context_frame
                retry_trace["react_stats"] = dict(result.metadata.get("react_stats") or {})
                return TurnRunResult(
                    reply=result.reply,
                    tools_used=tools_used,
                    tool_chain=tool_chain,
                    thinking=result.thinking,
                    streamed=result.streamed,
                    context_retry=retry_trace,
                )
            except ContentSafetyError:
                if attempt < len(attempts) - 1:
                    next_plan = attempts[attempt + 1]
                    logger.warning(
                        "安全拦截 (attempt=%d)，切到 plan=%s window=%d disabled=%s",
                        attempt + 1,
                        next_plan["name"],
                        next_plan["history_window"],
                        sorted(next_plan["disabled_sections"]),
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return TurnRunResult(
                        reply="你的消息触发了安全审查，无法处理。",
                        context_retry=retry_trace,
                    )
            except ContextLengthError:
                if attempt < len(attempts) - 1:
                    next_plan = attempts[attempt + 1]
                    logger.warning(
                        "上下文超长 (attempt=%d)，切到 plan=%s window=%d disabled=%s",
                        attempt + 1,
                        next_plan["name"],
                        next_plan["history_window"],
                        sorted(next_plan["disabled_sections"]),
                    )
                else:
                    logger.warning("上下文超长：所有窗口均失败，清空历史后仍超长")
                    return TurnRunResult(
                        reply="上下文过长无法处理，请尝试新建对话。",
                        context_retry=retry_trace,
                    )
            except asyncio.TimeoutError:
                logger.warning("LLM 流响应超时 (attempt=%d)，远端连接中断", attempt + 1)
                return TurnRunResult(
                    reply="模型流响应中断，请刷新对话重试。",
                    context_retry=retry_trace,
                )
        return TurnRunResult(reply="（安全重试异常）", context_retry=retry_trace)

    async def run(
        self,
        initial_messages: list[dict],
        *,
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = True,
        on_content_delta: Callable[[dict[str, str]], Awaitable[None]] | None = None,
        tool_event_session_key: str = "",
        tool_event_channel: str = "",
        tool_event_chat_id: str = "",
    ) -> ReasonerResult:
        # 1. 初始化消息上下文、本轮工具轨迹、循环检测状态。
        messages = initial_messages
        tools_used: list[str] = []
        tool_chain: list[dict] = []
        last_tool_signature = ""
        repeat_count = 0
        # 2. 初始化本轮可见工具集合。
        visible_names: set[str] | None = None
        streamed = False
        react_input_samples: list[int] = []
        react_cache_prompt_tokens = 0
        react_cache_hit_tokens = 0
        react_cache_seen = False
        if self._tool_search_enabled:
            always_on = self._tools.get_always_on_names()
            visible_names = always_on | (preloaded_tools or set())
            logger.info(
                "[tool_search] visible=%d 个工具 always_on=%d preloaded=%d need_search=%s",
                len(visible_names),
                len(always_on),
                len(preloaded_tools or set()),
                "yes" if len(visible_names) == len(always_on) else "maybe",
            )

        for iteration in range(self._llm_config.max_iterations):
            # 3. BeforeStep 模块链：token 估算、BeforeStep 事件、提示注入。
            step_ctx = await self._before_step.run(BeforeStepInput(
                session_key=tool_event_session_key,
                channel=tool_event_channel,
                chat_id=tool_event_chat_id,
                iteration=iteration,
                messages=messages,
                visible_names=visible_names,
            ))
            if step_ctx.early_stop:
                summary = await self._summarize_incomplete_progress(
                    messages,
                    reason="early_stop",
                    iteration=iteration + 1,
                    tools_used=tools_used,
                )
                return self._build_result(
                    reply=step_ctx.early_stop_reply or summary,
                    tools_used=tools_used,
                    tool_chain=tool_chain,
                    visible_names=visible_names,
                    thinking=None,
                    streamed=False,
                    react_input_samples=react_input_samples,
                    cache_prompt_tokens=react_cache_prompt_tokens,
                    cache_hit_tokens=react_cache_hit_tokens,
                    cache_seen=react_cache_seen,
                )
            # 4. 调用 LLM，带上当前可见工具 schema。
            react_input_samples.append(step_ctx.input_tokens_estimate)
            logger.info(
                "[LLM调用] 第%d轮，可见工具=%s input_tokens~=%d",
                iteration + 1,
                f"{len(visible_names)}个" if visible_names is not None else "全部（tool_search未开启）",
                step_ctx.input_tokens_estimate,
            )
            response = await self._llm.provider.chat(
                messages=messages,
                tools=self._tools.get_schemas(names=visible_names),
                model=self._llm_config.model,
                max_tokens=self._llm_config.max_tokens,
                tool_choice="auto",
                on_content_delta=on_content_delta,
            )
            if on_content_delta is not None and response.content:
                streamed = True
            if response.cache_prompt_tokens is not None:
                react_cache_seen = True
                react_cache_prompt_tokens += response.cache_prompt_tokens
                react_cache_hit_tokens += response.cache_hit_tokens or 0

            # 5. 模型返回 tool_calls 时，进入工具执行分支。
            if response.tool_calls:
                logger.info(
                    "[LLM决策→工具] 第%d轮，调用: %s",
                    iteration + 1,
                    [tc.name for tc in response.tool_calls],
                )
                signature = tool_call_signature(response.tool_calls)
                if signature and signature == last_tool_signature:
                    repeat_count += 1
                else:
                    repeat_count = 1
                    last_tool_signature = signature

                if repeat_count >= _TOOL_LOOP_REPEAT_LIMIT:
                    logger.warning(
                        "[循环检测] 工具调用连续重复%d次，强制收尾 (iteration=%d, signature=%s)",
                        repeat_count,
                        iteration + 1,
                        signature[:80] if signature else "",
                    )
                    summary = await self._summarize_incomplete_progress(
                        messages,
                        reason="tool_call_loop",
                        iteration=iteration + 1,
                        tools_used=tools_used,
                    )
                    return self._build_result(
                        reply=summary,
                        tools_used=tools_used,
                        tool_chain=tool_chain,
                        visible_names=visible_names,
                        thinking=None,
                        streamed=False,
                        react_input_samples=react_input_samples,
                        cache_prompt_tokens=react_cache_prompt_tokens,
                        cache_hit_tokens=react_cache_hit_tokens,
                        cache_seen=react_cache_seen,
                    )

                append_assistant_tool_calls(
                    messages,
                    content=response.content,
                    tool_calls=response.tool_calls,
                    provider_fields=response.provider_fields,
                )

                # 6. 逐个执行本轮工具调用。
                iter_calls: list[dict] = []
                for tool_call in response.tool_calls:
                    # 6.1 deferred 工具未解锁时，先回填 select: 引导错误。
                    if visible_names is not None and tool_call.name not in visible_names:
                        await self._observe_tool_call_started(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                        )
                        logger.warning(
                            "[工具未解锁] LLM 尝试调用 '%s'，但该工具 schema 不可见，引导模型先 tool_search",
                            tool_call.name,
                        )
                        result = (
                            f"工具 '{tool_call.name}' 当前未加载（schema 不可见）。"
                            f"请先调用 tool_search(query=\"select:{tool_call.name}\") 加载，"
                            "然后再调用该工具。不要放弃当前任务。"
                        )
                        append_tool_result(
                            messages,
                            tool_call_id=tool_call.id,
                            content=result,
                        )
                        await self._observe_tool_call_completed(
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                            iteration=iteration + 1,
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            final_arguments=tool_call.arguments,
                            status="blocked",
                            result_preview=support.log_preview(result),
                        )
                        iter_calls.append(
                            {
                                "call_id": tool_call.id,
                                "name": tool_call.name,
                                "arguments": tool_call.arguments,
                                "result": result,
                            }
                        )
                        continue

                    # 6.2 通过统一执行器跑 pre/post hooks + 真实工具。
                    # For tool_search: pass visible_names explicitly via
                    # set_excluded_names() instead of the old ContextVar channel.
                    if (
                        tool_call.name == "tool_search"
                        and visible_names is not None
                        and self._tool_search_tool is not None
                    ):
                        self._tool_search_tool.set_excluded_names(visible_names)
                    _args_preview = support.log_preview(tool_call.arguments, 120)
                    logger.info("[工具执行→] %s  args=%s", tool_call.name, _args_preview)
                    await self._observe_tool_call_started(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        iteration=iteration + 1,
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                    )
                    # 工具调用统一先过 ToolExecutor：
                    # pre_hook 可改参/拒绝，真实执行后再补 post_hook trace。
                    await self._bus.fanout(BeforeToolCallCtx(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        tool_name=tool_call.name,
                        arguments=dict(tool_call.arguments),
                    ))
                    exec_result = await self._tool_executor.execute(
                        ToolExecutionRequest(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            source="passive",
                            session_key=tool_event_session_key,
                            channel=tool_event_channel,
                            chat_id=tool_event_chat_id,
                        ),
                        # 真实工具执行入口仍是 ToolRegistry.execute；
                        # hook 只负责拦截与记录，不替代 registry。
                        self._tools.execute,
                    )
                    if exec_result.status == "success":
                        tools_used.append(tool_call.name)
                    result = exec_result.output
                    await self._bus.fanout(AfterToolResultCtx(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        tool_name=tool_call.name,
                        arguments=dict(exec_result.final_arguments),
                        result=str(result),
                        status=exec_result.status,
                    ))
                    normalized = normalize_tool_result(result)
                    _result_preview = support.log_preview(normalized.preview())
                    _result_len = len(normalized.preview() or "")
                    await self._observe_tool_call_completed(
                        session_key=tool_event_session_key,
                        channel=tool_event_channel,
                        chat_id=tool_event_chat_id,
                        iteration=iteration + 1,
                        call_id=tool_call.id,
                        tool_name=tool_call.name,
                        arguments=tool_call.arguments,
                        final_arguments=exec_result.final_arguments,
                        status=exec_result.status,
                        result_preview=normalized.preview(),
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
                    )

                    # 6.3 tool_search 的结果会扩展下一轮可见工具。
                    if (
                        exec_result.status == "success"
                        and tool_call.name == "tool_search"
                        and visible_names is not None
                    ):
                        _newly_unlocked = self._discovery.unlock_from_result(normalized.text)
                        _newly_unlocked -= visible_names  # keep only genuinely new ones
                        if _newly_unlocked:
                            visible_names.update(_newly_unlocked)
                            logger.info("[工具解锁] tool_search 新解锁: %s", sorted(_newly_unlocked))
                        else:
                            logger.info("[工具解锁] tool_search 未解锁新工具")
                    # tool_chain 持久化的是“执行后的事实”：
                    # 最终参数、hook trace、结果预览，供后续回放与 session 复原。
                    iter_calls.append(
                        {
                            "call_id": tool_call.id,
                            "name": tool_call.name,
                            "status": exec_result.status,
                            "arguments": tool_call.arguments,
                            "final_arguments": exec_result.final_arguments,
                            "pre_hook_trace": [
                                {
                                    "hook_name": item.hook_name,
                                    "event": item.event,
                                    "matched": item.matched,
                                    "decision": item.decision,
                                    "reason": item.reason,
                                    "extra_message": item.extra_message,
                                }
                                for item in exec_result.pre_hook_trace
                            ],
                            "post_hook_trace": [
                                {
                                    "hook_name": item.hook_name,
                                    "event": item.event,
                                    "matched": item.matched,
                                    "decision": item.decision,
                                    "reason": item.reason,
                                    "extra_message": item.extra_message,
                                }
                                for item in exec_result.post_hook_trace
                            ],
                            "result": normalized.preview(),
                        }
                    )

                # 7. 本轮工具执行完后，记录 tool_chain 并追加下一轮 loop_state 提示。
                tool_chain_group = {"text": response.content, "calls": iter_calls}
                if response.thinking is not None:
                    tool_chain_group["reasoning_content"] = response.thinking
                tool_chain.append(tool_chain_group)
                # 7a. AfterStep 模块链（工具分支）：通知观察者本轮工具执行完毕。
                _ = await self._after_step.run(AfterStepCtx(
                    session_key=tool_event_session_key,
                    channel=tool_event_channel,
                    chat_id=tool_event_chat_id,
                    iteration=iteration,
                    tools_called=tuple(tc.name for tc in response.tool_calls),
                    partial_reply=response.content or "",
                    tools_used_so_far=tuple(tools_used),
                    tool_chain_partial=tuple(tool_chain),
                    partial_thinking=response.thinking,
                    has_more=True,
                ))
                messages.append(
                    support.build_context_hint_message(
                        "loop_state",
                        build_loop_state_hint(
                            visible_names=visible_names,
                            always_on_names=(
                                self._tools.get_always_on_names()
                                if self._tool_search_enabled
                                else None
                            ),
                        ),
                    )
                )
                continue

            # 8. 没有 tool_calls 时，说明本轮得到最终回复。
            # 8a. 若 content 为空（模型只输出了 thinking），retry 一次。
            if not response.content and response.thinking:
                logger.warning(
                    "[空回复重试] 第%d轮，content为空但thinking非空，触发一次重试",
                    iteration + 1,
                )
                messages.append({"role": "assistant", "content": ""})
                messages.append({
                    "role": "user",
                    "content": "你刚才只输出了思考过程，没有给出正式回复。请直接回复用户，不要重复思考。",
                })
                retry_response = await self._llm.provider.chat(
                    messages=messages,
                    tools=[],
                    model=self._llm_config.model,
                    max_tokens=self._llm_config.max_tokens,
                    on_content_delta=on_content_delta,
                )
                if retry_response.cache_prompt_tokens is not None:
                    react_cache_seen = True
                    react_cache_prompt_tokens += retry_response.cache_prompt_tokens
                    react_cache_hit_tokens += retry_response.cache_hit_tokens or 0
                if retry_response.content:
                    response = retry_response
                    if on_content_delta is not None:
                        streamed = True
                    logger.info("[空回复重试] 重试成功，获得正常回复")
                else:
                    logger.warning("[空回复重试] 重试仍为空，使用fallback")

            logger.info(
                "[LLM决策→回复] 第%d轮，共调用工具%d次: %s",
                iteration + 1,
                len(tools_used),
                tools_used if tools_used else "无",
            )
            messages.append({"role": "assistant", "content": response.content})
            # 8b. AfterStep 模块链（最终回复分支）：通知观察者本轮推理结束。
            _ = await self._after_step.run(AfterStepCtx(
                session_key=tool_event_session_key,
                channel=tool_event_channel,
                chat_id=tool_event_chat_id,
                iteration=iteration,
                tools_called=(),
                partial_reply=response.content or "",
                tools_used_so_far=tuple(tools_used),
                tool_chain_partial=tuple(tool_chain),
                partial_thinking=response.thinking,
                has_more=False,
            ))
            return self._build_result(
                reply=response.content or "（无响应）",
                tools_used=tools_used,
                tool_chain=tool_chain,
                visible_names=visible_names,
                thinking=response.thinking,
                streamed=streamed,
                react_input_samples=react_input_samples,
                cache_prompt_tokens=react_cache_prompt_tokens,
                cache_hit_tokens=react_cache_hit_tokens,
                cache_seen=react_cache_seen,
            )

        # 9. 达到最大迭代次数后，生成不完整进展总结。
        logger.warning(
            "[迭代上限] 达到最大轮次%d，触发收尾总结，已调用工具: %s",
            self._llm_config.max_iterations,
            tools_used if tools_used else "无",
        )
        summary = await self._summarize_incomplete_progress(
            messages,
            reason="max_iterations",
            iteration=self._llm_config.max_iterations,
            tools_used=tools_used,
        )
        return self._build_result(
            reply=summary,
            tools_used=tools_used,
            tool_chain=tool_chain,
            visible_names=visible_names,
            thinking=None,
            streamed=False,
            react_input_samples=react_input_samples,
            cache_prompt_tokens=react_cache_prompt_tokens,
            cache_hit_tokens=react_cache_hit_tokens,
            cache_seen=react_cache_seen,
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
            )
        )

    async def _summarize_incomplete_progress(
        self,
        messages: list[dict],
        *,
        reason: str,
        iteration: int,
        tools_used: list[str],
    ) -> str:
        # 1. 先构造收尾总结 prompt。
        summary_prompt = (
            f"[收尾原因] {reason}\n"
            f"[已执行轮次] {iteration}\n"
            f"[已调用工具] {', '.join(tools_used[-8:]) if tools_used else '无'}\n\n"
            + _INCOMPLETE_SUMMARY_PROMPT
        )

        # 2. 先尝试让模型给一段中文收尾总结。
        try:
            response = await self._llm.provider.chat(
                messages=messages
                + [
                    support.build_context_hint_message(
                        "summary_request",
                        summary_prompt,
                    )
                ],
                tools=[],
                model=self._llm_config.model,
                max_tokens=min(_SUMMARY_MAX_TOKENS, self._llm_config.max_tokens),
            )
            text = (response.content or "").strip()
            if text:
                return text
        except Exception as exc:
            logger.warning("生成预算收尾总结失败: %s", exc)

        # 3. 模型收尾失败时，返回固定兜底文案。
        done = f"已尝试 {iteration} 轮，调用工具 {len(tools_used)} 次。"
        return (
            f"这次任务还没完全收束。{done}"
            "我先停在当前进度，后续会继续补齐缺失信息并给你最终结论。"
        )

    def _build_result(
        self,
        *,
        reply: str,
        tools_used: list[str],
        tool_chain: list[dict],
        visible_names: set[str] | None,
        thinking: str | None,
        streamed: bool,
        react_input_samples: list[int],
        cache_prompt_tokens: int,
        cache_hit_tokens: int,
        cache_seen: bool,
    ) -> ReasonerResult:
        # 1. 先把 tool_chain 扁平化成 invocations。
        invocations: list[LLMToolCall] = []
        for group in tool_chain:
            for call in group.get("calls") or []:
                args = call.get("arguments")
                invocations.append(
                    LLMToolCall(
                        id=str(call.get("call_id", "") or ""),
                        name=str(call.get("name", "") or ""),
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        # 2. 再把运行时元数据统一塞进 metadata。
        react_stats = {
            "iteration_count": len(react_input_samples),
            "turn_input_sum_tokens": sum(react_input_samples),
            "turn_input_peak_tokens": max(react_input_samples, default=0),
            "final_call_input_tokens": react_input_samples[-1] if react_input_samples else 0,
        }
        if cache_seen:
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
        metadata = {
            "tools_used": list(tools_used),
            "tool_chain": list(tool_chain),
            "visible_names": set(visible_names) if visible_names is not None else None,
            "react_stats": react_stats,
        }

        # 3. 最后返回标准 ReasonerResult。
        return ReasonerResult(
            reply=reply,
            invocations=invocations,
            thinking=thinking,
            streamed=streamed,
            metadata=metadata,
        )

    @staticmethod
    def _slice_history(source_history: list[dict], window: int) -> list[dict]:
        total_history = len(source_history)
        if window <= 0:
            return []
        if window >= total_history:
            return source_history
        return source_history[-window:]

    @staticmethod
    def _build_attempt_plans(total_history: int) -> list[dict]:
        attempts: list[dict] = []
        seen: set[tuple[tuple[str, ...], int]] = set()
        full_window = int(total_history * _SAFETY_RETRY_RATIOS[0])
        for trim_plan in DEFAULT_CONTEXT_TRIM_PLANS:
            disabled = set(trim_plan.drop_sections)
            key = (tuple(sorted(disabled)), full_window)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "name": trim_plan.name,
                    "disabled_sections": disabled,
                    "history_window": full_window,
                }
            )

        last_trim = set(DEFAULT_CONTEXT_TRIM_PLANS[-1].drop_sections)
        for ratio in _SAFETY_RETRY_RATIOS[1:]:
            window = int(total_history * ratio)
            key = (tuple(sorted(last_trim)), window)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "name": f"{DEFAULT_CONTEXT_TRIM_PLANS[-1].name}_history",
                    "disabled_sections": set(last_trim),
                    "history_window": window,
                }
            )
        return attempts

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



def get_history_since_consolidated(
    session: "SessionLike",
    memory_window: int,
) -> list[dict]:
    try:
        return session.get_history(
            max_messages=memory_window,
            start_index=session.last_consolidated,
        )
    except TypeError:
        return session.get_history(max_messages=memory_window)


def extract_model_facing_turn(
    messages: list[dict],
) -> tuple[object | None, str | None]:
    if not messages:
        return None, None
    user_content = (
        messages[-1].get("content")
        if messages[-1].get("role") == "user"
        else None
    )
    if len(messages) < 2:
        return user_content, None
    frame = messages[-2]
    frame_content = frame.get("content")
    if isinstance(frame_content, str) and is_context_frame(frame_content):
        return user_content, frame_content
    return user_content, None


def build_turn_injection_prompt(
    *,
    tools: "ToolRegistry",
    tool_search_enabled: bool,
    visible_names: set[str] | None,
) -> str:
    if not tool_search_enabled:
        return ""
    return build_deferred_tools_hint(tools, visible=visible_names)


def build_deferred_tools_hint(
    tools: "ToolRegistry",
    visible: set[str] | None = None,
) -> str:
    get_deferred_names = getattr(tools, "get_deferred_names", None)
    if not callable(get_deferred_names):
        return ""
    deferred_raw = get_deferred_names(visible=visible)
    if not isinstance(deferred_raw, dict):
        return ""
    builtin_raw = deferred_raw.get("builtin", [])
    mcp_raw = deferred_raw.get("mcp", {})
    builtin = [name for name in builtin_raw if isinstance(name, str)]
    mcp = {
        str(server): [name for name in names if isinstance(name, str)]
        for server, names in mcp_raw.items()
        if isinstance(server, str) and isinstance(names, list)
    }

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
        "- 已知工具名 → tool_search(query=\"select:工具名\")，支持逗号分隔多个\n"
        "- 描述功能   → tool_search(query=\"关键词\") 搜索匹配"
    )
    return "\n".join(lines) + "\n\n"


def build_loop_state_hint(
    visible_names: set[str] | None = None,
    always_on_names: set[str] | None = None,
) -> str:
    if visible_names is None or always_on_names is None:
        return "【当前工具状态】tool_search 未开启，本轮按现有工具继续。"

    unlocked_extra = visible_names - always_on_names - {"tool_search"}
    visible_text = ", ".join(sorted(unlocked_extra)) if unlocked_extra else "仅 always-on"
    return (
        f"【当前工具状态】已解锁: {visible_text}\n"
        "未知工具: tool_search(query=\"关键词\") 搜索\n"
        "已知工具名但未加载: tool_search(query=\"select:工具名\")"
    )
