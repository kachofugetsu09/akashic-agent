import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, TypeAlias

from agent.context import ContextBuilder
from agent.core.passive_turn import (
    AgentCore,
    AgentCoreDeps,
    DefaultContextStore,
    DefaultReasoner,
)
from agent.looping.interrupt import InterruptResult, TurnInterruptState
from agent.core.runner import CoreRunner, CoreRunnerDeps
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.consolidation import (
    ConsolidationRuntime,
    _select_consolidation_window,
)
from agent.looping.lifecycle_consumers import (
    register_turn_committed_consumers,
)
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    SessionServices,
    TurnScheduler,
)
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import MemoryRetrievalPipeline
from agent.turns.outbound import BusOutboundPort

# Re-export for backward-compat: existing callers import these from core.py
__all__ = [
    "AgentLoop",
]
from bus.event_bus import EventBus
from bus.events import InboundItem, InboundMessage, OutboundMessage, SpawnCompletionItem
from bus.events_lifecycle import (
    StreamDeltaReady,
    TurnStarted,
)
from bus.processing import ProcessingState
from bus.queue import MessageBus
from proactive_v2.presence import PresenceStore
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine
    from core.memory.runtime import MemoryRuntime
    from agent.tool_hooks.base import ToolHook

logger = logging.getLogger("agent.loop")
_MAX_PROCEDURE_RETRIEVE_K = 3

StreamDelta: TypeAlias = dict[str, str] | str
StreamSink: TypeAlias = Callable[[StreamDelta], Awaitable[None]]
StreamSinkFactory: TypeAlias = Callable[[object], StreamSink | None]
StreamSupportPolicy: TypeAlias = Callable[[str], bool]


def _is_positive_int(value: str) -> bool:
    try:
        return int(value) > 0
    except ValueError:
        return False


_STREAM_SUPPORT_POLICIES: dict[str, StreamSupportPolicy] = {
    "telegram": _is_positive_int,
}


def _supports_stream_events(channel: str, chat_id: str) -> bool:
    policy = _STREAM_SUPPORT_POLICIES.get(channel)
    return bool(policy is not None and policy(chat_id))


def _item_content(item: InboundItem) -> str:
    if isinstance(item, InboundMessage):
        return item.content
    return f"[后台任务完成] {item.event.label or item.event.status or item.event.job_id}"


class AgentLoop:
    """
    主循环：从 MessageBus 消费 InboundMessage，
    驱动 LLM + 工具调用，将结果发回 MessageBus。
    对话历史按 session_key 独立维护，格式为 OpenAI messages。
    """

    _CONSOLIDATION_WAIT_S: float = 30.0

    def __init__(
        self,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先挂基础运行时对象和配置。
        self._llm_config = config.llm
        self.bus = deps.bus
        self.tools = deps.tools
        self.memory_window = config.memory.window
        self._running = False
        self._processing_state = deps.processing_state
        self._event_bus = deps.event_bus or EventBus()

        # ── 中断控制面（纯内存态） ──
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._active_turn_states: dict[str, TurnInterruptState] = {}
        self._interrupt_states: dict[str, TurnInterruptState] = {}

        # 2. 再解析 memory runtime 入口。
        memory_engine = self._resolve_memory_runtime(deps)
        self._tool_search_enabled = bool(config.llm.tool_search_enabled)
        self._memory_engine = memory_engine
        self._context = deps.context or ContextBuilder(
            deps.workspace,
            memory=self._memory_engine,
            multimodal=config.llm.multimodal,
            vl_available=config.llm.vl_available,
        )
        self._llm_services = deps.llm_services or LLMServices(
            provider=deps.provider,
            light_provider=deps.light_provider or deps.provider,
        )
        self._session_services = deps.session_services or SessionServices(
            session_manager=deps.session_manager,
            presence=deps.presence,
        )

        # 3. 最后把 passive chain 装起来。
        self._assemble_passive_runtime(
            deps=deps,
            config=config,
        )
        self._configure_stream_events()

    def set_stream_sink_factory(self, factory: StreamSinkFactory | None) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            _ = setter(self._wrap_stream_sink_factory(factory))

    def _configure_stream_events(self) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            _ = setter(self._build_stream_event_sink)

    def _wrap_stream_sink_factory(
        self,
        factory: StreamSinkFactory | None,
    ) -> StreamSinkFactory | None:
        if factory is None:
            return None

        def _build(msg: object) -> StreamSink | None:
            downstream = factory(msg)
            channel = str(getattr(msg, "channel", ""))
            chat_id = str(getattr(msg, "chat_id", ""))
            session_key = str(getattr(msg, "session_key", f"{channel}:{chat_id}"))
            if downstream is None:
                return None

            async def _push(delta: StreamDelta) -> None:
                if isinstance(delta, str):
                    payload = {"content_delta": delta}
                else:
                    payload = delta
                content_delta = payload.get("content_delta")
                if isinstance(content_delta, str) and content_delta:
                    self._append_partial_reply(session_key, content_delta)
                thinking_delta = payload.get("thinking_delta")
                if isinstance(thinking_delta, str) and thinking_delta:
                    self._append_partial_thinking(session_key, thinking_delta)
                await downstream(payload)

            return _push

        return _build

    def _build_stream_event_sink(self, msg: object) -> StreamSink | None:
        channel = str(getattr(msg, "channel", ""))
        chat_id = str(getattr(msg, "chat_id", ""))
        if not _supports_stream_events(channel, chat_id):
            return None
        session_key = str(getattr(msg, "session_key", f"{channel}:{chat_id}"))

        async def _push(delta: StreamDelta) -> None:
            if isinstance(delta, str):
                payload = {"content_delta": delta}
            else:
                payload = delta
            content_delta = payload.get("content_delta")
            if isinstance(content_delta, str) and content_delta:
                self._append_partial_reply(session_key, content_delta)
            thinking_delta = payload.get("thinking_delta")
            if isinstance(thinking_delta, str) and thinking_delta:
                self._append_partial_thinking(session_key, thinking_delta)
            await self._event_bus.observe(
                StreamDeltaReady(
                    session_key=session_key,
                    channel=channel,
                    chat_id=chat_id,
                    content_delta=content_delta if isinstance(content_delta, str) else "",
                    thinking_delta=thinking_delta if isinstance(thinking_delta, str) else "",
                )
            )

        return _push

    def _append_partial_reply(self, session_key: str, delta: str) -> None:
        state = self._active_turn_states.get(session_key)
        if state is None or not delta:
            return
        state.partial_reply += delta

    def _append_partial_thinking(self, session_key: str, delta: str) -> None:
        state = self._active_turn_states.get(session_key)
        if state is None or not delta:
            return
        state.partial_thinking = (state.partial_thinking or "") + delta

    def _resolve_memory_runtime(
        self,
        deps: AgentLoopDeps,
    ) -> "MemoryEngine":
        if deps.memory_runtime is not None:
            return deps.memory_runtime.engine
        if deps.memory_services is not None and deps.memory_services.engine is not None:
            return deps.memory_services.engine
        raise ValueError("AgentLoop requires memory_runtime.engine")

    def _assemble_passive_runtime(
        self,
        *,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> None:
        # 1. 先组基础 service ports。
        llm_svc = self._llm_services
        memory_svc = deps.memory_services or MemoryServices(
            engine=getattr(deps.memory_runtime, "engine", None),
        )
        session_svc = self._session_services
        # 2. 再准备 retrieval / scheduler 依赖配置。
        handler_memory_config = MemoryConfig(
            window=config.memory.window,
            top_k_procedure=min(
                _MAX_PROCEDURE_RETRIEVE_K, max(1, int(config.memory.top_k_procedure))
            ),
            top_k_history=max(1, int(config.memory.top_k_history)),
            route_intention_enabled=config.memory.route_intention_enabled,
            procedure_guard_enabled=config.memory.procedure_guard_enabled,
            gate_llm_timeout_ms=max(100, int(config.memory.gate_llm_timeout_ms)),
            gate_max_tokens=max(32, int(config.memory.gate_max_tokens)),
            hyde_enabled=config.memory.hyde_enabled,
            hyde_timeout_ms=config.memory.hyde_timeout_ms,
        )

        # 3. 组执行层和 consolidation 相关组件。
        self._tool_discovery = deps.tool_discovery or ToolDiscoveryState()
        self._reasoner = deps.reasoner or DefaultReasoner(
            llm=llm_svc,
            llm_config=config.llm,
            tools=deps.tools,
            discovery=self._tool_discovery,
            tool_search_enabled=self._tool_search_enabled,
            memory_window=config.memory.keep_count,
            context=self._context,
            session_manager=self.session_manager,
            event_bus=self._event_bus,
        )
        self._scheduler = deps.scheduler or TurnScheduler(
            consolidation_runner=self._consolidate_and_save,
            keep_count=config.memory.keep_count,
        )
        self._consolidation_runtime = ConsolidationRuntime(
            session_manager=self.session_manager,
            scheduler=self._scheduler,
            memory=memory_svc.engine,
            keep_count=config.memory.keep_count,
            wait_timeout_s=self._CONSOLIDATION_WAIT_S,
        )

        # 4. 最后串 passive prepare / execute / commit 主链。
        retrieval_pipeline = deps.retrieval_pipeline or DefaultMemoryRetrievalPipeline(
            memory=memory_svc,
            memory_config=handler_memory_config,
            llm=llm_svc,
            workspace=deps.workspace,
            light_model=self.light_model,
        )
        self._retrieval_pipeline = retrieval_pipeline
        register_turn_committed_consumers(
            event_bus=self._event_bus,
            session_manager=self.session_manager,
            scheduler=self._scheduler,
            memory_engine=memory_svc.engine,
        )
        passive_context_store = DefaultContextStore(
            retrieval=retrieval_pipeline,
            context=self._context,
            history_window=config.memory.keep_count,
        )
        agent_core = AgentCore(
            AgentCoreDeps(
                session=session_svc,
                context_store=passive_context_store,
                context=self._context,
                tools=deps.tools,
                reasoner=self._reasoner,
                event_bus=self._event_bus,
                outbound_port=BusOutboundPort(self.bus),
                history_window=config.memory.keep_count,
            )
        )
        self._agent_core = agent_core
        self._core_runner = deps.core_runner or CoreRunner(
            CoreRunnerDeps(
                agent_core=agent_core,
                session=session_svc,
                context=self._context,
                tools=deps.tools,
                memory_window=config.memory.keep_count,
                run_agent_loop_fn=self._run_agent_loop,
                prompt_render_fn=self._reasoner.render_prompt,
            )
        )

    @property
    def light_model(self) -> str:
        # 1. 兼容外部读取 loop.light_model，真实值统一来自 llm 配置。
        return self._llm_config.light_model or self._llm_config.model

    @property
    def context(self) -> ContextBuilder:
        # 1. 兼容外部读取 loop.context，真实值统一来自私有 context 依赖。
        return self._context

    @property
    def light_provider(self):
        # 1. 兼容外部读取 loop.light_provider，真实值统一来自 llm services。
        return self._llm_services.light_provider

    @property
    def session_manager(self):
        # 1. 兼容外部读取 loop.session_manager，真实值统一来自 session services。
        return self._session_services.session_manager

    @light_model.setter
    def light_model(self, value: str) -> None:
        # 1. 兼容初始化期和少量外部覆写，统一回写到 llm 配置。
        self._llm_config.light_model = value

    @property
    def max_iterations(self) -> int:
        # 1. 兼容外部读取 loop.max_iterations，真实值统一来自 llm 配置。
        return int(self._llm_config.max_iterations)

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        # 1. 兼容测试或外部直接改 loop.max_iterations，真实执行也同步生效。
        self._llm_config.max_iterations = int(value)

    async def run(self) -> None:
        self._running = True
        logger.info(f"AgentLoop 启动  max_iter={self.max_iterations}")
        while self._running:
            try:
                item = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            key = item.session_key
            self._active_turn_states[key] = self._build_initial_turn_state(item, key)
            task = asyncio.create_task(self._process(item))
            self._active_tasks[key] = task
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Turn cancelled for {key}")
            except Exception as e:
                logger.error(f"处理消息出错: {e}", exc_info=True)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=item.channel,
                        chat_id=item.chat_id,
                        content=f"出错：{e}",
                    )
                )
            finally:
                self._active_tasks.pop(key, None)
                self._active_turn_states.pop(key, None)

    @property
    def processing_state(self) -> ProcessingState | None:
        return self._processing_state

    @property
    def active_turn_states(self) -> dict[str, TurnInterruptState]:
        return self._active_turn_states

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

    def add_tool_hooks(self, hooks: list["ToolHook"]) -> None:
        self._reasoner.add_tool_hooks(hooks)

    def add_before_turn_plugin_modules(
        self,
        early: list[object],
        late: list[object],
    ) -> None:
        self._agent_core.add_before_turn_plugin_modules(early, late)

    def add_before_reasoning_plugin_modules(
        self,
        before_emit: list[object],
        after_emit: list[object],
    ) -> None:
        self._agent_core.add_before_reasoning_plugin_modules(before_emit, after_emit)

    def add_after_reasoning_plugin_modules(
        self,
        before_emit: list[object],
        before_persist: list[object],
    ) -> None:
        self._agent_core.add_after_reasoning_plugin_modules(
            before_emit,
            before_persist,
        )

    def add_after_turn_plugin_modules(
        self,
        before_commit: list[object],
        before_fanout: list[object],
    ) -> None:
        self._agent_core.add_after_turn_plugin_modules(before_commit, before_fanout)

    def add_prompt_render_plugin_modules(
        self,
        top: list[object],
        bottom: list[object],
    ) -> None:
        self._reasoner.add_prompt_render_plugin_modules(top, bottom)

    def add_before_step_plugin_modules(
        self,
        before_emit: list[object],
        after_emit: list[object],
    ) -> None:
        self._reasoner.add_before_step_plugin_modules(before_emit, after_emit)

    def add_after_step_plugin_modules(
        self,
        before_fanout: list[object],
        after_fanout: list[object],
    ) -> None:
        self._reasoner.add_after_step_plugin_modules(before_fanout, after_fanout)

    # ── 中断控制面 ────────────────────────────────────────────────

    def request_interrupt(
        self,
        session_key: str,
        sender: str = "",
        command: str = "/stop",
    ) -> InterruptResult:
        """Channel 层调用的中断入口，不走 MessageBus。"""
        task = self._active_tasks.get(session_key)
        if task is None or task.done():
            return InterruptResult(
                status="idle",
                session_key=session_key,
                message="当前没有正在执行的任务。",
            )

        # 保存中断态（纯内存，不落库）
        active_state = self._active_turn_states.get(session_key)
        if active_state is None:
            active_state = TurnInterruptState(
                session_key=session_key,
                original_user_message="",
            )
        self._interrupt_states[session_key] = replace(
            active_state,
            interrupted_by=command,
            interrupted_at=time.monotonic(),
        )
        task.cancel()
        logger.info(
            f"Turn interrupted  session_key={session_key}  "
            f"sender={sender}  command={command}"
        )
        return InterruptResult(
            status="interrupted",
            session_key=session_key,
            message="本轮已中断。你可以继续补充要求，我会接着这件事处理。",
        )

    def _get_interrupt_state(self, session_key: str) -> TurnInterruptState | None:
        """读取中断态（含 TTL 过期检查），不提前消费。"""
        state = self._interrupt_states.get(session_key)
        if state is None:
            return None
        if state.expired:
            logger.info(f"Interrupt state expired for {session_key}, discarding")
            self._interrupt_states.pop(session_key, None)
            return None
        return state

    def _build_initial_turn_state(
        self,
        item: InboundItem,
        key: str,
    ) -> TurnInterruptState:
        # 1. 普通消息保留真实用户输入，spawn 回传用固定 marker 表示内部工作项。
        match item:
            case InboundMessage():
                return TurnInterruptState(
                    session_key=key,
                    original_user_message=item.content,
                    original_metadata=dict(item.metadata or {}),
                )
            case SpawnCompletionItem():
                return TurnInterruptState(
                    session_key=key,
                    original_user_message=_item_content(item),
                    original_metadata={},
                )
        raise TypeError(f"unsupported inbound item: {type(item).__name__}")

    def _resume_interrupted_message(
        self,
        msg: InboundItem,
        key: str,
    ) -> tuple[InboundItem, bool]:
        # 1. 只有普通入站消息参与续跑，内部工作项不消费中断态。
        if not isinstance(msg, InboundMessage):
            return msg, False
        interrupted = self._get_interrupt_state(key)
        if interrupted is None:
            return msg, False

        # 2. 有中断态时，把上一轮进度和本轮补充拼成新的用户消息。
        resumed = InboundMessage(
            channel=msg.channel,
            sender=msg.sender,
            chat_id=msg.chat_id,
            content=_build_resume_content(interrupted, msg.content),
            timestamp=msg.timestamp,
            media=msg.media,
            metadata={**(msg.metadata or {}), "resumed_from_interrupt": True},
        )
        logger.info(f"Resuming interrupted turn for {key}")
        self._active_turn_states[key] = TurnInterruptState(
            session_key=key,
            original_user_message=resumed.content,
            original_metadata=dict(resumed.metadata or {}),
        )
        return resumed, True

    async def _observe_turn_started(
        self,
        msg: InboundItem,
        key: str,
    ) -> None:
        # 1. 对外发布被动 turn 开始事件，具体副作用由 observer 决定。
        await self._event_bus.observe(
            TurnStarted(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=_item_content(msg),
                timestamp=msg.timestamp,
            )
        )

    # ── 被动 turn 处理 ────────────────────────────────────────────

    async def _process(
        self,
        msg: InboundItem,
        session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        started = time.time()
        key = session_key or msg.session_key

        # 1. 先处理可能存在的续跑态，并发布 turn started。
        msg, resumed_from_interrupt = self._resume_interrupted_message(msg, key)
        await self._observe_turn_started(msg, key)
        content = _item_content(msg)
        preview = content[:60] + "..." if len(content) > 60 else content
        logger.info(f"Processing message from {msg.channel}: {preview}")

        # 2. 再进入 busy 状态并执行核心处理。
        if self._processing_state:
            self._processing_state.enter(key)
        try:
            outbound = await self._core_runner.process(
                msg,
                key,
                dispatch_outbound=dispatch_outbound,
            )
            if resumed_from_interrupt:
                self._interrupt_states.pop(key, None)
            return outbound
        finally:
            # 3. 最后无论成功失败都直接释放 busy 状态。
            if self._processing_state:
                self._processing_state.exit(key)
            _ = started

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        omit_user_turn: bool = False,
    ) -> str:
        msg = InboundMessage(
            channel=channel,
            sender="user",
            chat_id=chat_id,
            content=content,
            metadata={"omit_user_turn": True} if omit_user_turn else {},
        )
        response = await self._process(
            msg,
            session_key=session_key,
            dispatch_outbound=False,
        )
        return response.content if response else ""

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
    ) -> tuple[str, list[str], list[dict], set[str] | None, str | None]:
        from agent.core.passive_turn import build_turn_injection_prompt
        from agent.prompting import (
            PromptSectionRender,
            build_context_frame_content,
            build_context_frame_message,
        )

        # 1. 补充 deferred tools hint（与 run_turn 路径保持一致）。
        visible = preloaded_tools if self._tool_search_enabled else None
        hint = build_turn_injection_prompt(
            tools=self.tools,
            tool_search_enabled=self._tool_search_enabled,
            visible_names=visible,
        )
        if hint:
            hint_message = build_context_frame_message(
                build_context_frame_content(
                    [
                        PromptSectionRender(
                            name="turn_injection",
                            content=hint,
                            is_static=False,
                        )
                    ]
                )
            )
            if initial_messages and initial_messages[-1].get("role") == "user":
                initial_messages = initial_messages[:-1] + [
                    hint_message,
                    initial_messages[-1],
                ]
            else:
                initial_messages = initial_messages + [hint_message]

        # 2. 内部事件链统一直接走新 Reasoner。
        result = await self._reasoner.run(
            initial_messages,
            request_time=request_time,
            preloaded_tools=preloaded_tools,
            preflight_injected=True,
        )
        tools_used = list(result.metadata.get("tools_used") or [])
        tool_chain = list(result.metadata.get("tool_chain") or [])
        visible_names = result.metadata.get("visible_names")
        return result.reply, tools_used, tool_chain, visible_names, result.thinking

    async def _consolidate_memory(
        self,
        session,
        archive_all: bool = False,
        force: bool = False,
    ) -> None:
        await self._consolidation_runtime.consolidate_memory(
            session,
            archive_all=archive_all,
            force=force,
        )

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
        force: bool = False,
    ) -> bool:
        return await self._consolidation_runtime.trigger_memory_consolidation(
            session_key,
            archive_all=archive_all,
            force=force,
            consolidate_fn=self._consolidate_memory,
        )

    async def _wait_for_consolidation_idle(self, session_key: str) -> None:
        await self._consolidation_runtime.wait_for_consolidation_idle(session_key)

    async def _consolidate_and_save(self, session: object) -> None:
        await self._consolidation_runtime.consolidate_and_save(session)


# ── 模块级辅助 ────────────────────────────────────────────────────


def _build_resume_content(state: TurnInterruptState, new_message: str) -> str:
    """将中断态 + 用户补充消息拼装为续跑输入。"""
    parts = [
        "【上一轮任务（被用户中断）】",
        state.original_user_message,
        "",
        "【上一轮已生成但未完成的中间结果】",
        state.partial_reply or "（无）",
    ]
    if state.tools_used:
        parts.append(f"已使用工具：{', '.join(state.tools_used)}")
    parts += [
        "",
        "【用户补充要求】",
        new_message,
    ]
    return "\n".join(parts)
