import asyncio
import logging
import time
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from agent.context import ContextBuilder
from agent.core.agent_core import AgentCore, AgentCoreDeps
from agent.looping.interrupt import InterruptResult, TurnInterruptState
from agent.core.context_store import DefaultContextStore
from agent.core.reasoner import DefaultReasoner
from agent.core.runner import CoreRunner, CoreRunnerDeps
from agent.core.runtime_support import ToolDiscoveryState
from agent.looping.consolidation import (
    ConsolidationRuntime,
    ConsolidationService,
    _select_consolidation_window,
)
from agent.looping.ports import (
    AgentLoopConfig,
    AgentLoopDeps,
    LLMConfig,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    ObservabilityServices,
    SessionServices,
    TurnScheduler,
)
from agent.postturn.default_pipeline import DefaultPostTurnPipeline
from agent.postturn.protocol import PostTurnPipeline
from agent.retrieval.default_pipeline import DefaultMemoryRetrievalPipeline
from agent.retrieval.protocol import MemoryRetrievalPipeline
from agent.turns.outbound import BusOutboundPort

# Re-export for backward-compat: existing callers import these from core.py
__all__ = [
    "AgentLoop",
]
from agent.memes.catalog import MemeCatalog
from agent.memes.decorator import MemeDecorator
from bus.events import InboundMessage, OutboundMessage
from bus.processing import ProcessingState
from bus.queue import MessageBus
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.profile_extractor import ProfileFactExtractor
from memory2.query_rewriter import QueryRewriter
from memory2.sufficiency_checker import SufficiencyChecker
from proactive_v2.presence import PresenceStore
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.port import MemoryPort
    from core.memory.runtime import MemoryRuntime
    from memory2.hyde_enhancer import HyDEEnhancer

logger = logging.getLogger("agent.loop")
_MAX_PROCEDURE_RETRIEVE_K = 3

class AgentLoop:
    """
    主循环：从 MessageBus 消费 InboundMessage，
    驱动 LLM + 工具调用，将结果发回 MessageBus。
    对话历史按 session_key 独立维护，格式为 OpenAI messages。
    """

    _MESSAGE_TIMEOUT_S: float = 600.0
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

        # ── 中断控制面（纯内存态） ──
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._active_turn_states: dict[str, TurnInterruptState] = {}
        self._interrupt_states: dict[str, TurnInterruptState] = {}

        # 2. 再解析 memory runtime 入口。
        memory_port, post_mem_worker = self._resolve_memory_runtime(deps)
        self._tool_search_enabled = bool(config.llm.tool_search_enabled)
        self._memory_port = memory_port
        self._context = deps.context or ContextBuilder(
            deps.workspace,
            memory=(
                getattr(deps.memory_runtime, "profile_reader", None) or self._memory_port
            ),
        )
        self._post_mem_worker = post_mem_worker
        self._llm_services = deps.llm_services or LLMServices(
            provider=deps.provider,
            light_provider=deps.light_provider or deps.provider,
        )
        self._session_services = deps.session_services or SessionServices(
            session_manager=deps.session_manager,
            presence=deps.presence,
        )

        # 3. 然后组装 retrieval 需要的轻量能力。
        hyde_enhancer = self._build_hyde_enhancer(deps, config)
        self._hyde_enhancer = hyde_enhancer

        # 4. 最后把 passive chain 装起来。
        self._assemble_passive_runtime(
            deps=deps,
            config=config,
            memory_port=memory_port,
            post_mem_worker=post_mem_worker,
            hyde_enhancer=hyde_enhancer,
        )
        self._configure_interrupt_progress_tracking()

    def set_stream_sink_factory(self, factory) -> None:
        setter = getattr(self._reasoner, "set_stream_sink_factory", None)
        if callable(setter):
            setter(self._wrap_stream_sink_factory(factory))

    def _configure_interrupt_progress_tracking(self) -> None:
        progress_setter = getattr(self._reasoner, "set_progress_sink_factory", None)
        if callable(progress_setter):
            progress_setter(self._build_progress_sink)

    def _wrap_stream_sink_factory(self, factory):
        if factory is None:
            return None

        def _build(msg):
            downstream = factory(msg)
            session_key = getattr(msg, "session_key", f"{msg.channel}:{msg.chat_id}")
            if downstream is None:
                return None

            async def _push(delta: dict[str, str] | str) -> None:
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

    def _build_progress_sink(self, msg):
        session_key = getattr(msg, "session_key", f"{msg.channel}:{msg.chat_id}")

        async def _progress(payload: dict[str, object]) -> None:
            state = self._active_turn_states.get(session_key)
            if state is None:
                return
            partial_reply = payload.get("partial_reply")
            if isinstance(partial_reply, str) and partial_reply:
                state.partial_reply = partial_reply
            partial_thinking = payload.get("partial_thinking")
            if isinstance(partial_thinking, str) and partial_thinking:
                state.partial_thinking = partial_thinking
            tools_used = payload.get("tools_used")
            if isinstance(tools_used, list):
                state.tools_used = list(tools_used)
            tool_chain_partial = payload.get("tool_chain_partial")
            if isinstance(tool_chain_partial, list):
                state.tool_chain_partial = list(tool_chain_partial)

        return _progress

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
    ) -> tuple["MemoryPort", "PostResponseMemoryWorker | None"]:
        # 1. 先取显式注入的 memory 对象。
        memory_port = deps.memory_port
        post_mem_worker = deps.post_mem_worker

        # 2. 如果给了 memory_runtime，就优先使用 runtime 里的实现。
        if deps.memory_runtime is not None:
            memory_port = deps.memory_runtime.port
            post_mem_worker = deps.memory_runtime.post_response_worker

        # 3. 当前主链必须拿到 memory_port。
        if memory_port is None:
            raise ValueError("AgentLoop requires memory_port or memory_runtime")
        return memory_port, post_mem_worker

    def _build_hyde_enhancer(
        self,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
    ) -> "HyDEEnhancer | None":
        # 1. 先尊重外部显式注入。
        hyde_enhancer = deps.hyde_enhancer

        # 2. 没开 HyDE 或已经注入时，直接返回。
        if hyde_enhancer is not None or not config.memory.hyde_enabled:
            return hyde_enhancer

        # 3. 没配独立 light_model 时，不启用 HyDE。
        if not config.llm.light_model:
            logger.warning(
                "hyde_enabled=True 但未配置独立 light_model，"
                "为避免主模型被额外调用，HyDE 已自动禁用。"
                "请在配置中设置 light_model 后重启。"
            )
            return None

        # 4. 需要时再构造默认 HyDE 能力。
        from memory2.hyde_enhancer import HyDEEnhancer

        return HyDEEnhancer(
            light_provider=self.light_provider,
            light_model=self.light_model,
            timeout_s=config.memory.hyde_timeout_ms / 1000.0,
        )

    def _assemble_passive_runtime(
        self,
        *,
        deps: AgentLoopDeps,
        config: AgentLoopConfig,
        memory_port: "MemoryPort",
        post_mem_worker: "PostResponseMemoryWorker | None",
        hyde_enhancer: "HyDEEnhancer | None",
    ) -> None:
        # 1. 先组基础 service ports。
        llm_svc = self._llm_services
        memory_svc = deps.memory_services or MemoryServices(
            engine=getattr(deps.memory_runtime, "engine", None),
            facade=getattr(deps.memory_runtime, "facade", None),
            query_rewriter=deps.query_rewriter,
            hyde_enhancer=hyde_enhancer,
            sufficiency_checker=deps.sufficiency_checker,
        )
        session_svc = self._session_services
        trace_svc = deps.observability_services or ObservabilityServices(
            workspace=deps.workspace,
            observe_writer=deps.observe_writer,
        )

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
        )
        consolidation_service = deps.consolidation_service or ConsolidationService(
            memory_port=self._memory_port,
            profile_maint=(
                getattr(deps.memory_runtime, "profile_maint", None) or self._memory_port
            ),
            provider=deps.provider,
            model=config.llm.model,
            keep_count=config.memory.keep_count,
            profile_extractor=deps.profile_extractor,
            recent_context_provider=deps.light_provider or deps.provider,
            recent_context_model=config.llm.light_model or config.llm.model,
        )
        if memory_svc.facade is not None:
            memory_svc.facade.bind_consolidation_runner(
                lambda session, archive_all: consolidation_service.consolidate(
                    session,
                    archive_all=archive_all,
                )
            )
        self._scheduler = deps.scheduler or TurnScheduler(
            post_mem_worker=post_mem_worker,
            consolidation_runner=self._consolidate_and_save,
            keep_count=config.memory.keep_count,
        )
        self._consolidation_runtime = ConsolidationRuntime(
            session_manager=self.session_manager,
            scheduler=self._scheduler,
            consolidation=consolidation_service,
            facade=memory_svc.facade,
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
        post_turn_pipeline = deps.post_turn_pipeline or DefaultPostTurnPipeline(
            scheduler=self._scheduler,
            engine=memory_svc.engine,
            recent_context_refresher=lambda event: consolidation_service.refresh_recent_turns(
                session=event.session,
            ),
        )
        passive_meme_decorator = MemeDecorator(MemeCatalog(deps.workspace / "memes"))
        passive_context_store = DefaultContextStore(
            retrieval=retrieval_pipeline,
            context=self._context,
            history_window=config.memory.keep_count,
            session=session_svc,
            trace=trace_svc,
            post_turn=post_turn_pipeline,
            outbound=BusOutboundPort(self.bus),
            meme_decorator=passive_meme_decorator,
        )
        agent_core = AgentCore(
            AgentCoreDeps(
                session=session_svc,
                context_store=passive_context_store,
                context=self._context,
                tools=deps.tools,
                reasoner=self._reasoner,
            )
        )
        self._core_runner = deps.core_runner or CoreRunner(
            CoreRunnerDeps(
                agent_core=agent_core,
                session=session_svc,
                context=self._context,
                context_store=passive_context_store,
                tools=deps.tools,
                memory_window=config.memory.keep_count,
                run_agent_loop_fn=self._run_agent_loop,
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
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            key = msg.session_key
            self._active_turn_states[key] = TurnInterruptState(
                session_key=key,
                original_user_message=msg.content,
                original_metadata=dict(msg.metadata or {}),
            )
            task = asyncio.create_task(self._process(msg))
            self._active_tasks[key] = task
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Turn cancelled for {key}")
            except Exception as e:
                logger.error(f"处理消息出错: {e}", exc_info=True)
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"出错：{e}",
                    )
                )
            finally:
                self._active_tasks.pop(key, None)
                self._active_turn_states.pop(key, None)

    @property
    def processing_state(self) -> ProcessingState | None:
        return self._processing_state

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

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

    # ── 被动 turn 处理 ────────────────────────────────────────────

    async def _process(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        started = time.time()
        key = session_key or msg.session_key

        # ── 续跑拼装：消费上一轮中断态 ──
        interrupted = self._get_interrupt_state(key)
        resumed_from_interrupt = interrupted is not None
        if interrupted is not None:
            msg = InboundMessage(
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
                original_user_message=msg.content,
                original_metadata=dict(msg.metadata or {}),
            )

        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        if self._processing_state:
            self._processing_state.enter(key)
        try:
            outbound = await asyncio.wait_for(
                self._core_runner.process(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                ),
                timeout=self._MESSAGE_TIMEOUT_S,
            )
            if resumed_from_interrupt:
                self._interrupt_states.pop(key, None)
            return outbound
        except asyncio.TimeoutError:
            logger.error(
                f"消息处理超时 ({self._MESSAGE_TIMEOUT_S}s)  "
                f"channel={msg.channel} chat_id={msg.chat_id}"
            )
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="（处理超时，请重试）",
            )
        finally:
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
            metadata={"omit_user_turn": True} if omit_user_turn else None,
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
        from agent.core.reasoner import build_turn_injection_prompt

        # 1. 补充 deferred tools hint（与 run_turn 路径保持一致）。
        visible = preloaded_tools if self._tool_search_enabled else None
        hint = build_turn_injection_prompt(
            tools=self.tools,
            tool_search_enabled=self._tool_search_enabled,
            visible_names=visible,
        )
        if hint:
            initial_messages = initial_messages + [{"role": "user", "content": hint}]

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
    ) -> None:
        await self._consolidation_runtime.consolidate_memory(
            session,
            archive_all=archive_all,
        )

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
    ) -> bool:
        return await self._consolidation_runtime.trigger_memory_consolidation(
            session_key,
            archive_all=archive_all,
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
