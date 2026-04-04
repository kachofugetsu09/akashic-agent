import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

from agent.context import ContextBuilder
from agent.core.agent_core import AgentCore, AgentCoreDeps
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
        # 1. 先保留 llm 配置对象，后面的兼容属性直接从这里读。
        self._llm_config = config.llm
        self.bus = deps.bus
        self.provider = deps.provider
        self.tools = deps.tools
        self.session_manager = deps.session_manager
        self.workspace = deps.workspace
        self.light_model = self._llm_config.light_model or self._llm_config.model
        self.light_provider = deps.light_provider or deps.provider
        self.max_iterations = self._llm_config.max_iterations
        self.memory_window = config.memory.window
        self._presence = deps.presence
        self._running = False
        self._processing_state = deps.processing_state

        memory_port = deps.memory_port
        post_mem_worker = deps.post_mem_worker
        if deps.memory_runtime is not None:
            memory_port = deps.memory_runtime.port
            post_mem_worker = deps.memory_runtime.post_response_worker
        if memory_port is None:
            raise ValueError("AgentLoop requires memory_port or memory_runtime")
        self._post_mem_worker = post_mem_worker

        self._tool_search_enabled = bool(config.llm.tool_search_enabled)

        self._memory_port = memory_port
        self.context = deps.context or ContextBuilder(self.workspace, memory=self._memory_port)
        self._profile_extractor = deps.profile_extractor

        # ── Build HyDE enhancer ────────────────────────────────────────────────
        hyde_enhancer = deps.hyde_enhancer
        if hyde_enhancer is None and config.memory.hyde_enabled:
            if not config.llm.light_model:
                logger.warning(
                    "hyde_enabled=True 但未配置独立 light_model，"
                    "为避免主模型被额外调用，HyDE 已自动禁用。"
                    "请在配置中设置 light_model 后重启。"
                )
            else:
                from memory2.hyde_enhancer import HyDEEnhancer

                hyde_enhancer = HyDEEnhancer(
                    light_provider=self.light_provider,
                    light_model=self.light_model,
                    timeout_s=config.memory.hyde_timeout_ms / 1000.0,
                )
        self._hyde_enhancer = hyde_enhancer

        # ── Assemble ports ─────────────────────────────────────────────────────
        llm_svc = deps.llm_services or LLMServices(
            provider=deps.provider,
            light_provider=self.light_provider,
        )
        memory_svc = deps.memory_services or MemoryServices(
            port=memory_port,
            query_rewriter=deps.query_rewriter,
            hyde_enhancer=hyde_enhancer,
            sufficiency_checker=deps.sufficiency_checker,
        )
        session_svc = deps.session_services or SessionServices(
            session_manager=deps.session_manager,
            presence=deps.presence,
        )
        trace_svc = deps.observability_services or ObservabilityServices(
            workspace=deps.workspace,
            observe_writer=deps.observe_writer,
        )
        self._tool_discovery = deps.tool_discovery or ToolDiscoveryState()
        self._reasoner = DefaultReasoner(
            llm=llm_svc,
            llm_config=config.llm,
            tools=deps.tools,
            discovery=self._tool_discovery,
            memory_port=self._memory_port,
            tool_search_enabled=self._tool_search_enabled,
            memory_window=self.memory_window,
            context=self.context,
            session_manager=self.session_manager,
        )
        if deps.reasoner is not None:
            self._reasoner = deps.reasoner
        self._consolidation = deps.consolidation_service or ConsolidationService(
            memory_port=self._memory_port,
            provider=self.provider,
            model=config.llm.model,
            memory_window=self.memory_window,
            profile_extractor=self._profile_extractor,
        )

        # Resolved MemoryConfig with clamped values for the handler
        handler_memory_config = MemoryConfig(
            window=config.memory.window,
            top_k_procedure=min(
                _MAX_PROCEDURE_RETRIEVE_K, max(1, int(config.memory.top_k_procedure))
            ),
            top_k_history=max(1, int(config.memory.top_k_history)),
            route_intention_enabled=config.memory.route_intention_enabled,
            sop_guard_enabled=config.memory.sop_guard_enabled,
            gate_llm_timeout_ms=max(100, int(config.memory.gate_llm_timeout_ms)),
            gate_max_tokens=max(32, int(config.memory.gate_max_tokens)),
            hyde_enabled=config.memory.hyde_enabled,
            hyde_timeout_ms=config.memory.hyde_timeout_ms,
        )

        self._scheduler = deps.scheduler or TurnScheduler(
            post_mem_worker=post_mem_worker,
            consolidation_runner=self._consolidate_and_save,
            memory_window=config.memory.window,
        )
        self._consolidation_runtime = ConsolidationRuntime(
            session_manager=self.session_manager,
            scheduler=self._scheduler,
            consolidation=self._consolidation,
            memory_window=self.memory_window,
            wait_timeout_s=self._CONSOLIDATION_WAIT_S,
        )

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
            post_mem_worker=post_mem_worker,
        )
        passive_meme_decorator = MemeDecorator(MemeCatalog(self.workspace / "memes"))
        passive_context_store = DefaultContextStore(
            retrieval=retrieval_pipeline,
            context=self.context,
            session=session_svc,
            trace=trace_svc,
            post_turn=post_turn_pipeline,
            outbound=BusOutboundPort(self.bus),
            meme_decorator=passive_meme_decorator,
        )
        self._agent_core = AgentCore(
            AgentCoreDeps(
                session=session_svc,
                context_store=passive_context_store,
                context=self.context,
                tools=deps.tools,
                reasoner=self._reasoner,
            )
        )
        self._core_runner = deps.core_runner or CoreRunner(
            CoreRunnerDeps(
                agent_core=self._agent_core,
                session=session_svc,
                context=self.context,
                context_store=passive_context_store,
                tools=deps.tools,
                memory_window=config.memory.window,
                run_agent_loop_fn=self._run_agent_loop,
            )
        )

    @property
    def light_model(self) -> str:
        # 1. 兼容外部读取 loop.light_model，真实值统一来自 llm 配置。
        return self._llm_config.light_model or self._llm_config.model

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
                try:
                    await self._process(msg)
                except Exception as e:
                    logger.error(f"处理消息出错: {e}", exc_info=True)
                    await self.bus.publish_outbound(
                        OutboundMessage(
                            channel=msg.channel,
                            chat_id=msg.chat_id,
                            content=f"出错：{e}",
                        )
                    )
            except asyncio.TimeoutError:
                continue

    @property
    def processing_state(self) -> ProcessingState | None:
        return self._processing_state

    def stop(self) -> None:
        self._running = False
        logger.info("AgentLoop 停止")

    async def _process(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        dispatch_outbound: bool = True,
    ) -> OutboundMessage:
        started = time.time()
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        key = session_key or msg.session_key
        if self._processing_state:
            self._processing_state.enter(key)
        try:
            return await asyncio.wait_for(
                self._core_runner.process(
                    msg,
                    key,
                    dispatch_outbound=dispatch_outbound,
                ),
                timeout=self._MESSAGE_TIMEOUT_S,
            )
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
    ) -> str:
        msg = InboundMessage(
            channel=channel,
            sender="user",
            chat_id=chat_id,
            content=content,
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
        # 1. 内部事件链统一直接走新 Reasoner。
        result = await self._reasoner.run(
            initial_messages,
            request_time=request_time,
            preloaded_tools=preloaded_tools,
        )
        tools_used = list(result.metadata.get("tools_used") or [])
        tool_chain = list(result.metadata.get("tool_chain") or [])
        visible_names = result.metadata.get("visible_names")
        return result.reply, tools_used, tool_chain, visible_names, result.thinking

    async def _consolidate_memory(
        self,
        session,
        archive_all: bool = False,
        await_vector_store: bool = False,
    ) -> None:
        await self._consolidation_runtime.consolidate_memory(
            session,
            archive_all=archive_all,
            await_vector_store=await_vector_store,
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
