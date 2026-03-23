import asyncio
import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.context import ContextBuilder
from agent.looping.consolidation import (
    AgentLoopConsolidationMixin,
    _select_consolidation_window,
)
from agent.looping.handlers import ConversationTurnHandler, InternalEventHandler
from agent.looping.ports import (
    LLMConfig,
    LLMServices,
    MemoryConfig,
    MemoryServices,
    ObservabilityServices,
    SessionServices,
    TurnScheduler,
)

# Re-export for backward-compat: existing callers import these from core.py
__all__ = [
    "AgentLoop",
    "AgentLoopConfig",
    "AgentLoopDeps",
    "LLMConfig",
    "MemoryConfig",
]
from agent.looping.safety_retry import AgentLoopSafetyRetryMixin
from agent.looping.tool_execution import AgentLoopToolExecutionMixin
from bus.events import InboundMessage, OutboundMessage
from bus.internal_events import is_spawn_completion_message
from bus.processing import ProcessingState
from bus.queue import MessageBus
from memory2.post_response_worker import PostResponseMemoryWorker
from memory2.profile_extractor import ProfileFactExtractor
from memory2.query_rewriter import QueryRewriter
from memory2.sufficiency_checker import SufficiencyChecker
from proactive.presence import PresenceStore
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.port import MemoryPort
    from core.memory.runtime import MemoryRuntime
    from memory2.hyde_enhancer import HyDEEnhancer

logger = logging.getLogger("agent.loop")
_MAX_PROCEDURE_RETRIEVE_K = 3


@dataclass
class AgentLoopDeps:
    bus: MessageBus
    provider: LLMProvider
    tools: ToolRegistry
    session_manager: SessionManager
    workspace: Path
    presence: PresenceStore | None = None
    light_provider: LLMProvider | None = None
    processing_state: ProcessingState | None = None
    memory_runtime: "MemoryRuntime | None" = None
    memory_port: "MemoryPort | None" = None
    post_mem_worker: PostResponseMemoryWorker | None = None
    observe_writer: object | None = None
    query_rewriter: QueryRewriter | None = None
    sufficiency_checker: SufficiencyChecker | None = None
    profile_extractor: ProfileFactExtractor | None = None


@dataclass
class AgentLoopConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


class AgentLoop(
    AgentLoopSafetyRetryMixin,
    AgentLoopToolExecutionMixin,
    AgentLoopConsolidationMixin,
):
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
        self.bus = deps.bus
        self.provider = deps.provider
        self.tools = deps.tools
        self.session_manager = deps.session_manager
        self.workspace = deps.workspace
        self.model = config.llm.model
        self.light_model = config.llm.light_model or config.llm.model
        self.light_provider = deps.light_provider or deps.provider
        self.max_iterations = config.llm.max_iterations
        self.max_tokens = config.llm.max_tokens
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

        self._tool_search_enabled = bool(config.llm.tool_search_enabled)

        # Processes-internal LRU: session_key → recently-used non-core tools (cap 5)
        # Used by AgentLoopSafetyRetryMixin; cleared on restart.
        self._unlocked_tools: dict[str, OrderedDict[str, None]] = {}
        self._memory_port = memory_port
        self.context = ContextBuilder(self.workspace, memory=self._memory_port)
        self._profile_extractor = deps.profile_extractor

        # ── Build HyDE enhancer ────────────────────────────────────────────────
        hyde_enhancer: HyDEEnhancer | None = None
        if config.memory.hyde_enabled:
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

        # ── Assemble ports ─────────────────────────────────────────────────────
        llm_svc = LLMServices(
            provider=deps.provider,
            light_provider=self.light_provider,
            run_turn_fn=self._run_with_safety_retry,
        )
        memory_svc = MemoryServices(
            port=memory_port,
            query_rewriter=deps.query_rewriter,
            hyde_enhancer=hyde_enhancer,
            sufficiency_checker=deps.sufficiency_checker,
        )
        session_svc = SessionServices(
            session_manager=deps.session_manager,
            presence=deps.presence,
        )
        trace_svc = ObservabilityServices(
            workspace=deps.workspace,
            observe_writer=deps.observe_writer,
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

        self._scheduler = TurnScheduler(
            post_mem_worker=post_mem_worker,
            consolidation_runner=self._consolidate_and_save,
            memory_window=config.memory.window,
        )

        self._conversation_handler = ConversationTurnHandler(
            llm=llm_svc,
            llm_config=config.llm,
            memory=memory_svc,
            memory_config=handler_memory_config,
            session=session_svc,
            scheduler=self._scheduler,
            trace=trace_svc,
            tools=deps.tools,
            context=self.context,
        )
        self._internal_event_handler = InternalEventHandler(
            session_svc=session_svc,
            context=self.context,
            tools=deps.tools,
            memory_window=config.memory.window,
            run_agent_loop_fn=self._run_agent_loop,
        )

    async def run(self) -> None:
        self._running = True
        logger.info(
            f"AgentLoop 启动  model={self.model}  max_iter={self.max_iterations}"
        )
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                try:
                    response = await self._process(msg)
                    await self.bus.publish_outbound(response)
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

    def _set_tool_context(self, channel: str, chat_id: str) -> None:
        self.tools.set_context(channel=channel, chat_id=chat_id)

    def _collect_skill_mentions(self, user_message: str) -> list[str]:
        raw_names = re.findall(r"\$([a-zA-Z0-9_-]+)", user_message)
        if not raw_names:
            return []
        available = {
            s["name"] for s in self.context.skills.list_skills(filter_unavailable=False)
        }
        seen: set[str] = set()
        result: list[str] = []
        for name in raw_names:
            if name in available and name not in seen:
                seen.add(name)
                result.append(name)
        return result

    async def _process(
        self, msg: InboundMessage, session_key: str | None = None
    ) -> OutboundMessage:
        started = time.time()
        preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender}: {preview}")

        key = session_key or msg.session_key
        if self._processing_state:
            self._processing_state.enter(key)
        try:
            return await asyncio.wait_for(
                self._process_inner(msg, key),
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

    async def _process_inner(self, msg: InboundMessage, key: str) -> OutboundMessage:
        if self._is_spawn_completion(msg):
            return await self._internal_event_handler.process_spawn_completion(msg, key)
        return await self._conversation_handler.process(msg, key)

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
        response = await self._process(msg, session_key=session_key)
        return response.content if response else ""

    async def trigger_memory_consolidation(
        self,
        session_key: str,
        *,
        archive_all: bool = False,
    ) -> bool:
        # 1. 先读取真实 session，并判断当前是否真的需要 consolidation。
        session = self.session_manager.get_or_create(session_key)
        window = _select_consolidation_window(
            session,
            memory_window=self.memory_window,
            archive_all=archive_all,
        )
        if window is None:
            return False
        if self._scheduler.is_consolidating(session_key):
            # 2. 若后台已在跑，同步等待那次 consolidation 完成，避免返回语义含糊的 False。
            await self._wait_for_consolidation_idle(session_key)
            session = self.session_manager.get_or_create(session_key)
            window = _select_consolidation_window(
                session,
                memory_window=self.memory_window,
                archive_all=archive_all,
            )
            if window is None:
                return True
        # 2. 再复用现有真实 consolidation 逻辑执行一次，避免测试绕过主实现。
        if not self._scheduler.mark_manual_start(session_key):
            return False
        try:
            await self._consolidate_memory(
                session,
                archive_all=archive_all,
                await_vector_store=True,
            )
            await self.session_manager.save_async(session)
            return True
        finally:
            self._scheduler.mark_manual_end(session_key)

    async def _wait_for_consolidation_idle(self, session_key: str) -> None:
        # 后台 consolidation 是异步任务，这里短轮询等待它退出运行态。
        deadline = time.perf_counter() + self._CONSOLIDATION_WAIT_S
        while self._scheduler.is_consolidating(session_key):
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"等待 consolidation 完成超时: session_key={session_key}"
                )
            await asyncio.sleep(0.05)

    @staticmethod
    def _is_spawn_completion(msg: InboundMessage) -> bool:
        return is_spawn_completion_message(msg)

    async def _consolidate_and_save(self, session: object) -> None:
        """Consolidation runner passed to TurnScheduler.

        Runs _consolidate_memory + session_manager.save_async.
        _consolidating set management is TurnScheduler's responsibility.
        """
        await self._consolidate_memory(session)  # type: ignore[arg-type]
        await self.session_manager.save_async(session)  # type: ignore[arg-type]
