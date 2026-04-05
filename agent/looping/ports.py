from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.reasoner import Reasoner
    from agent.core.runner import CoreRunner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.looping.consolidation import ConsolidationService
    from agent.postturn.protocol import PostTurnPipeline
    from agent.provider import LLMProvider
    from agent.retrieval.protocol import MemoryRetrievalPipeline
    from agent.tools.registry import ToolRegistry
    from bus.processing import ProcessingState
    from bus.queue import MessageBus
    from core.memory.engine import MemoryEngine, PassiveMemoryEngine
    from core.memory.port import MemoryPort
    from core.memory.runtime import MemoryRuntime
    from memory2.hyde_enhancer import HyDEEnhancer
    from memory2.post_response_worker import PostResponseMemoryWorker
    from memory2.profile_extractor import ProfileFactExtractor
    from memory2.query_rewriter import QueryRewriter
    from memory2.sufficiency_checker import SufficiencyChecker
    from proactive_v2.presence import PresenceStore
    from session.manager import SessionManager

logger = logging.getLogger("agent.loop")


# ── Config dataclasses（参数，不含服务对象）───────────────────────────────────


@dataclass
class LLMConfig:
    model: str = "deepseek-chat"
    light_model: str = ""
    max_iterations: int = 10
    max_tokens: int = 8192
    tool_search_enabled: bool = False


@dataclass
class MemoryConfig:
    window: int = 40
    top_k_procedure: int = 4
    top_k_history: int = 8
    route_intention_enabled: bool = False
    procedure_guard_enabled: bool = True
    gate_llm_timeout_ms: int = 800
    gate_max_tokens: int = 96
    hyde_enabled: bool = False
    hyde_timeout_ms: int = 2000


# ── 服务对象分组（仅放对象，不放配置参数）──────────────────────────────────────


@dataclass
class LLMServices:
    """LLM provider services."""
    provider: LLMProvider
    light_provider: LLMProvider


@dataclass
class MemoryServices:
    port: MemoryPort
    engine: MemoryEngine | None = None
    passive_engine: PassiveMemoryEngine | None = None
    query_rewriter: QueryRewriter | None = None
    hyde_enhancer: HyDEEnhancer | None = None
    sufficiency_checker: SufficiencyChecker | None = None


@dataclass
class SessionServices:
    session_manager: SessionManager
    presence: PresenceStore | None = None


@dataclass
class ObservabilityServices:
    workspace: Path
    observe_writer: object | None = None


@dataclass
class AgentLoopDeps:
    bus: "MessageBus"
    provider: "LLMProvider"
    tools: "ToolRegistry"
    session_manager: "SessionManager"
    workspace: Path
    presence: "PresenceStore | None" = None
    light_provider: "LLMProvider | None" = None
    processing_state: "ProcessingState | None" = None
    memory_runtime: "MemoryRuntime | None" = None
    memory_port: "MemoryPort | None" = None
    post_mem_worker: "PostResponseMemoryWorker | None" = None
    observe_writer: object | None = None
    query_rewriter: "QueryRewriter | None" = None
    sufficiency_checker: "SufficiencyChecker | None" = None
    profile_extractor: "ProfileFactExtractor | None" = None
    retrieval_pipeline: "MemoryRetrievalPipeline | None" = None
    post_turn_pipeline: "PostTurnPipeline | None" = None
    context: "ContextBuilder | None" = None
    llm_services: LLMServices | None = None
    memory_services: MemoryServices | None = None
    session_services: SessionServices | None = None
    observability_services: ObservabilityServices | None = None
    hyde_enhancer: "HyDEEnhancer | None" = None
    tool_discovery: "ToolDiscoveryState | None" = None
    reasoner: "Reasoner | None" = None
    consolidation_service: "ConsolidationService | None" = None
    scheduler: "TurnScheduler | None" = None
    core_runner: "CoreRunner | None" = None


@dataclass
class AgentLoopConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


# ── TurnScheduler：封装调度行为 ────────────────────────────────────────────────


class TurnScheduler:
    """Encapsulates async task scheduling for consolidation and post-response memory.

    Owns _consolidating dedup set and _post_mem_failures counter, which previously
    lived in AgentLoop (core.py / consolidation.py).

    consolidation_runner: async (session) -> None — runs _consolidate_memory +
        session_manager.save_async without touching _consolidating (that's our job).
    """

    def __init__(
        self,
        post_mem_worker: PostResponseMemoryWorker | None,
        consolidation_runner: ConsolidationRunner,
        memory_window: int,
    ) -> None:
        self._post_mem_worker = post_mem_worker
        self._consolidation_runner = consolidation_runner
        self._memory_window = memory_window
        self._consolidating: set[str] = set()

    def is_consolidating(self, key: str) -> bool:
        return key in self._consolidating

    def mark_manual_start(self, key: str) -> bool:
        if key in self._consolidating:
            return False
        self._consolidating.add(key)
        return True

    def mark_manual_end(self, key: str) -> None:
        self._consolidating.discard(key)

    def schedule_consolidation(self, session: SessionLike, key: str) -> None:
        """Fire-and-forget consolidation; deduplicates by key."""
        # 1. 只有消息数超过 memory_window，且当前 session 没在 consolidate 中，才起后台任务。
        if len(session.messages) > self._memory_window and key not in self._consolidating:
            self._consolidating.add(key)
            task = asyncio.create_task(
                self._run_consolidation_bg(session, key),
                name=f"consolidation:{key}",
            )
            task.add_done_callback(lambda t: self._on_consolidation_done(t, key))

    async def _run_consolidation_bg(self, session: SessionLike, key: str) -> None:
        try:
            # 2. 真正的 consolidate/save 细节由外部注入的 runner 负责。
            await self._consolidation_runner(session)
        finally:
            self._consolidating.discard(key)

    def _on_consolidation_done(self, task: asyncio.Task, key: str) -> None:
        if task.cancelled():
            logger.info("consolidation task cancelled: %s", key)
            return
        try:
            exc = task.exception()
        except Exception as e:
            logger.warning(
                "consolidation task inspection failed: session=%s err=%s", key, e
            )
            return
        if exc is not None:
            logger.warning("consolidation task failed: session=%s err=%s", key, exc)

    def schedule_post_response_memory(
        self,
        *,
        msg: InboundMessage,
        key: str,
        final_content: str,
        tool_chain: list[dict],
    ) -> None:
        """Fire-and-forget post-response memory extraction."""
        if not self._post_mem_worker:
            return
        task = asyncio.create_task(
            self._post_mem_worker.run(
                user_msg=msg.content,
                agent_response=final_content,
                tool_chain=tool_chain,
                source_ref=f"{key}@post_response",
                session_key=key,
            ),
            name=f"post_mem:{key}",
        )
        task.add_done_callback(lambda t: self._on_post_mem_done(t, key))

    def _on_post_mem_done(self, task: asyncio.Task, key: str) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            logger.info("post_response_memorize task cancelled: %s", key)
            return
        except Exception as e:
            logger.warning(
                "post_response_memorize task inspection failed session=%s err=%s", key, e,
            )
            return

        if exc is not None:
            logger.warning(
                "post_response_memorize task failed session=%s err=%s", key, exc,
            )
