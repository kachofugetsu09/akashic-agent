from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.provider import LLMProvider
    from bus.events import InboundMessage
    from core.memory.port import MemoryPort
    from memory2.hyde_enhancer import HyDEEnhancer
    from memory2.post_response_worker import PostResponseMemoryWorker
    from memory2.query_rewriter import QueryRewriter
    from memory2.sufficiency_checker import SufficiencyChecker
    from proactive.presence import PresenceStore
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
    sop_guard_enabled: bool = True
    gate_llm_timeout_ms: int = 800
    gate_max_tokens: int = 96
    hyde_enabled: bool = False
    hyde_timeout_ms: int = 2000


# ── 服务对象分组（仅放对象，不放配置参数）──────────────────────────────────────


@dataclass
class LLMServices:
    """LLM provider services.

    run_turn_fn: Phase 2 → Phase 4 transition callable.  Wraps AgentLoop._run_with_safety_retry
    until TurnExecutor is properly extracted in Phase 4.
    """

    provider: LLMProvider
    light_provider: LLMProvider
    # Temporary: Phase 4 replaces this with an injected TurnExecutor
    run_turn_fn: Any = field(default=None)


@dataclass
class MemoryServices:
    port: MemoryPort
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
        consolidation_runner: Any,  # Callable[[session], Coroutine[None]]
        memory_window: int,
    ) -> None:
        self._post_mem_worker = post_mem_worker
        self._consolidation_runner = consolidation_runner
        self._memory_window = memory_window
        self._consolidating: set[str] = set()
        self._post_mem_failures: int = 0

    def is_consolidating(self, key: str) -> bool:
        return key in self._consolidating

    def mark_manual_start(self, key: str) -> bool:
        if key in self._consolidating:
            return False
        self._consolidating.add(key)
        return True

    def mark_manual_end(self, key: str) -> None:
        self._consolidating.discard(key)

    def schedule_consolidation(self, session: Any, key: str) -> None:
        """Fire-and-forget consolidation; deduplicates by key."""
        if len(session.messages) > self._memory_window and key not in self._consolidating:
            self._consolidating.add(key)
            task = asyncio.create_task(
                self._run_consolidation_bg(session, key),
                name=f"consolidation:{key}",
            )
            task.add_done_callback(lambda t: self._on_consolidation_done(t, key))

    async def _run_consolidation_bg(self, session: Any, key: str) -> None:
        try:
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
            self._post_mem_failures += 1
            logger.warning(
                "post_response_memorize task inspection failed session=%s failures=%d err=%s",
                key,
                self._post_mem_failures,
                e,
            )
            return

        if exc is not None:
            self._post_mem_failures += 1
            logger.warning(
                "post_response_memorize task failed session=%s failures=%d err=%s",
                key,
                self._post_mem_failures,
                exc,
            )
