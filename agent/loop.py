import asyncio
import logging
import re
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.context import ContextBuilder
from agent.loop_consolidation import AgentLoopConsolidationMixin
from agent.loop_memory_gate import AgentLoopMemoryGateMixin
from agent.loop_safety_retry import AgentLoopSafetyRetryMixin
from agent.loop_tool_execution import AgentLoopToolExecutionMixin
from bus.events import InboundMessage, OutboundMessage
from bus.processing import ProcessingState
from bus.queue import MessageBus
from memory2.post_response_worker import PostResponseMemoryWorker
from proactive.presence import PresenceStore
from agent.provider import LLMProvider
from agent.tools.registry import ToolRegistry
from session.manager import SessionManager

if TYPE_CHECKING:
    from core.memory.port import MemoryPort
    from core.memory.runtime import MemoryRuntime

logger = logging.getLogger(__name__)


class AgentLoop(
    AgentLoopSafetyRetryMixin,
    AgentLoopMemoryGateMixin,
    AgentLoopToolExecutionMixin,
    AgentLoopConsolidationMixin,
):
    """
    主循环：从 MessageBus 消费 InboundMessage，
    驱动 LLM + 工具调用，将结果发回 MessageBus。
    对话历史按 session_key 独立维护，格式为 OpenAI messages。
    """

    _MESSAGE_TIMEOUT_S: float = 600.0

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        tools: ToolRegistry,
        session_manager: SessionManager,
        workspace: Path,
        model: str = "deepseek-chat",
        max_iterations: int = 10,
        max_tokens: int = 8192,
        memory_window: int = 40,
        presence: PresenceStore | None = None,
        light_model: str = "",
        light_provider: LLMProvider | None = None,
        processing_state: ProcessingState | None = None,
        memory_top_k_procedure: int = 4,
        memory_top_k_history: int = 8,
        memory_route_intention_enabled: bool = False,
        memory_sop_guard_enabled: bool = True,
        memory_gate_llm_timeout_ms: int = 800,
        memory_gate_max_tokens: int = 96,
        memory_port: "MemoryPort | None" = None,
        post_mem_worker: PostResponseMemoryWorker | None = None,
        memory_runtime: "MemoryRuntime | None" = None,
        tool_search_enabled: bool = False,
    ) -> None:
        self.bus = bus
        self.provider = provider
        self.tools = tools
        self.session_manager = session_manager
        self.workspace = workspace
        self.model = model
        self.light_model = light_model or model
        self.light_provider = light_provider or provider
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self._presence = presence
        self._running = False
        self._consolidating: set[str] = set()
        self._processing_state = processing_state
        self._memory_top_k_procedure = max(1, int(memory_top_k_procedure))
        self._memory_top_k_history = max(1, int(memory_top_k_history))
        self._memory_route_intention_enabled = bool(memory_route_intention_enabled)
        self._memory_sop_guard_enabled = bool(memory_sop_guard_enabled)
        self._memory_gate_llm_timeout_ms = max(100, int(memory_gate_llm_timeout_ms))
        self._memory_gate_max_tokens = max(32, int(memory_gate_max_tokens))

        if memory_runtime is not None:
            memory_port = memory_runtime.port
            post_mem_worker = memory_runtime.post_response_worker
        if memory_port is None:
            raise ValueError("AgentLoop requires memory_port or memory_runtime")

        self._tool_search_enabled = bool(tool_search_enabled)
        # 进程内 LRU 缓存：session_key → 最近实际调用的非核心工具（容量 5）
        # 只记录 agent 真正调用过的工具，发现未用的不写入
        # 重启后清空（重新搜索一次即可恢复），不写入 session 持久化
        self._unlocked_tools: dict[str, OrderedDict[str, None]] = {}
        self._post_mem_worker = post_mem_worker
        self._memory_port = memory_port
        self.context = ContextBuilder(workspace, memory=self._memory_port)
        self._post_mem_failures = 0

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
        session = self.session_manager.get_or_create(key)
        skill_mentions = self._collect_skill_mentions(msg.content)
        if skill_mentions:
            logger.info(f"检测到 $skill 提及，直接注入完整内容: {skill_mentions}")

        main_history = session.get_history(max_messages=self.memory_window)
        retrieved_block = ""
        try:
            route_decision = "RETRIEVE"
            rewritten_query = msg.content
            fallback_reason = ""
            gate_latency_ms: dict[str, int] = {}
            runtime_md = session.metadata if isinstance(session.metadata, dict) else {}

            p_query = f"{msg.content} 操作规范"
            recent_turns = self._format_gate_history(main_history, max_turns=3)
            p_task = asyncio.create_task(
                self._memory_port.retrieve_related(
                    p_query,
                    memory_types=["procedure", "preference"],
                    top_k=self._memory_top_k_procedure,
                )
            )
            route_task = asyncio.create_task(
                self._decide_history_retrieval(
                    user_msg=msg.content,
                    metadata=runtime_md,
                    recent_history=recent_turns,
                )
            )
            p_items, (
                needs_history,
                rewritten_query,
                route_reason,
                route_ms,
            ) = await asyncio.gather(p_task, route_task)

            gate_latency_ms["route"] = route_ms
            if route_reason != "ok":
                fallback_reason = route_reason
            route_decision = "RETRIEVE" if needs_history else "NO_RETRIEVE"

            h_items: list[dict] = []
            if needs_history:
                h_items = await self._memory_port.retrieve_related(
                    rewritten_query,
                    memory_types=["event", "profile"],
                    top_k=self._memory_top_k_history,
                )

            seen_ids: set[str] = set()
            items = []
            for item in p_items + h_items:
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id:
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                items.append(item)

            selected_items = self._memory_port.select_for_injection(items)
            retrieved_block, injected_item_ids = (
                self._memory_port.format_injection_with_ids(selected_items)
            )
            if retrieved_block:
                logger.info(
                    f"memory2 retrieve: {len(items)} 条命中，筛选后 {len(selected_items)} 条注入"
                )

            protected_ids = {
                str(i.get("id", ""))
                for i in p_items
                if isinstance(i, dict)
                and i.get("memory_type") == "procedure"
                and (i.get("extra_json") or {}).get("tool_requirement")
                and i.get("id")
            }
            sop_guard_applied = bool(
                self._memory_sop_guard_enabled
                and any(item_id in protected_ids for item_id in injected_item_ids)
            )

            self._trace_memory_retrieve(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                user_msg=msg.content,
                items=selected_items,
                injected_block=retrieved_block,
                route_decision=route_decision,
                rewritten_query=rewritten_query,
                fallback_reason=fallback_reason,
                sop_guard_applied=sop_guard_applied,
                procedure_hits=len(p_items),
                history_hits=len(h_items),
                injected_item_ids=injected_item_ids,
                gate_latency_ms=gate_latency_ms,
            )
        except Exception as e:
            logger.warning(f"memory2 retrieve 失败，跳过: {e}")
            self._trace_memory_retrieve(
                session_key=key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                user_msg=msg.content,
                items=[],
                injected_block="",
                fallback_reason="retrieve_exception",
                error=str(e),
            )

        self._set_tool_context(msg.channel, msg.chat_id)
        final_content, tools_used, tool_chain = await self._run_with_safety_retry(
            msg,
            session,
            skill_names=skill_mentions or None,
            base_history=main_history,
            retrieved_memory_block=retrieved_block,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        preview = (
            final_content[:120] + "..." if len(final_content) > 120 else final_content
        )
        logger.info(f"Response to {msg.channel}:{msg.sender}: {preview}")

        if self._presence:
            self._presence.record_user_message(key)
        session.add_message("user", msg.content, media=msg.media if msg.media else None)
        session.add_message(
            "assistant",
            final_content,
            tools_used=tools_used if tools_used else None,
            tool_chain=tool_chain if tool_chain else None,
        )
        self._update_session_runtime_metadata(
            session,
            tools_used=tools_used,
            tool_chain=tool_chain,
        )
        await self.session_manager.append_messages(session, session.messages[-2:])

        if len(session.messages) > self.memory_window and key not in self._consolidating:
            self._consolidating.add(key)
            asyncio.create_task(self._consolidate_memory_bg(session, key))

        if self._post_mem_worker:
            task = asyncio.create_task(
                self._post_mem_worker.run(
                    user_msg=msg.content,
                    agent_response=final_content,
                    tool_chain=tool_chain,
                    source_ref=f"{key}@post_response",
                ),
                name=f"post_mem:{key}",
            )
            task.add_done_callback(lambda t: self._on_post_mem_task_done(t, key))

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata={
                **(msg.metadata or {}),
                "tools_used": tools_used,
                "tool_chain": tool_chain,
            },
        )

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
