import asyncio
import logging
import re
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from agent.context import ContextBuilder
from agent.looping.consolidation import AgentLoopConsolidationMixin
from agent.looping.handlers import ConversationTurnHandler, InternalEventHandler
from agent.looping.memory_gate import AgentLoopMemoryGateMixin
from agent.looping.safety_retry import AgentLoopSafetyRetryMixin
from agent.looping.tool_execution import AgentLoopToolExecutionMixin
from bus.events import InboundMessage, OutboundMessage
from bus.internal_events import is_spawn_completion_message
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

logger = logging.getLogger("agent.loop")
_MAX_PROCEDURE_RETRIEVE_K = 3


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
        memory_hyde_enabled: bool = False,
        memory_hyde_timeout_ms: int = 2000,
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
        self._memory_top_k_procedure = min(
            _MAX_PROCEDURE_RETRIEVE_K,
            max(1, int(memory_top_k_procedure)),
        )
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

        if memory_hyde_enabled:
            if not light_model:
                logger.warning(
                    "hyde_enabled=True 但未配置独立 light_model，"
                    "为避免主模型被额外调用，HyDE 已自动禁用。"
                    "请在配置中设置 light_model 后重启。"
                )
                self._hyde_enhancer: HyDEEnhancer | None = None
            else:
                from memory2.hyde_enhancer import HyDEEnhancer

                self._hyde_enhancer = HyDEEnhancer(
                    light_provider=self.light_provider,
                    light_model=self.light_model,
                    timeout_s=memory_hyde_timeout_ms / 1000.0,
                )
        else:
            self._hyde_enhancer = None

        # 进程内 LRU 缓存：session_key → 最近实际调用的非核心工具（容量 5）
        # 只记录 agent 真正调用过的工具，发现未用的不写入
        # 重启后清空（重新搜索一次即可恢复），不写入 session 持久化
        self._unlocked_tools: dict[str, OrderedDict[str, None]] = {}
        self._post_mem_worker = post_mem_worker
        self._memory_port = memory_port
        self.context = ContextBuilder(workspace, memory=self._memory_port)
        self._post_mem_failures = 0
        self._conversation_handler = ConversationTurnHandler(self)
        self._internal_event_handler = InternalEventHandler(self)

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

    @staticmethod
    def _is_spawn_completion(msg: InboundMessage) -> bool:
        return is_spawn_completion_message(msg)

    def _schedule_consolidation_if_needed(self, session, key: str) -> None:
        if (
            len(session.messages) > self.memory_window
            and key not in self._consolidating
        ):
            self._consolidating.add(key)
            asyncio.create_task(self._consolidate_memory_bg(session, key))

    def _schedule_post_response_memory(
        self,
        *,
        msg: InboundMessage,
        key: str,
        final_content: str,
        tool_chain: list[dict],
    ) -> None:
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
