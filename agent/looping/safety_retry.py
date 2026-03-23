import asyncio
import logging
from typing import TYPE_CHECKING

from agent.looping.constants import _SAFETY_RETRY_RATIOS
from agent.provider import ContentSafetyError, ContextLengthError

logger = logging.getLogger("agent.loop")

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.looping.tool_execution import ToolDiscoveryState, TurnExecutor
    from agent.tools.registry import ToolRegistry
    from session.manager import SessionManager

class SafetyRetryService:
    def __init__(
        self,
        executor: "TurnExecutor",
        context: "ContextBuilder",
        session_manager: "SessionManager",
        tools: "ToolRegistry",
        discovery: "ToolDiscoveryState",
        *,
        tool_search_enabled: bool,
        memory_window: int,
    ) -> None:
        self._executor = executor
        self._context = context
        self._session_manager = session_manager
        self._tools = tools
        self._discovery = discovery
        self._tool_search_enabled = tool_search_enabled
        self._memory_window = memory_window

    async def run(
        self,
        msg,
        session,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> tuple[str, list[str], list[dict], str | None]:
        source_history = base_history or session.get_history(max_messages=self._memory_window)
        total_history = len(source_history)

        preloaded: set[str] | None = None
        if self._tool_search_enabled:
            preloaded = self._discovery.get_preloaded(session.key)
            logger.info(
                "[tool_search] LRU preloaded=%s",
                sorted(preloaded) if preloaded else "[]",
            )

        for attempt, ratio in enumerate(_SAFETY_RETRY_RATIOS):
            window = int(total_history * ratio)
            if window <= 0:
                history_for_attempt: list[dict] = []
            elif window >= total_history:
                history_for_attempt = source_history
            else:
                history_for_attempt = source_history[-window:]
            initial_messages = self._context.build_messages(
                history=history_for_attempt,
                current_message=msg.content,
                media=msg.media if msg.media else None,
                skill_names=skill_names,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
                retrieved_memory_block=retrieved_memory_block,
            )
            try:
                content, tools_used, tool_chain, _visible, thinking = await self._executor.execute(
                    initial_messages,
                    request_time=msg.timestamp,
                    preloaded_tools=preloaded,
                )
                if attempt > 0:
                    logger.warning("安全拦截后以 window=%d 成功，修剪 session 历史", window)
                    if window == 0:
                        session.messages.clear()
                    else:
                        session.messages = session.messages[-window:]
                    session.last_consolidated = 0
                    await self._session_manager.save_async(session)

                if self._tool_search_enabled and tools_used:
                    self._discovery.update(
                        session.key,
                        tools_used,
                        self._tools.get_always_on_names(),
                    )
                return content, tools_used, tool_chain, thinking
            except ContentSafetyError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(total_history * _SAFETY_RETRY_RATIOS[attempt + 1])
                    logger.warning(
                        "安全拦截 (attempt=%d)，缩小历史窗口重试 %d → %d",
                        attempt + 1,
                        window,
                        next_window,
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return "你的消息触发了安全审查，无法处理。", [], [], None
            except ContextLengthError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(total_history * _SAFETY_RETRY_RATIOS[attempt + 1])
                    logger.warning(
                        "上下文超长 (attempt=%d)，缩小历史窗口重试 %d → %d",
                        attempt + 1,
                        window,
                        next_window,
                    )
                else:
                    logger.warning("上下文超长：所有窗口均失败，清空历史后仍超长")
                    return "上下文过长无法处理，请尝试新建对话。", [], [], None

        return "（安全重试异常）", [], [], None
