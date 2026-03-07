import asyncio
import logging

from agent.loop_constants import _SAFETY_RETRY_RATIOS
from agent.provider import ContentSafetyError, ContextLengthError

logger = logging.getLogger("agent.loop")


class AgentLoopSafetyRetryMixin:
    async def _run_with_safety_retry(
        self,
        msg,
        session,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> tuple[str, list[str], list[dict]]:
        """递减历史窗口重试，处理 LLM 安全拦截错误。"""
        source_history = base_history or session.get_history(
            max_messages=self.memory_window
        )
        total_history = len(source_history)

        for attempt, ratio in enumerate(_SAFETY_RETRY_RATIOS):
            window = int(total_history * ratio)
            if window <= 0:
                history_for_attempt: list[dict] = []
            elif window >= total_history:
                history_for_attempt = source_history
            else:
                history_for_attempt = source_history[-window:]
            initial_messages = self.context.build_messages(
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
                result = await self._run_agent_loop(
                    initial_messages,
                    request_time=msg.timestamp,
                )
                if attempt > 0:
                    logger.warning(
                        f"安全拦截后以 window={window} 成功，修剪 session 历史"
                    )
                    if window == 0:
                        session.messages.clear()
                    else:
                        session.messages = session.messages[-window:]
                    session.last_consolidated = 0
                    await self.session_manager.save_async(session)
                return result
            except ContentSafetyError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(total_history * _SAFETY_RETRY_RATIOS[attempt + 1])
                    logger.warning(
                        f"安全拦截 (attempt={attempt + 1})，"
                        f"缩小历史窗口重试 {window} → {next_window}"
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return "你的消息触发了安全审查，无法处理。", [], []
            except ContextLengthError:
                if attempt < len(_SAFETY_RETRY_RATIOS) - 1:
                    next_window = int(total_history * _SAFETY_RETRY_RATIOS[attempt + 1])
                    logger.warning(
                        f"上下文超长 (attempt={attempt + 1})，"
                        f"缩小历史窗口重试 {window} → {next_window}"
                    )
                else:
                    logger.warning("上下文超长：所有窗口均失败，清空历史后仍超长")
                    return "上下文过长无法处理，请尝试新建对话。", [], []

        return "（安全重试异常）", [], []
