import asyncio
import logging
from typing import TYPE_CHECKING

from agent.core.reasoner import build_preflight_prompt
from agent.looping.constants import _SAFETY_RETRY_RATIOS
from agent.prompting import DEFAULT_CONTEXT_TRIM_PLANS
from agent.provider import ContentSafetyError, ContextLengthError

logger = logging.getLogger("agent.loop")

if TYPE_CHECKING:
    from agent.core.reasoner import Reasoner
    from agent.core.runtime_support import ToolDiscoveryState
    from agent.context import ContextBuilder
    from agent.tools.registry import ToolRegistry
    from session.manager import SessionManager


class SafetyRetryService:
    def __init__(
        self,
        context: "ContextBuilder",
        session_manager: "SessionManager",
        tools: "ToolRegistry",
        discovery: "ToolDiscoveryState",
        *,
        reasoner: "Reasoner",
        tool_search_enabled: bool,
        memory_window: int,
    ) -> None:
        self._reasoner = reasoner
        self._context = context
        self._session_manager = session_manager
        self._tools = tools
        self._discovery = discovery
        self._tool_search_enabled = tool_search_enabled
        self._memory_window = memory_window
        self._last_retry_trace: dict[str, object] = {}

    @property
    def last_retry_trace(self) -> dict[str, object]:
        return dict(self._last_retry_trace)

    async def run(
        self,
        msg,
        session,
        skill_names: list[str] | None = None,
        base_history: list[dict] | None = None,
        retrieved_memory_block: str = "",
    ) -> tuple[str, list[str], list[dict], str | None]:
        self._last_retry_trace = {
            "attempts": [],
            "selected_plan": None,
            "trimmed_sections": [],
        }
        source_history = base_history or session.get_history(max_messages=self._memory_window)
        total_history = len(source_history)

        preloaded: set[str] | None = None
        if self._tool_search_enabled:
            preloaded = self._discovery.get_preloaded(session.key)
            logger.info(
                "[tool_search] LRU preloaded=%s",
                sorted(preloaded) if preloaded else "[]",
            )

        attempts = self._build_attempt_plans(total_history)
        for attempt, plan in enumerate(attempts):
            self._last_retry_trace["attempts"].append(
                {
                    "name": plan["name"],
                    "history_window": plan["history_window"],
                    "disabled_sections": sorted(plan["disabled_sections"]),
                }
            )
            history_for_attempt = self._slice_history(source_history, plan["history_window"])
            preflight_prompt = build_preflight_prompt(
                request_time=msg.timestamp,
                tools=self._tools,
                tool_search_enabled=self._tool_search_enabled,
                visible_names=preloaded if self._tool_search_enabled else None,
            )
            # 每次重试都重新走 assembled input，确保裁掉的 section 真正体现在消息里。
            runtime_guard_context = self._context.build_runtime_guard_context(
                preflight_prompt=preflight_prompt
            )
            initial_messages = self._context.build_messages(
                history=history_for_attempt,
                current_message=msg.content,
                media=msg.media if msg.media else None,
                skill_names=skill_names,
                channel=msg.channel,
                chat_id=msg.chat_id,
                message_timestamp=msg.timestamp,
                retrieved_memory_block=retrieved_memory_block,
                disabled_sections=plan["disabled_sections"],
                runtime_guard_context=runtime_guard_context,
            )
            try:
                content, tools_used, tool_chain, thinking = await self._run_attempt(
                    initial_messages,
                    request_time=msg.timestamp,
                    preloaded_tools=preloaded,
                    preflight_injected=True,
                )
                if attempt > 0:
                    window = plan["history_window"]
                    self._last_retry_trace["selected_plan"] = plan["name"]
                    self._last_retry_trace["trimmed_sections"] = sorted(
                        plan["disabled_sections"]
                    )
                    logger.warning(
                        "重试成功 plan=%s window=%d disabled=%s，修剪 session 历史",
                        plan["name"],
                        window,
                        sorted(plan["disabled_sections"]),
                    )
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
                if attempt == 0:
                    self._last_retry_trace["selected_plan"] = plan["name"]
                    self._last_retry_trace["trimmed_sections"] = sorted(
                        plan["disabled_sections"]
                    )
                return content, tools_used, tool_chain, thinking
            except ContentSafetyError:
                if attempt < len(attempts) - 1:
                    next_plan = attempts[attempt + 1]
                    logger.warning(
                        "安全拦截 (attempt=%d)，切到 plan=%s window=%d disabled=%s",
                        attempt + 1,
                        next_plan["name"],
                        next_plan["history_window"],
                        sorted(next_plan["disabled_sections"]),
                    )
                else:
                    logger.warning("安全拦截：所有窗口均失败，当前消息本身可能违规")
                    return "你的消息触发了安全审查，无法处理。", [], [], None
            except ContextLengthError:
                if attempt < len(attempts) - 1:
                    next_plan = attempts[attempt + 1]
                    logger.warning(
                        "上下文超长 (attempt=%d)，切到 plan=%s window=%d disabled=%s",
                        attempt + 1,
                        next_plan["name"],
                        next_plan["history_window"],
                        sorted(next_plan["disabled_sections"]),
                    )
                else:
                    logger.warning("上下文超长：所有窗口均失败，清空历史后仍超长")
                    return "上下文过长无法处理，请尝试新建对话。", [], [], None

        return "（安全重试异常）", [], [], None

    async def _run_attempt(
        self,
        initial_messages: list[dict],
        *,
        request_time,
        preloaded_tools: set[str] | None,
        preflight_injected: bool,
    ) -> tuple[str, list[str], list[dict], str | None]:
        # 1. 统一直接走新 ReasonerResult，再在这里做一次本地解包。
        result = await self._reasoner.run(
            initial_messages,
            request_time=request_time,
            preloaded_tools=preloaded_tools,
            preflight_injected=preflight_injected,
        )
        tools_used = list(result.metadata.get("tools_used") or [])
        tool_chain = list(result.metadata.get("tool_chain") or [])
        return result.reply, tools_used, tool_chain, result.thinking

    @staticmethod
    def _slice_history(source_history: list[dict], window: int) -> list[dict]:
        total_history = len(source_history)
        if window <= 0:
            return []
        if window >= total_history:
            return source_history
        return source_history[-window:]

    def _build_attempt_plans(self, total_history: int) -> list[dict]:
        attempts: list[dict] = []
        seen: set[tuple[tuple[str, ...], int]] = set()
        full_window = int(total_history * _SAFETY_RETRY_RATIOS[0])
        # 先裁动态/次要 section，再在最后一个 trim 状态下缩 history。
        for trim_plan in DEFAULT_CONTEXT_TRIM_PLANS:
            disabled = set(trim_plan.drop_sections)
            key = (tuple(sorted(disabled)), full_window)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "name": trim_plan.name,
                    "disabled_sections": disabled,
                    "history_window": full_window,
                }
            )

        last_trim = set(DEFAULT_CONTEXT_TRIM_PLANS[-1].drop_sections)
        for ratio in _SAFETY_RETRY_RATIOS[1:]:
            window = int(total_history * ratio)
            key = (tuple(sorted(last_trim)), window)
            if key in seen:
                continue
            seen.add(key)
            attempts.append(
                {
                    "name": f"{DEFAULT_CONTEXT_TRIM_PLANS[-1].name}_history",
                    "disabled_sections": set(last_trim),
                    "history_window": window,
                }
            )
        return attempts
