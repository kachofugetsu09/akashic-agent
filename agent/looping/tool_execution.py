import logging
from datetime import datetime
from typing import TYPE_CHECKING

from agent.core.reasoner import (
    DefaultReasoner,
    build_preflight_prompt,
)
from agent.core.runtime_support import ToolDiscoveryState

logger = logging.getLogger("agent.loop")

if TYPE_CHECKING:
    from agent.looping.ports import LLMConfig, LLMServices
    from agent.tools.registry import ToolRegistry
    from core.memory.port import MemoryPort


class TurnExecutor:
    def __init__(
        self,
        llm: "LLMServices",
        llm_config: "LLMConfig",
        tools: "ToolRegistry",
        discovery: ToolDiscoveryState,
        memory_port: "MemoryPort",
        *,
        tool_search_enabled: bool,
    ) -> None:
        self._llm = llm
        self._llm_config = llm_config
        self._tools = tools
        self._discovery = discovery
        self._memory_port = memory_port
        self._tool_search_enabled = tool_search_enabled
        self._reasoner = DefaultReasoner(
            llm=llm,
            llm_config=llm_config,
            tools=tools,
            discovery=discovery,
            memory_port=memory_port,
            tool_search_enabled=tool_search_enabled,
        )

    async def execute(
        self,
        initial_messages: list[dict],
        request_time: datetime | None = None,
        preloaded_tools: set[str] | None = None,
        preflight_injected: bool = False,
    ) -> tuple[str, list[str], list[dict], set[str] | None, str | None]:
        # 1. 统一走新的 DefaultReasoner。
        result = await self._reasoner.run(
            initial_messages,
            request_time=request_time,
            preloaded_tools=preloaded_tools,
            preflight_injected=preflight_injected,
        )
        tools_used = list(result.metadata.get("tools_used") or [])
        tool_chain = list(result.metadata.get("tool_chain") or [])
        visible_names = result.metadata.get("visible_names")
        return result.reply, tools_used, tool_chain, visible_names, result.thinking

    @staticmethod
    def _format_request_time_anchor(ts: datetime | None) -> str:
        return DefaultReasoner.format_request_time_anchor(ts)
