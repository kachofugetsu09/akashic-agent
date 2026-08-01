from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent.tools.base import Tool, ToolExecutionContext, get_current_tool_context
from agent.tools.filesystem import ListDirTool, ReadFileTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_fetch_spill import WebFetchSpillStore
from agent.tools.web_search import WebSearchTool
from core.net.http import HttpRequester


def build_readonly_research_tools(
    *,
    fetch_requester: HttpRequester,
    allowed_dir: Path | None = None,
    include_list_dir: bool = False,
    multimodal: bool = True,
    vl_available: bool = False,
    context_provider: Callable[[], ToolExecutionContext | None] | None = None,
) -> list[Tool]:
    tools: list[Tool] = [ReadFileTool(allowed_dir=allowed_dir, multimodal=multimodal, vl_available=vl_available)]
    if include_list_dir:
        tools.append(ListDirTool(allowed_dir=allowed_dir))
    spill_store = None
    if allowed_dir is not None:
        spill_store = WebFetchSpillStore(allowed_dir / ".tmp" / "web-fetch")
    tools.append(
        WebFetchTool(
            fetch_requester,
            spill_store=spill_store,
            context_provider=context_provider or get_current_tool_context,
        )
    )
    tools.append(WebSearchTool())
    return tools
