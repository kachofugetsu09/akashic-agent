from __future__ import annotations

from pathlib import Path

from agent.tools.base import Tool
from agent.tools.filesystem import ListDirTool, ReadFileTool
from agent.tools.fitbit import FitbitHealthSnapshotTool, FitbitSleepReportTool
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from core.net.http import HttpRequester


def build_readonly_research_tools(
    *,
    fetch_requester: HttpRequester,
    allowed_dir: Path | None = None,
    include_list_dir: bool = False,
    multimodal: bool = True,
    vl_available: bool = False,
) -> list[Tool]:
    tools: list[Tool] = [ReadFileTool(allowed_dir=allowed_dir, multimodal=multimodal, vl_available=vl_available)]
    if include_list_dir:
        tools.append(ListDirTool(allowed_dir=allowed_dir))
    tools.append(WebFetchTool(fetch_requester))
    tools.append(WebSearchTool())
    return tools


def build_fitbit_tools(
    *,
    fitbit_url: str,
    requester: HttpRequester,
) -> list[Tool]:
    if not fitbit_url:
        return []
    return [
        FitbitHealthSnapshotTool(fitbit_url, requester=requester),
        FitbitSleepReportTool(fitbit_url, requester=requester),
    ]
