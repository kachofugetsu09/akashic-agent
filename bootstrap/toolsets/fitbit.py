from __future__ import annotations

from agent.config_models import Config
from agent.tool_bundles import build_fitbit_tools
from agent.tools.registry import ToolRegistry
from core.net.http import SharedHttpResources


def register_fitbit_tools(
    tools: ToolRegistry,
    config: Config,
    http_resources: SharedHttpResources,
) -> None:
    if not getattr(config.proactive, "fitbit_enabled", False):
        return
    fitbit_tools = {
        tool.name: tool
        for tool in build_fitbit_tools(
            fitbit_url=getattr(config.proactive, "fitbit_url", "http://127.0.0.1:18765"),
            requester=http_resources.local_service,
        )
    }
    tools.register(fitbit_tools["fitbit_health_snapshot"], tags=["health", "fitbit"], risk="read-only", search_keywords=["健康数据", "运动数据", "fitbit", "心率", "步数", "卡路里"])
    tools.register(fitbit_tools["fitbit_sleep_report"], tags=["health", "fitbit"], risk="read-only", search_keywords=["睡眠报告", "睡眠数据", "睡眠质量", "fitbit", "sleep"])
