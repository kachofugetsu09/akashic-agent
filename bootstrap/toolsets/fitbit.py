from __future__ import annotations

from pathlib import Path

from agent.config_models import Config
from agent.tool_bundles import build_fitbit_tools
from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetProvider,
    build_registration_result,
)
from core.net.http import SharedHttpResources

_FITBIT_DEFAULT_URL = "http://127.0.0.1:18765"


class FitbitToolsetProvider(ToolsetProvider):
    def register(self, registry: ToolRegistry, deps: ToolsetDeps):
        before = set(registry._tools.keys())
        config = deps.config
        http_resources = deps.http_resources
        if config is None or http_resources is None:
            raise ValueError("fitbit toolset 缺少必要依赖")
        if not getattr(config.fitbit, "enabled", False):
            return build_registration_result(
                registry=registry,
                source_name="fitbit",
                before=before,
            )
        fitbit_tools = {
            tool.name: tool
            for tool in build_fitbit_tools(
                fitbit_url=_FITBIT_DEFAULT_URL,
                requester=http_resources.local_service,
            )
        }
        registry.register(
            fitbit_tools["fitbit_health_snapshot"],
            risk="read-only",
            search_hint="卡路里 运动数据",
        )
        registry.register(fitbit_tools["fitbit_sleep_report"], risk="read-only")
        return build_registration_result(
            registry=registry,
            source_name="fitbit",
            before=before,
        )


def register_fitbit_tools(
    tools: ToolRegistry,
    config: Config,
    http_resources: SharedHttpResources,
) -> None:
    FitbitToolsetProvider().register(
        tools,
        ToolsetDeps(
            config=config,
            workspace=Path("."),
            http_resources=http_resources,
        ),
    )
