from __future__ import annotations
from typing import Any, cast

import pytest

from pathlib import Path
from types import SimpleNamespace

from agent.config_models import Config, WiringConfig
from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.protocol import (
    ToolsetDeps,
    ToolsetRegistrationResult,
    build_registration_result,
)
from bootstrap.toolsets.memory import MemoryToolsetProvider
from bootstrap.tools import build_registered_tools
from bus.event_bus import EventBus


def test_build_registered_tools_uses_toolset_providers(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    class _MemoryProvider:
        def register(self, registry, deps):
            calls.append("memory")
            runtime = SimpleNamespace(markdown=object())
            return ToolsetRegistrationResult(
                source_name="markdown_memory",
                extras={"memory_runtime": runtime},
            )

    class _MetaProvider:
        def __init__(self, readonly_tools):
            self._readonly_tools = readonly_tools

        def register(self, registry, deps):
            calls.append("meta")
            return ToolsetRegistrationResult(source_name="meta_common")

    class _McpProvider:
        def register(self, registry, deps):
            calls.append("mcp")
            return ToolsetRegistrationResult(
                source_name="mcp",
                extras={},
            )

    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: _MemoryProvider(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: {
            "meta_common": _MetaProvider(readonly_tools),
            "mcp": _McpProvider(),
        }[name],
    )
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_, **__: {})
    tools, push_tool, memory_runtime = build_registered_tools(
        config=Config(
            system_prompt="s",
            wiring=WiringConfig(toolsets=["meta_common"]),
        ),
        workspace=tmp_path,
        http_resources=cast(Any, SimpleNamespace()),
        bus=cast(Any, SimpleNamespace(chat_lane=None)),
        runtime_snapshot_store=cast(Any, object()),
        session_store=object(),
        tools=ToolRegistry(),
        event_publisher=EventBus(),
        agent_loop_provider=lambda: None,
    )

    assert calls == ["memory", "meta"]
    assert push_tool is not None
    assert memory_runtime.markdown is not None


def test_build_registration_result_uses_public_registry_names():
    registry = SimpleNamespace(
        get_registered_names=lambda: {"a", "b", "always"},
        get_always_on_names=lambda: {"always"},
    )

    result = build_registration_result(
        registry=cast(Any, registry),
        source_name="demo",
        before={"a"},
    )

    assert result.tool_names == ["always", "b"]
    assert result.always_on_names == ["always"]


def test_memory_rejects_missing_runtime_snapshot_store(tmp_path: Path):
    config = Config(
        system_prompt="s",
    )

    with pytest.raises(ValueError, match="runtime_snapshot_store"):
        MemoryToolsetProvider().register(
            ToolRegistry(),
            ToolsetDeps(
                config=config,
                workspace=tmp_path,
                http_resources=cast(Any, object()),
            ),
        )
