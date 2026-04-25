from __future__ import annotations
from typing import Any, cast

from pathlib import Path
from types import SimpleNamespace

from agent.tools.registry import ToolRegistry
from bootstrap.toolsets.protocol import (
    ToolsetRegistrationResult,
    build_registration_result,
)
from bootstrap.toolsets.schedule import SchedulerToolsetProvider
from bootstrap.tools import build_registered_tools


def test_scheduler_toolset_provider_registers_expected_tools(tmp_path: Path):
    registry = ToolRegistry()
    scheduler = SimpleNamespace()

    result = SchedulerToolsetProvider().register(
        registry,
        cast(Any, SimpleNamespace(
            config=None,
            workspace=tmp_path,
            scheduler=scheduler,
        )),
    )

    assert result.source_name == "schedule"
    assert set(result.tool_names) == {
        "schedule",
        "list_schedules",
        "cancel_schedule",
    }
    assert result.always_on_names == []


def test_build_registered_tools_uses_toolset_providers(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    class _MemoryProvider:
        def register(self, registry, deps):
            calls.append("memory")
            runtime = SimpleNamespace(port=object())
            return ToolsetRegistrationResult(
                source_name="memory",
                tool_names=[],
                extras={"memory_runtime": runtime},
            )

    class _MetaProvider:
        def __init__(self, readonly_tools):
            self._readonly_tools = readonly_tools

        def register(self, registry, deps):
            calls.append("meta")
            return ToolsetRegistrationResult(source_name="meta_common")

    class _SpawnProvider:
        def register(self, registry, deps):
            calls.append("spawn")
            return ToolsetRegistrationResult(source_name="spawn")

    class _ScheduleProvider:
        def register(self, registry, deps):
            calls.append("schedule")
            return ToolsetRegistrationResult(source_name="schedule")

    class _McpProvider:
        def register(self, registry, deps):
            calls.append("mcp")
            return ToolsetRegistrationResult(
                source_name="mcp",
                extras={"mcp_registry": object()},
            )

    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: _MemoryProvider(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: {
            "meta_common": _MetaProvider(readonly_tools),
            "spawn": _SpawnProvider(),
            "schedule": _ScheduleProvider(),
            "mcp": _McpProvider(),
        }[name],
    )
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_, **__: {})
    monkeypatch.setattr(
        "bootstrap.tools.build_scheduler",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.build_peer_agent_resources",
        lambda *_args, **_kwargs: (None, None),
    )

    tools, push_tool, scheduler, mcp_registry, memory_runtime, peer_pm, peer_poller = (
        build_registered_tools(
            config=cast(Any, SimpleNamespace(spawn_enabled=False, proactive=SimpleNamespace())),
            workspace=tmp_path,
            http_resources=cast(Any, SimpleNamespace()),
            bus=cast(Any, SimpleNamespace()),
            provider=object(),
            light_provider=object(),
            session_store=object(),
            tools=ToolRegistry(),
            observe_writer=None,
            agent_loop_provider=lambda: None,
        )
    )

    assert calls == ["memory", "meta", "spawn", "schedule", "mcp"]
    assert push_tool is not None
    assert scheduler is not None
    assert mcp_registry is not None
    assert memory_runtime.port is not None
    assert peer_pm is None
    assert peer_poller is None


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
