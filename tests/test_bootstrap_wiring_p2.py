from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.config import Config
from agent.config_models import Config as ConfigModel, WiringConfig
from agent.tools.registry import ToolRegistry
from bootstrap.tools import _build_loop_deps, build_registered_tools
from bootstrap.wiring import (
    resolve_context_factory,
    resolve_memory_engine_builder,
    resolve_memory_toolset_provider,
    resolve_toolset_provider,
)


def test_config_load_reads_wiring_block(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "m",
                "api_key": "k",
                "system_prompt": "s",
                "wiring": {
                    "context": "default",
                    "memory": "default",
                    "memory_engine": "default",
                    "toolsets": ["schedule", "mcp"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = Config.load(cfg_path)

    assert cfg.wiring.context == "default"
    assert cfg.wiring.memory == "default"
    assert cfg.wiring.memory_engine == "default"
    assert cfg.wiring.toolsets == ["schedule", "mcp"]


def test_config_load_reads_memory_window_and_socket(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps(
            {
                "provider": "openai",
                "model": "m",
                "api_key": "k",
                "system_prompt": "s",
                "memory_window": 0,
                "channels": {
                    "socket": "/tmp/dev-akasic.sock",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = Config.load(cfg_path)

    assert cfg.memory_window == 0
    assert cfg.channels.socket == "/tmp/dev-akasic.sock"


def test_build_registered_tools_respects_toolset_order_and_subset(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    class _MemoryProvider:
        def register(self, registry, deps):
            calls.append("memory")
            runtime = SimpleNamespace(port=object())
            return SimpleNamespace(extras={"memory_runtime": runtime})

    class _ToolsetProvider:
        def __init__(self, name: str) -> None:
            self._name = name

        def register(self, registry, deps):
            calls.append(self._name)
            extras = {"mcp_registry": object()} if self._name == "mcp" else {}
            return SimpleNamespace(extras=extras)

    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: _MemoryProvider(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: _ToolsetProvider(name),
    )
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_: {})
    monkeypatch.setattr(
        "bootstrap.tools.build_scheduler",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.build_peer_agent_resources",
        lambda *_args, **_kwargs: (None, None),
    )

    config = ConfigModel(
        provider="openai",
        model="m",
        api_key="k",
        system_prompt="s",
        wiring=WiringConfig(toolsets=["schedule", "mcp"]),
    )
    build_registered_tools(
        config=config,
        workspace=tmp_path,
        http_resources=SimpleNamespace(),
        bus=SimpleNamespace(),
        provider=object(),
        light_provider=object(),
        session_store=object(),
        tools=ToolRegistry(),
        observe_writer=None,
        agent_loop_provider=lambda: None,
    )

    assert calls == ["memory", "schedule", "mcp"]


def test_build_loop_deps_uses_context_factory(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    fake_context = object()

    monkeypatch.setattr(
        "bootstrap.tools.resolve_context_factory",
        lambda name: (
            lambda workspace, memory_port: observed.update(
                {"name": name, "workspace": workspace, "memory_port": memory_port}
            )
            or fake_context
        ),
    )

    config = ConfigModel(
        provider="openai",
        model="m",
        api_key="k",
        system_prompt="s",
        wiring=WiringConfig(context="default"),
    )
    deps = _build_loop_deps(
        config=config,
        workspace=tmp_path,
        bus=SimpleNamespace(),
        provider=object(),
        light_provider=None,
        tools=ToolRegistry(),
        session_manager=SimpleNamespace(),
        presence=None,
        processing_state=SimpleNamespace(),
        memory_runtime=SimpleNamespace(port=object(), post_response_worker=None),
        observe_writer=None,
    )

    assert observed["name"] == "default"
    assert observed["workspace"] == tmp_path
    assert deps.context is fake_context


def test_wiring_error_messages_list_available_choices():
    try:
        resolve_context_factory("bad")
    except ValueError as exc:
        assert "可选值" in str(exc)
        assert "default" in str(exc)
    else:
        raise AssertionError("resolve_context_factory should fail for bad name")

    try:
        resolve_memory_toolset_provider("bad")
    except ValueError as exc:
        assert "可选值" in str(exc)
        assert "default" in str(exc)
    else:
        raise AssertionError("resolve_memory_toolset_provider should fail for bad name")

    try:
        resolve_memory_engine_builder("bad")
    except ValueError as exc:
        assert "可选值" in str(exc)
        assert "default" in str(exc)
        assert "memu" in str(exc)
    else:
        raise AssertionError("resolve_memory_engine_builder should fail for bad name")

    try:
        resolve_toolset_provider("bad")
    except ValueError as exc:
        assert "可选值" in str(exc)
        assert "meta_common" in str(exc)
    else:
        raise AssertionError("resolve_toolset_provider should fail for bad name")


def test_resolve_memory_engine_builder_supports_memu():
    builder = resolve_memory_engine_builder("memu")

    assert callable(builder)


def test_build_registered_tools_without_mcp_toolset_still_returns_empty_registry(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: SimpleNamespace(
            register=lambda registry, deps: SimpleNamespace(
                extras={"memory_runtime": SimpleNamespace(port=object())}
            )
        ),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: SimpleNamespace(
            register=lambda registry, deps: SimpleNamespace(extras={})
        ),
    )
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_: {})
    monkeypatch.setattr(
        "bootstrap.tools.build_scheduler",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.build_peer_agent_resources",
        lambda *_args, **_kwargs: (None, None),
    )

    config = ConfigModel(
        provider="openai",
        model="m",
        api_key="k",
        system_prompt="s",
        wiring=WiringConfig(toolsets=["schedule"]),
    )
    _, _, _, mcp_registry, _, _, _ = build_registered_tools(
        config=config,
        workspace=tmp_path,
        http_resources=SimpleNamespace(),
        bus=SimpleNamespace(),
        provider=object(),
        light_provider=object(),
        session_store=object(),
        tools=ToolRegistry(),
        observe_writer=None,
        agent_loop_provider=lambda: None,
    )

    assert mcp_registry is not None
    assert mcp_registry.list_servers() == "当前没有已注册的 MCP server。"
