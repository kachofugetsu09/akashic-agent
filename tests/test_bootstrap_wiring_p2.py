from __future__ import annotations
from typing import Any, cast

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import Config
from agent.config_models import Config as ConfigModel, WiringConfig
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.types import AfterStepCtx
from agent.looping.interrupt import TurnInterruptState
from agent.tools.registry import ToolRegistry
from bootstrap.tools import _build_loop_deps, build_registered_tools
from bootstrap.wiring import (
    wire_turn_lifecycle,
    resolve_context_factory,
    resolve_memory_engine_builder,
    resolve_memory_toolset_provider,
    resolve_toolset_provider,
)
from bus.event_bus import EventBus


def _toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _dump_toml(data: dict, prefix: tuple[str, ...] = ()) -> list[str]:
    lines: list[str] = []
    scalar_lines: list[str] = []

    for key, value in data.items():
        if isinstance(value, dict):
            continue
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            continue
        scalar_lines.append(f"{key} = {_toml_value(value)}")

    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    lines.extend(scalar_lines)
    if scalar_lines:
        lines.append("")

    for key, value in data.items():
        if isinstance(value, dict):
            lines.extend(_dump_toml(value, prefix + (key,)))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            for item in value:
                lines.append(f"[[{'.'.join(prefix + (key,))}]]")
                for item_key, item_value in item.items():
                    lines.append(f"{item_key} = {_toml_value(item_value)}")
                lines.append("")
    return lines


def _write_toml(path: Path, payload: dict) -> None:
    path.write_text("\n".join(_dump_toml(payload)).strip() + "\n", encoding="utf-8")


def test_config_load_reads_wiring_block(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
                "wiring": {
                    "context": "default",
                    "memory": "default",
                    "memory_engine": "default",
                    "toolsets": ["schedule", "mcp"],
                },
            },
        },
    )

    cfg = Config.load(cfg_path)

    assert cfg.wiring.context == "default"
    assert cfg.wiring.memory == "default"
    assert cfg.wiring.memory_engine == "default"
    assert cfg.wiring.toolsets == ["schedule", "mcp"]


def test_config_load_reads_memory_window_and_socket(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
                "context": {
                    "memory_window": 20,
                },
            },
            "channels": {
                "socket": "/tmp/dev-akashic.sock",
            },
        },
    )

    cfg = Config.load(cfg_path)

    assert cfg.memory_window == 20


def test_config_load_reads_agent_dev_mode(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
                "dev_mode": True,
            },
        },
    )

    cfg = Config.load(cfg_path)

    assert cfg.dev_mode is True


def test_config_load_accepts_dev_model_alias(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
                "dev_model": True,
            },
        },
    )

    cfg = Config.load(cfg_path)

    assert cfg.dev_mode is True


def test_config_load_skips_unfilled_channels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
            },
            "channels": {
                "telegram": {
                    "token": "${TELEGRAM_BOT_TOKEN}",
                    "allow_from": ["user1"],
                },
                "qq": {
                    "bot_uin": "",
                    "allow_from": ["42"],
                },
                "qqbot": {
                    "app_id": "app",
                    "client_secret": "${QQBOT_SECRET}",
                    "allow_from": ["user-openid"],
                    "groups": [
                        {
                            "group_openid": "group-openid",
                            "allow_from": ["member-openid"],
                            "require_at": True,
                            "allow_proactive": True,
                        }
                    ],
                },
            },
        },
    )

    monkeypatch.setenv("QQBOT_SECRET", "secret")
    cfg = Config.load(cfg_path)

    assert cfg.channels.telegram is None
    assert cfg.channels.qq is None
    assert cfg.channels.qqbot is not None
    assert cfg.channels.qqbot.app_id == "app"
    assert cfg.channels.qqbot.client_secret == "secret"
    assert cfg.channels.qqbot.allow_from == ["user-openid"]
    assert cfg.channels.qqbot.groups[0].group_openid == "group-openid"
    assert cfg.channels.qqbot.groups[0].allow_from == ["member-openid"]
    assert cfg.channels.qqbot.groups[0].require_at is True
    assert cfg.channels.qqbot.groups[0].allow_proactive is True
    assert cfg.channels.socket == "/tmp/akashic.sock"


def test_config_load_reads_fitbit_integration_block(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
            },
            "integrations": {
                "fitbit": {
                    "enabled": True,
                }
            },
        },
    )

    cfg = Config.load(cfg_path)

    assert cfg.fitbit.enabled is True


def test_config_load_reads_toml_layout(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[llm]
provider = "openai"

[llm.main]
model = "m"
api_key = "k"

[agent]
system_prompt = "s"
max_tokens = 256

[agent.context]
memory_window = 12

[channels]
socket = "/tmp/toml-akashic.sock"

[integrations.fitbit]
enabled = true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = Config.load(cfg_path)

    assert cfg.provider == "openai"
    assert cfg.model == "m"
    assert cfg.max_tokens == 256
    assert cfg.memory_window == 12
    assert cfg.channels.socket == "/tmp/toml-akashic.sock"
    assert cfg.fitbit.enabled is True


def test_config_load_reads_qq_websocket_timeout(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {
                "provider": "openai",
                "main": {
                    "model": "m",
                    "api_key": "k",
                },
            },
            "agent": {
                "system_prompt": "s",
            },
            "channels": {
                "qq": {
                    "bot_uin": "10001",
                    "allow_from": ["42"],
                    "websocket_open_timeout_seconds": 9.5,
                },
            },
        },
    )

    cfg = Config.load(cfg_path)

    assert cfg.channels.qq is not None
    assert cfg.channels.qq.websocket_open_timeout_seconds == 9.5


def test_build_registered_tools_respects_toolset_order_and_subset(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    class _MemoryProvider:
        def register(self, registry, deps):
            calls.append("memory")
            runtime = SimpleNamespace(engine=object())
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
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_, **__: {})
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
        http_resources=cast(Any, SimpleNamespace()),
        bus=cast(Any, SimpleNamespace()),
        provider=object(),
        light_provider=object(),
        session_store=object(),
        tools=ToolRegistry(),
        event_publisher=EventBus(),
        agent_loop_provider=lambda: None,
    )

    assert calls == ["memory", "schedule", "mcp"]


def test_build_loop_deps_uses_context_factory(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    fake_context = object()

    monkeypatch.setattr(
        "bootstrap.tools.resolve_context_factory",
        lambda name: (
            lambda workspace, memory_engine: observed.update(
                {"name": name, "workspace": workspace, "memory_engine": memory_engine}
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
        bus=cast(Any, SimpleNamespace()),
        provider=cast(Any, object()),
        light_provider=None,
        tools=ToolRegistry(),
        session_manager=cast(Any, SimpleNamespace()),
        presence=cast(Any, None),
        processing_state=cast(Any, SimpleNamespace()),
        event_bus=EventBus(),
        memory_runtime=cast(Any, SimpleNamespace(engine=object())),
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
    else:
        raise AssertionError("resolve_memory_engine_builder should fail for bad name")

    try:
        resolve_toolset_provider("bad")
    except ValueError as exc:
        assert "可选值" in str(exc)
        assert "meta_common" in str(exc)
    else:
        raise AssertionError("resolve_toolset_provider should fail for bad name")


@pytest.mark.asyncio
async def test_wire_turn_lifecycle_registers_afterstep_progress_handler():
    bus = EventBus()
    states: dict[str, TurnInterruptState] = {
        "telegram:1": TurnInterruptState(
            session_key="telegram:1",
            original_user_message="hello",
        )
    }
    wire_turn_lifecycle(
        lifecycle=TurnLifecycle(bus),
        active_turn_states=states,
    )

    await bus.emit(
        AfterStepCtx(
            session_key="telegram:1",
            channel="telegram",
            chat_id="1",
            iteration=0,
            tools_called=("noop",),
            partial_reply="部分回复",
            tools_used_so_far=("a", "b"),
            tool_chain_partial=(
                {"text": "tool", "calls": []},
            ),
            partial_thinking="思考",
            has_more=True,
        )
    )

    state = states["telegram:1"]
    assert state.partial_reply == "部分回复"
    assert state.partial_thinking == "思考"
    assert state.tools_used == ["a", "b"]
    assert state.tool_chain_partial == [{"text": "tool", "calls": []}]


def test_build_registered_tools_without_mcp_toolset_still_returns_empty_registry(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: SimpleNamespace(
            register=lambda registry, deps: SimpleNamespace(
                extras={"memory_runtime": SimpleNamespace(engine=object())}
            )
        ),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: SimpleNamespace(
            register=lambda registry, deps: SimpleNamespace(extras={})
        ),
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
        http_resources=cast(Any, SimpleNamespace()),
        bus=cast(Any, SimpleNamespace()),
        provider=object(),
        light_provider=object(),
        session_store=object(),
        tools=ToolRegistry(),
        event_publisher=EventBus(),
        agent_loop_provider=lambda: None,
    )

    assert mcp_registry is not None
    assert mcp_registry.list_servers() == "当前没有已注册的 MCP server。"
