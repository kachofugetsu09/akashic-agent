from __future__ import annotations
from copy import deepcopy
from typing import Any, cast

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.config import Config, DEFAULT_SOCKET
from agent.config_models import Config as ConfigModel, WiringConfig
from agent.lifecycle.facade import TurnLifecycle
from agent.lifecycle.types import AfterStepCtx
from agent.looping.interrupt import TurnInterruptState
from agent.tools.registry import ToolRegistry
from bootstrap.tools import _build_loop_deps, build_registered_tools
from bootstrap.wiring import (
    wire_turn_lifecycle,
    resolve_context_factory,
    resolve_toolset_provider,
)
from bus.event_bus import EventBus
from session.store import SessionStore


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
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ):
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
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ):
            for item in value:
                lines.append(f"[[{'.'.join(prefix + (key,))}]]")
                for item_key, item_value in item.items():
                    lines.append(f"{item_key} = {_toml_value(item_value)}")
                lines.append("")
    return lines


def _write_toml(path: Path, payload: dict) -> None:
    normalized = deepcopy(payload)
    normalized.pop("llm", None)
    path.write_text("\n".join(_dump_toml(normalized)).strip() + "\n", encoding="utf-8")


def _write_wiring_config(path: Path, wiring: object) -> None:
    _write_toml(
        path,
        {
            "agent": {"system_prompt": "s", "wiring": wiring},
        },
    )


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
                    "toolsets": ["fixture", "mcp"],
                },
            },
        },
    )

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.wiring.context == "default"
    assert cfg.wiring.memory == "default"
    assert cfg.wiring.toolsets == ["fixture", "mcp"]


def test_config_load_reads_memory_wiring(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_wiring_config(cfg_path, {"memory": "default"})

    cfg = Config.load(cfg_path, workspace=tmp_path)
    assert cfg.wiring.memory == "default"


@pytest.mark.parametrize("toolsets", ["fixture", [1, 2], ["fixture", ""]])
def test_config_load_rejects_invalid_wiring_toolsets(
    tmp_path: Path,
    toolsets: object,
):
    cfg_path = tmp_path / "config.toml"
    _write_wiring_config(cfg_path, {"toolsets": toolsets})

    with pytest.raises(ValueError, match="agent.wiring.toolsets 必须是字符串数组"):
        Config.load(cfg_path, workspace=tmp_path)


def test_config_load_preserves_empty_wiring_toolsets(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_wiring_config(cfg_path, {"toolsets": []})

    assert Config.load(cfg_path, workspace=tmp_path).wiring.toolsets == []


def test_config_load_rejects_invalid_wiring_table(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_wiring_config(cfg_path, "invalid")

    with pytest.raises(ValueError, match="agent.wiring 必须是 TOML table"):
        Config.load(cfg_path, workspace=tmp_path)


def test_config_load_ignores_legacy_memory_v2_enabled(tmp_path: Path):
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
            "agent": {"system_prompt": "s"},
            "memory_v2": {
                "enabled": True,
            },
        },
    )

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert not hasattr(cfg, "memory_v2")
    assert not hasattr(cfg, "memory")


def test_config_load_rejects_retired_memory_table(tmp_path: Path):
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
            "agent": {"system_prompt": "s"},
            "memory": {
                "enabled": True,
                "embedding": {
                    "model_ref": "embedding-a",
                },
                "retrieval": {
                    "score_threshold": 0.99,
                    "thresholds": {"event": 0.99},
                },
                "hyde": {"enabled": True},
            },
        },
    )

    with pytest.raises(ValueError, match=r"\[memory\].*普通模型插件"):
        Config.load(cfg_path, workspace=tmp_path)


def test_config_load_reads_compaction_and_app_server(tmp_path: Path):
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
                    "compaction": {
                        "keep_recent_tokens": 21000,
                    },
                },
            },
            "app_server": {
                "listen": "/tmp/dev-akashic.sock",
            },
        },
    )

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.context_compaction.keep_recent_tokens == 21000
    assert cfg.app_server.listen == "/tmp/dev-akashic.sock"


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

    cfg = Config.load(cfg_path, workspace=tmp_path)

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

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.dev_mode is True


def test_config_load_skips_unfilled_channels(tmp_path: Path):
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
            },
        },
    )

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.channels.telegram is None
    assert cfg.channels.qq is None
    assert cfg.app_server.listen == DEFAULT_SOCKET


def test_config_load_reads_toml_layout(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[agent]
system_prompt = "s"

[agent.context]
[agent.context.compaction]
keep_recent_tokens = 20000

[app_server]
listen = "/tmp/toml-akashic.sock"

""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.system_prompt == "s"
    assert cfg.context_compaction.keep_recent_tokens == 20000
    if sys.platform == "win32":
        assert cfg.app_server.listen != "/tmp/toml-akashic.sock"
        assert cfg.app_server.listen.startswith("127.0.0.1:")
    else:
        assert cfg.app_server.listen == "/tmp/toml-akashic.sock"


def test_config_rejects_legacy_cli_socket(tmp_path: Path):
    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        {
            "llm": {"provider": "openai", "main": {"model": "m", "api_key": "k"}},
            "agent": {"system_prompt": "s"},
            "channels": {"socket": "/tmp/legacy.sock"},
        },
    )
    with pytest.raises(ValueError, match="app_server"):
        _ = Config.load(cfg_path, workspace=tmp_path)


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

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.channels.qq is not None
    assert cfg.channels.qq.websocket_open_timeout_seconds == 9.5


def test_config_load_reads_web_chat_config(tmp_path: Path):
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
                "chat": {
                    "enabled": True,
                    "host": "127.0.0.2",
                    "port": 6324,
                },
            },
        },
    )

    cfg = Config.load(cfg_path, workspace=tmp_path)

    assert cfg.channels.chat.enabled is True
    assert not hasattr(cfg.channels.chat, "channel_name")
    assert not hasattr(cfg.channels.chat, "host")
    assert not hasattr(cfg.channels.chat, "port")


def test_build_registered_tools_respects_toolset_order_and_subset(
    monkeypatch, tmp_path: Path
):
    calls: list[str] = []

    class _MemoryProvider:
        def register(self, registry, deps):
            calls.append("memory")
            return SimpleNamespace(
                extras={"memory_runtime": SimpleNamespace(markdown=object())}
            )

    class _ToolsetProvider:
        def __init__(self, name: str) -> None:
            self._name = name

        def register(self, registry, deps):
            calls.append(self._name)
            extras = {}
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
    config = ConfigModel(
        system_prompt="s",
        wiring=WiringConfig(toolsets=["fixture", "mcp"]),
    )
    build_registered_tools(
        config=config,
        workspace=tmp_path,
        http_resources=cast(Any, SimpleNamespace()),
        bus=cast(Any, SimpleNamespace(chat_lane=None)),
        runtime_snapshot_store=cast(Any, object()),
        session_store=object(),
        tools=ToolRegistry(),
        event_publisher=EventBus(),
        agent_loop_provider=lambda: None,
    )

    assert calls == ["memory", "fixture", "mcp"]


def test_build_registered_tools_failure_preserves_external_session_store(
    monkeypatch,
    tmp_path: Path,
):
    class _MemoryProvider:
        def register(self, registry, deps):
            return SimpleNamespace(
                extras={"memory_runtime": SimpleNamespace(markdown=object())}
            )

    class _FailingToolsetProvider:
        def register(self, registry, deps):
            raise RuntimeError("toolset registration failed")

    monkeypatch.setattr(
        "bootstrap.tools.resolve_memory_toolset_provider",
        lambda name: _MemoryProvider(),
    )
    monkeypatch.setattr(
        "bootstrap.tools.resolve_toolset_provider",
        lambda name, readonly_tools=None: _FailingToolsetProvider(),
    )
    monkeypatch.setattr("bootstrap.tools.build_readonly_tools", lambda *_, **__: {})
    store = SessionStore(tmp_path / "sessions.db")
    try:
        config = ConfigModel(
            system_prompt="s",
            wiring=WiringConfig(toolsets=["fixture"]),
        )
        with pytest.raises(RuntimeError, match="toolset registration failed"):
            build_registered_tools(
                config=config,
                workspace=tmp_path,
                http_resources=cast(Any, SimpleNamespace()),
                bus=cast(Any, SimpleNamespace(chat_lane=None)),
                runtime_snapshot_store=cast(Any, object()),
                session_store=store,
                tools=ToolRegistry(),
            )
        assert store._closed is False
    finally:
        store.close()


def test_build_loop_deps_uses_context_factory(monkeypatch, tmp_path: Path):
    observed: dict[str, object] = {}
    fake_context = object()
    markdown_store = object()
    markdown_maintenance = SimpleNamespace()
    monkeypatch.setattr(
        "bootstrap.tools.resolve_context_factory",
        lambda name: (
            lambda workspace, memory_store: observed.update(
                {"name": name, "workspace": workspace, "memory_store": memory_store}
            )
            or fake_context
        ),
    )

    config = ConfigModel(
        system_prompt="s",
        wiring=WiringConfig(context="default"),
    )
    deps = _build_loop_deps(
        config=config,
        workspace=tmp_path,
        bus=cast(Any, SimpleNamespace(chat_lane=None)),
        tools=ToolRegistry(),
        session_manager=cast(
            Any,
            SimpleNamespace(
                get_or_create=lambda key: None,
                save_async=lambda session: None,
            ),
        ),
        presence=cast(Any, None),
        processing_state=cast(Any, SimpleNamespace()),
        event_bus=EventBus(),
        memory_runtime=cast(
            Any,
            SimpleNamespace(
                markdown=SimpleNamespace(
                    store=markdown_store,
                    maintenance=markdown_maintenance,
                )
            ),
        ),
    )

    assert observed["name"] == "default"
    assert observed["workspace"] == tmp_path
    assert observed["memory_store"] is markdown_store
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
            context_tokens_estimate=0,
            tools_called=("noop",),
            partial_reply="部分回复",
            tools_used_so_far=("a", "b"),
            tool_chain_partial=({"text": "tool", "calls": []},),
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
                extras={"memory_runtime": SimpleNamespace(markdown=object())}
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
    config = ConfigModel(
        system_prompt="s",
        wiring=WiringConfig(toolsets=["fixture"]),
    )
    tools, push_tool, _memory_runtime = build_registered_tools(
        config=config,
        workspace=tmp_path,
        http_resources=cast(Any, SimpleNamespace()),
        bus=cast(Any, SimpleNamespace(chat_lane=None)),
        runtime_snapshot_store=cast(Any, object()),
        session_store=object(),
        tools=ToolRegistry(),
        event_publisher=EventBus(),
        agent_loop_provider=lambda: None,
    )

    assert tools.get_registered_names() == set()
    assert push_tool is not None
