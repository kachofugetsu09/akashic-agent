import asyncio
import json
import sys
import types
from pathlib import Path
from typing import cast, Any

import pytest

import main
from bootstrap import app as bootstrap_app
from bootstrap import init_workspace as workspace_init
from bootstrap.channels import start_channels
from agent.config import (
    ChannelsConfig,
    Config,
    QQChannelConfig,
    QQGroupConfig,
    TelegramChannelConfig,
)
from agent.memory import DEFAULT_SELF_MD
from bus.event_bus import EventBus
from core.net.http import SharedHttpResources


class _FakeDashboardServer:
    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


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
        scalar_lines.append(f"{key} = {_toml_value(value)}")
    if prefix:
        lines.append(f"[{'.'.join(prefix)}]")
    lines.extend(scalar_lines)
    if scalar_lines:
        lines.append("")
    for key, value in data.items():
        if isinstance(value, dict):
            lines.extend(_dump_toml(value, prefix + (key,)))
    return lines


def _write_config(path: Path, socket_path: Path) -> None:
    payload = {
        "llm": {
            "provider": "openai",
            "main": {
                "model": "test-model",
                "api_key": "test-key",
            },
        },
        "agent": {
            "system_prompt": "test system prompt",
            "max_tokens": 256,
            "max_iterations": 2,
            "maintenance": {
                "memory_optimizer_enabled": False,
            },
        },
        "proactive": {
            "enabled": False,
            "profile": "quiet",
        },
        "channels": {
            "cli": {
                "socket": str(socket_path),
            }
        },
    }
    path.write_text("\n".join(_dump_toml(payload)).strip() + "\n", encoding="utf-8")


@pytest.mark.asyncio
async def test_serve_smoke_loads_config_and_runs_shutdown(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    socket_path = tmp_path / "akashic.sock"
    _write_config(config_path, socket_path)

    original_build_core_runtime = bootstrap_app.build_core_runtime
    observed: dict[str, object] = {}

    def _patched_build_core_runtime(config, workspace, http_resources):
        runtime = original_build_core_runtime(config, workspace, http_resources)
        agent_loop = runtime.loop
        bus = runtime.bus
        scheduler = runtime.scheduler

        async def _agent_loop_run():
            return None

        async def _bus_dispatch_outbound():
            return None

        async def _scheduler_run():
            return None

        agent_loop.run = _agent_loop_run  # type: ignore[assignment]
        bus.dispatch_outbound = _bus_dispatch_outbound  # type: ignore[assignment]
        scheduler.run = _scheduler_run  # type: ignore[assignment]
        observed["scheduler"] = scheduler
        observed["bus"] = bus
        observed["http_resources"] = http_resources
        return runtime

    monkeypatch.setattr(
        bootstrap_app, "build_core_runtime", _patched_build_core_runtime
    )
    monkeypatch.setattr(
        bootstrap_app, "build_dashboard_server", lambda **_: _FakeDashboardServer()
    )
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)

    await main.serve(str(config_path))

    assert socket_path.exists() is False
    assert "scheduler" in observed
    assert "bus" in observed
    assert cast(SharedHttpResources, observed["http_resources"]).closed is True


@pytest.mark.asyncio
async def test_run_cleanup_steps_continues_after_failure():
    calls: list[str] = []

    async def _fail() -> None:
        calls.append("fail")
        raise RuntimeError("stop failed")

    async def _cleanup() -> None:
        calls.append("cleanup")

    with pytest.raises(RuntimeError, match="stop failed"):
        await bootstrap_app._run_cleanup_steps(
            ("fail", _fail),
            ("cleanup", _cleanup),
        )

    assert calls == ["fail", "cleanup"]


def test_connect_cli_uses_socket_from_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    socket_path = tmp_path / "cli.sock"
    _write_config(config_path, socket_path)
    observed: dict[str, str] = {}

    fake_cli_tui = types.ModuleType("infra.channels.cli_tui")

    def _run_tui(socket: str) -> None:
        observed["socket"] = socket

    fake_cli_tui.run_tui = _run_tui  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra.channels.cli_tui", fake_cli_tui)

    main.connect_cli(str(config_path))

    assert observed["socket"] == str(socket_path)


def test_init_workspace_creates_expected_assets(tmp_path):
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    summary = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )

    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "multimodal = false" in config_text
    assert "[llm.vl]" in config_text
    assert 'model = "qwen-vl-plus"' in config_text
    assert (workspace / "sessions.db").exists()
    assert (workspace / "observe").is_dir()
    assert (workspace / "memory" / "consolidation_writes.db").exists()
    assert (workspace / "memory" / "journal").is_dir()
    assert (workspace / "memory" / "memory2.db").exists()
    assert "Proactive Context" in (
        workspace / "PROACTIVE_CONTEXT.md"
    ).read_text(encoding="utf-8")
    assert json.loads(
        (workspace / "mcp_servers.json").read_text(encoding="utf-8")
    ) == {"servers": {}}
    assert json.loads(
        (workspace / "proactive_sources.json").read_text(encoding="utf-8")
    ) == {"sources": []}
    assert (workspace / "proactive.db").exists()
    assert (workspace / "skills").is_dir()
    assert (workspace / "drift" / "skills").is_dir()
    assert any(path == config_path for path in summary.created)


def test_init_workspace_respects_force_for_text_assets(tmp_path):
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )
    self_path = workspace / "memory" / "SELF.md"
    self_path.write_text("custom\n", encoding="utf-8")

    summary_skip = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )
    assert self_path.read_text(encoding="utf-8") == "custom\n"
    assert any(path == self_path for path in summary_skip.skipped)

    summary_force = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
        force=True,
    )
    assert self_path.read_text(encoding="utf-8") == DEFAULT_SELF_MD
    assert any(path == self_path for path in summary_force.overwritten)


@pytest.mark.asyncio
async def test_start_channels_wires_telegram_and_qq(monkeypatch, tmp_path):
    starts: list[str] = []
    registrations: list[str] = []

    fake_ipc_server = types.ModuleType("infra.channels.ipc_server")
    fake_telegram_channel = types.ModuleType("infra.channels.telegram_channel")
    fake_qq_channel = types.ModuleType("infra.channels.qq_channel")

    class _IPCServerChannel:
        def __init__(self, bus, socket):
            self.bus = bus
            self.socket = socket

        async def start(self) -> None:
            starts.append("ipc")

        async def stop(self) -> None:
            starts.append("ipc.stop")

    class _TelegramChannel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            starts.append("telegram")

        async def stop(self) -> None:
            starts.append("telegram.stop")

        async def send(self, *args, **kwargs):
            return None

        async def send_stream(self, *args, **kwargs):
            return None

        async def send_file(self, *args, **kwargs):
            return None

        async def send_image(self, *args, **kwargs):
            return None

    class _QQChannel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def start(self) -> None:
            starts.append("qq")

        async def stop(self) -> None:
            starts.append("qq.stop")

        async def send(self, *args, **kwargs):
            return None

        async def send_file(self, *args, **kwargs):
            return None

        async def send_image(self, *args, **kwargs):
            return None

    fake_ipc_server.IPCServerChannel = _IPCServerChannel  # type: ignore[attr-defined]
    fake_telegram_channel.TelegramChannel = _TelegramChannel  # type: ignore[attr-defined]
    fake_qq_channel.QQChannel = _QQChannel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra.channels.ipc_server", fake_ipc_server)
    monkeypatch.setitem(sys.modules, "infra.channels.telegram_channel", fake_telegram_channel)
    monkeypatch.setitem(sys.modules, "infra.channels.qq_channel", fake_qq_channel)

    class _PushTool:
        def register_channel(self, name: str, **kwargs) -> None:
            registrations.append(name)

    config = Config(
        provider="openai",
        model="m",
        api_key="k",
        system_prompt="s",
        channels=ChannelsConfig(
            telegram=TelegramChannelConfig(token="tg-token", allow_from=["1"]),
            qq=QQChannelConfig(
                bot_uin="10001",
                allow_from=["2"],
                groups=[QQGroupConfig(group_id="3")],
            ),
            socket=str(tmp_path / "sock"),
        ),
    )
    resources = SharedHttpResources()
    event_bus = EventBus()
    try:
        controller = object()
        ipc, tg, qq = await start_channels(
            config,
            bus=cast(Any, object()),
            session_manager=cast(Any, object()),
            push_tool=cast(Any, _PushTool()),
            http_resources=resources,
            event_bus=event_bus,
            interrupt_controller=cast(Any, controller),
        )
    finally:
        await resources.aclose()

    assert ipc is not None
    assert tg is not None
    assert qq is not None
    assert starts == ["ipc", "telegram", "qq"]
    assert registrations == ["telegram", "qq"]
    assert tg.kwargs["event_bus"] is event_bus
    assert tg.kwargs["interrupt_controller"] is controller
    assert qq.kwargs["interrupt_controller"] is controller


@pytest.mark.asyncio
async def test_start_channels_skips_unfilled_optional_channels(monkeypatch, tmp_path):
    starts: list[str] = []

    fake_ipc_server = types.ModuleType("infra.channels.ipc_server")
    fake_telegram_channel = types.ModuleType("infra.channels.telegram_channel")
    fake_qq_channel = types.ModuleType("infra.channels.qq_channel")

    class _IPCServerChannel:
        def __init__(self, bus, socket):
            self.bus = bus
            self.socket = socket

        async def start(self) -> None:
            starts.append("ipc")

        async def stop(self) -> None:
            starts.append("ipc.stop")

    class _TelegramChannel:
        async def start(self) -> None:
            starts.append("telegram")

    class _QQChannel:
        async def start(self) -> None:
            starts.append("qq")

    fake_ipc_server.IPCServerChannel = _IPCServerChannel  # type: ignore[attr-defined]
    fake_telegram_channel.TelegramChannel = _TelegramChannel  # type: ignore[attr-defined]
    fake_qq_channel.QQChannel = _QQChannel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra.channels.ipc_server", fake_ipc_server)
    monkeypatch.setitem(sys.modules, "infra.channels.telegram_channel", fake_telegram_channel)
    monkeypatch.setitem(sys.modules, "infra.channels.qq_channel", fake_qq_channel)

    class _PushTool:
        def register_channel(self, name: str, **kwargs) -> None:
            raise AssertionError(f"unexpected channel registration: {name}")

    config = Config(
        provider="openai",
        model="m",
        api_key="k",
        system_prompt="s",
        channels=ChannelsConfig(
            telegram=None,
            qq=None,
            socket=str(tmp_path / "sock"),
        ),
    )
    resources = SharedHttpResources()
    try:
        ipc, tg, qq = await start_channels(
            config,
            bus=cast(Any, object()),
            session_manager=cast(Any, object()),
            push_tool=cast(Any, _PushTool()),
            http_resources=resources,
            event_bus=EventBus(),
        )
    finally:
        await resources.aclose()

    assert ipc is not None
    assert tg is None
    assert qq is None
    assert starts == ["ipc"]
