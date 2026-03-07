import json
import sys
import types
from pathlib import Path
from typing import cast

import pytest

import main
from bootstrap import app as bootstrap_app
from bootstrap.channels import start_channels
from agent.config import (
    ChannelsConfig,
    Config,
    QQChannelConfig,
    QQGroupConfig,
    TelegramChannelConfig,
)
from core.net.http import SharedHttpResources


def _write_config(path: Path, socket_path: Path) -> None:
    payload = {
        "provider": "openai",
        "model": "test-model",
        "api_key": "test-key",
        "system_prompt": "test system prompt",
        "max_tokens": 256,
        "max_iterations": 2,
        "memory_optimizer_enabled": False,
        "proactive": {
            "enabled": False,
        },
        "channels": {
            "cli": {
                "socket": str(socket_path),
            }
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.mark.asyncio
async def test_serve_smoke_loads_config_and_runs_shutdown(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    socket_path = tmp_path / "akasic.sock"
    _write_config(config_path, socket_path)

    original_build_core_runtime = bootstrap_app.build_core_runtime
    observed: dict[str, object] = {}

    def _patched_build_core_runtime(config, workspace, http_resources):
        runtime = original_build_core_runtime(config, workspace, http_resources)
        (
            agent_loop,
            bus,
            _tools,
            _push_tool,
            _session_manager,
            scheduler,
            _provider,
            _light_provider,
            _mcp_registry,
            _memory_runtime,
            _presence,
        ) = runtime

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

    monkeypatch.setattr(bootstrap_app, "build_core_runtime", _patched_build_core_runtime)
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
    config_path = tmp_path / "config.json"
    socket_path = tmp_path / "cli.sock"
    _write_config(config_path, socket_path)
    observed: dict[str, str] = {}

    fake_cli_tui = types.ModuleType("channels.cli_tui")

    def _run_tui(socket: str) -> None:
        observed["socket"] = socket

    fake_cli_tui.run_tui = _run_tui  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "channels.cli_tui", fake_cli_tui)

    main.connect_cli(str(config_path))

    assert observed["socket"] == str(socket_path)


@pytest.mark.asyncio
async def test_start_channels_wires_telegram_and_qq(monkeypatch, tmp_path):
    starts: list[str] = []
    registrations: list[str] = []

    fake_ipc_server = types.ModuleType("channels.ipc_server")
    fake_telegram_channel = types.ModuleType("channels.telegram_channel")
    fake_qq_channel = types.ModuleType("channels.qq_channel")

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
    monkeypatch.setitem(sys.modules, "channels.ipc_server", fake_ipc_server)
    monkeypatch.setitem(sys.modules, "channels.telegram_channel", fake_telegram_channel)
    monkeypatch.setitem(sys.modules, "channels.qq_channel", fake_qq_channel)

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
    try:
        ipc, tg, qq = await start_channels(
            config,
            bus=object(),
            session_manager=object(),
            push_tool=_PushTool(),
            http_resources=resources,
        )
    finally:
        await resources.aclose()

    assert ipc is not None
    assert tg is not None
    assert qq is not None
    assert starts == ["ipc", "telegram", "qq"]
    assert registrations == ["telegram", "qq"]
