from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from bootstrap.app import AppRuntime
from bus.event_bus import EventBus
from infra.channels.group_filter import DefaultGroupFilter, strip_at_segments


@pytest.mark.asyncio
async def test_app_runtime_start_passes_markdown_store_to_memory_optimizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    markdown_store = MagicMock(name="markdown_store")
    memory_runtime = SimpleNamespace(
        markdown=SimpleNamespace(store=markdown_store),
        aclose=AsyncMock(),
    )
    startup_order: list[str] = []
    plugin_manager = MagicMock()
    plugin_manager.bind_core_channel_definitions = AsyncMock(
        side_effect=lambda definitions: startup_order.append("bindings")
    )
    plugin_manager.run_runtime_services = AsyncMock()
    snapshot_store = MagicMock()
    plugin_manager.snapshot_store = snapshot_store
    core = SimpleNamespace(
        loop=SimpleNamespace(
            run=lambda: "loop-task",
            bind_plugin_rollout_fact_provider=MagicMock(),
        ),
        bus=SimpleNamespace(dispatch_outbound=lambda: "bus-task"),
        event_bus=EventBus(),
        tools=MagicMock(),
        push_tool=MagicMock(),
        session_manager=MagicMock(),
        memory_runtime=memory_runtime,
        channel_attachment_store=MagicMock(),
        presence=MagicMock(),
        plugin_manager=plugin_manager,
        bind_conversation_runtime=MagicMock(),
        start=AsyncMock(),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(
        "bootstrap.app.build_core_runtime", lambda *args, **kwargs: core
    )
    channel_host = SimpleNamespace(
        start_all=AsyncMock(side_effect=lambda: startup_order.append("providers")),
        stop_all=AsyncMock(),
        bind_plugin_channels=MagicMock(),
        swap_plugin_channels=AsyncMock(),
        channels=(),
    )
    monkeypatch.setattr(
        "bootstrap.app.start_channels",
        AsyncMock(return_value=channel_host),
    )
    memory_optimizer = MagicMock()
    build_memory_optimizer_task = MagicMock(return_value=([], memory_optimizer))
    monkeypatch.setattr(
        "bootstrap.app.build_memory_optimizer_task", build_memory_optimizer_task
    )
    monkeypatch.setattr(
        "bootstrap.app.build_dashboard_server",
        lambda **kwargs: SimpleNamespace(
            should_exit=False,
            serve=AsyncMock(return_value=None),
            manual_memory_optimizer=kwargs["manual_memory_optimizer"],
        ),
    )

    app = AppRuntime(
        config=cast(
            Any,
            SimpleNamespace(
                app_server=SimpleNamespace(enabled=False),
                channels=SimpleNamespace(chat=SimpleNamespace(enabled=False)),
                mobile_realtime=SimpleNamespace(enabled=False),
            ),
        ),
        workspace=tmp_path,
    )
    await app.start()

    build_memory_optimizer_task.assert_called_once()
    kwargs = build_memory_optimizer_task.call_args.kwargs
    assert kwargs["memory_store"] is markdown_store
    assert kwargs["runtime_snapshot_store"] is snapshot_store
    assert app.dashboard_server.manual_memory_optimizer is memory_optimizer
    assert startup_order == ["bindings", "providers"]
    await app.shutdown()


@pytest.mark.asyncio
async def test_group_filter_paths() -> None:
    group = SimpleNamespace(group_id="1", allow_from=["42"], require_at=True)
    event = SimpleNamespace(user_id="42", raw_message="[CQ:at,qq=10001] hi")

    assert (
        await DefaultGroupFilter("10001").should_process(event, cast(Any, group))
        is True
    )
    assert strip_at_segments("x [CQ:at,qq=10001] y") == "x  y".strip()

    bad_user = SimpleNamespace(user_id="9", raw_message="hi")
    assert (
        await DefaultGroupFilter("10001").should_process(bad_user, cast(Any, group))
        is False
    )


@pytest.mark.asyncio
async def test_bootstrap_trigger_and_entrypoints_cover_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agent.migrations import MigrationOutcome

    supervisor_calls: list[tuple[Path, Path]] = []

    def _fake_supervisor(
        *,
        config_path: Path,
        workspace: Path,
        readiness_timeout_s: float = 15.0,
    ) -> int:
        supervisor_calls.append((config_path, workspace))
        return 0

    def _fake_migration(config_path: Path, workspace: Path) -> MigrationOutcome:
        return MigrationOutcome(state="current")

    monkeypatch.setattr("agent.supervisor.run_supervisor", _fake_supervisor)
    monkeypatch.setattr("agent.migrations.migrate_installation", _fake_migration)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--config", "missing.json", "--workspace", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")
    assert exc.value.code == 0
    assert supervisor_calls == [(Path("missing.json"), tmp_path)]

    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    supervisor_calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--workspace", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("main", run_name="__main__")
    assert exc.value.code == 0
    assert supervisor_calls == [(Path("config.toml"), tmp_path)]
