import asyncio
import hashlib
import json
import subprocess
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
    DEFAULT_SOCKET,
    QQChannelConfig,
    QQGroupConfig,
    TelegramChannelConfig,
    load_config,
    resolve_app_server_endpoint,
)
from agent.persona import reset_veda
from bus.event_bus import EventBus
from core.net.http import SharedHttpResources


class _FakeDashboardServer:
    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


class _FakeChatServer:
    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


def test_plugin_uninstall_passes_active_turn_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[runtime]\nworkspace='workspace'\n", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        main.Config,
        "load",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            app_server=types.SimpleNamespace(listen="runtime.sock")
        ),
    )
    monkeypatch.setattr(
        main, "resolve_app_server_endpoint", lambda *_args: "runtime.sock"
    )

    async def request(
        _endpoint: str,
        plugin_id: str,
        _workspace: Path,
        *,
        owner_turn_id: str,
    ) -> dict[str, object]:
        calls.append(owner_turn_id)
        return {
            "pluginId": plugin_id,
            "publicationState": "pending_turn_end",
        }

    monkeypatch.setattr(main, "_request_plugin_uninstall", request)
    monkeypatch.delenv("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", raising=False)
    outside = main._uninstall_via_runtime(
        str(config_path),
        "context_pressure@github",
        tmp_path / "workspace",
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", "turn:owner")
    inside = main._uninstall_via_runtime(
        str(config_path),
        "context_pressure@github",
        tmp_path / "workspace",
    )

    assert calls == ["", "turn:owner"]
    assert outside["publicationState"] == "pending_turn_end"
    assert inside["publicationState"] == "pending_turn_end"


def test_agent_turn_rejects_internal_plugin_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", "turn:owner")

    with pytest.raises(ValueError, match="Core 内部维护动作"):
        main._reject_agent_internal_plugin_action("plugin-promote")

    main._reject_agent_internal_plugin_action("plugin-install")
    main._reject_agent_internal_plugin_action("plugin-uninstall")
    main._reject_agent_internal_plugin_action("plugin-revert")


def test_app_runtime_does_not_own_public_web_listener(tmp_path: Path) -> None:
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)

    assert not hasattr(runtime, "dashboard_host")
    assert not hasattr(runtime, "dashboard_port")


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
        "agent": {
            "system_prompt": "test system prompt",
            "max_iterations": 2,
            "plugins": {"disabled_builtin": ["akasha", "wake"]},
        },
        "app_server": {
            "listen": str(socket_path),
        },
    }
    path.write_text("\n".join(_dump_toml(payload)).strip() + "\n", encoding="utf-8")


def test_load_config_keeps_internal_max_iterations_default(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
system_prompt = "test"
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert cfg.max_iterations == 10


def test_load_config_has_no_pending_optimizer_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
system_prompt = "test"
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert not hasattr(cfg, "memory_window")
    assert not hasattr(cfg, "context_compaction")
    assert not hasattr(cfg, "memory_optimizer_enabled")
    assert not hasattr(cfg, "memory_optimizer_interval_seconds")


def test_load_config_rejects_retired_pending_optimizer_keys(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[agent]\nsystem_prompt = "test"\n\n'
        "[agent.maintenance]\nmemory_optimizer_enabled = false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PENDING/MemoryOptimizer 已移除"):
        _ = load_config(config_path, workspace=tmp_path)


def test_load_config_projects_generic_disabled_builtin_plugins(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
system_prompt = "test"

[agent.plugins]
disabled_builtin = ["subagent", "scheduler"]
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert cfg.disabled_builtin_plugins == frozenset({"subagent", "scheduler"})


def test_load_config_rejects_removed_spawn_switch(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent.tools]
spawn_enabled = false
""".strip() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="spawn_enabled 已移除"):
        load_config(config_path, workspace=tmp_path)


@pytest.mark.parametrize("body", ["[proactive]\n", "[proactive]\nenabled = false\n"])
def test_load_config_rejects_retired_proactive_before_workspace_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
) -> None:
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    config_path.write_text(body, encoding="utf-8")

    def reject_store_access(_: Path):
        raise AssertionError("legacy config must fail before opening the model store")

    monkeypatch.setattr(
        "agent.model_runtime.store.ModelRegistryStore.for_workspace",
        reject_store_access,
    )

    with pytest.raises(ValueError, match=r"\[proactive\] 已移除"):
        load_config(config_path, workspace=workspace)

    assert not workspace.exists()


def test_config_load_resolves_channel_secret_from_explicit_workspace(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    for workspace, token in (
        (first_workspace, "first-token"),
        (second_workspace, "second-token"),
    ):
        memory = workspace / "memory"
        memory.mkdir(parents=True)
        (memory / "TG_TOKEN").write_text(token, encoding="utf-8")
    config_path.write_text(
        """
[agent]
system_prompt = "test"

[channels.telegram]
token = "${TG_TOKEN}"
""".strip() + "\n",
        encoding="utf-8",
    )

    first = load_config(config_path, workspace=first_workspace)
    second = Config.load(config_path, workspace=second_workspace)

    assert first.channels.telegram is not None
    assert first.channels.telegram.token == "first-token"
    assert second.channels.telegram is not None
    assert second.channels.telegram.token == "second-token"


def test_default_socket_is_derived_from_workspace(tmp_path: Path) -> None:
    endpoint = resolve_app_server_endpoint(DEFAULT_SOCKET, tmp_path)

    if sys.platform == "win32":
        assert endpoint.startswith("127.0.0.1:")
    else:
        assert endpoint == str(tmp_path / "akashic.sock")


def test_main_help_does_not_start_runtime() -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "用法: python main.py" in result.stdout
    assert "Agent 已启动" not in result.stdout


def test_workspace_selection_prefers_cli_then_env_then_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[runtime]\nworkspace = "~/configured-workspace"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AKASHIC_WORKSPACE", raising=False)

    assert (
        main._workspace_from_args([], config_path)
        == (tmp_path / "configured-workspace").resolve()
    )

    environment_workspace = tmp_path / "environment-workspace"
    monkeypatch.setenv("AKASHIC_WORKSPACE", str(environment_workspace))
    assert (
        main._workspace_from_args(
            [],
            config_path,
        )
        == environment_workspace.resolve()
    )

    cli_workspace = tmp_path / "cli-workspace"
    assert (
        main._workspace_from_args(
            ["--workspace", str(cli_workspace)],
            config_path,
        )
        == cli_workspace.resolve()
    )


def test_workspace_selection_uses_default_only_for_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "missing.toml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AKASHIC_WORKSPACE", raising=False)

    assert (
        main._workspace_from_args(
            [],
            config_path,
            allow_default=True,
        )
        == (tmp_path / ".akashic" / "workspace").resolve()
    )
    with pytest.raises(ValueError, match="找不到配置文件"):
        main._workspace_from_args([], config_path)


@pytest.mark.asyncio
async def test_inspect_modules_closes_all_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[str] = []

    class _Runtime:
        async def inspect_modules(self) -> str:
            return "graph"

        async def stop(self) -> None:
            closed.append("core")

    class _Resources:
        async def aclose(self) -> None:
            closed.append("http")

    monkeypatch.setattr(
        main.Config,
        "load",
        lambda _path, **_kwargs: object(),
    )
    monkeypatch.setattr(main, "SharedHttpResources", _Resources)
    monkeypatch.setattr(
        "bootstrap.tools.build_core_runtime",
        lambda *_args: _Runtime(),
    )

    await main.inspect_modules("config.toml", tmp_path)

    assert closed == ["core", "http"]


@pytest.mark.parametrize(
    ("field", "snippet"),
    [
        ("agent.context", "[agent]\ncontext = []"),
        ("channels", "channels = []"),
    ],
)
def test_load_config_rejects_non_table_sections(
    tmp_path: Path,
    field: str,
    snippet: str,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(f"{snippet}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="必须是 TOML table") as exc_info:
        load_config(config_path, workspace=tmp_path)

    assert field in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "snippet"),
    [
        ("agent.dev_mode", '[agent]\ndev_mode = "false"'),
        ("channels.chat.enabled", '[channels.chat]\nenabled = "false"'),
    ],
)
def test_load_config_rejects_string_booleans(
    tmp_path: Path,
    field: str,
    snippet: str,
):
    config_path = tmp_path / "config.toml"
    contents = f"{snippet}\n"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        load_config(config_path, workspace=tmp_path)


@pytest.mark.asyncio
async def test_serve_smoke_loads_config_and_runs_shutdown(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    socket_path = tmp_path / "akashic.sock"
    _write_config(config_path, socket_path)
    _ = reset_veda(tmp_path)

    original_build_core_runtime = bootstrap_app.build_core_runtime
    observed: dict[str, object] = {}

    def _patched_build_core_runtime(config, workspace, http_resources, **kwargs):
        runtime = original_build_core_runtime(
            config, workspace, http_resources, **kwargs
        )
        agent_loop = runtime.loop
        bus = runtime.bus

        async def _agent_loop_run():
            return None

        agent_loop.run = _agent_loop_run  # type: ignore[assignment]
        monkeypatch.setattr(
            bootstrap_app,
            "PassiveMessageWorker",
            lambda *args, **kwargs: types.SimpleNamespace(run=_agent_loop_run),
        )
        monkeypatch.setattr(bus, "dispatch_outbound", _agent_loop_run)
        assert runtime.plugin_manager is not None
        monkeypatch.setattr(
            runtime.plugin_manager,
            "run_runtime_services",
            _agent_loop_run,
        )
        observed["bus"] = bus
        observed["http_resources"] = http_resources
        return runtime

    monkeypatch.setattr(
        bootstrap_app, "build_core_runtime", _patched_build_core_runtime
    )
    monkeypatch.setattr(
        bootstrap_app, "build_dashboard_server", lambda **_: _FakeDashboardServer()
    )
    monkeypatch.setattr(
        bootstrap_app, "build_chat_server", lambda **_: _FakeChatServer()
    )

    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)

    await main.serve(str(config_path), tmp_path)

    assert socket_path.exists() is False
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


@pytest.mark.asyncio
async def test_run_cleanup_steps_continues_after_cancellation():
    calls: list[str] = []

    async def _cancel() -> None:
        calls.append("cancel")
        raise asyncio.CancelledError

    async def _cleanup() -> None:
        calls.append("cleanup")

    with pytest.raises(asyncio.CancelledError):
        await bootstrap_app._run_cleanup_steps(
            ("cancel", _cancel),
            ("cleanup", _cleanup),
        )

    assert calls == ["cancel", "cleanup"]


@pytest.mark.asyncio
async def test_shutdown_stops_mobile_channel_before_closing_gateway_storage(tmp_path):
    events: list[str] = []

    class Gateway:
        closed = False

        def close(self) -> None:
            self.closed = True
            events.append("gateway.close")

    gateway = Gateway()

    class ConversationRuntime:
        async def shutdown(self) -> None:
            events.append("conversation.shutdown")

    class ChannelHost:
        async def stop_all(self) -> None:
            assert gateway.closed is False
            events.append("channels.stop")

    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    runtime.mobile_gateway_runtime = gateway
    runtime.conversation_runtime = cast(Any, ConversationRuntime())
    runtime.channel_host = cast(Any, ChannelHost())

    await runtime.shutdown()

    assert events == ["conversation.shutdown", "channels.stop", "gateway.close"]


@pytest.mark.asyncio
async def test_app_runtime_run_stops_primary_tasks_after_server_failure(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    runtime.dashboard_server = _FakeDashboardServer()

    async def _failed_server() -> None:
        raise RuntimeError("dashboard crashed")

    stopped = asyncio.Event()

    async def _primary() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def _start() -> None:
        runtime.dashboard_task = asyncio.create_task(_failed_server())
        runtime.tasks = [_primary()]

    runtime.start = _start  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="dashboard crashed"):
        await runtime.run()

    assert stopped.is_set()
    assert runtime.http_resources.closed is True
    assert runtime.dashboard_task is None


@pytest.mark.asyncio
async def test_app_runtime_run_stops_primary_tasks_after_server_return(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    runtime.dashboard_server = _FakeDashboardServer()
    stopped = asyncio.Event()

    async def _returned_server() -> None:
        return None

    async def _primary() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def _start() -> None:
        runtime.dashboard_task = asyncio.create_task(_returned_server())
        runtime.tasks = [_primary()]

    runtime.start = _start  # type: ignore[method-assign]

    await runtime.run()

    assert stopped.is_set()
    assert runtime.http_resources.closed is True
    assert runtime.dashboard_task is None


@pytest.mark.asyncio
async def test_app_runtime_run_preserves_server_error_when_shutdown_fails(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    runtime.dashboard_server = _FakeDashboardServer()
    server_error = RuntimeError("dashboard crashed")
    shutdown_error = RuntimeError("core stop failed")

    async def _failed_server() -> None:
        raise server_error

    async def _primary() -> None:
        await asyncio.Event().wait()

    class _Core:
        async def stop(self) -> None:
            raise shutdown_error

    async def _start() -> None:
        runtime.core = cast(Any, _Core())
        runtime.dashboard_task = asyncio.create_task(_failed_server())
        runtime.tasks = [_primary()]

    runtime.start = _start  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="dashboard crashed") as caught:
        await runtime.run()

    assert caught.value is server_error
    assert caught.value.__cause__ is shutdown_error
    assert runtime.dashboard_task is None


@pytest.mark.asyncio
async def test_app_runtime_run_stops_other_tasks_after_primary_failure(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    stopped = asyncio.Event()

    async def _failed() -> None:
        raise RuntimeError("primary task failed")

    async def _other() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def _start() -> None:
        runtime.tasks = [_failed(), _other()]

    runtime.start = _start  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="primary task failed"):
        await runtime.run()

    assert stopped.is_set()


@pytest.mark.asyncio
async def test_app_runtime_run_waits_for_primary_sibling_cleanup(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def _failed() -> None:
        raise RuntimeError("primary task failed")

    async def _other() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await asyncio.sleep(0)
            await cleanup_release.wait()
            cleanup_finished.set()

    async def _start() -> None:
        runtime.tasks = [_failed(), _other()]

    runtime.start = _start  # type: ignore[method-assign]
    running = asyncio.create_task(runtime.run())
    await cleanup_started.wait()

    assert not running.done()
    cleanup_release.set()

    with pytest.raises(RuntimeError, match="primary task failed"):
        await running

    assert cleanup_finished.is_set()


@pytest.mark.asyncio
async def test_app_runtime_run_rethrows_external_cancellation_after_shutdown(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    stopped = asyncio.Event()
    shutdown_calls: list[str] = []

    async def _primary() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    class _Core:
        async def stop(self) -> None:
            shutdown_calls.append("core.stop")

    async def _start() -> None:
        runtime.core = cast(Any, _Core())
        runtime.tasks = [_primary()]

    runtime.start = _start  # type: ignore[method-assign]
    running = asyncio.create_task(runtime.run())
    await asyncio.sleep(0)
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running

    assert stopped.is_set()
    assert shutdown_calls == ["core.stop"]
    assert runtime.http_resources.closed is True


@pytest.mark.asyncio
async def test_primary_task_cancellation_waits_for_async_finally() -> None:
    cleanup_finished = asyncio.Event()

    async def _primary() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleanup_finished.set()

    child = asyncio.create_task(_primary())
    supervisor = asyncio.create_task(bootstrap_app._run_primary_tasks([child]))
    await asyncio.sleep(0)
    supervisor.cancel()

    with pytest.raises(asyncio.CancelledError):
        await supervisor

    assert cleanup_finished.is_set()


@pytest.mark.asyncio
async def test_app_runtime_task_cleanup_exposes_non_cancel_failure(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)

    async def _failed() -> None:
        raise RuntimeError("primary task failed")

    failed = asyncio.create_task(_failed())
    await asyncio.sleep(0)
    runtime._runtime_tasks.add(failed)

    with pytest.raises(RuntimeError, match="primary task failed"):
        await runtime._cancel_runtime_tasks()

    assert not runtime._runtime_tasks


@pytest.mark.asyncio
async def test_app_runtime_candidate_cleanup_exposes_non_cancel_failure(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)

    async def _failed() -> None:
        raise RuntimeError("candidate task failed")

    failed = asyncio.create_task(_failed())
    await asyncio.sleep(0)
    runtime._plugin_candidate_tasks.add(failed)

    with pytest.raises(RuntimeError, match="candidate task failed"):
        await runtime._cancel_plugin_candidate_tasks()

    assert not runtime._plugin_candidate_tasks


@pytest.mark.asyncio
async def test_app_runtime_start_preserves_startup_error_when_rollback_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    startup_error = RuntimeError("startup failed")
    rollback_error = RuntimeError("rollback failed")

    async def _start() -> None:
        raise startup_error

    async def _stop() -> None:
        raise rollback_error

    core = types.SimpleNamespace(
        loop=object(),
        bus=object(),
        event_bus=EventBus(),
        tools=object(),
        push_tool=object(),
        session_manager=object(),
        provider=object(),
        light_provider=None,
        presence=object(),
        plugin_manager=None,
        start=_start,
        stop=_stop,
    )
    monkeypatch.setattr(bootstrap_app, "build_core_runtime", lambda *_, **__: core)
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)

    with pytest.raises(RuntimeError, match="startup failed") as caught:
        await runtime.start()

    assert caught.value is startup_error
    assert caught.value.__cause__ is rollback_error


@pytest.mark.asyncio
async def test_app_runtime_shutdown_cleans_up_after_server_failure(tmp_path):
    calls: list[str] = []

    async def _failed_server() -> None:
        raise RuntimeError("dashboard crashed")

    class _Core:
        async def stop(self) -> None:
            calls.append("core.stop")

    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    runtime.core = cast(Any, _Core())
    runtime.dashboard_server = _FakeDashboardServer()
    runtime.dashboard_task = asyncio.create_task(_failed_server())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="dashboard crashed"):
        await runtime.shutdown()

    assert calls == ["core.stop"]
    assert runtime.dashboard_server.should_exit is True
    assert runtime.http_resources.closed is True


def test_init_workspace_creates_expected_assets(tmp_path):
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    summary = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )

    assert config_path.exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert "[llm]" not in config_text
    assert "[memory]" not in config_text
    assert "2236 的“模型”页" in config_text
    assert "[channels.chat]" in config_text
    assert "6322" not in config_text
    assert "[runtime]\n" in config_text
    assert 'workspace = "~/.akashic/workspace"' in config_text
    assert any("http://127.0.0.1:2236" in step for step in summary.next_steps)
    assert any("默认聊天模型" in step for step in summary.next_steps)
    assert any("embedding 模型" in step for step in summary.next_steps)
    assert not any("llm.main" in step for step in summary.next_steps)
    assert not any("memory.embedding" in step for step in summary.next_steps)
    assert (workspace / "sessions.db").exists()
    assert (workspace / "observe").is_dir()
    assert not (workspace / "memory" / "consolidation_writes.db").exists()
    assert not (workspace / "memory" / "journal").exists()
    assert not (workspace / "memory" / "memory2.db").exists()
    assert "你是 Akashic" in (workspace / "memory" / "VEDA.md").read_text(
        encoding="utf-8"
    )
    assert not (workspace / "PROACTIVE_CONTEXT.md").exists()
    assert (workspace / "mcp" / "servers").is_dir()
    assert not (workspace / "proactive_sources.json").exists()
    assert not (workspace / "proactive.db").exists()
    assert (workspace / "skills").is_dir()
    assert (workspace / "drift" / "skills").is_dir()
    assert any(path == config_path for path in summary.created)


def test_init_workspace_preserves_legacy_proactive_assets(tmp_path):
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context_path = workspace / "PROACTIVE_CONTEXT.md"
    database_path = workspace / "proactive.db"
    context_bytes = b"legacy proactive context\n"
    database_bytes = b"legacy proactive database\x00"
    context_path.write_bytes(context_bytes)
    database_path.write_bytes(database_bytes)
    before = {
        path: (path.stat().st_ino, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (context_path, database_path)
    }

    summary = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
        force=True,
    )

    assert context_path.read_bytes() == context_bytes
    assert database_path.read_bytes() == database_bytes
    assert {
        path: (path.stat().st_ino, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (context_path, database_path)
    } == before
    assert context_path not in summary.created + summary.overwritten
    assert database_path not in summary.created + summary.overwritten


def test_init_workspace_leaves_markdown_profiles_to_plugin(tmp_path):
    config_path = tmp_path / "config.toml"
    workspace = tmp_path / "workspace"

    workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )
    self_path = workspace / "memory" / "SELF.md"
    veda_path = workspace / "memory" / "VEDA.md"
    assert not self_path.exists()
    veda_path.write_text("custom veda\n", encoding="utf-8")

    summary_skip = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )
    assert not self_path.exists()
    assert veda_path.read_text(encoding="utf-8") == "custom veda\n"
    assert any(path == veda_path for path in summary_skip.skipped)

    summary_force = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
        force=True,
    )
    assert not self_path.exists()
    assert veda_path.read_text(encoding="utf-8") == "custom veda\n"
    assert self_path not in summary_force.created + summary_force.overwritten
    assert any(path == veda_path for path in summary_force.skipped)


@pytest.mark.asyncio
async def test_start_channels_wires_telegram_qq_and_extra_channel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    starts: list[str] = []
    attachment_roots: list[Path] = []
    mobile_catalogs: list[list[tuple[str, str]]] = []
    fake_telegram = types.ModuleType("infra.channels.telegram_channel")
    fake_qq = types.ModuleType("infra.channels.qq_channel")

    class _TelegramChannel:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.name = str(kwargs.get("channel_name") or "telegram")

        async def start(self, ctx: Any) -> None:
            starts.append("telegram")

        async def stop(self) -> None:
            starts.append("telegram.stop")

        async def send(self, *args: object, **kwargs: object) -> None:
            return None

        async def send_stream(self, *args: object, **kwargs: object) -> None:
            return None

        async def send_file(self, *args: object, **kwargs: object) -> None:
            return None

        async def send_image(self, *args: object, **kwargs: object) -> None:
            return None

    class _QQChannel:
        name = "qq"

        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        async def start(self, ctx: Any) -> None:
            starts.append("qq")

        async def stop(self) -> None:
            starts.append("qq.stop")

        async def send(self, *args: object, **kwargs: object) -> None:
            return None

        async def send_file(self, *args: object, **kwargs: object) -> None:
            return None

        async def send_image(self, *args: object, **kwargs: object) -> None:
            return None

    class _PluginChannel:
        name = "plugin"

        async def start(self, ctx: Any) -> None:
            starts.append("plugin")
            attachment_roots.append(ctx.attachment_store.root)
            provider = ctx.command_catalog_provider
            mobile_catalogs.append([] if provider is None else list(provider()))

        async def stop(self) -> None:
            starts.append("plugin.stop")

        async def send(self, *args: object, **kwargs: object) -> None:
            return None

    fake_telegram.TelegramChannel = _TelegramChannel  # type: ignore[attr-defined]
    fake_qq.QQChannel = _QQChannel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra.channels.telegram_channel", fake_telegram)
    monkeypatch.setitem(sys.modules, "infra.channels.qq_channel", fake_qq)

    class _PushTool:
        pass

    config = Config(
        system_prompt="s",
        channels=ChannelsConfig(
            telegram=TelegramChannelConfig(token="tg-token", allow_from=["1"]),
            qq=QQChannelConfig(
                bot_uin="10001",
                allow_from=["2"],
                groups=[QQGroupConfig(group_id="3")],
            ),
        ),
    )
    resources = SharedHttpResources()
    event_bus = EventBus()
    controller = object()
    host = await start_channels(
        config,
        bus=cast(Any, object()),
        session_manager=cast(Any, types.SimpleNamespace(workspace=tmp_path)),
        push_tool=cast(Any, _PushTool()),
        http_resources=resources,
        event_bus=event_bus,
        telegram_command_catalog_provider=lambda: (("telegram_only", "仅 Telegram"),),
        mobile_command_catalog_provider=lambda: (("mobile_only", "仅 mobile"),),
        interrupt_controller=cast(Any, controller),
        extra_channels=[cast(Any, _PluginChannel())],
    )
    try:
        await host.start_all()

        telegram, qq, plugin = host.channels
        assert starts == ["telegram", "qq", "plugin"]
        assert telegram.kwargs["event_bus"] is event_bus
        assert telegram.kwargs["interrupt_controller"] is controller
        assert telegram.kwargs["command_catalog_provider"]() == (
            ("telegram_only", "仅 Telegram"),
        )
        assert qq.kwargs["interrupt_controller"] is controller
        assert plugin.name == "plugin"
        assert attachment_roots == [tmp_path / "uploads"]
        assert mobile_catalogs == [[("mobile_only", "仅 mobile")]]
    finally:
        await host.stop_all()
        await resources.aclose()


@pytest.mark.asyncio
async def test_start_channels_skips_unfilled_optional_channels(tmp_path: Path) -> None:
    config = Config(
        system_prompt="s",
        channels=ChannelsConfig(telegram=None, qq=None),
    )
    resources = SharedHttpResources()
    try:
        host = await start_channels(
            config,
            bus=cast(Any, object()),
            session_manager=cast(Any, types.SimpleNamespace(workspace=tmp_path)),
            push_tool=cast(Any, object()),
            http_resources=resources,
            event_bus=EventBus(),
        )
    finally:
        await resources.aclose()

    assert host.channels == []
