import asyncio
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
    _validated_timezone,
    load_config,
    resolve_app_server_endpoint,
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


class _FakeChatServer:
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
        "app_server": {
            "listen": str(socket_path),
        },
    }
    path.write_text("\n".join(_dump_toml(payload)).strip() + "\n", encoding="utf-8")


def test_load_config_keeps_internal_max_iterations_default(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "openai"

[llm.main]
model = "test-model"
api_key = "test-key"

[agent]
system_prompt = "test"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert cfg.max_iterations == 10


def test_load_config_defaults_memory_window_and_optimizer_interval(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "openai"

[llm.main]
model = "test-model"
api_key = "test-key"

[agent]
system_prompt = "test"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert cfg.memory_window == 40
    assert cfg.memory_optimizer_interval_seconds == 64800


def test_config_load_resolves_secrets_from_explicit_workspace(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    for workspace, api_key, token in (
        (first_workspace, "first-key", "first-token"),
        (second_workspace, "second-key", "second-token"),
    ):
        memory = workspace / "memory"
        memory.mkdir(parents=True)
        (memory / "API_KEY").write_text(api_key, encoding="utf-8")
        (memory / "TG_TOKEN").write_text(token, encoding="utf-8")
    config_path.write_text(
        """
[llm]
provider = "openai"

[llm.main]
model = "test-model"
api_key = "${API_KEY}"

[agent]
system_prompt = "test"

[channels.telegram]
token = "${TG_TOKEN}"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    first = load_config(config_path, workspace=first_workspace)
    second = Config.load(config_path, workspace=second_workspace)

    assert first.api_key == "first-key"
    assert first.channels.telegram is not None
    assert first.channels.telegram.token == "first-token"
    assert second.api_key == "second-key"
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

    assert main._workspace_from_args([], config_path) == (
        tmp_path / "configured-workspace"
    ).resolve()

    environment_workspace = tmp_path / "environment-workspace"
    monkeypatch.setenv("AKASHIC_WORKSPACE", str(environment_workspace))
    assert main._workspace_from_args(
        [],
        config_path,
    ) == environment_workspace.resolve()

    cli_workspace = tmp_path / "cli-workspace"
    assert main._workspace_from_args(
        ["--workspace", str(cli_workspace)],
        config_path,
    ) == cli_workspace.resolve()


def test_workspace_selection_uses_default_only_for_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "missing.toml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AKASHIC_WORKSPACE", raising=False)

    assert main._workspace_from_args(
        [],
        config_path,
        allow_default=True,
    ) == (tmp_path / ".akashic" / "workspace").resolve()
    with pytest.raises(ValueError, match="找不到配置文件"):
        main._workspace_from_args([], config_path)


@pytest.mark.asyncio
async def test_inspect_modules_closes_all_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[str] = []

    class _MemoryRuntime:
        async def aclose(self) -> None:
            closed.append("memory")

    class _Runtime:
        memory_runtime = _MemoryRuntime()

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

    assert closed == ["core", "memory", "http"]


@pytest.mark.parametrize(
    ("field", "snippet"),
    [
        ("llm", 'llm = []'),
        ("llm.main", '[llm]\nmain = []'),
        ("agent.context", '[agent]\ncontext = []'),
        ("channels", 'channels = []'),
        ("memory.embedding", '[memory]\nembedding = []'),
        ("extra_body", 'extra_body = []'),
    ],
)
def test_load_config_rejects_non_table_sections(
    tmp_path: Path,
    field: str,
    snippet: str,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "provider = \"openai\"\nmodel = \"test-model\"\n"
        f"{snippet}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="必须是 TOML table") as exc_info:
        load_config(config_path, workspace=tmp_path)

    assert field in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "snippet"),
    [
        ("agent.dev_mode", '[agent]\ndev_mode = "false"'),
        ("llm.main.enable_thinking", '[llm.main]\nenable_thinking = "true"'),
        ("channels.chat.enabled", '[channels.chat]\nenabled = "false"'),
        ("memory.enabled", '[memory]\nenabled = "false"'),
    ],
)
def test_load_config_rejects_string_booleans(
    tmp_path: Path,
    field: str,
    snippet: str,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'provider = "openai"\nmodel = "test-model"\n' + snippet + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        load_config(config_path, workspace=tmp_path)


@pytest.mark.parametrize("tz_name", ["", "Not/AZone"])
def test_validated_timezone_rejects_invalid_names(tz_name: str):
    with pytest.raises(ValueError, match="IANA"):
        _validated_timezone(tz_name, enabled=True)


@pytest.mark.asyncio
async def test_serve_smoke_loads_config_and_runs_shutdown(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    socket_path = tmp_path / "akashic.sock"
    _write_config(config_path, socket_path)

    original_build_core_runtime = bootstrap_app.build_core_runtime
    observed: dict[str, object] = {}

    def _patched_build_core_runtime(config, workspace, http_resources, **kwargs):
        runtime = original_build_core_runtime(config, workspace, http_resources, **kwargs)
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
        monkeypatch.setattr(
            bootstrap_app,
            "PassiveMessageWorker",
            lambda *args, **kwargs: types.SimpleNamespace(run=_agent_loop_run),
        )
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
    monkeypatch.setattr(
        bootstrap_app, "build_chat_server", lambda **_: _FakeChatServer()
    )

    class _FakePluginJobRuntime:
        def __init__(self, **_: object) -> None:
            pass

        async def run(self) -> None:
            return None

        def stop(self) -> None:
            return None

        async def wait_stopped(self) -> None:
            return None

    monkeypatch.setattr(bootstrap_app, "PluginJobRuntime", _FakePluginJobRuntime)
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)

    await main.serve(str(config_path), tmp_path)

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
async def test_app_runtime_supervises_workspace_mcp_watcher_failure(tmp_path):
    runtime = bootstrap_app.AppRuntime(cast(Any, object()), tmp_path)
    primary_stopped = asyncio.Event()
    core_stopped = asyncio.Event()

    async def _failed_watcher() -> None:
        raise KeyError("unexpected watcher failure")

    async def _primary() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            primary_stopped.set()

    class _Core:
        async def stop(self) -> None:
            core_stopped.set()

    async def _start() -> None:
        runtime.core = cast(Any, _Core())
        runtime.workspace_mcp_watcher_task = asyncio.create_task(
            _failed_watcher()
        )
        runtime.tasks = [_primary()]

    runtime.start = _start  # type: ignore[method-assign]
    with pytest.raises(KeyError, match="unexpected watcher failure"):
        await runtime.run()

    assert primary_stopped.is_set()
    assert core_stopped.is_set()
    assert runtime.workspace_mcp_watcher_task is None


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
        scheduler=object(),
        provider=object(),
        light_provider=None,
        memory_runtime=types.SimpleNamespace(aclose=bootstrap_app._noop_async),
        presence=object(),
        peer_process_manager=None,
        peer_poller=None,
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
    assert 'input_modalities = ["text"]' in config_text
    assert "[llm.runtimes.qwen_vl]" in config_text
    assert 'model = "qwen-vl-plus"' in config_text
    assert "[channels.chat]" in config_text
    assert "port = 6322" in config_text
    assert '[runtime]\n' in config_text
    assert 'workspace = "~/.akashic/workspace"' in config_text
    assert any("http://127.0.0.1:6322" in step for step in summary.next_steps)
    assert (workspace / "sessions.db").exists()
    assert (workspace / "observe").is_dir()
    assert (workspace / "memory" / "consolidation_writes.db").exists()
    assert not (workspace / "memory" / "journal").exists()
    assert (workspace / "memory" / "memory2.db").exists()
    assert "Proactive Context" in (
        workspace / "PROACTIVE_CONTEXT.md"
    ).read_text(encoding="utf-8")
    assert (workspace / "mcp" / "servers").is_dir()
    assert not (workspace / "proactive_sources.json").exists()
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
async def test_start_channels_wires_telegram_qq_and_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    starts: list[str] = []
    registrations: list[tuple[str, list[str]]] = []
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
            ctx.push_tool.register_channel(
                self.name,
                text=self.send,
                stream_text=self.send_stream,
                file=self.send_file,
                image=self.send_image,
            )

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
            ctx.push_tool.register_channel(
                self.name,
                text=self.send,
                file=self.send_file,
                image=self.send_image,
            )

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
            mobile_catalogs.append(ctx.mobile_bot_commands)
            ctx.push_tool.register_channel(self.name, text=self.send)

        async def stop(self) -> None:
            starts.append("plugin.stop")

        async def send(self, *args: object, **kwargs: object) -> None:
            return None

    fake_telegram.TelegramChannel = _TelegramChannel  # type: ignore[attr-defined]
    fake_qq.QQChannel = _QQChannel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "infra.channels.telegram_channel", fake_telegram)
    monkeypatch.setitem(sys.modules, "infra.channels.qq_channel", fake_qq)

    class _PushTool:
        def register_channel(self, name: str, **kwargs: object) -> None:
            registrations.append((name, sorted(kwargs)))

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
        telegram_bot_commands=[("telegram_only", "仅 Telegram")],
        mobile_bot_commands=[("mobile_only", "仅 mobile")],
        interrupt_controller=cast(Any, controller),
        plugin_channels=[cast(Any, _PluginChannel())],
    )
    try:
        await host.start_all()

        telegram, qq, plugin = host.channels
        assert starts == ["telegram", "qq", "plugin"]
        assert registrations == [
            ("telegram", ["file", "image", "stream_text", "text"]),
            ("qq", ["file", "image", "text"]),
            ("plugin", ["text"]),
        ]
        assert telegram.kwargs["event_bus"] is event_bus
        assert telegram.kwargs["interrupt_controller"] is controller
        assert telegram.kwargs["bot_commands"] == [("telegram_only", "仅 Telegram")]
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
        provider="openai",
        model="m",
        api_key="k",
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
