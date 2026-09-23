import asyncio
import ast
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import types
from pathlib import Path
from typing import Any, Required, TypedDict, Unpack, cast
from collections.abc import Callable, Iterable

import pytest
import uvicorn

import main
from bootstrap import app as bootstrap_app
from bootstrap import app_server as bootstrap_app_server
from bootstrap import init_workspace as workspace_init
from bootstrap import tools as bootstrap_tools
from bootstrap.web_runtime import dashboard_socket_path
from bootstrap.workspace_lock import WorkspaceInstanceLock
from agent.config import (
    Config,
    DEFAULT_SOCKET,
    load_config,
    resolve_app_server_endpoint,
)
from plugins.prompt.persona import reset_veda
from agent.plugin_composition import (
    CompositionError,
    Context,
    FiberState,
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    ServiceKey,
)
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from agent.restart import RestartGate
from agent.supervisor import RESTART_EXIT_CODE
from bus.event_bus import EventBus
from core.net.http import SharedHttpResources
from tests.fixtures.plugin_workspace import initialize_plugin_workspace


class _CoreKwargs(TypedDict, total=False):
    restart_gate: RestartGate | None
    clear_stale_session_admissions: bool
    plugin_dirs: Iterable[Path] | None


class _DashboardKwargs(TypedDict, total=False):
    workspace: Required[Path]
    host: str | None
    port: int | None
    uds: str | None
    plugin_manager: object | None


class _AppKwargs(TypedDict, total=False):
    restart_gate: RestartGate | None
    readiness: bootstrap_app.RuntimeReadiness | None


class _FakeDashboardServer(uvicorn.Server):
    """Expose the only server shutdown flag used by fault injection tests."""

    def __init__(self) -> None:
        self.should_exit = False


def test_plugin_uninstall_uses_runtime_control_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[runtime]\nworkspace='workspace'\n", encoding="utf-8")
    calls: list[tuple[str, str, Path]] = []

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
        workspace: Path,
    ) -> dict[str, object]:
        calls.append((_endpoint, plugin_id, workspace))
        return {
            "plugin_id": plugin_id,
            "state": "accepted",
            "selection_ref": "selection-ref",
        }

    monkeypatch.setattr(main, "_request_plugin_uninstall", request)
    result = main._uninstall_via_runtime(
        str(config_path),
        "context_pressure@github",
        tmp_path / "workspace",
    )

    assert calls == [
        ("runtime.sock", "context_pressure@github", tmp_path / "workspace")
    ]
    assert result == {
        "plugin_id": "context_pressure@github",
        "state": "accepted",
        "selection_ref": "selection-ref",
    }


def test_agent_turn_rejects_internal_plugin_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN", "turn:owner")

    with pytest.raises(ValueError, match="Core 内部维护动作"):
        main._reject_agent_internal_plugin_action("plugin-enable")
    with pytest.raises(ValueError, match="Core 内部维护动作"):
        main._reject_agent_internal_plugin_action("plugin-disable")

    main._reject_agent_internal_plugin_action("plugin-install")
    main._reject_agent_internal_plugin_action("plugin-uninstall")


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
            "plugins": {"disabled_builtin": ["akasha", "wake"]},
        },
        "app_server": {
            "listen": str(socket_path),
        },
    }
    path.write_text("\n".join(_dump_toml(payload)).strip() + "\n", encoding="utf-8")
    _ = workspace_init.init_workspace(config_path=path, workspace=path.parent)
    initialize_plugin_workspace(path.parent)


def _write_cold_start_plugins(source_root: Path) -> Path:
    """Write statically compiled plugin sources for one real cold-start graph."""

    sources = {
        "a-healthy-peer": """
from agent.plugin_composition import Context, RUNTIME_STARTED, RUNTIME_STARTING, ServiceKey

api_version = 3
name = "a-healthy-peer"
version = "1.0.0"
inject = ()
PEER_SERVICE = ServiceKey("cold-start.peer")


async def apply(ctx: Context) -> None:
    state = {"events": [], "effect": 0, "cleanup": 0}
    await ctx.on(RUNTIME_STARTING, lambda _event: state["events"].append("starting"))
    await ctx.on(RUNTIME_STARTED, lambda _event: state["events"].append("started"))

    async def setup():
        state["effect"] += 1

        async def cleanup() -> None:
            state["cleanup"] += 1

        return cleanup

    await ctx.effect(setup, label="cold-start-peer")
    await ctx.provide(PEER_SERVICE, state)
""",
        "b-bad-owner": """
from agent.plugin_composition import Context, ServiceKey

api_version = 3
name = "b-bad-owner"
version = "1.0.0"
inject = ()
BAD_SERVICE = ServiceKey("cold-start.bad")


async def apply(ctx: Context) -> None:
    await ctx.provide(BAD_SERVICE, "bad owner service")
    raise RuntimeError("cold-start bad owner apply failed")
""",
        "c-required-downstream": """
from agent.plugin_composition import Context, ServiceKey

api_version = 3
name = "c-required-downstream"
version = "1.0.0"
BAD_SERVICE = ServiceKey("cold-start.bad")
inject = (BAD_SERVICE,)


async def apply(ctx: Context) -> None:
    _ = ctx.require(BAD_SERVICE)
""",
        "z-catalog-observer": """
from agent.plugin_composition import Context
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG

api_version = 3
name = "z-catalog-observer"
version = "1.0.0"
inject = (RUNTIME_CATALOG,)


async def apply(ctx: Context) -> None:
    ctx.require(RUNTIME_CATALOG)
""",
    }
    source_root.mkdir(parents=True, exist_ok=True)
    for plugin_name, source in sources.items():
        path = source_root / plugin_name / "plugin.py"
        path.parent.mkdir()
        source = source.strip() + "\n"
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")
        path.write_text(source, encoding="utf-8")
    return source_root


def _write_first_null_source_plugins(source_root: Path) -> Path:
    """Write a real host graph with one preflight-invalid source and a peer."""
    sources = {
        "a-first-null-peer": """
from agent.plugin_composition import Context, RUNTIME_STARTED, RUNTIME_STARTING, ServiceKey

api_version = 3
name = "a-first-null-peer"
version = "1.0.0"
inject = ()
PEER_SERVICE = ServiceKey("first-null.peer")


async def apply(ctx: Context) -> None:
    state = {"events": [], "effect": 0, "cleanup": 0}
    await ctx.on(RUNTIME_STARTING, lambda _event: state["events"].append("starting"))
    await ctx.on(RUNTIME_STARTED, lambda _event: state["events"].append("started"))

    async def setup():
        state["effect"] += 1
        async def cleanup() -> None:
            state["cleanup"] += 1
        return cleanup

    await ctx.effect(setup, label="first-null-peer")
    await ctx.provide(PEER_SERVICE, state)
""",
        "c-first-null-downstream": """
from agent.plugin_composition import Context, ServiceKey

api_version = 3
name = "c-first-null-downstream"
version = "1.0.0"
MISSING_SERVICE = ServiceKey("first-null.missing")
inject = (MISSING_SERVICE,)


async def apply(ctx: Context) -> None:
    _ = ctx.require(MISSING_SERVICE)
""",
    }
    source_root.mkdir(parents=True, exist_ok=True)
    for plugin_name, source in sources.items():
        path = source_root / plugin_name / "plugin.py"
        path.parent.mkdir()
        source = source.strip() + "\n"
        tree = ast.parse(source, filename=str(path))
        compile(tree, str(path), "exec")
        path.write_text(source, encoding="utf-8")
    bad = source_root / "b-first-null-bad"
    bad.mkdir()
    (bad / "plugin.py").write_text("def broken(:\n    return 1\n", encoding="utf-8")
    return source_root


def _write_cold_import_plugins(source_root: Path) -> Path:
    """Write a fixed-archive graph whose second boot fails during import."""

    sources = {
        "a-import-peer": """
from agent.plugin_composition import (
    Context,
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    ServiceKey,
)

api_version = 3
name = "a-import-peer"
version = "1.0.0"
inject = ()
PEER_SERVICE = ServiceKey("cold-import.peer")


async def apply(ctx: Context) -> None:
    state = {"cleanup": 0}
    state["events"] = []
    state["effect"] = 0

    await ctx.on(RUNTIME_STARTING, lambda _event: state["events"].append("starting"))
    await ctx.on(RUNTIME_STARTED, lambda _event: state["events"].append("started"))

    async def setup():
        state["effect"] += 1
        async def cleanup() -> None:
            state["cleanup"] += 1

        return cleanup

    await ctx.effect(setup, label="cold-import-peer")
    await ctx.provide(PEER_SERVICE, state)
""",
        "b-import-bad": """
import os
from agent.plugin_composition import Context, ServiceKey

api_version = 3
name = "b-import-bad"
version = "1.0.0"
BAD_SERVICE = ServiceKey("cold-import.bad")
inject = ()
if os.environ.get("COLD_IMPORT_RESTART") == "yes":
    raise ImportError("cold selected archive import blocked")


async def apply(ctx: Context) -> None:
    await ctx.provide(BAD_SERVICE, {"healthy": True})
""",
        "c-import-downstream": """
from agent.plugin_composition import Context, ServiceKey

api_version = 3
name = "c-import-downstream"
version = "1.0.0"
BAD_SERVICE = ServiceKey("cold-import.bad")
inject = (BAD_SERVICE,)


async def apply(ctx: Context) -> None:
    _ = ctx.require(BAD_SERVICE)
""",
        "z-import-catalog": """
from agent.plugin_composition import Context
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG

api_version = 3
name = "z-import-catalog"
version = "1.0.0"
inject = (RUNTIME_CATALOG,)


async def apply(ctx: Context) -> None:
    ctx.require(RUNTIME_CATALOG)
""",
    }
    source_root.mkdir(parents=True, exist_ok=True)
    for plugin_name, source in sources.items():
        entry = source_root / plugin_name / "plugin.py"
        entry.parent.mkdir()
        source = source.strip() + "\n"
        tree = ast.parse(source, filename=str(entry))
        compile(tree, str(entry), "exec")
        entry.write_text(source, encoding="utf-8")
    return source_root


def test_load_config_has_no_legacy_agent_fields(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert not hasattr(cfg, "max_iterations")


@pytest.mark.parametrize(
    "snippet",
    [
        '[agent]\nsystem_prompt = "old"',
        "[agent]\nmax_iterations = 1",
        "[agent.tools]\nsearch_enabled = true",
        "[agent]\ndev_mode = false",
        '[agent.wiring]\ntoolsets = ["meta_common"]',
    ],
)
def test_load_config_rejects_retired_agent_fields(
    tmp_path: Path,
    snippet: str,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(snippet + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Core 配置不支持字段"):
        load_config(config_path, workspace=tmp_path)


def test_load_config_has_no_pending_optimizer_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
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
        "[agent.maintenance]\nmemory_optimizer_enabled = false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Core 配置不支持字段"):
        _ = load_config(config_path, workspace=tmp_path)


def test_load_config_projects_generic_disabled_builtin_plugins(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent.plugins]
disabled_builtin = ["subagent", "scheduler"]
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_config(config_path, workspace=tmp_path)

    assert cfg.disabled_builtin_plugins == frozenset({"subagent", "scheduler"})


def test_runtime_validates_disabled_plugin_ids_only_for_explicit_roots(monkeypatch) -> None:
    from agent.config_models import Config
    from bootstrap.tools import _disabled_builtin_plugins_for_runtime

    monkeypatch.delenv("AKASHIC_WORKLOAD_SOCKET", raising=False)
    repo_plugins = Path(__file__).parents[1] / "plugins"
    existing = frozenset(
        {"akasha", "scheduler", "wake", "compaction", "markdown_memory"}
    )
    assert (
        _disabled_builtin_plugins_for_runtime(
            Config(disabled_builtin_plugins=existing), [repo_plugins]
        )
        == (existing, ())
    )
    assert _disabled_builtin_plugins_for_runtime(
        Config(disabled_builtin_plugins=frozenset({"future-plugin"}))
    ) == (frozenset({"future-plugin"}), ())
    with pytest.raises(ValueError, match="未知内置插件: agent_restart, skills"):
        _disabled_builtin_plugins_for_runtime(
            Config(disabled_builtin_plugins=frozenset({"skills", "agent_restart"})),
            [repo_plugins],
        )


def test_runtime_source_preflight_keeps_healthy_ids_when_content_is_bad(
    tmp_path: Path,
) -> None:
    repo_plugins = tmp_path / "plugins"
    healthy = repo_plugins / "healthy"
    healthy.mkdir(parents=True)
    (healthy / "plugin.py").write_text(
        'name = "healthy"\nversion = "1.0.0"\napi_version = 3\n',
        encoding="utf-8",
    )
    broken = repo_plugins / "broken"
    broken.mkdir()
    (broken / "plugin.py").write_text("this is not Python !!!\n", encoding="utf-8")

    from bootstrap.tools import _disabled_builtin_plugins_for_runtime

    disabled, failures = _disabled_builtin_plugins_for_runtime(
        Config(disabled_builtin_plugins=frozenset({"healthy"})),
        [repo_plugins],
    )

    assert disabled == frozenset({"healthy"})
    assert len(failures) == 1
    assert failures[0].plugin_id is None
    assert failures[0].source_root == broken.resolve()


def test_load_config_rejects_removed_spawn_switch(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent.tools]
spawn_enabled = false
""".strip() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Core 配置不支持字段"):
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

    with pytest.raises(ValueError, match="Core 配置不支持字段"):
        load_config(config_path, workspace=workspace)

    assert not workspace.exists()


def test_config_load_rejects_legacy_channel_owner_after_migration(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[channels.telegram]
token = "legacy-token"
""".strip() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Core 配置不支持字段"):
        load_config(config_path, workspace=tmp_path / "workspace")


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

    with pytest.raises(ValueError, match="Core 配置不支持字段|必须是 TOML table") as exc_info:
        load_config(config_path, workspace=tmp_path)

    assert field in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "snippet"),
    [
        ("agent.dev_mode", '[agent]\ndev_mode = "false"'),
        ("app_server.enabled", '[app_server]\nenabled = "false"'),
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
        observed["bus"] = runtime.bus
        observed["http_resources"] = http_resources
        return runtime

    monkeypatch.setattr(
        bootstrap_app, "build_core_runtime", _patched_build_core_runtime
    )
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def _build_dashboard_server(**kwargs):
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        manager = kwargs["plugin_manager"]
        assert manager.live_root is None
        observed["pre_root_manager_root"] = manager.live_root
        observed["pre_root_routes"] = tuple(server.config.app.routes)

        async def serve_without_network() -> None:
            return None

        monkeypatch.setattr(server, "serve", serve_without_network)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", _build_dashboard_server)
    monkeypatch.setattr(main.Path, "home", lambda: tmp_path)

    await main.serve(str(config_path), tmp_path)

    assert socket_path.exists() is False
    assert "bus" in observed
    assert cast(SharedHttpResources, observed["http_resources"]).closed is True
    server = observed["dashboard_server"]
    assert observed["pre_root_manager_root"] is None
    assert tuple(server.config.app.routes) == observed["pre_root_routes"]
    assert server.config.uds is not None
    assert Path(server.config.uds).resolve() == dashboard_socket_path(tmp_path).resolve()


def _prepare_real_host_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Create an isolated real Config/workspace/plugin-home boundary."""

    monkeypatch.setenv("AKASHIC_EXECUTION_MODE", "local")
    for name in (
        "AKASHIC_EXTRA_PLUGIN_DIRS",
        "AKASHIC_HOST_BRIDGE_SOCKET",
        "AKASHIC_HOST_BRIDGE_TOKEN",
        "AKASHIC_BOOT_ID",
        "AKASHIC_RUNTIME_COMMIT",
        "AKASHIC_HOST_TOOLCHAIN_DIGEST",
    ):
        monkeypatch.delenv(name, raising=False)
    plugin_home = tmp_path / "plugin-home"
    monkeypatch.setattr(bootstrap_tools, "plugins_root", lambda: plugin_home)
    config_path = tmp_path / "config.toml"
    socket_path = tmp_path / "control.sock"
    _write_config(config_path, socket_path)
    return config_path, socket_path


def _capture_real_app_runtime(
    monkeypatch: pytest.MonkeyPatch,
    observed: dict[str, object],
    *,
    plugin_dirs: tuple[Path, ...] = (),
) -> None:
    """Install observers around real AppRuntime-owned Core and lock construction."""

    real_build_core_runtime = bootstrap_app.build_core_runtime

    def build_core_runtime(
        config: Config,
        workspace: Path,
        http_resources: SharedHttpResources,
        **kwargs: Unpack[_CoreKwargs],
    ) -> bootstrap_tools.CoreRuntime:
        kwargs["plugin_dirs"] = plugin_dirs
        core = real_build_core_runtime(
            config,
            workspace,
            http_resources,
            **kwargs,
        )
        observed["core"] = core
        return core

    monkeypatch.setattr(bootstrap_app, "build_core_runtime", build_core_runtime)
    real_lock = bootstrap_app.WorkspaceInstanceLock

    def workspace_lock(workspace: Path) -> WorkspaceInstanceLock:
        lock = real_lock(workspace)
        observed["workspace_lock"] = lock
        return lock

    monkeypatch.setattr(bootstrap_app, "WorkspaceInstanceLock", workspace_lock)


def _assert_real_app_closed(
    observed: dict[str, object],
    tmp_path: Path,
    socket_path: Path,
) -> None:
    """Check physical Core, sockets, HTTP clients, and workspace lock settlement."""

    core = cast(Any, observed["core"])
    runtime = cast(Any, observed["app_runtime"])
    lock = cast(Any, observed["workspace_lock"])
    assert core.bus._closed is True
    assert core.event_bus._closed is True
    assert core.plugin_manager.live_root is None
    assert runtime.http_resources.closed is True
    assert lock._stream is None
    assert socket_path.exists() is False
    assert dashboard_socket_path(tmp_path).exists() is False


@pytest.mark.asyncio
async def test_real_app_first_null_source_subset_keeps_host_and_owner_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Real Core/AppRuntime commits one healthy subset and exposes source diagnostics."""
    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'disabled_builtin = ["akasha", "wake"]',
            "disabled_builtin = []",
        ),
        encoding="utf-8",
    )
    source_root = _write_first_null_source_plugins(tmp_path / "first-null-plugins")
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed, plugin_dirs=(source_root,))
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    commit_calls: list[tuple[tuple[str, ...], str | None]] = []
    from agent.plugins.selection import PluginSelection
    real_commit = PluginSelection.commit

    def observed_commit(
        selection: Any,
        components: tuple[str, ...], *, expected_ref: str | None,
    ) -> str:
        commit_calls.append((components, expected_ref))
        return real_commit(selection, components, expected_ref=expected_ref)

    monkeypatch.setattr(PluginSelection, "commit", observed_commit)
    running: asyncio.Task[None] | None = None
    runner_retrieved = False
    try:
        running = asyncio.create_task(runtime.run(), name="real-app-first-null-subset")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        assert not running.done()
        core = cast(Any, observed["core"])
        manager = core.plugin_manager
        selection = manager._selection.read()
        assert selection is not None
        assert len(commit_calls) == 1
        assert commit_calls[0][1] is None
        selected_ids = {
            manager._archive.read_descriptor(ref)["plugin_id"]
            for ref in manager._selection_components(selection)
        }
        assert selected_ids == {
            "a-first-null-peer", "c-first-null-downstream",
        }
        assert manager.generation("b-first-null-bad") is None
        assert all(
            fiber.runtime is None
            or fiber.runtime.plugin_id != "b-first-null-bad"
            for fiber in manager.live_root._fibers.values()  # pyright: ignore[union-attr, reportPrivateUsage]
        )
        control_service = runtime.control_service
        assert control_service is not None
        failures = control_service.plugin_status()["source_failures"]
        assert isinstance(failures, list)
        assert len(failures) == 1
        assert failures[0]["source_root"] == str(
            (source_root / "b-first-null-bad").resolve()
        )
        assert failures[0]["error_type"] == "SyntaxError"
        assert "line" in failures[0]["error_text"]

        root = manager.live_root
        assert root is not None
        root_token = root.instance_token
        fibers = {
            fiber.runtime.plugin_id: fiber
            for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.runtime is not None
        }
        peer = fibers["a-first-null-peer"]
        downstream = fibers["c-first-null-downstream"]
        assert peer.state is FiberState.ACTIVE
        peer_context = peer.context
        peer_activation = peer_context.fiber.activation_token
        peer_effects = tuple(peer.effects)
        peer_state = root.service_value(ServiceKey("first-null.peer"))
        assert isinstance(peer_state, dict)
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert not peer._in_flight_calls
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("first-null.peer")) is peer_state
        assert peer_context.fiber.activation_token is peer_activation
        assert tuple(peer.effects) == peer_effects
        assert not peer._in_flight_calls
        assert root.instance_token is root_token
        assert downstream.state is FiberState.PENDING
        with pytest.raises(CompositionError) as unavailable:
            async with downstream.context.runtime_scope():
                pass
        assert unavailable.value.code == "OWNER_UNAVAILABLE"
        assert runtime._shutdown is False
        assert runtime.dashboard_server is not None
        assert runtime.dashboard_server.should_exit is False
        assert runtime.control_service is not None
        assert core.bus._closed is False
        assert core.event_bus._closed is False
        assert manager.live_root is root
        cast(Any, observed["dashboard_server"]).should_exit = True
        async with asyncio.timeout(10):
            await asyncio.wait((running,))
        try:
            running.result()
        finally:
            runner_retrieved = True
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if running is not None and not runner_retrieved:
            cancel_requested = False
            if not running.done():
                cancel_requested = running.cancel()
            await asyncio.wait((running,))
            try:
                running.result()
            except asyncio.CancelledError as error:
                if not (cancel_requested and error.__cause__ is None):
                    raise
            finally:
                runner_retrieved = True
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.asyncio
async def test_real_app_first_null_all_fail_commits_empty_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Real Core/AppRuntime turns an all-invalid first input into durable empty selection."""
    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'disabled_builtin = ["akasha", "wake"]',
            "disabled_builtin = []",
        ),
        encoding="utf-8",
    )
    source_root = tmp_path / "all-fail-plugins"
    for name in ("first-bad", "second-bad"):
        plugin_dir = source_root / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text(
            "def broken(:\n    return 1\n", encoding="utf-8",
        )
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed, plugin_dirs=(source_root,))
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    commit_calls: list[tuple[tuple[str, ...], str | None]] = []
    from agent.plugins.selection import PluginSelection
    real_commit = PluginSelection.commit

    def observed_commit(
        selection: Any,
        components: tuple[str, ...], *, expected_ref: str | None,
    ) -> str:
        commit_calls.append((components, expected_ref))
        return real_commit(selection, components, expected_ref=expected_ref)

    monkeypatch.setattr(PluginSelection, "commit", observed_commit)
    running: asyncio.Task[None] | None = None
    runner_retrieved = False
    try:
        running = asyncio.create_task(runtime.run(), name="real-app-first-null-all-fail")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        assert not running.done()
        core = cast(Any, observed["core"])
        manager = core.plugin_manager
        selection = manager._selection.read()
        assert selection is not None
        assert manager._selection_components(selection) == ()
        assert commit_calls == [((), None)]
        root = manager.live_root
        assert root is not None
        control_service = runtime.control_service
        assert control_service is not None
        assert control_service.plugin_status()["source_failures"]
        assert manager._active_generations == {}
        assert all(fiber.runtime is None for fiber in root._fibers.values())
        assert runtime._shutdown is False
        assert runtime.dashboard_server is not None
        assert runtime.dashboard_server.should_exit is False
        assert core.bus._closed is False
        assert core.event_bus._closed is False
        cast(Any, observed["dashboard_server"]).should_exit = True
        async with asyncio.timeout(10):
            await asyncio.wait((running,))
        try:
            running.result()
        finally:
            runner_retrieved = True
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if running is not None and not runner_retrieved:
            cancel_requested = False
            if not running.done():
                cancel_requested = running.cancel()
            await asyncio.wait((running,))
            try:
                running.result()
            except asyncio.CancelledError as error:
                if not (cancel_requested and error.__cause__ is None):
                    raise
            finally:
                runner_retrieved = True
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.asyncio
async def test_real_app_runtime_waits_without_primary_until_dashboard_releases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The real Dashboard/Watcher keep AppRuntime alive when HostBridge is absent."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        observed["pre_root"] = kwargs["plugin_manager"].live_root  # type: ignore[attr-defined]
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    running: asyncio.Task[None] | None = None
    task_retrieved = False
    try:
        running = asyncio.create_task(runtime.run(), name="real-app-no-primary")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        core = cast(Any, observed["core"])
        assert observed["pre_root"] is None
        assert core.plugin_manager.live_root is not None
        assert runtime._shutdown is False
        assert cast(Any, observed["dashboard_server"]).should_exit is False
        assert runtime.tasks == []
        assert runtime._primary_task is None
        assert not running.done()

        cast(Any, observed["dashboard_server"]).should_exit = True
        async with asyncio.timeout(10):
            await running
        task_retrieved = True
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if running is not None and not task_retrieved:
            if not running.done():
                running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.asyncio
async def test_real_live_root_local_failure_preserves_peer_identity_and_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bad live-Root branch stays local while a healthy peer keeps its owner."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    running: asyncio.Task[None] | None = None
    task_retrieved = False
    bad_fiber: Any = None
    downstream_fiber: Any = None
    try:
        running = asyncio.create_task(runtime.run(), name="real-app-local-failure")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        core = cast(Any, observed["core"])
        root = core.plugin_manager.live_root
        assert root is not None
        server = cast(Any, observed["dashboard_server"])
        assert runtime._shutdown is False
        assert server.should_exit is False
        root_token = root.instance_token
        peer_key = ServiceKey[str]("test.host.peer")
        missing_key = ServiceKey[str]("test.host.missing")
        bad_key = ServiceKey[str]("test.host.bad")
        peer_events: list[str] = []
        peer_body = asyncio.Event()
        peer_effect_closed = asyncio.Event()

        async def peer(ctx: Context) -> None:
            await ctx.on(RUNTIME_STARTING, lambda _event: peer_events.append("starting"))
            await ctx.on(RUNTIME_STARTED, lambda _event: peer_events.append("started"))
            await ctx.effect(
                lambda: lambda: peer_effect_closed.set(),
                label="test-peer-effect",
            )
            await ctx.provide(peer_key, "peer")
            peer_body.set()

        async with asyncio.timeout(10):
            peer_fiber = await root.mount(peer, name="healthy-peer")
            await peer_body.wait()
        peer_context = peer_fiber.context
        peer_activation = peer_context.fiber.activation_token
        peer_effects = tuple(peer_fiber.effects)
        assert peer_events == ["starting", "started"]
        assert peer_fiber.state == FiberState.ACTIVE

        bad_body = asyncio.Event()

        async def bad_owner(ctx: Context) -> None:
            ctx.require(missing_key)
            await ctx.provide(bad_key, "unreachable")
            bad_body.set()

        async def downstream(ctx: Context) -> None:
            ctx.require(bad_key)

        async with asyncio.timeout(10):
            bad_fiber = await root.mount(
                bad_owner,
                name="bad-owner",
                inject=(missing_key,),
            )
            downstream_fiber = await root.mount(
                downstream,
                name="bad-downstream",
                inject=(bad_key,),
            )
        receipt = root.receipt()
        assert receipt.ready is False
        assert bad_fiber.state == FiberState.PENDING
        assert downstream_fiber.state == FiberState.PENDING
        assert not bad_body.is_set()
        assert runtime._shutdown is False
        assert server.should_exit is False
        for fiber in (bad_fiber, downstream_fiber):
            with pytest.raises(CompositionError) as unavailable:
                async with fiber.context.runtime_scope():
                    pass
            assert unavailable.value.code == "OWNER_UNAVAILABLE"

        async with peer_context.runtime_scope():
            assert peer_context.require(peer_key) == "peer"
        assert root.instance_token is root_token
        assert peer_fiber.context is peer_context
        assert peer_context.fiber.activation_token is peer_activation
        assert tuple(peer_fiber.effects) == peer_effects
        assert peer_events.count("starting") == 1
        assert peer_events.count("started") == 1
        assert not running.done()
        assert runtime._shutdown is False
        assert server.should_exit is False

        await downstream_fiber.dispose()
        await bad_fiber.dispose()
        assert peer_fiber.state == FiberState.ACTIVE
        async with peer_context.runtime_scope():
            assert peer_context.require(peer_key) == "peer"
        assert peer_effect_closed.is_set() is False
        assert runtime._shutdown is False
        assert server.should_exit is False

        server.should_exit = True
        async with asyncio.timeout(10):
            await running
        task_retrieved = True
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        try:
            if downstream_fiber is not None and downstream_fiber.state is not FiberState.DISPOSED:
                await downstream_fiber.dispose()
        finally:
            try:
                if bad_fiber is not None and bad_fiber.state is not FiberState.DISPOSED:
                    await bad_fiber.dispose()
            finally:
                if running is not None and not task_retrieved:
                    if not running.done():
                        running.cancel()
                    try:
                        await running
                    except asyncio.CancelledError:
                        pass
    _assert_real_app_closed(observed, tmp_path, socket_path)
    assert peer_effect_closed.is_set()


@pytest.mark.asyncio
async def test_real_cold_start_local_failure_preserves_peer_and_host_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cold-start bad Fiber stays local while the real host and peer continue."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'disabled_builtin = ["akasha", "wake"]',
            "disabled_builtin = []",
        ),
        encoding="utf-8",
    )
    source_root = _write_cold_start_plugins(tmp_path / "cold-start-plugins")
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed, plugin_dirs=(source_root,))
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    from agent.plugins.manager import PluginManager

    real_mount = PluginManager._mount_generation_composition
    mount_order: list[tuple[str, str]] = []
    peer_baseline: dict[str, object] | None = None

    async def observe_mount(manager: Any, root: Any, generation: Any) -> None:
        nonlocal peer_baseline
        plugin_id = generation.plugin_id
        mount_order.append(("before", plugin_id))
        await real_mount(manager, root, generation)
        mount_order.append(("after", plugin_id))
        if plugin_id != "a-healthy-peer":
            return
        peer_fiber = generation.fiber
        assert peer_fiber is not None
        peer_service = root.service_value(ServiceKey("cold-start.peer"))
        assert isinstance(peer_service, dict)
        peer_baseline = {
            "root": root,
            "fiber": peer_fiber,
            "context": peer_fiber.context,
            "activation": peer_fiber.context.fiber.activation_token,
            "effects": tuple(peer_fiber.effects),
            "service": peer_service,
            "events": tuple(cast(list[str], peer_service["events"])),
            "effect": peer_service["effect"],
            "cleanup": peer_service["cleanup"],
        }
        observed["peer_baseline"] = peer_baseline

    observed["mount_order"] = mount_order
    monkeypatch.setattr(PluginManager, "_mount_generation_composition", observe_mount)
    running: asyncio.Task[None] | None = None
    task_retrieved = False
    peer_state: dict[str, object] | None = None
    bad_fiber: Any = None
    downstream_fiber: Any = None
    try:
        running = asyncio.create_task(runtime.run(), name="real-cold-start-local-failure")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        core = cast(Any, observed["core"])
        root = core.plugin_manager.live_root
        assert root is not None
        server = cast(Any, observed["dashboard_server"])
        assert runtime._shutdown is False
        assert server.should_exit is False
        assert running.done() is False

        fibers = {
            fiber.runtime.plugin_id: fiber
            for fiber in root._fibers.values()
            if fiber.runtime is not None
        }
        peer_fiber = fibers["a-healthy-peer"]
        bad_fiber = fibers["b-bad-owner"]
        downstream_fiber = fibers["c-required-downstream"]
        catalog_fiber = fibers["z-catalog-observer"]
        baseline = cast(dict[str, object], observed["peer_baseline"])
        mount_order = cast(list[tuple[str, str]], observed["mount_order"])
        assert mount_order.index(("after", "a-healthy-peer")) < mount_order.index(
            ("before", "b-bad-owner")
        )
        peer_context = peer_fiber.context
        root_token = root.instance_token
        peer_activation = peer_context.fiber.activation_token
        peer_effects = tuple(peer_fiber.effects)
        assert baseline["root"] is root
        assert baseline["fiber"] is peer_fiber
        assert baseline["context"] is peer_context
        assert baseline["activation"] is peer_activation
        assert baseline["effects"] == peer_effects
        peer_state = cast(dict[str, object], baseline["service"])
        assert root.service_value(ServiceKey("cold-start.peer")) is peer_state
        assert isinstance(peer_state, dict)
        assert tuple(cast(list[str], peer_state["events"])) == baseline["events"]
        assert peer_state["effect"] == baseline["effect"] == 1
        assert peer_state["cleanup"] == baseline["cleanup"] == 0
        assert peer_fiber.state == FiberState.ACTIVE

        assert bad_fiber.state == FiberState.FAILED
        assert isinstance(bad_fiber.error, RuntimeError)
        assert str(bad_fiber.error) == "cold-start bad owner apply failed"
        assert bad_fiber.missing_services == ()
        assert downstream_fiber.state == FiberState.PENDING
        assert downstream_fiber.missing_services == ("cold-start.bad",)
        assert downstream_fiber.error is None

        receipt = root.receipt()
        assert receipt.ready is False
        assert "b-bad-owner" in receipt.required_pending
        assert "c-required-downstream" in receipt.required_pending
        for fiber in (bad_fiber, downstream_fiber):
            with pytest.raises(CompositionError) as unavailable:
                async with fiber.context.runtime_scope():
                    pass
            assert unavailable.value.code == "OWNER_UNAVAILABLE"

        catalog_context = catalog_fiber.context
        async with catalog_context.runtime_scope():
            reader = catalog_context.require(RUNTIME_CATALOG)
            catalog = reader(catalog_context)
        assert isinstance(catalog, dict)
        catalog_plugins = {
            item["id"]: item
            for item in cast(list[dict[str, object]], catalog["plugins"])
        }
        bad_view = cast(dict[str, object], catalog_plugins["b-bad-owner"])
        bad_composition = cast(dict[str, object], bad_view["composition"])
        assert bad_composition["ready"] is False
        assert bad_view["load_error"] is None
        assert bad_view["cleanup_pending"] is False
        bad_fiber_view = next(
            item
            for item in cast(list[dict[str, object]], bad_composition["fibers"])
            if item["name"] == "b-bad-owner"
        )
        assert bad_fiber_view["state"] == "failed"
        assert bad_fiber_view["error"] == "cold-start bad owner apply failed"
        assert any(
            incident["kind"] == "runtime_error"
            and incident["message"] == "cold-start bad owner apply failed"
            for incident in cast(list[dict[str, object]], bad_composition["recent_incidents"])
        )
        downstream_view = cast(dict[str, object], catalog_plugins["c-required-downstream"])
        downstream_composition = cast(dict[str, object], downstream_view["composition"])
        assert downstream_composition["ready"] is False
        downstream_fiber_view = next(
            item
            for item in cast(list[dict[str, object]], downstream_composition["fibers"])
            if item["name"] == "c-required-downstream"
        )
        assert downstream_fiber_view["state"] == "pending"
        assert downstream_fiber_view["missing_services"] == ["cold-start.bad"]

        assert root.instance_token is root_token
        assert peer_fiber.context is peer_context
        assert peer_context.fiber.activation_token is peer_activation
        assert tuple(peer_fiber.effects) == peer_effects
        assert root.service_value(ServiceKey("cold-start.peer")) is peer_state
        assert tuple(cast(list[str], peer_state["events"])) == baseline["events"]
        assert peer_state["effect"] == baseline["effect"]
        assert peer_state["cleanup"] == baseline["cleanup"] == 0

        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("cold-start.peer")) is peer_state
        assert not peer_context.fiber._fiber._in_flight_calls
        assert bad_fiber.state == FiberState.FAILED
        assert downstream_fiber.state == FiberState.PENDING
        assert runtime._shutdown is False
        assert server.should_exit is False
        assert running.done() is False

        try:
            await downstream_fiber.dispose()
        finally:
            await bad_fiber.dispose()
        assert downstream_fiber.state == FiberState.DISPOSED
        assert bad_fiber.state == FiberState.DISPOSED
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("cold-start.peer")) is peer_state
        assert not peer_context.fiber._fiber._in_flight_calls
        assert root.instance_token is baseline["root"].instance_token
        assert peer_fiber is baseline["fiber"]
        assert peer_fiber.context is baseline["context"]
        assert peer_context.fiber.activation_token is baseline["activation"]
        assert tuple(peer_fiber.effects) == baseline["effects"]
        assert tuple(cast(list[str], peer_state["events"])) == baseline["events"]
        assert peer_state["effect"] == baseline["effect"]
        assert peer_state["cleanup"] == baseline["cleanup"] == 0
        assert runtime._shutdown is False
        assert server.should_exit is False
        assert running.done() is False

        server.should_exit = True
        async with asyncio.timeout(10):
            await running
        task_retrieved = True
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if running is not None and not task_retrieved:
            if not running.done():
                running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
    _assert_real_app_closed(observed, tmp_path, socket_path)
    assert isinstance(peer_state, dict)
    events = peer_state["events"]
    assert isinstance(events, list)
    assert events[:2] == ["starting", "started"]
    assert events.count("starting") == 1
    assert events.count("started") == 1
    assert peer_state["cleanup"] == 1


@pytest.mark.asyncio
async def test_real_selected_archive_import_failure_keeps_peer_and_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fixed archive can fail before Fiber while the real host and peer continue."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            'disabled_builtin = ["akasha", "wake"]',
            "disabled_builtin = []",
        ),
        encoding="utf-8",
    )
    source_root = _write_cold_import_plugins(tmp_path / "cold-import-plugins")
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed, plugin_dirs=(source_root,))
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    assert config.disabled_builtin_plugins == frozenset()
    monkeypatch.setenv("COLD_IMPORT_RESTART", "no")
    first_runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = first_runtime
    first_running: asyncio.Task[None] | None = None
    first_task_retrieved = False
    selected_before: str | None = None

    async def settle_runner(task: asyncio.Task[None]) -> None:
        """Wait for physical shutdown, then retrieve the runner's raw terminal state."""
        cancel_requested = False
        if not task.done():
            cancel_requested = task.cancel()
            await asyncio.wait((task,))
        try:
            task.result()
        except asyncio.CancelledError as error:
            if not cancel_requested or error.__cause__ is not None:
                raise

    try:
        first_running = asyncio.create_task(first_runtime.run(), name="cold-import-first")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        first_core = cast(Any, observed["core"])
        first_root = first_core.plugin_manager.live_root
        assert first_root is not None
        first_fibers = {
            fiber.runtime.plugin_id: fiber
            for fiber in first_root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.runtime is not None
        }
        assert first_fibers["b-import-bad"].state is FiberState.ACTIVE
        assert first_fibers["c-import-downstream"].state is FiberState.ACTIVE
        selected_before = first_core.plugin_manager._selection.read()
        assert selected_before is not None
        cast(Any, observed["dashboard_server"]).should_exit = True
        try:
            async with asyncio.timeout(10):
                await asyncio.wait((first_running,))
                try:
                    first_running.result()
                finally:
                    first_task_retrieved = True
        except asyncio.TimeoutError:
            raise
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if first_running is not None and not first_task_retrieved:
            await settle_runner(first_running)
    _assert_real_app_closed(observed, tmp_path, socket_path)

    dashboard_started.clear()
    monkeypatch.setenv("COLD_IMPORT_RESTART", "yes")
    second_runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = second_runtime
    second_running: asyncio.Task[None] | None = None
    second_task_retrieved = False
    try:
        second_running = asyncio.create_task(second_runtime.run(), name="cold-import-second")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        core = cast(Any, observed["core"])
        manager = core.plugin_manager
        root = manager.live_root
        server = cast(Any, observed["dashboard_server"])
        assert root is not None
        assert manager._selection.read() == selected_before
        assert second_runtime._shutdown is False
        assert server.should_exit is False
        assert second_running.done() is False

        failed = manager.generation("b-import-bad")
        peer = manager.generation("a-import-peer")
        assert failed is not None and peer is not None
        assert failed.archive_ref in manager._selection_components(selected_before)
        assert failed.state == "failed"
        assert isinstance(failed.load_error, ImportError)
        assert failed.fiber is None
        assert failed.scope.closed
        assert not manager._draining_generations.get("b-import-bad")
        assert peer.fiber is not None and peer.fiber.state is FiberState.ACTIVE

        fibers = {
            fiber.runtime.plugin_id: fiber
            for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.runtime is not None
        }
        downstream = fibers["c-import-downstream"]
        catalog_fiber = fibers["z-import-catalog"]
        assert downstream.state is FiberState.PENDING
        assert downstream.missing_services == ("cold-import.bad",)
        with pytest.raises(CompositionError) as unavailable:
            async with downstream.context.runtime_scope():
                pass
        assert unavailable.value.code == "OWNER_UNAVAILABLE"

        peer_context = peer.fiber.context
        peer_identity = (peer.fiber, peer_context, peer_context.fiber.activation_token)
        peer_effects = tuple(peer.fiber.effects)
        async with peer_context.runtime_scope():
            peer_state = peer_context.require(ServiceKey("cold-import.peer"))
        assert isinstance(peer_state, dict)
        assert peer_identity[0] is peer.fiber
        assert peer_identity[1] is peer_context
        assert peer_identity[2] is peer_context.fiber.activation_token
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        assert peer_effects == tuple(peer.fiber.effects)
        assert not peer_context.fiber._fiber._in_flight_calls

        async with catalog_fiber.context.runtime_scope():
            reader = catalog_fiber.context.require(RUNTIME_CATALOG)
            catalog = reader(catalog_fiber.context)
        bad_view = next(
            item for item in cast(list[dict[str, object]], catalog["plugins"])
            if item["id"] == "b-import-bad"
        )
        bad_composition = cast(dict[str, object], bad_view["composition"])
        assert failed.static_manifest is not None
        assert bad_view["archive_ref"] == failed.archive_ref
        assert bad_view["generation_id"] == failed.generation_id
        assert bad_view["api_version"] == failed.static_manifest.api_version
        assert bad_view["state"] == "failed"
        assert bad_view["load_error"] == "cold selected archive import blocked"
        assert bad_view["cleanup_pending"] is False
        assert bad_composition["ready"] is False
        assert bad_composition["fibers"] == []
        assert bad_composition["incident_count"] == 0

        await manager._dispose_generation(failed, state="discarded")
        assert manager.generation("b-import-bad") is None
        assert manager.live_root is root
        assert peer.fiber is peer_identity[0]
        assert peer.fiber.context is peer_identity[1]
        assert peer_context.fiber.activation_token is peer_identity[2]
        assert tuple(peer.fiber.effects) == peer_effects
        assert peer_state["events"] == ["starting", "started"]
        assert peer_state["effect"] == 1
        assert peer_state["cleanup"] == 0
        async with peer_context.runtime_scope():
            assert peer_context.require(ServiceKey("cold-import.peer")) is peer_state
        assert not peer_context.fiber._fiber._in_flight_calls
        assert second_runtime._shutdown is False
        assert server.should_exit is False
        assert second_running.done() is False

        server.should_exit = True
        try:
            async with asyncio.timeout(10):
                await asyncio.wait((second_running,))
                try:
                    second_running.result()
                finally:
                    second_task_retrieved = True
        except asyncio.TimeoutError:
            raise
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if second_running is not None and not second_task_retrieved:
            await settle_runner(second_running)
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.parametrize("failing_task", ["dashboard", "watcher", "host_bridge"])
@pytest.mark.asyncio
async def test_real_app_host_task_error_propagates_and_cleans_every_owner(
    failing_task: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Real supervised host-task failures retain their type/message and cleanup."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    expected = RuntimeError(f"{failing_task} task failed")
    failure_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve
        if failing_task == "dashboard":
            async def fail_dashboard() -> None:
                failure_started.set()
                raise expected

            monkeypatch.setattr(server, "serve", fail_dashboard)
        else:
            async def observe_serve() -> None:
                await real_serve()

            monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    if failing_task == "watcher":
        async def fail_watcher(_watcher: object) -> None:
            failure_started.set()
            raise expected

        monkeypatch.setattr(bootstrap_app.PluginWatcher, "run", fail_watcher)
    elif failing_task == "host_bridge":
        async def fail_host_bridge() -> None:
            failure_started.set()
            raise expected

        monkeypatch.setattr(bootstrap_app, "build_host_bridge_monitor", lambda: fail_host_bridge())

    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    running: asyncio.Task[None] | None = None
    task_retrieved = False
    try:
        running = asyncio.create_task(runtime.run(), name=f"real-app-error-{failing_task}")
        with pytest.raises(RuntimeError) as caught:
            async with asyncio.timeout(10):
                await running
        task_retrieved = True
        assert caught.value is expected
        assert type(caught.value) is RuntimeError
        assert str(caught.value) == str(expected)
        async with asyncio.timeout(10):
            await failure_started.wait()
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if running is not None and not task_retrieved:
            if not running.done():
                running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.asyncio
async def test_real_app_external_cancel_waits_for_physical_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Caller cancellation still lets the real host cleanup finish first."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        real_serve = server.serve

        async def observe_serve() -> None:
            dashboard_started.set()
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        observed["dashboard_server"] = server
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)
    config = Config.load(config_path, workspace=tmp_path)
    runtime = bootstrap_app.build_app_runtime(config, tmp_path)
    observed["app_runtime"] = runtime
    running: asyncio.Task[None] | None = None
    task_retrieved = False
    try:
        running = asyncio.create_task(runtime.run(), name="real-app-cancel")
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        assert running is not None
        assert runtime._shutdown is False
        assert cast(Any, observed["dashboard_server"]).should_exit is False
        assert not running.done()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        task_retrieved = True
    finally:
        server = observed.get("dashboard_server")
        if server is not None:
            cast(Any, server).should_exit = True
        if running is not None and not task_retrieved:
            if not running.done():
                running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
    _assert_real_app_closed(observed, tmp_path, socket_path)


class _ControlledStdin:
    """Control only the physical stdio readline used by the real server."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        outcome: str,
        started: asyncio.Event,
    ) -> None:
        self.buffer = self
        self._loop = loop
        self._outcome = outcome
        self._started = started
        self.release = threading.Event()
        self.finished = threading.Event()

    def readline(self) -> bytes:
        self._loop.call_soon_threadsafe(self._started.set)
        try:
            self.release.wait()
            if self._outcome == "error":
                raise OSError("stdio input failed")
            return b""
        finally:
            self.finished.set()


@pytest.mark.parametrize("outcome", ["eof", "error"])
@pytest.mark.asyncio
async def test_real_stdio_entry_waits_for_physical_input_and_cleans_resources(
    outcome: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The real stdio entry preserves EOF/error and settles every real owner."""

    config_path, _socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    config = Config.load(config_path, workspace=tmp_path)
    observed: dict[str, object] = {}
    real_http = bootstrap_app_server.SharedHttpResources

    def shared_http_resources() -> SharedHttpResources:
        http = real_http()
        observed["http"] = http
        return http

    monkeypatch.setattr(bootstrap_app_server, "SharedHttpResources", shared_http_resources)
    real_lock = bootstrap_app_server.WorkspaceInstanceLock

    def workspace_lock(workspace: Path) -> WorkspaceInstanceLock:
        lock = real_lock(workspace)
        observed["lock"] = lock
        return lock

    monkeypatch.setattr(bootstrap_app_server, "WorkspaceInstanceLock", workspace_lock)
    real_build_core = bootstrap_app_server.build_core_runtime

    def build_core(
        cfg: Config,
        workspace: Path,
        http: SharedHttpResources,
        **kwargs: Unpack[_CoreKwargs],
    ) -> bootstrap_tools.CoreRuntime:
        kwargs["plugin_dirs"] = ()
        core = real_build_core(cfg, workspace, http, **kwargs)
        observed["core"] = core
        return core

    monkeypatch.setattr(bootstrap_app_server, "build_core_runtime", build_core)
    real_build_service = bootstrap_app_server.build_control_service

    def build_service(core: object, **kwargs: object) -> object:
        service = real_build_service(core, **kwargs)  # type: ignore[arg-type]
        observed["service"] = service
        return service

    monkeypatch.setattr(bootstrap_app_server, "build_control_service", build_service)
    read_started = asyncio.Event()
    stdin = _ControlledStdin(asyncio.get_running_loop(), outcome, read_started)
    monkeypatch.setattr(sys, "stdin", stdin)
    running: asyncio.Task[None] | None = None
    task_retrieved = False
    try:
        running = asyncio.create_task(
            bootstrap_app_server.run_stdio_app_server(config, tmp_path),
            name=f"real-stdio-{outcome}",
        )
        async with asyncio.timeout(10):
            await read_started.wait()
        assert running is not None
        assert not running.done()
        stdin.release.set()
        if outcome == "error":
            with pytest.raises(OSError, match="stdio input failed"):
                async with asyncio.timeout(10):
                    await asyncio.shield(running)
        else:
            async with asyncio.timeout(10):
                await asyncio.shield(running)
        task_retrieved = True
    finally:
        stdin.release.set()
        try:
            if read_started.is_set():
                physical_finished = await asyncio.to_thread(stdin.finished.wait, 10)
                assert physical_finished is True
        finally:
            if running is not None and not task_retrieved:
                try:
                    await running
                except asyncio.CancelledError:
                    pass
    assert stdin.finished.is_set() is True
    core = cast(Any, observed["core"])
    service = cast(Any, observed["service"])
    http = cast(Any, observed["http"])
    lock = cast(Any, observed["lock"])
    assert service._closed is True
    assert core.bus._closed is True
    assert core.event_bus._closed is True
    assert core.plugin_manager.live_root is None
    assert http.closed is True
    assert lock._stream is None
    reacquired = WorkspaceInstanceLock(tmp_path)
    reacquired.acquire()
    reacquired.release()


def _prepare_real_cli_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    """Isolate real CLI serve from inherited supervisor and host identity state."""

    config_path, socket_path = _prepare_real_host_fixture(monkeypatch, tmp_path)
    for name in (
        "AKASHIC_SUPERVISED",
        "AKASHIC_LIFECYCLE_FD",
        "AKASHIC_RESTART_NONCE",
        "AKASHIC_BOOT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    return config_path, socket_path


def _capture_cli_runtime(
    monkeypatch: pytest.MonkeyPatch,
    observed: dict[str, object],
) -> None:
    """Observe the exact real AppRuntime and Task created by main.serve."""

    real_build_app_runtime = main.build_app_runtime

    def build_app_runtime(
        config: Config,
        workspace: Path,
        **kwargs: Unpack[_AppKwargs],
    ) -> bootstrap_app.AppRuntime:
        runtime = real_build_app_runtime(config, workspace, **kwargs)
        real_run = runtime.run  # type: ignore[attr-defined]

        async def observe_run() -> None:
            task = asyncio.current_task()
            assert task is not None
            observed["runtime_task"] = task
            try:
                await real_run()
            except asyncio.CancelledError as error:
                observed["runtime_cancel_error"] = error
                raise

        runtime.run = observe_run  # type: ignore[attr-defined]
        observed["app_runtime"] = runtime
        return runtime

    monkeypatch.setattr(main, "build_app_runtime", build_app_runtime)


def _observe_cli_dashboard(
    monkeypatch: pytest.MonkeyPatch,
    observed: dict[str, object],
    started: asyncio.Event,
    *,
    return_immediately: bool = False,
    failure: BaseException | None = None,
) -> None:
    """Observe a real Dashboard server while keeping its boundary controllable."""

    real_build_dashboard_server = bootstrap_app.build_dashboard_server

    def build_dashboard_server(**kwargs: Unpack[_DashboardKwargs]) -> object:
        server = real_build_dashboard_server(**kwargs)
        observed["dashboard_server"] = server
        real_serve = server.serve

        async def observe_serve() -> None:
            started.set()
            if failure is not None:
                raise failure
            if return_immediately:
                return
            await real_serve()

        monkeypatch.setattr(server, "serve", observe_serve)
        return server

    monkeypatch.setattr(bootstrap_app, "build_dashboard_server", build_dashboard_server)


async def _mount_cli_cleanup_fiber(
    runtime: Any,
    cleanup_started: asyncio.Event,
    cleanup_release: asyncio.Event,
    cleanup_finished: asyncio.Event,
    cleanup_calls: list[str],
) -> None:
    """Mount a real Fiber whose Effect makes physical shutdown observable."""

    root = runtime.core.plugin_manager.live_root
    assert root is not None
    body_ready = asyncio.Event()

    async def owner(ctx: Context) -> None:
        async def cleanup() -> None:
            cleanup_calls.append("cleanup")
            cleanup_started.set()
            await cleanup_release.wait()
            cleanup_finished.set()

        await ctx.effect(lambda: cleanup, label="cli-serve-cleanup")
        body_ready.set()
        return

    await root.mount(owner, name="cli-serve-cleanup-owner")
    async with asyncio.timeout(10):
        await body_ready.wait()


@pytest.mark.asyncio
async def test_cli_serve_external_cancel_settles_exact_runtime_after_repeated_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Outer cancellation waits for one real AppRuntime cleanup and both cancels."""

    config_path, socket_path = _prepare_real_cli_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    _capture_cli_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    _observe_cli_dashboard(monkeypatch, observed, dashboard_started)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_calls: list[str] = []
    serving = asyncio.create_task(main.serve(str(config_path), tmp_path))
    try:
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        runtime = cast(Any, observed["app_runtime"])
        await _mount_cli_cleanup_fiber(
            runtime,
            cleanup_started,
            cleanup_release,
            cleanup_finished,
            cleanup_calls,
        )
        serving.cancel()
        async with asyncio.timeout(10):
            await cleanup_started.wait()
        serving.cancel()
        loop = asyncio.get_running_loop()
        second_cancel_delivered = asyncio.Event()
        loop.call_soon(second_cancel_delivered.set)
        async with asyncio.timeout(10):
            await second_cancel_delivered.wait()
        runtime_task = cast(asyncio.Task[object], observed["runtime_task"])
        assert serving.done() is False
        assert runtime_task.done() is False
        assert serving.cancelling() >= 2
        assert runtime_task.cancelling() == 1
        assert cleanup_finished.is_set() is False
        assert cleanup_calls == ["cleanup"]
        cleanup_release.set()
        with pytest.raises(asyncio.CancelledError):
            await serving
        assert cleanup_finished.is_set()
        assert runtime_task.done()
    finally:
        cleanup_release.set()
        if not serving.done():
            serving.cancel()
        try:
            await serving
        except asyncio.CancelledError:
            pass
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime_first", [False, True])
async def test_cli_serve_cancel_preserves_real_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_first: bool,
) -> None:
    """Outer cancellation keeps the real Core.stop error as CancelledError cause."""

    config_path, socket_path = _prepare_real_cli_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    _capture_cli_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    _observe_cli_dashboard(monkeypatch, observed, dashboard_started)
    serving = asyncio.create_task(main.serve(str(config_path), tmp_path))
    serving_retrieved = False
    cleanup_error = RuntimeError("cli core cleanup failed")
    try:
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        runtime = cast(Any, observed["app_runtime"])
        core = runtime.core
        assert core is not None
        real_stop = core.stop

        async def stop_with_error() -> None:
            await real_stop()
            raise cleanup_error

        core.stop = stop_with_error
        runtime_task = cast(asyncio.Task[object], observed["runtime_task"])
        if runtime_first:
            runtime_task.add_done_callback(lambda _task: serving.cancel())
            runtime_task.cancel()
        else:
            serving.cancel()
        try:
            with pytest.raises(asyncio.CancelledError) as caught:
                async with asyncio.timeout(10):
                    await serving
        finally:
            if serving.done():
                serving_retrieved = True
        runtime_cancel_error = cast(
            asyncio.CancelledError,
            observed["runtime_cancel_error"],
        )
        assert caught.value is runtime_cancel_error
        assert runtime_cancel_error.__cause__ is cleanup_error
        assert cast(Any, observed["runtime_task"]).done()
    finally:
        if not serving_retrieved:
            if not serving.done():
                serving.cancel()
            try:
                await serving
            except asyncio.CancelledError:
                pass
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.asyncio
async def test_cli_serve_same_turn_cancel_wins_real_runtime_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stop cleanup completion can cancel the outer serve before it returns."""

    config_path, socket_path = _prepare_real_cli_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    _capture_cli_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    _observe_cli_dashboard(monkeypatch, observed, dashboard_started)
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_calls: list[str] = []
    loop = asyncio.get_running_loop()
    callbacks: dict[int, Callable[..., object]] = {}
    real_add = loop.add_signal_handler

    def add_signal_handler(sig: int, callback: Callable[..., object], *args: object) -> None:
        callbacks[sig] = callback
        real_add(sig, callback, *args)

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    serving = asyncio.create_task(main.serve(str(config_path), tmp_path))
    serving_retrieved = False
    cancel_results: list[bool] = []
    cancel_delivered = asyncio.Event()

    def cancel_serving(_task: asyncio.Task[object]) -> None:
        cancel_results.append(serving.cancel())
        cancel_delivered.set()

    try:
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        runtime = cast(Any, observed["app_runtime"])
        await _mount_cli_cleanup_fiber(
            runtime,
            cleanup_started,
            cleanup_release,
            cleanup_finished,
            cleanup_calls,
        )
        runtime_task = cast(asyncio.Task[object], observed["runtime_task"])
        runtime_task.add_done_callback(cancel_serving)
        cast(Any, callbacks[signal.SIGTERM])()
        async with asyncio.timeout(10):
            await cleanup_started.wait()
        assert serving.done() is False
        assert cleanup_finished.is_set() is False
        cleanup_release.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                async with asyncio.timeout(10):
                    await serving
        finally:
            if serving.done():
                serving_retrieved = True
        async with asyncio.timeout(10):
            await cancel_delivered.wait()
        assert cancel_results == [True]
        assert runtime_task.done()
        assert cleanup_finished.is_set()
        assert cleanup_calls == ["cleanup"]
    finally:
        cleanup_release.set()
        if not serving.done():
            serving.cancel()
        if not serving_retrieved:
            try:
                await serving
            except asyncio.CancelledError:
                pass
    assert cleanup_finished.is_set()
    assert cleanup_calls == ["cleanup"]
    assert cast(Any, observed["runtime_task"]).done()
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.asyncio
async def test_cli_serve_normal_return_and_real_host_error_settle_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: bool,
) -> None:
    """Normal Dashboard return is zero; a real host error remains the same instance."""

    config_path, socket_path = _prepare_real_cli_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    _capture_cli_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    expected = RuntimeError("cli dashboard failed") if failure else None
    _observe_cli_dashboard(
        monkeypatch,
        observed,
        dashboard_started,
        return_immediately=not failure,
        failure=expected,
    )
    if failure:
        with pytest.raises(RuntimeError) as caught:
            await main.serve(str(config_path), tmp_path)
        assert caught.value is expected
    else:
        assert await main.serve(str(config_path), tmp_path) == 0
    assert cast(Any, observed["runtime_task"]).done()
    _assert_real_app_closed(observed, tmp_path, socket_path)

@pytest.mark.asyncio
async def test_cli_serve_stop_signal_returns_zero_and_removes_handler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real registered stop callback requests one normal runtime shutdown."""

    config_path, socket_path = _prepare_real_cli_fixture(monkeypatch, tmp_path)
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    _capture_cli_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    _observe_cli_dashboard(monkeypatch, observed, dashboard_started)
    loop = asyncio.get_running_loop()
    callbacks: dict[int, Callable[..., object]] = {}
    removed: list[int] = []
    real_add = loop.add_signal_handler
    real_remove = loop.remove_signal_handler

    def add_signal_handler(sig: int, callback: Callable[..., object], *args: object) -> None:
        callbacks[sig] = callback
        real_add(sig, callback, *args)

    def remove_signal_handler(sig: int) -> bool:
        removed.append(sig)
        return real_remove(sig)

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", remove_signal_handler)
    serving = asyncio.create_task(main.serve(str(config_path), tmp_path))
    try:
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        cast(Any, callbacks[signal.SIGTERM])()
        assert await serving == 0
        assert signal.SIGTERM in removed
        assert signal.SIGINT in removed
    finally:
        if not serving.done():
            serving.cancel()
        try:
            await serving
        except asyncio.CancelledError:
            pass
    _assert_real_app_closed(observed, tmp_path, socket_path)


@pytest.mark.parametrize("transport_failure", [False, True])
@pytest.mark.asyncio
async def test_cli_serve_settings_restart_commits_real_channel_and_returns_75(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport_failure: bool,
) -> None:
    """A real settings restart commits its pipe frame and settles the runtime."""

    if not hasattr(signal, "SIGUSR2"):
        pytest.skip("SIGUSR2 不可用")
    config_path, socket_path = _prepare_real_cli_fixture(monkeypatch, tmp_path)
    read_fd, write_fd = os.pipe()
    monkeypatch.setenv("AKASHIC_SUPERVISED", "1")
    monkeypatch.setenv("AKASHIC_BOOT_ID", "boot-cli-test")
    monkeypatch.setenv("AKASHIC_RESTART_NONCE", "n" * 64)
    monkeypatch.setenv("AKASHIC_LIFECYCLE_FD", str(write_fd))
    observed: dict[str, object] = {}
    _capture_real_app_runtime(monkeypatch, observed)
    _capture_cli_runtime(monkeypatch, observed)
    dashboard_started = asyncio.Event()
    _observe_cli_dashboard(monkeypatch, observed, dashboard_started)
    readiness_marked = asyncio.Event()
    real_mark_ready = main.RuntimeReadiness.mark_ready

    def mark_ready(readiness: Any) -> None:
        real_mark_ready(readiness)
        readiness_marked.set()

    monkeypatch.setattr(main.RuntimeReadiness, "mark_ready", mark_ready)
    loop = asyncio.get_running_loop()
    callbacks: dict[int, Callable[..., object]] = {}
    real_add = loop.add_signal_handler
    real_remove = loop.remove_signal_handler

    def add_signal_handler(sig: int, callback: Callable[..., object], *args: object) -> None:
        callbacks[sig] = callback
        real_add(sig, callback, *args)

    monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
    monkeypatch.setattr(loop, "remove_signal_handler", real_remove)
    serving = asyncio.create_task(main.serve(str(config_path), tmp_path))
    serving_retrieved = False
    try:
        async with asyncio.timeout(10):
            await dashboard_started.wait()
        async with asyncio.timeout(10):
            await readiness_marked.wait()
        ready_frames = [
            json.loads(line)
            for line in os.read(read_fd, 65536).splitlines()
        ]
        assert any(
            frame["type"] == "ready" and frame["bootId"] == "boot-cli-test"
            for frame in ready_frames
        )
        if transport_failure:
            runtime = cast(Any, observed["app_runtime"])
            core = runtime.core
            assert core is not None
            cleanup_error = RuntimeError("cli restart cleanup failed")
            real_stop = core.stop

            async def stop_with_error() -> None:
                await real_stop()
                raise cleanup_error

            core.stop = stop_with_error
            os.close(read_fd)
            read_fd = -1
        cast(Any, callbacks[signal.SIGUSR2])()
        if transport_failure:
            try:
                with pytest.raises(BrokenPipeError) as caught:
                    async with asyncio.timeout(10):
                        await serving
            finally:
                if serving.done():
                    serving_retrieved = True
            runtime_cancel_error = cast(
                asyncio.CancelledError,
                observed["runtime_cancel_error"],
            )
            assert caught.value.__cause__ is runtime_cancel_error
            assert runtime_cancel_error.__cause__ is cleanup_error
        else:
            try:
                async with asyncio.timeout(10):
                    serving_result = await serving
            finally:
                if serving.done():
                    serving_retrieved = True
            assert serving_result == RESTART_EXIT_CODE
            os.close(write_fd)
            write_fd = -1
            frames = [json.loads(line) for line in os.read(read_fd, 65536).splitlines()]
            commits = [frame for frame in frames if frame["type"] == "commit"]
            assert len(commits) == 1
            assert commits[0]["bootId"] == "boot-cli-test"
            assert commits[0]["nonce"] == "n" * 64
            assert commits[0]["requestId"].startswith("settings_")
        assert cast(Any, observed["runtime_task"]).done()
    finally:
        try:
            if not serving_retrieved:
                if not serving.done():
                    serving.cancel()
                try:
                    await serving
                except asyncio.CancelledError:
                    pass
        finally:
            if write_fd >= 0:
                os.close(write_fd)
            if read_fd >= 0:
                os.close(read_fd)
    _assert_real_app_closed(observed, tmp_path, socket_path)


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
        plugin_manager=types.SimpleNamespace(
            bind_endpoint_switcher=lambda _: None,
            configure_dashboard_routes=lambda _: None,
        ),
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
    assert "模型" not in config_text
    assert "Telegram" not in config_text
    assert "QQ" not in config_text
    assert "plugin-data" in config_text
    assert "[channels.chat]" not in config_text
    assert "[mobile_realtime]" not in config_text
    assert "6322" not in config_text
    assert "[runtime]\n" in config_text
    assert 'workspace = "~/.akashic/workspace"' in config_text
    # 启动迁移完成后由 MessageLog owner 创建 canonical schema。
    assert not (workspace / "sessions.db").exists()
    assert not (workspace / "observe").exists()
    assert not (workspace / "memory" / "consolidation_writes.db").exists()
    assert not (workspace / "memory" / "journal").exists()
    assert not (workspace / "memory" / "memory2.db").exists()
    assert not (workspace / "memory" / "VEDA.md").exists()
    assert not (workspace / "plugin-data").exists()
    assert not (workspace / "memes").exists()
    assert not (workspace / "PROACTIVE_CONTEXT.md").exists()
    assert not (workspace / "mcp").exists()
    assert not (workspace / "proactive_sources.json").exists()
    assert not (workspace / "proactive.db").exists()
    assert not (workspace / "skills").exists()
    assert not (workspace / "drift").exists()
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
    veda_path.parent.mkdir(parents=True, exist_ok=True)
    veda_path.write_text("custom veda\n", encoding="utf-8")

    summary_skip = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
    )
    assert not self_path.exists()
    assert veda_path.read_text(encoding="utf-8") == "custom veda\n"
    assert veda_path not in summary_skip.created + summary_skip.overwritten

    summary_force = workspace_init.init_workspace(
        config_path=config_path,
        workspace=workspace,
        force=True,
    )
    assert not self_path.exists()
    assert veda_path.read_text(encoding="utf-8") == "custom veda\n"
    assert self_path not in summary_force.created + summary_force.overwritten
    assert veda_path not in summary_force.created + summary_force.overwritten
