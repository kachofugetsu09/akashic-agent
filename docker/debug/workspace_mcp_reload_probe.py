from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bootstrap.app as bootstrap_app
from agent.config_models import (
    AppServerConfig,
    ChannelsConfig,
    Config,
    WebChatConfig,
)
from agent.plugins.manager import PluginManager
from agent.tools.base import ToolResult
from proactive_v2.config import ProactiveConfig
from docker.debug.programmatic_control_probe import (
    _prepare_host_sandbox,
    _repository_digest,
)


SERVER_SOURCE = r'''from __future__ import annotations
import json
import os
from pathlib import Path
import sys

log = Path(os.environ["LIFECYCLE_LOG"])
version = os.environ["VERSION"]
instance = os.environ["INSTANCE"]
pid = os.getpid()
starttime = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[19]
marker = Path(os.environ["MARKER_PATH"]).read_text(encoding="utf-8").strip()

def record(event: str) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, "pid": pid, "starttime": starttime, "version": version, "instance": instance}) + "\n")

record("started")
try:
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25"}
        elif method == "tools/list":
            result = {"tools": [{"name": "version", "description": "Return server version", "inputSchema": {"type": "object", "properties": {}}}]}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": f"{version}:{marker}"}]}
        else:
            result = {}
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
finally:
    record("stopped")
'''


def _config() -> Config:
    return Config(
        provider="openai",
        model="workspace-mcp-gate",
        api_key="gate",
        system_prompt="workspace MCP gate",
        base_url="http://127.0.0.1:9/v1",
        channels=ChannelsConfig(chat=WebChatConfig(enabled=False)),
        app_server=AppServerConfig(enabled=False),
        proactive=ProactiveConfig(enabled=False),
        memory_optimizer_enabled=False,
        multimodal=False,
        spawn_enabled=False,
    )


@contextmanager
def _ephemeral_dashboard() -> Iterator[None]:
    """让真实 AppRuntime 的 dashboard 绑定随机端口。"""

    original = bootstrap_app.build_dashboard_server

    def build(**kwargs: Any) -> object:
        return original(host="127.0.0.1", port=0, **kwargs)

    bootstrap_app.build_dashboard_server = build  # type: ignore[assignment]
    try:
        yield
    finally:
        bootstrap_app.build_dashboard_server = original


def _write_server(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "watch.txt").write_text("one", encoding="utf-8")
    server = root / "synthetic_mcp.py"
    server.write_text(SERVER_SOURCE, encoding="utf-8")
    return server


def _write_declaration(
    declarations: Path,
    name: str,
    server: Path,
    lifecycle: Path,
    version: str,
    *,
    watch_path: str = "",
) -> Path:
    declarations.mkdir(parents=True, exist_ok=True)
    watch = f'watch_paths = ["{watch_path}"]\n' if watch_path else ""
    path = declarations / f"{name}.toml"
    path.write_text(
        "schema_version = 1\n"
        f'name = "{name}"\n'
        f'command = ["{sys.executable}", "{server}"]\n'
        f"{watch}"
        "[env]\n"
        f'VERSION = "{version}"\n'
        f'INSTANCE = "{name}"\n'
        f'LIFECYCLE_LOG = "{lifecycle}"\n'
        f'MARKER_PATH = "{server.parent / "watch.txt"}"\n',
        encoding="utf-8",
    )
    return path


def _lifecycle(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _pid_starttime(pid: int) -> str | None:
    stat = Path(f"/proc/{pid}/stat")
    try:
        content = stat.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return content.rsplit(")", 1)[1].split()[19]


def _running_pids(
    path: Path,
    *,
    instance: str = "",
    version: str = "",
) -> list[int]:
    """按 PID 与 starttime 判断 fixture 启动的同一进程是否仍存活。"""

    started = [
        item
        for item in _lifecycle(path)
        if item["event"] == "started"
        and (not instance or item["instance"] == instance)
        and (not version or item["version"] == version)
    ]
    return sorted(
        {
            int(item["pid"])
            for item in started
            if _pid_starttime(int(item["pid"])) == item["starttime"]
        }
    )


async def _wait_until(predicate: Any, label: str, timeout: float = 8.0) -> None:
    """在固定 deadline 内等待真实 watcher 状态。"""

    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.02)
    if not predicate():
        raise AssertionError(f"等待失败: {label}")


async def _call_version(lease: Any, server_name: str = "docs") -> str:
    registry = lease.snapshot.tool_registry
    if registry is None:
        raise AssertionError("snapshot 缺少 tool registry")
    result = await registry.execute(
        f"mcp_{server_name}__version",
        {},
        raise_errors=True,
    )
    return result.text if isinstance(result, ToolResult) else result


def _check(checks: list[dict[str, object]], name: str, passed: bool, **evidence: object) -> None:
    checks.append({"name": name, "passed": passed, "evidence": evidence})
    if not passed:
        raise AssertionError(f"{name}: {evidence}")


async def _run_reload_sequence(root: Path, checks: list[dict[str, object]]) -> None:
    workspace = root / "workspace"
    mcp_root = workspace / "mcp"
    declarations = mcp_root / "servers"
    runtime_root = mcp_root / "synthetic"
    lifecycle = root / "lifecycle.jsonl"
    server = _write_server(runtime_root)
    marker = runtime_root / "watch.txt"
    marker.write_text("one", encoding="utf-8")
    _write_declaration(
        declarations,
        "docs",
        server,
        lifecycle,
        "v1",
        watch_path="../synthetic/watch.txt",
    )
    (workspace / "mcp_servers.json").write_text(
        json.dumps(
            {
                "servers": {
                    "legacy": {
                        "command": [sys.executable, str(server)],
                        "env": {
                            "VERSION": "legacy",
                            "INSTANCE": "legacy",
                            "LIFECYCLE_LOG": str(lifecycle),
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    app = bootstrap_app.AppRuntime(_config(), workspace)
    with _ephemeral_dashboard():
        await app.start()
    manager = app.core.plugin_manager
    if manager is None:
        raise AssertionError("AppRuntime 缺少 PluginManager")
    try:
        first = manager.active_workspace_mcp
        _check(
            checks,
            "initial-v1",
            first is not None and await _current_version(manager) == "v1:one",
            generation=first.generation_id if first else None,
        )
        legacy_started = [
            item
            for item in _lifecycle(lifecycle)
            if item["event"] == "started" and item["instance"] == "legacy"
        ]
        _check(
            checks,
            "legacy-json-ignored",
            "mcp_legacy__version" not in manager.current_snapshot.tool_registry.get_registered_names()  # type: ignore[union-attr]
            and not legacy_started,
            path=str(workspace / "mcp_servers.json"),
            lifecycle=legacy_started,
        )

        old_turn_started = asyncio.Event()
        release_old_turn = asyncio.Event()

        async def old_turn() -> tuple[str, str]:
            lease = await manager.snapshot_store.acquire()
            old_turn_started.set()
            await release_old_turn.wait()
            try:
                return await _call_version(lease), lease.snapshot.snapshot_id
            finally:
                await lease.release()

        old_turn_task = asyncio.create_task(old_turn(), name="workspace-mcp-old-turn")
        await old_turn_started.wait()
        old_generation = manager.active_workspace_mcp
        _write_declaration(
            declarations,
            "docs",
            server,
            lifecycle,
            "v2",
            watch_path="../synthetic/watch.txt",
        )
        await _wait_until(
            lambda: manager.active_workspace_mcp is not old_generation,
            "v2 generation",
        )
        new_lease = await manager.snapshot_store.acquire()
        new_version = await _call_version(new_lease)
        new_snapshot_id = new_lease.snapshot.snapshot_id
        await new_lease.release()
        release_old_turn.set()
        old_version, old_snapshot_id = await old_turn_task
        _check(
            checks,
            "old-new-isolation",
            old_version == "v1:one" and new_version == "v2:one",
            oldTurnResult=old_version,
            oldSnapshot=old_snapshot_id,
            newSnapshot=new_snapshot_id,
        )
        await _wait_until(
            lambda: not _running_pids(lifecycle, version="v1"),
            "v1 lease drain",
        )

        before_watch = manager.active_workspace_mcp
        marker.write_text("two", encoding="utf-8")
        await _wait_until(
            lambda: manager.active_workspace_mcp is not before_watch,
            "watch content generation",
        )
        _check(
            checks,
            "watch-content-reload",
            await _current_version(manager) == "v2:two",
            before=before_watch.generation_id if before_watch else None,
            after=manager.active_workspace_mcp.generation_id,
            toolResult=await _current_version(manager),
        )

        stable = manager.active_workspace_mcp
        docs = declarations / "docs.toml"
        docs.write_text("schema_version = 1\nname = [\n", encoding="utf-8")
        watcher = app.core.workspace_mcp_watcher
        await _wait_until(lambda: watcher.last_error is not None, "bad TOML rejection")
        _check(
            checks,
            "bad-toml-rollback",
            manager.active_workspace_mcp is stable
            and await _current_version(manager) == "v2:two",
            error=watcher.last_error,
        )

        _write_declaration(declarations, "a_good", server, lifecycle, "partial")
        _write_declaration(
            declarations,
            "z_bad",
            root / "missing-server.py",
            lifecycle,
            "bad",
        )
        _write_declaration(
            declarations,
            "docs",
            server,
            lifecycle,
            "v2",
            watch_path="../synthetic/watch.txt",
        )
        previous_error = watcher.last_error
        await _wait_until(
            lambda: watcher.last_error is not None
            and watcher.last_error != previous_error,
            "partial candidate rejection",
        )
        await _wait_until(
            lambda: not _running_pids(lifecycle, instance="a_good"),
            "partial candidate process cleanup",
        )
        _check(
            checks,
            "partial-candidate-cleanup",
            manager.active_workspace_mcp is stable
            and not _running_pids(lifecycle, instance="a_good"),
            error=watcher.last_error,
            runningCandidatePids=_running_pids(
                lifecycle,
                instance="a_good",
            ),
            lifecycle=_lifecycle(lifecycle),
        )

        _write_declaration(declarations, "z_bad", server, lifecycle, "repaired")
        await _wait_until(
            lambda: manager.active_workspace_mcp is not stable
            and watcher.last_error is None,
            "automatic recovery",
        )
        repaired_lease = await manager.snapshot_store.acquire()
        _check(
            checks,
            "automatic-recovery",
            await _call_version(repaired_lease, "z_bad") == "repaired:two",
            generation=manager.active_workspace_mcp.generation_id,
        )

        for declaration in declarations.glob("*.toml"):
            declaration.unlink()
        repaired_generation = manager.active_workspace_mcp
        await _wait_until(
            lambda: manager.active_workspace_mcp is not repaired_generation
            and not manager.active_workspace_mcp.catalog.servers,
            "empty generation",
        )
        empty_lease = await manager.snapshot_store.acquire()
        empty_registry = empty_lease.snapshot.tool_registry
        if empty_registry is None:
            raise AssertionError("empty snapshot 缺少 tool registry")
        empty_names = sorted(empty_registry.get_registered_names())
        await empty_lease.release()
        leased_pids = _running_pids(lifecycle)
        await repaired_lease.release()
        await _wait_until(lambda: not _running_pids(lifecycle), "empty lease drain")
        _check(
            checks,
            "delete-all-drains",
            bool(leased_pids)
            and not manager.active_workspace_mcp.catalog.servers
            and "mcp_docs__version" not in empty_names
            and "mcp_a_good__version" not in empty_names
            and "mcp_z_bad__version" not in empty_names,
            leasedPids=leased_pids,
            emptyRegistryNames=empty_names,
            lifecycle=_lifecycle(lifecycle),
        )
    finally:
        await app.shutdown()
    _check(
        checks,
        "shutdown-no-residual",
        app.core.workspace_mcp_watcher_task.done()
        and not _running_pids(lifecycle)
        and not [
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name() == "workspace_mcp_watcher"
            and not task.done()
        ],
        runningPids=_running_pids(lifecycle),
    )


async def _current_version(manager: PluginManager) -> str:
    lease = await manager.snapshot_store.acquire()
    try:
        return await _call_version(lease)
    finally:
        await lease.release()


async def _run_plugin_conflict(root: Path, checks: list[dict[str, object]]) -> None:
    workspace = root / "conflict-workspace"
    declarations = workspace / "mcp" / "servers"
    server = _write_server(workspace / "mcp" / "server")
    lifecycle = root / "conflict-lifecycle.jsonl"
    _write_declaration(declarations, "docs", server, lifecycle, "workspace")
    plugins = root / "conflict-plugins"
    plugin = plugins / "collision"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_text(
        "from agent.plugins import McpServerSpec, Plugin\n"
        "class CollisionPlugin(Plugin):\n"
        "    name = 'collision'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='docs', command=('python', 'unused.py'))]\n",
        encoding="utf-8",
    )
    previous = os.environ.get("AKASHIC_EXTRA_PLUGIN_DIRS")
    os.environ["AKASHIC_EXTRA_PLUGIN_DIRS"] = str(plugins)
    app = bootstrap_app.AppRuntime(_config(), workspace)
    error = ""
    try:
        with _ephemeral_dashboard():
            await app.start()
    except RuntimeError as caught:
        error = str(caught)
    finally:
        if previous is None:
            os.environ.pop("AKASHIC_EXTRA_PLUGIN_DIRS", None)
        else:
            os.environ["AKASHIC_EXTRA_PLUGIN_DIRS"] = previous
    _check(
        checks,
        "plugin-name-conflict-fail-loud",
        "workspace MCP 与插件声明冲突" in error
        and not _running_pids(lifecycle),
        error=error,
        runningPids=_running_pids(lifecycle),
    )


async def _run_watcher_fault(root: Path, checks: list[dict[str, object]]) -> None:
    import agent.mcp.watcher as watcher_module

    workspace = root / "fault-workspace"
    app = bootstrap_app.AppRuntime(_config(), workspace)
    original = watcher_module.declarations_input_revision

    def fail(*_args: object, **_kwargs: object) -> str:
        raise KeyError("synthetic watcher fault")

    watcher_module.declarations_input_revision = fail
    error = ""
    try:
        with _ephemeral_dashboard():
            await asyncio.wait_for(app.run(), timeout=8)
    except KeyError as caught:
        error = str(caught)
    finally:
        watcher_module.declarations_input_revision = original
    task = app.core.workspace_mcp_watcher_task
    _check(
        checks,
        "watcher-fault-supervised",
        "synthetic watcher fault" in error and task is not None and task.done(),
        error=error,
        taskDone=task.done() if task else None,
    )


async def run_gate(root: Path) -> dict[str, object]:
    """运行完整 AppRuntime workspace MCP Docker gate。"""

    checks: list[dict[str, object]] = []
    run_root = root / f"run-{uuid4().hex}"
    home = run_root / "home"
    home.mkdir(parents=True, exist_ok=True)
    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    try:
        await _run_reload_sequence(run_root, checks)
        await _run_plugin_conflict(run_root, checks)
        await _run_watcher_fault(run_root, checks)
    finally:
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
    return {"status": "passed", "checks": checks, "root": str(run_root)}


def _manifest_digest(manifest: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sandbox_manifest(app_root: Path, source: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in source:
        path = app_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"sandbox 源码缺失: {relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _git_state(repo: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    dirty = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return head, dirty


def _compose_command(repo: Path, project: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.control-gate.yml"),
    ]


def _compose_environment(sandbox: Path) -> dict[str, str]:
    env = {
        **os.environ,
        "AKASHIC_CONTROL_SANDBOX": str(sandbox),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    env.pop("AKASHIC_EXTRA_PLUGIN_DIRS", None)
    return env


def _project_resources(resource: str, project: str) -> list[str]:
    """按 Compose project 标签返回指定类型的残留资源。"""

    list_flags = "-aq" if resource == "container" else "-q"
    command = [
        "docker",
        resource,
        "ls",
        list_flags,
        "--filter",
        f"label=com.docker.compose.project={project}",
    ]
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Docker {resource} residual query failed: {completed.stderr.strip()}"
        )
    return completed.stdout.split()


def _run_host(report_root: Path | None) -> int:
    """在独立 Compose project 中运行只读 sandbox gate。"""

    repo = Path(__file__).resolve().parents[2]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    report_dir = (
        report_root / run_id
        if report_root is not None
        else repo / "docker/debug/reports/workspace-mcp" / run_id
    )
    report_dir.mkdir(parents=True)
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-workspace-mcp-gate-", dir="/tmp"))
    _prepare_host_sandbox(sandbox, repo)

    head_before, dirty_before = _git_state(repo)
    source_before = _repository_digest(repo)
    app_before = _sandbox_manifest(sandbox / "app", source_before)
    project = f"akashic-workspace-mcp-{run_id.lower()}"
    compose = _compose_command(repo, project)
    env = _compose_environment(sandbox)
    image = "akashic-agent-control-gate:latest"
    controller_error = ""
    internal: dict[str, object] = {}
    cleanup_returncode = -1
    residual_containers: list[str] = []
    residual_networks: list[str] = []
    residual_volumes: list[str] = []
    image_id = ""
    try:
        build = subprocess.run(
            [*compose, "build", "model-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(f"control-gate image build failed: {build.returncode}")
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        image_id = inspected.stdout.strip()
        completed = subprocess.run(
            [
                *compose,
                "run",
                "--rm",
                "-T",
                "--no-deps",
                "-e",
                "AKASHIC_WORKSPACE_MCP_DOCKER_GATE=1",
                "control-probe",
                "python",
                "docker/debug/workspace_mcp_reload_probe.py",
                "--internal",
                "--workspace",
                "/sandbox/workspace-mcp-gate",
                "--report",
                "/sandbox/reports/inside.json",
            ],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        (report_dir / "container.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (report_dir / "container.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        inside_path = sandbox / "reports/inside.json"
        if inside_path.exists():
            internal = json.loads(inside_path.read_text(encoding="utf-8"))
            shutil.copy2(inside_path, report_dir / "inside.json")
        if completed.returncode != 0:
            raise RuntimeError(
                f"container gate failed: {completed.returncode}; "
                f"stderr={completed.stderr[-2000:]}"
            )
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        cleanup = subprocess.run(
            [*compose, "down", "--remove-orphans", "--volumes"],
            cwd=repo,
            env=env,
            check=False,
        )
        cleanup_returncode = cleanup.returncode
        residual_containers = _project_resources("container", project)
        residual_networks = _project_resources("network", project)
        residual_volumes = _project_resources("volume", project)

    head_after, dirty_after = _git_state(repo)
    source_after = _repository_digest(repo)
    app_after = _sandbox_manifest(sandbox / "app", source_before)
    source_digest = _manifest_digest(source_before)
    app_digest = _manifest_digest(app_before)
    integrity = (
        head_before == head_after
        and dirty_before == dirty_after
        and source_before == source_after
        and source_before == app_before == app_after
        and cleanup_returncode == 0
        and not residual_containers
        and not residual_networks
        and not residual_volumes
    )
    isolation = internal.get("isolation")
    passed = (
        not controller_error
        and internal.get("status") == "passed"
        and isinstance(isolation, dict)
        and isolation.get("passed") is True
        and integrity
    )
    report = {
        "status": "passed" if passed else "failed",
        "runId": run_id,
        "head": head_before,
        "dirtyStatus": dirty_before.splitlines(),
        "sourceDigest": source_digest,
        "sourceFileCount": len(source_before),
        "sandboxAppDigest": app_digest,
        "composeProject": project,
        "composeFile": str(repo / "docker/debug/docker-compose.control-gate.yml"),
        "image": image,
        "imageId": image_id,
        "sandbox": str(sandbox),
        "controllerError": controller_error,
        "cleanupReturncode": cleanup_returncode,
        "residualContainers": residual_containers,
        "residualNetworks": residual_networks,
        "residualVolumes": residual_volumes,
        "repositoryUnchanged": integrity,
        "internal": internal,
        "reportDir": str(report_dir),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    (report_dir / "gate.json").write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    shutil.rmtree(sandbox)
    return 0 if passed else 1


def _container_isolation() -> dict[str, object]:
    docker_gate = os.environ.get("AKASHIC_WORKSPACE_MCP_DOCKER_GATE") == "1"
    extra_present = "AKASHIC_EXTRA_PLUGIN_DIRS" in os.environ
    mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    host_cache_mounts = [
        line
        for line in mountinfo.splitlines()
        if ".akashic-plugin/cache" in line
    ]
    home = str(Path.home())
    passed = (
        not docker_gate
        or (
            not extra_present
            and home == "/sandbox/home"
            and not host_cache_mounts
        )
    )
    return {
        "passed": passed,
        "dockerGate": docker_gate,
        "home": home,
        "extraPluginDirsPresent": extra_present,
        "hostPluginCacheMounts": host_cache_mounts,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="workspace MCP 真实 Docker 热重载 gate")
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-root", type=Path)
    return parser


def _run_internal(workspace: Path | None, report: Path | None) -> int:
    temporary: tempfile.TemporaryDirectory[str] | None = None
    root = workspace
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="workspace-mcp-gate-")
        root = Path(temporary.name)
    try:
        payload = asyncio.run(run_gate(root.resolve()))
        payload["isolation"] = _container_isolation()
        if not payload["isolation"]["passed"]:  # type: ignore[index]
            raise RuntimeError(f"container isolation failed: {payload['isolation']}")
        exit_code = 0
    except Exception as error:
        payload = {
            "status": "failed",
            "error": {"type": type(error).__name__, "message": str(error)},
            "root": str(root),
        }
        exit_code = 1
    finally:
        if temporary is not None:
            temporary.cleanup()
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(encoded + "\n", encoding="utf-8")
    return exit_code


def main() -> int:
    args = _build_parser().parse_args()
    if args.internal:
        return _run_internal(args.workspace, args.report)
    if args.workspace is not None or args.report is not None:
        raise SystemExit("--workspace/--report 仅供 --internal 使用")
    return _run_host(args.report_root)


if __name__ == "__main__":
    raise SystemExit(main())
