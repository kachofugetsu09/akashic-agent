#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.control.client import ControlClient


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    evidence: object


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: str
    checks: list[CheckResult]


def _gate_status(checks: list[CheckResult]) -> str:
    return "passed" if all(check.passed for check in checks) else "failed"


def _sandbox_is_protected(sandbox: Path, protected: list[Path]) -> bool:
    return any(sandbox == path or sandbox.is_relative_to(path) for path in protected)


def _controller_gate_passed(
    *,
    build_returncode: int,
    integrity_returncode: int,
    smoke_passed: bool,
    cleanup_returncode: int,
    unchanged: bool,
    controller_error: str,
) -> bool:
    return (
        build_returncode == 0
        and integrity_returncode == 0
        and smoke_passed
        and cleanup_returncode == 0
        and unchanged
        and not controller_error
    )


def _run(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _repository_digest(repo: Path) -> str:
    digest = hashlib.sha256()
    commands = (
        ("status", "--porcelain=v1", "--untracked-files=all"),
        ("diff", "--binary", "--no-ext-diff"),
        ("diff", "--binary", "--cached", "--no-ext-diff"),
        ("submodule", "status", "--recursive"),
    )
    for command in commands:
        digest.update(b"\0".join(part.encode() for part in command))
        digest.update(_run(repo, *command))
    paths = _run(
        repo,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ).split(b"\0")
    for raw_path in paths:
        if not raw_path:
            continue
        path = repo / os.fsdecode(raw_path)
        if not path.is_file() or path.is_symlink():
            continue
        digest.update(raw_path)
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _mounted_tree_digest(root: Path, *, excluded_roots: set[Path] | None = None) -> str:
    """不依赖 Git 元数据，摘要只读挂载中容器实际可见的源码树。"""

    digest = hashlib.sha256()
    excluded = {path.resolve() for path in (excluded_roots or set())}
    for directory, names, files in os.walk(root, topdown=True):
        current = Path(directory)
        names[:] = sorted(
            name
            for name in names
            if name not in {".codegraph", ".git", "__pycache__"}
            and (current / name).resolve() not in excluded
        )
        for name in sorted(files):
            path = current / name
            if path.resolve() in excluded:
                continue
            relative = path.relative_to(root)
            digest.update(os.fsencode(relative))
            if path.is_symlink():
                digest.update(b"symlink\0" + os.fsencode(os.readlink(path)))
                continue
            if not path.is_file():
                continue
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _repositories() -> list[Path]:
    repositories = [Path("/app")]
    plugin_root = Path("/fixtures/plugins")
    repositories.extend(
        path
        for path in sorted(plugin_root.iterdir())
        if path.is_dir() and (path / ".git").exists()
    )
    return repositories


def _host_repositories(repo: Path, plugin_root: Path) -> list[Path]:
    repositories = [repo]
    repositories.extend(
        path
        for path in sorted(plugin_root.iterdir())
        if path.is_dir() and (path / ".git").exists()
    )
    index = 0
    while index < len(repositories):
        parent = repositories[index]
        output = _run(parent, "submodule", "status", "--recursive").decode()
        for line in output.splitlines():
            match = re.match(r"^[ +\-U]?[0-9a-f]{40} (.+?)(?: \(.+\))?$", line)
            if match is None:
                continue
            submodule = (parent / match.group(1)).resolve()
            if submodule not in repositories:
                repositories.append(submodule)
        index += 1
    return repositories


def _mount_options(path: Path) -> set[str]:
    output = subprocess.run(
        ["findmnt", "--target", str(path), "--noheadings", "--output", "OPTIONS"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    return set(output.split(","))


def _path_check(check_id: str, actual: Path, expected: Path) -> CheckResult:
    resolved = actual.resolve()
    return CheckResult(
        check_id=check_id,
        passed=resolved == expected,
        evidence={"actual": str(resolved), "expected": str(expected)},
    )


def _sandbox_integrity() -> GateResult:
    repositories = _repositories()
    excluded = {
        Path("/app/docker/debug/profiles").resolve(),
        Path("/app/logs").resolve(),
        Path("/app/static").resolve(),
    }
    before = {
        str(repo): _mounted_tree_digest(
            repo,
            excluded_roots=excluded if repo == Path("/app") else None,
        )
        for repo in repositories
    }

    sandbox = Path("/sandbox")
    cache = Path.home() / ".akashic-plugin" / "cache"
    test_plugin = cache / "gate" / "integrity" / "1.0.0" / "plugin.py"
    test_plugin.parent.mkdir(parents=True, exist_ok=True)
    _ = test_plugin.write_text("REVISION = 1\n", encoding="utf-8")
    _ = test_plugin.write_text("REVISION = 2\n", encoding="utf-8")

    after = {
        str(repo): _mounted_tree_digest(
            repo,
            excluded_roots=excluded if repo == Path("/app") else None,
        )
        for repo in repositories
    }
    app_options = _mount_options(Path("/app"))
    fixtures_options = _mount_options(Path("/fixtures/plugins"))
    sandbox_options = _mount_options(sandbox)
    checks = [
        CheckResult("app_read_only", "ro" in app_options, sorted(app_options)),
        CheckResult(
            "plugin_fixtures_read_only",
            "ro" in fixtures_options,
            sorted(fixtures_options),
        ),
        CheckResult(
            "sandbox_writable", "rw" in sandbox_options, sorted(sandbox_options)
        ),
        _path_check("home_isolated", Path.home(), Path("/sandbox/home")),
        _path_check(
            "workspace_isolated",
            Path(os.environ["AKASHIC_DEBUG_WORKSPACE"]),
            Path("/sandbox/workspace"),
        ),
        _path_check(
            "config_isolated",
            Path(os.environ["AKASHIC_DEBUG_CONFIG"]),
            Path("/sandbox/config.toml"),
        ),
        CheckResult(
            "plugin_cache_isolated",
            cache.resolve() == Path("/sandbox/home/.akashic-plugin/cache"),
            str(cache.resolve()),
        ),
        CheckResult(
            "repositories_unchanged",
            before == after,
            {
                "repositories": len(repositories),
                "before": before,
                "after": after,
            },
        ),
        CheckResult(
            "isolated_plugin_updated",
            test_plugin.read_text(encoding="utf-8") == "REVISION = 2\n",
            str(test_plugin),
        ),
    ]
    status = _gate_status(checks)
    result = GateResult(gate_id="G-1", status=status, checks=checks)
    report_dir = sandbox / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    _ = (report_dir / "sandbox-integrity.json").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(report)
    return result


def _run_controller(*, scenario: str, phase: str) -> int:
    repo = Path(__file__).resolve().parents[2]
    plugin_root = Path(
        os.environ.get("AKASHIC_PLUGIN_SOURCE", "/mnt/data/coding/akashic-plugin")
    ).resolve()
    host_cache = (Path.home() / ".akashic-plugin" / "cache").resolve()
    sandbox = Path(
        tempfile.mkdtemp(prefix="akashic-plugin-gate-", dir="/tmp")
    ).resolve()
    (sandbox / "static").mkdir()
    protected = [repo.resolve(), plugin_root, host_cache]
    if _sandbox_is_protected(sandbox, protected):
        shutil.rmtree(sandbox)
        raise SystemExit(f"Gate sandbox 不能位于受保护路径内：{sandbox}")

    repositories = _host_repositories(repo, plugin_root)
    before = {str(path): _repository_digest(path) for path in repositories}
    env = {
        **os.environ,
        "AKASHIC_GATE_SANDBOX": str(sandbox),
        "AKASHIC_PLUGIN_SOURCE": str(plugin_root),
        "UID": str(os.getuid()),
        "GID": str(os.getgid()),
    }
    project = sandbox.name.replace("_", "-")
    compose = [
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(repo / "docker/debug/docker-compose.plugin-gate.yml"),
    ]
    command = [
        *compose,
        "run",
        "--rm",
        "akashic-plugin-gate",
        "python",
        "docker/debug/plugin_hot_reload_probe.py",
        "--scenario",
        "sandbox-integrity",
        "--inside-container",
    ]
    build_returncode = -1
    integrity_returncode = -1
    smoke_passed = False
    smoke_evidence: dict[str, object] = {"skipped": True}
    cleanup_returncode = -1
    controller_error = ""
    try:
        build = subprocess.run(
            [*compose, "build", "akashic-plugin-gate"],
            cwd=repo,
            env=env,
            check=False,
        )
        build_returncode = build.returncode
        if build_returncode == 0:
            integrity = subprocess.run(command, cwd=repo, env=env, check=False)
            integrity_returncode = integrity.returncode
        if integrity_returncode == 0:
            smoke_passed, smoke_evidence = _run_runtime_smoke(
                repo=repo,
                sandbox=sandbox,
                compose=compose,
                env=env,
                phase=phase if scenario == "full-runtime" else "smoke",
            )
    except Exception as error:
        controller_error = f"{type(error).__name__}: {error}"
    finally:
        try:
            cleanup = subprocess.run(
                [*compose, "down", "--remove-orphans"],
                cwd=repo,
                env=env,
                check=False,
            )
            cleanup_returncode = cleanup.returncode
        except Exception as error:
            suffix = f"{type(error).__name__}: {error}"
            controller_error = f"{controller_error}; {suffix}".strip("; ")
    after = {str(path): _repository_digest(path) for path in repositories}
    unchanged = before == after
    passed = _controller_gate_passed(
        build_returncode=build_returncode,
        integrity_returncode=integrity_returncode,
        smoke_passed=smoke_passed,
        cleanup_returncode=cleanup_returncode,
        unchanged=unchanged,
        controller_error=controller_error,
    )
    report: dict[str, object] = {
        "gate_id": (
            "G-1-host" if scenario == "sandbox-integrity" else f"runtime:{phase}"
        ),
        "status": "passed" if passed else "failed",
        "checks": {
            "image_build_passed": build_returncode == 0,
            "container_gate_passed": integrity_returncode == 0,
            "runtime_smoke_passed": smoke_passed,
            "runtime": smoke_evidence,
            "cleanup_passed": cleanup_returncode == 0,
            "repositories_unchanged": unchanged,
            "repositories": len(repositories),
            "sandbox": str(sandbox),
            "compose_project": project,
            "controller_error": controller_error,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


def _write_smoke_config(
    sandbox: Path,
    *,
    proactive_enabled: bool = False,
    fast_tick: bool = False,
) -> None:
    config = sandbox / "config.toml"
    lines = [
        'provider = "openai"',
        'model = "plugin-gate"',
        'api_key = "gate-not-used"',
        'system_prompt = "plugin gate"',
        "max_iterations = 1",
        "max_tokens = 64",
        "memory_window = 4",
        "memory_optimizer_enabled = false",
        "spawn_enabled = false",
        "",
        "[app_server]",
        "enabled = true",
        'listen = "/sandbox/akashic.sock"',
        "max_connections = 8",
        "ingress_queue_size = 32",
        "outbound_queue_size = 64",
        "",
        "[channels.chat]",
        "enabled = false",
        "",
        "[channels.telegram]",
        "enabled = false",
        'token = ""',
        "",
        "[channels.qq]",
        "enabled = false",
        'bot_uin = ""',
        "",
        "[proactive]",
        'profile = "quiet"',
        f"enabled = {'true' if proactive_enabled else 'false'}",
        "",
    ]
    if fast_tick:
        lines.extend(
            [
                "[proactive.overrides.trigger]",
                "tick_interval_s0 = 1",
                "tick_interval_s1 = 1",
                "tick_jitter = 0.0",
                "",
                "[proactive.target]",
                'channel = "cli"',
                'chat_id = "gate"',
                "",
            ]
        )
    _ = config.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _install_scope_plugin(sandbox: Path) -> Path:
    plugin_dir = sandbox / "home/.akashic-plugin/cache/gate/scope_gate/1.0.0"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from agent.plugins import ManagedServiceSpec, Plugin\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class ScopeGateChannel:\n"
        "    name = 'scope-gate-channel'\n"
        "    def __init__(self, plugin): self.plugin = plugin\n"
        "    async def start(self, ctx): self.plugin.context.kv_store.set('channel_started', True)\n"
        "    async def stop(self): self.plugin.context.kv_store.set('channel_stopped', True)\n"
        "class ScopeGatePlugin(Plugin):\n"
        "    name = 'scope_gate'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def managed_services(cls):\n"
        "        return [ManagedServiceSpec(id='probe', command=('python', 'service.py'), readiness_url='http://127.0.0.1:18766/')]\n"
        "    def channels(self): return [ScopeGateChannel(self)]\n"
        "    async def initialize(self):\n"
        "        self.context.kv_store.set('initialized', True)\n"
        "        self.context.kv_store.set('generation', self.context.generation_id)\n"
        "        self.context.defer('subscription_check', self._check_subscription)\n"
        "        self.subscription = self.context.event_bus.on(TurnCommitted, self._handle)\n"
        "        self.context.create_task(self._worker(), name='scope-gate-worker')\n"
        "        self.context.create_task(self._emit(), name='scope-gate-emit')\n"
        "    async def terminate(self):\n"
        "        self.context.kv_store.set('terminated', True)\n"
        "    async def _worker(self):\n"
        "        try:\n"
        "            await asyncio.Event().wait()\n"
        "        finally:\n"
        "            self.context.kv_store.set('task_cancelled', True)\n"
        "    async def _emit(self):\n"
        "        await asyncio.sleep(0)\n"
        "        self.context.event_bus.enqueue(TurnCommitted(\n"
        "            session_key='gate:scope', channel='gate', chat_id='scope',\n"
        "            input_message='scope', persisted_user_message='scope',\n"
        "            assistant_response='scope', tools_used=[]))\n"
        "    def _check_subscription(self):\n"
        "        self.context.kv_store.set('subscription_closed', not self.subscription.active)\n"
        "    async def _handle(self, event):\n"
        "        self.context.kv_store.set('event_started', True)\n"
        "        await asyncio.sleep(2)\n"
        "        self.context.kv_store.increment('events')\n",
        encoding="utf-8",
    )
    _ = (plugin_dir / "service.py").write_text(
        "import os, signal, sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "data = Path(os.environ['AKA_PLUGIN_DATA_DIR'])\n"
        "(data / 'service.started').write_text('started')\n"
        "def stop(*args):\n"
        "    (data / 'service.stopped').write_text('stopped')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', 18766), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    return sandbox / "workspace/plugin-data/scope_gate-gate/.kv.json"


def _install_fitbit_plugin(sandbox: Path, plugin_root: Path) -> Path:
    source = plugin_root / "fitbit-mcp"
    target = sandbox / "home/.akashic-plugin/cache/gate/fitbit/1.1.0"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache"),
    )
    return sandbox / "workspace/plugin-data/fitbit-gate"


def _install_management_plugin(sandbox: Path) -> tuple[Path, Path, Path]:
    cache = sandbox / "home/.akashic-plugin/cache/gate/management/1.0.0"
    data = sandbox / "workspace/plugin-data/management-gate"
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    cache.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    _ = (cache / "plugin.py").write_text(
        "from agent.plugins import ManagedServiceSpec, Plugin, tool\n"
        "class ManagementPlugin(Plugin):\n"
        "    name = 'management'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def managed_services(cls):\n"
        "        return [ManagedServiceSpec(id='probe', command=('python', 'management_service.py'), readiness_url='http://127.0.0.1:18768/')]\n"
        "    @tool(name='management_probe')\n"
        "    async def probe(self, event):\n"
        '        """Return the management probe state."""\n'
        "        return 'ok'\n",
        encoding="utf-8",
    )
    _ = (cache / "management_service.py").write_text(
        "import os, signal, sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "data = Path(os.environ['AKA_PLUGIN_DATA_DIR'])\n"
        "(data / 'service.started').write_text('started')\n"
        "def stop(*args):\n"
        "    (data / 'service.stopped').write_text('stopped')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self): self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', 18768), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    _ = (data / "retained.txt").write_text("keep", encoding="utf-8")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _ = manifest.write_text(
        '[plugins."management@gate"]\nenabled = true\n',
        encoding="utf-8",
    )
    return cache.parent, data, manifest


def _install_proactive_fetch_plugin(sandbox: Path) -> Path:
    cache = sandbox / "home/.akashic-plugin/cache/gate/proactive_fetch/1.0.0"
    data = sandbox / "workspace/plugin-data/proactive_fetch-gate"
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    cache.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    _ = (cache / "plugin.py").write_text(
        "from agent.plugins import McpServerSpec, Plugin, ProactiveSourceSpec\n"
        "class ProactiveFetchPlugin(Plugin):\n"
        "    name = 'proactive_fetch'\n"
        "    version = '1.0.0'\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        "        return [McpServerSpec(name='proactive_fetch', command=('python', 'mcp_server.py'))]\n"
        "    def proactive_sources(self):\n"
        "        return [ProactiveSourceSpec(id='context', channels=('context',), server='proactive_fetch', fetch_tool='get_context')]\n",
        encoding="utf-8",
    )
    _ = (cache / "mcp_server.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "calls = Path(os.environ['AKA_PLUGIN_DATA_DIR']) / 'fetch_calls.jsonl'\n"
        "tools = [{'name': 'get_context', 'description': 'Get context.', 'inputSchema': {'type': 'object', 'properties': {}}}]\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    if 'id' not in message: continue\n"
        "    method = message.get('method')\n"
        "    result = {'tools': tools} if method == 'tools/list' else {}\n"
        "    if method == 'tools/call':\n"
        "        with calls.open('a', encoding='utf-8') as stream: stream.write(json.dumps(message['params']['name']) + '\\n')\n"
        "        result = {'content': [{'type': 'text', 'text': '{\"available\": true}'}]}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _ = manifest.write_text(
        '[plugins."proactive_fetch@gate"]\nenabled = true\n',
        encoding="utf-8",
    )
    return data / "fetch_calls.jsonl"


def _install_migrated_plugins(sandbox: Path, plugin_root: Path) -> Path:
    cache = sandbox / "home/.akashic-plugin/cache/gate"
    entries: list[str] = []
    observe_source = Path()
    for source_name, plugin_name in (
        ("citation", "citation"),
        ("emotion", "emotion"),
        ("meme", "meme"),
        ("observe", "observe"),
        ("proactive_feedback", "proactive_feedback"),
        ("status_commands", "status_commands"),
    ):
        target = cache / plugin_name / "1.0.0"
        shutil.copytree(
            plugin_root / source_name,
            target,
            ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", ".pytest_cache"
            ),
        )
        entries.append(f'[plugins."{plugin_name}@gate"]\nenabled = true\n')
        if plugin_name == "observe":
            observe_source = target / "plugin.py"
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    driver = cache / "zz_gate_driver" / "1.0.0"
    driver.mkdir(parents=True)
    _ = (driver / "plugin.py").write_text(
        "from __future__ import annotations\n"
        "import asyncio\n"
        "import sqlite3\n"
        "from agent.plugins import Plugin\n"
        "from agent.plugins.snapshot import get_current_runtime_snapshot\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class InspectRuntimeModules:\n"
        "    slot = 'gate_driver.inspect_runtime_modules'\n"
        "    requires = ('before_turn.acquire_session',)\n"
        "    def __init__(self, plugin): self.plugin = plugin\n"
        "    async def run(self, frame):\n"
        "        snapshot = get_current_runtime_snapshot()\n"
        "        slots = [] if snapshot is None else [getattr(item, 'slot', '') for item in snapshot.before_turn_modules + snapshot.prompt_render_modules]\n"
        "        self.plugin.context.kv_store.set('phase_slots', slots)\n"
        "        return frame\n"
        "class GateDriverPlugin(Plugin):\n"
        "    name = 'zz_gate_driver'\n"
        "    def before_turn_modules(self): return [InspectRuntimeModules(self)]\n"
        "    async def initialize(self):\n"
        "        self.context.create_task(self._drive_feedback(), name='gate-feedback-driver')\n"
        "    async def _drive_feedback(self):\n"
        "        await asyncio.sleep(0.2)\n"
        "        content = '/memorystatus 被回复消息：这是一条用于验证主动反馈工作链路的提醒消息【你当前新消息】谢谢'\n"
        "        with sqlite3.connect(self.context.workspace / 'sessions.db') as connection:\n"
        "            connection.execute(\"INSERT INTO messages (id, session_key, seq, role, content, extra, ts) VALUES (?, ?, ?, ?, ?, ?, ?)\", ('cli:snapshot-gate:1', 'cli:snapshot-gate', 1, 'user', content, '{}', '2026-07-11T00:01:00+00:00'))\n"
        "            connection.execute(\"INSERT INTO messages (id, session_key, seq, role, content, extra, ts) VALUES (?, ?, ?, ?, ?, ?, ?)\", ('cli:snapshot-gate:2', 'cli:snapshot-gate', 2, 'assistant', '状态回复', '{}', '2026-07-11T00:01:01+00:00'))\n"
        "            connection.execute(\"UPDATE sessions SET next_seq = 3 WHERE key = ?\", ('cli:snapshot-gate',))\n"
        "        await self.context.event_bus.fanout(TurnCommitted(session_key='cli:snapshot-gate', channel='cli', chat_id='snapshot-gate', input_message=content, persisted_user_message=content, assistant_response='状态回复', tools_used=[]))\n",
        encoding="utf-8",
    )
    entries.append('[plugins."zz_gate_driver@gate"]\nenabled = true\n')
    _ = manifest.write_text("\n".join(entries), encoding="utf-8")
    workspace = sandbox / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(workspace / "sessions.db") as connection:
        _ = connection.execute(
            "CREATE TABLE sessions (key TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL, last_consolidated INTEGER NOT NULL DEFAULT 0, "
            "metadata TEXT, last_user_at TEXT, last_proactive_at TEXT, "
            "next_seq INTEGER NOT NULL DEFAULT 0)"
        )
        _ = connection.execute(
            "CREATE TABLE messages (id TEXT PRIMARY KEY, session_key TEXT NOT NULL, "
            "seq INTEGER NOT NULL, role TEXT NOT NULL, content TEXT, tool_chain TEXT, "
            "extra TEXT, ts TEXT NOT NULL, UNIQUE (session_key, seq))"
        )
        _ = connection.execute(
            "INSERT INTO sessions "
            "(key, created_at, updated_at, metadata, next_seq) VALUES (?, ?, ?, ?, ?)",
            (
                "cli:snapshot-gate",
                "2026-07-11T00:00:00+00:00",
                "2026-07-11T00:00:00+00:00",
                "{}",
                1,
            ),
        )
        _ = connection.execute(
            "INSERT INTO messages "
            "(id, session_key, seq, role, content, extra, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "cli:snapshot-gate:0",
                "cli:snapshot-gate",
                0,
                "assistant",
                "这是一条用于验证主动反馈工作链路的提醒消息",
                '{"proactive": true}',
                "2026-07-11T00:00:00+00:00",
            ),
        )
    return observe_source


def _install_all_plugins(
    sandbox: Path,
    plugin_root: Path,
) -> tuple[dict[str, Path], Path]:
    cache = sandbox / "home/.akashic-plugin/cache/gate"
    sources: dict[str, Path] = {}
    entries: list[str] = []
    for source_name, plugin_name in (
        ("calendar-mcp", "calendar"),
        ("citation", "citation"),
        ("context_pressure", "context_pressure"),
        ("daynight_gate", "daynight_gate"),
        ("emotion", "emotion"),
        ("feed-mcp", "feed"),
        ("feishu", "feishu"),
        ("huayue-skills", "huayue-skills"),
        ("meme", "meme"),
        ("observe", "observe"),
        ("plugin_undo", "plugin_undo"),
        ("proactive_feedback", "proactive_feedback"),
        ("qqbot", "qqbot"),
        ("setup_helper", "setup_helper"),
        ("shell_restore", "shell_restore"),
        ("shell_safety", "shell_safety"),
        ("status_commands", "status_commands"),
        ("steam-mcp", "steam"),
        ("tool_loop_guard", "tool_loop_guard"),
    ):
        target = cache / plugin_name / "1.0.0"
        ignored = (
            shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache")
            if source_name == "calendar-mcp"
            else shutil.ignore_patterns(
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
            )
        )
        shutil.copytree(plugin_root / source_name, target, ignore=ignored)
        plugin_id = f"{plugin_name}@gate"
        sources[plugin_id] = target / "plugin.py"
        entries.append(f'[plugins."{plugin_id}"]\nenabled = true\n')
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _ = manifest.write_text("\n".join(entries), encoding="utf-8")
    return sources, manifest


def _install_candidate_plugins(
    sandbox: Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    cache = sandbox / "home/.akashic-plugin/cache/gate"
    valid = cache / "candidate_valid/1.0.0"
    invalid = cache / "candidate_invalid/1.0.0"
    failed = cache / "candidate_failed/1.0.0"
    observer = cache / "candidate_observer/1.0.0"
    reload_plugin = cache / "candidate_reload/1.0.0"
    valid.mkdir(parents=True, exist_ok=True)
    invalid.mkdir(parents=True, exist_ok=True)
    failed.mkdir(parents=True, exist_ok=True)
    observer.mkdir(parents=True, exist_ok=True)
    reload_plugin.mkdir(parents=True, exist_ok=True)
    _ = (valid / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class CandidateValidPlugin(Plugin):\n"
        "    name = 'candidate_valid'\n"
        "    async def initialize(self):\n"
        "        self.context.kv_store.set('initialized', True)\n"
        "        self.context.kv_store.set('generation', self.context.generation_id)\n",
        encoding="utf-8",
    )
    _ = (invalid / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class CandidateInvalidPlugin(Plugin):\n"
        "    name = 'candidate_invalid'\n"
        "    api_version = 2\n"
        "    async def initialize(self):\n"
        "        self.context.kv_store.set('initialized', True)\n",
        encoding="utf-8",
    )
    _ = (failed / "plugin.py").write_text(
        "from agent.plugins import Plugin, tool\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class CandidateFailedPlugin(Plugin):\n"
        "    name = 'candidate_failed'\n"
        "    @tool(name='candidate_failed_tool')\n"
        "    async def failed_tool(self, event):\n"
        '        """Failed candidate tool."""\n'
        "        return 'leaked'\n"
        "    async def initialize(self):\n"
        "        self.context.event_bus.on(TurnCommitted, self._on_turn)\n"
        "        raise RuntimeError('candidate init failed')\n"
        "    def _on_turn(self, event):\n"
        "        self.context.kv_store.set('handler_leaked', True)\n",
        encoding="utf-8",
    )
    _ = (observer / "plugin.py").write_text(
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from agent.plugins import Plugin\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class CandidateObserverPlugin(Plugin):\n"
        "    name = 'candidate_observer'\n"
        "    async def initialize(self):\n"
        "        self.context.create_task(self._verify(), name='candidate-observer')\n"
        "    async def _verify(self):\n"
        "        await asyncio.sleep(0.2)\n"
        "        registry = self.context.tool_registry\n"
        "        self.context.kv_store.set(\n"
        "            'failed_tool_visible',\n"
        "            registry is not None and registry.has_tool('candidate_failed_tool'),\n"
        "        )\n"
        "        await self.context.event_bus.fanout(TurnCommitted(\n"
        "            session_key='gate:candidate', channel='gate', chat_id='candidate',\n"
        "            input_message='candidate', persisted_user_message='candidate',\n"
        "            assistant_response='candidate', tools_used=[]))\n"
        "        self.context.kv_store.set('event_sent', True)\n",
        encoding="utf-8",
    )
    reload_source = reload_plugin / "plugin.py"
    for version in ("v1", "v2"):
        reload_skill = reload_plugin / f"skills-{version}" / "candidate-skill"
        reload_skill.mkdir(parents=True)
        _ = (reload_skill / "SKILL.md").write_text(
            f"---\ndescription: candidate {version} skill\n---\ncandidate {version}\n",
            encoding="utf-8",
        )
        reload_drift_skill = reload_plugin / f"drift-{version}" / "candidate-drift"
        reload_drift_skill.mkdir(parents=True)
        _ = (reload_drift_skill / "SKILL.md").write_text(
            f"---\ndescription: candidate drift {version}\n---\ndrift {version}\n",
            encoding="utf-8",
        )
    _ = (reload_plugin / "candidate_mcp_server.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "version = os.environ['CANDIDATE_VERSION']\n"
        "calls = Path(os.environ['AKA_PLUGIN_DATA_DIR']) / 'candidate_mcp_calls.jsonl'\n"
        "tools = [\n"
        "    {'name': name, 'description': name, 'inputSchema': {'type': 'object', 'properties': {}}}\n"
        "    for name in ('fetch_events', 'ack_events', 'candidate_version')\n"
        "]\n"
        "for line in sys.stdin:\n"
        "    msg = json.loads(line)\n"
        "    if 'id' not in msg:\n"
        "        continue\n"
        "    method = msg.get('method')\n"
        "    result = {'tools': tools} if method == 'tools/list' else {}\n"
        "    if method == 'tools/call':\n"
        "        tool = msg.get('params', {}).get('name')\n"
        "        with calls.open('a', encoding='utf-8') as stream:\n"
        "            stream.write(json.dumps({'version': version, 'tool': tool}) + '\\n')\n"
        "        text = version if tool == 'candidate_version' else '[]'\n"
        "        result = {'content': [{'type': 'text', 'text': text}]}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}), flush=True)\n",
        encoding="utf-8",
    )
    _ = (reload_plugin / "candidate_service.py").write_text(
        "import os\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        self.send_response(200); self.end_headers()\n"
        "        self.wfile.write(os.environ['CANDIDATE_VERSION'].encode())\n"
        "    def log_message(self, *args): pass\n"
        "HTTPServer(('127.0.0.1', 18767), Handler).serve_forever()\n",
        encoding="utf-8",
    )
    _ = reload_source.write_text(_candidate_reload_source("v1"), encoding="utf-8")
    data = sandbox / "workspace/plugin-data"
    (data / "candidate_reload-gate").mkdir(parents=True, exist_ok=True)
    return (
        data / "candidate_valid-gate/.kv.json",
        data / "candidate_invalid-gate/.kv.json",
        data / "candidate_failed-gate/.kv.json",
        data / "candidate_observer-gate/.kv.json",
        reload_source,
        data / "candidate_reload-gate/.kv.json",
    )


def _candidate_reload_source(version: str) -> str:
    initialize = (
        "        self.context.event_bus.on(TurnCommitted, self._on_committed)\n"
        "        self.context.kv_store.set('initialized_version', 'v1')\n"
        "        self.context.kv_store.set('active_generation', self.context.generation_id)\n"
        "        self.context.create_task(self._heartbeat(), name='candidate-reload-heartbeat')\n"
        "    async def _heartbeat(self):\n"
        "        while True:\n"
        "            self.context.kv_store.increment('heartbeats')\n"
        "            self.context.kv_store.set('active_generation', self.context.generation_id)\n"
        "            registry = self.context.tool_registry\n"
        "            if registry is not None and registry.has_tool('mcp_candidate_feed__candidate_version'):\n"
        "                result = await registry.execute('mcp_candidate_feed__candidate_version', {}, raise_errors=True)\n"
        "                self.context.kv_store.set('live_mcp_version', getattr(result, 'text', str(result)))\n"
        "            await asyncio.sleep(0.05)\n"
        if version == "v1"
        else "        self.context.event_bus.on(TurnCommitted, self._on_committed)\n"
        "        self.context.kv_store.set('initialized_version', 'v2')\n"
    )
    skills = (
        "    @classmethod\n"
        "    def skill_roots(cls):\n"
        f"        return ('skills-{version}',)\n"
        "    @classmethod\n"
        "    def drift_skill_roots(cls):\n"
        f"        return ('drift-{version}',)\n"
    )
    return (
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from agent.plugins.snapshot import get_current_runtime_snapshot\n"
        "from agent.skills import SkillsLoader\n"
        "from agent.tool_hooks import ToolExecutionRequest, ToolExecutor\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "from agent.plugins import (IntervalTrigger, ManagedServiceSpec, McpServerSpec, Plugin, PluginJobSpec, "
        "PluginSemanticCheck, ProactiveSourceSpec, on_tool_pre, tool)\n"
        "class SnapshotBeforeTurn:\n"
        "    slot = 'candidate_reload.before_turn'\n"
        "    requires = ('before_turn.emit',)\n"
        "    def __init__(self, plugin): self.plugin = plugin\n"
        "    async def run(self, frame):\n"
        f"        self.plugin.context.kv_store.increment('phase_runs_{version}')\n"
        "        registry = self.plugin.context.tool_registry\n"
        "        if registry is not None:\n"
        "            execution = await ToolExecutor().execute(ToolExecutionRequest(call_id='gate', tool_name='candidate_reload_tool', arguments={}, source='passive'), lambda name, args: registry.execute(name, args, raise_errors=True))\n"
        "            value = execution.output\n"
        f"            self.plugin.context.kv_store.set('phase_tool_version_{version}', str(value))\n"
        "        skill_body = SkillsLoader(self.plugin.context.workspace, runtime_catalog='normal').load_skill_body('candidate-skill')\n"
        f"        self.plugin.context.kv_store.set('phase_skill_body_{version}', skill_body)\n"
        "        await self.plugin.context.event_bus.fanout(TurnCommitted(session_key='gate:event', channel='gate', chat_id='event', input_message='event', persisted_user_message='event', assistant_response='event', tools_used=[]))\n"
        "        self.plugin.context.create_task(self._probe_detached(), name='snapshot-detached-probe')\n"
        "        ctx = frame.slots['session:ctx']\n"
        f"        if '{version}' == 'v1' and ctx.content == 'block snapshot':\n"
        "            self.plugin.context.kv_store.set('blocked_v1_turn', True)\n"
        "            release = self.plugin.context.data_dir / 'release-v1-turn'\n"
        "            while not release.exists():\n"
        "                await asyncio.sleep(0.01)\n"
        "        ctx.abort = True\n"
        f"        ctx.abort_reply = 'snapshot-{version}'\n"
        "        return frame\n"
        "    async def _probe_detached(self):\n"
        "        release = self.plugin.context.data_dir / 'release-detached-probe'\n"
        "        while not release.exists():\n"
        "            await asyncio.sleep(0.01)\n"
        "        self.plugin.context.kv_store.set('detached_snapshot_visible', get_current_runtime_snapshot() is not None)\n"
        "class CandidateReloadPlugin(Plugin):\n"
        "    name = 'candidate_reload'\n"
        f"{skills}"
        "    def before_turn_modules(self):\n"
        "        return [SnapshotBeforeTurn(self)]\n"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        f"        return [McpServerSpec(name='candidate_feed', command=('python', 'candidate_mcp_server.py'), env={{'CANDIDATE_VERSION': '{version}'}})]\n"
        "    @classmethod\n"
        "    def managed_services(cls):\n"
        f"        return [ManagedServiceSpec(id='candidate_http', command=('python', 'candidate_service.py'), env={{'CANDIDATE_VERSION': '{version}'}}, readiness_url='http://127.0.0.1:18767/')]\n"
        "    def proactive_sources(self):\n"
        "        return [ProactiveSourceSpec(id='candidate_feed', channels=('content',), "
        "server='candidate_feed', fetch_tool='fetch_events', ack_tool='ack_events')]\n"
        "    def jobs(self):\n"
        f"        return [PluginJobSpec(id='refresh', triggers=[IntervalTrigger({1 if version == 'v1' else 2})], handler=self.refresh)]\n"
        "    async def refresh(self, context):\n"
        f"        self.context.kv_store.increment('job_runs_{version}')\n"
        "        snapshot = get_current_runtime_snapshot()\n"
        f"        self.context.kv_store.set('job_snapshot_bound_{version}', snapshot is not None and snapshot.generations.get('candidate_reload@gate').instance is self)\n"
        "    async def readiness_semantic_checks(self, context):\n"
        "        server = context.mcp_catalog.servers['candidate_feed']\n"
        "        value = await server.client.call('candidate_version', {})\n"
        "        job = context.job_catalog.jobs['candidate_reload@gate:refresh']\n"
        "        source = context.proactive_catalog.sources['candidate_reload@gate:candidate_feed']\n"
        "        owned = getattr(job.spec.handler, '__self__', None) is self\n"
        "        job_interval = job.spec.triggers[0].seconds\n"
        "        evidence = {'mcp': value, 'job_owned': owned, 'source': source.spec.id, "
        "'job_interval': job_interval}\n"
        f"        return [PluginSemanticCheck('candidate_capabilities', value == '{version}' and owned and job_interval == {1 if version == 'v1' else 2}, evidence)]\n"
        "    @tool(name='candidate_reload_tool')\n"
        "    async def run(self, event):\n"
        '        """Candidate reload tool."""\n'
        f"        return '{version}'\n"
        "    @on_tool_pre(tool_name='candidate_reload_tool')\n"
        "    async def before_candidate_tool(self, event):\n"
        f"        self.context.kv_store.increment('phase_hook_version_{version}')\n"
        "    async def initialize(self):\n"
        f"{initialize}"
        "    async def _on_committed(self, event):\n"
        f"        self.context.kv_store.increment('phase_event_version_{version}')\n"
    )


def _skill_fixture_hash(version: str, *, drift: bool) -> str:
    label = f"candidate drift {version}" if drift else f"candidate {version} skill"
    body = f"drift {version}" if drift else f"candidate {version}"
    return hashlib.sha256(
        f"---\ndescription: {label}\n---\n{body}\n".encode()
    ).hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    raw: object = {}
    for _ in range(20):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            break
        except json.JSONDecodeError:
            time.sleep(0.01)
    if not isinstance(raw, dict):
        return {}
    mapping = cast(dict[object, object], raw)
    return {str(key): value for key, value in mapping.items()}


def _wait_json_value(path: Path, key: str, expected: object) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = _read_json_object(path)
        if state.get(key) == expected:
            return state
        time.sleep(0.05)
    return _read_json_object(path)


def _integer(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _mcp_call_counts(path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        item = cast(dict[object, object], raw)
        version = str(item.get("version") or "")
        tool = str(item.get("tool") or "")
        if version and tool:
            version_counts = counts.setdefault(version, {})
            version_counts[tool] = version_counts.get(tool, 0) + 1
    return counts


def _candidate_statuses(container_id: str) -> list[dict[str, object]]:
    logs = subprocess.run(
        ["docker", "logs", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    marker = "plugin_candidate_status_detail "
    statuses: list[dict[str, object]] = []
    for line in logs.splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            raw: object = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            mapping = cast(dict[object, object], raw)
            statuses.append({str(key): value for key, value in mapping.items()})
    return statuses


def _snapshot_statuses(container_id: str) -> list[dict[str, object]]:
    logs = subprocess.run(
        ["docker", "logs", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    marker = "plugin_snapshot_status "
    statuses: list[dict[str, object]] = []
    for line in logs.splitlines():
        if marker not in line:
            continue
        try:
            raw: object = json.loads(line.split(marker, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            mapping = cast(dict[object, object], raw)
            statuses.append({str(key): value for key, value in mapping.items()})
    return statuses


def _container_process_count(container_id: str, marker: str) -> int:
    script = (
        "import os\n"
        "from pathlib import Path\n"
        "count = 0\n"
        "for entry in Path('/proc').iterdir():\n"
        "    if not entry.name.isdigit() or int(entry.name) == os.getpid():\n"
        "        continue\n"
        "    try:\n"
        "        command = (entry / 'cmdline').read_bytes().replace(b'\\0', b' ').decode()\n"
        "    except (FileNotFoundError, PermissionError, ProcessLookupError):\n"
        "        continue\n"
        f"    if {marker!r} in command:\n"
        "        count += 1\n"
        "print(count)\n"
    )
    result = subprocess.run(
        ["docker", "exec", container_id, "python", "-c", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1


def _candidate_service_version(container_id: str) -> str:
    result = subprocess.run(
        [
            "docker",
            "exec",
            container_id,
            "python",
            "-c",
            (
                "import urllib.request; "
                "print(urllib.request.urlopen('http://127.0.0.1:18767/', "
                "timeout=2).read().decode())"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _dashboard_plugins(container_id: str) -> list[str]:
    result = subprocess.run(
        [
            "docker",
            "exec",
            container_id,
            "python",
            "-c",
            (
                "import json, urllib.request; "
                "items=json.loads(urllib.request.urlopen("
                "'http://127.0.0.1:2236/api/dashboard/plugins', timeout=2).read()); "
                "print(json.dumps(sorted(item['id'] for item in items)))"
            ),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _wait_sqlite_count(path: Path, query: str) -> int:
    deadline = time.monotonic() + 8
    count = 0
    while time.monotonic() < deadline:
        if path.exists():
            try:
                with sqlite3.connect(path) as connection:
                    row = connection.execute(query).fetchone()
                count = int(row[0]) if row is not None else 0
            except sqlite3.OperationalError:
                count = 0
            if count:
                return count
        time.sleep(0.1)
    return count


def _exercise_migrated_plugins(
    container_id: str,
    observe_source: Path,
    sandbox: Path,
) -> dict[str, object]:
    status_response = _control_roundtrip(
        sandbox / "akashic.sock",
        "/memorystatus 被回复消息：这是一条用于验证主动反馈工作链路的提醒消息"
        "【你当前新消息】谢谢",
    )
    feedback_count = _wait_sqlite_count(
        sandbox / "workspace/proactive_feedback/proactive_feedback.db",
        "SELECT count(*) FROM proactive_feedback_events",
    )
    driver_state = _read_json_object(
        sandbox / "workspace/plugin-data/zz_gate_driver-gate/.kv.json"
    )
    raw_slots = driver_state.get("phase_slots", [])
    lifecycle_slots = (
        {str(item) for item in raw_slots if isinstance(item, str)}
        if isinstance(raw_slots, list)
        else set()
    )
    before = _snapshot_statuses(container_id)
    _ = observe_source.write_text(
        observe_source.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    _, reloaded = _wait_snapshot_status(
        container_id,
        after=len(before),
        publication_state="committed",
        plugin_id="observe@gate",
    )
    plugins = _dashboard_plugins(container_id)
    expected = {
        "emotion@gate",
        "meme@gate",
        "observe@gate",
        "proactive_feedback@gate",
        "status_commands@gate",
    }
    observe_db = sandbox / "workspace/observe/observe.db"
    passed = (
        str(status_response.get("content", "")).startswith("🧠 记忆整理状态")
        and feedback_count >= 1
        and {
            "meme.prompt",
            "status_commands.memory_status",
            "gate_driver.inspect_runtime_modules",
        }.issubset(lifecycle_slots)
        and reloaded.get("old_generation") != reloaded.get("new_generation")
        and isinstance(reloaded.get("new_generation"), str)
        and expected.issubset(plugins)
        and observe_db.exists()
    )
    return {
        "passed": passed,
        "status_response": status_response.get("content"),
        "proactive_feedback_events": feedback_count,
        "lifecycle_slots": sorted(lifecycle_slots),
        "reloaded": reloaded,
        "dashboard_plugins": plugins,
        "observe_db": observe_db.exists(),
    }


def _exercise_all_plugins(
    container_id: str,
    sources: dict[str, Path],
    manifest: Path,
) -> dict[str, object]:
    reloads: dict[str, object] = {}
    disables: dict[str, object] = {}
    for plugin_id, source in sources.items():
        before = _snapshot_statuses(container_id)
        _ = source.write_text(
            source.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        _, reloads[plugin_id] = _wait_snapshot_status(
            container_id,
            after=len(before),
            publication_state="committed",
            plugin_id=plugin_id,
        )
    manifest_text = manifest.read_text(encoding="utf-8")
    for plugin_id in reversed(sources):
        before = _snapshot_statuses(container_id)
        manifest_text = manifest_text.replace(
            f'[plugins."{plugin_id}"]\nenabled = true',
            f'[plugins."{plugin_id}"]\nenabled = false',
        )
        _ = manifest.write_text(manifest_text, encoding="utf-8")
        _, disables[plugin_id] = _wait_snapshot_status(
            container_id,
            after=len(before),
            publication_state="disabled",
            plugin_id=plugin_id,
        )
    passed = all(
        isinstance(result, dict)
        and result.get("publication_state") == "committed"
        and result.get("old_generation") != result.get("new_generation")
        for result in reloads.values()
    ) and all(
        isinstance(result, dict) and result.get("publication_state") == "disabled"
        for result in disables.values()
    )
    return {
        "passed": passed,
        "plugin_count": len(sources),
        "reloaded": sorted(
            plugin_id
            for plugin_id, result in reloads.items()
            if isinstance(result, dict)
            and result.get("publication_state") == "committed"
        ),
        "disabled": sorted(
            plugin_id
            for plugin_id, result in disables.items()
            if isinstance(result, dict)
            and result.get("publication_state") == "disabled"
        ),
        "reload_failures": {
            plugin_id: result
            for plugin_id, result in reloads.items()
            if not isinstance(result, dict)
            or result.get("publication_state") != "committed"
        },
        "disable_failures": {
            plugin_id: result
            for plugin_id, result in disables.items()
            if not isinstance(result, dict)
            or result.get("publication_state") != "disabled"
        },
    }


def _exercise_plugin_management(
    container_id: str,
    cache: Path,
    data: Path,
    manifest: Path,
) -> dict[str, object]:
    commands: dict[str, dict[str, object]] = {}
    service_states = {"initial": _management_service_ready(container_id)}

    def run(command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "docker",
                "exec",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                container_id,
                "python",
                "main.py",
                command,
                "management@gate",
                "--config",
                "/sandbox/config.toml",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    before = len(_snapshot_statuses(container_id))
    disabled_command = run("plugin-disable")
    _, disabled = _wait_snapshot_status(
        container_id,
        after=before,
        publication_state="disabled",
        plugin_id="management@gate",
    )
    commands["disable"] = {
        "returncode": disabled_command.returncode,
        "output": disabled_command.stdout.strip(),
        "status": disabled,
    }
    service_states["disabled"] = (
        _wait_process_count(
            container_id,
            "management_service.py",
            lambda count: count == 0,
        )
        == 0
    )
    (data / "service.stopped").unlink(missing_ok=True)

    before = len(_snapshot_statuses(container_id))
    enabled_command = run("plugin-enable")
    _, enabled = _wait_snapshot_status(
        container_id,
        after=before,
        publication_state="committed",
        plugin_id="management@gate",
    )
    commands["enable"] = {
        "returncode": enabled_command.returncode,
        "output": enabled_command.stdout.strip(),
        "status": enabled,
    }
    service_states["enabled"] = _management_service_ready(container_id)

    before = len(_snapshot_statuses(container_id))
    uninstall_command = run("plugin-uninstall")
    _, uninstalled = _wait_snapshot_status(
        container_id,
        after=before,
        publication_state="disabled",
        plugin_id="management@gate",
    )
    commands["uninstall"] = {
        "returncode": uninstall_command.returncode,
        "output": uninstall_command.stdout.strip(),
        "status": uninstalled,
    }
    service_states["uninstalled"] = (
        _wait_process_count(
            container_id,
            "management_service.py",
            lambda count: count == 0,
        )
        == 0
    )
    manifest_data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    manifest_plugins = manifest_data.get("plugins", {})
    passed = (
        all(item["returncode"] == 0 for item in commands.values())
        and disabled.get("publication_state") == "disabled"
        and enabled.get("publication_state") == "committed"
        and uninstalled.get("publication_state") == "disabled"
        and service_states
        == {
            "initial": True,
            "disabled": True,
            "enabled": True,
            "uninstalled": True,
        }
        and (data / "service.stopped").exists()
        and not cache.exists()
        and (data / "retained.txt").read_text(encoding="utf-8") == "keep"
        and "management@gate" not in manifest_plugins
    )
    return {
        "passed": passed,
        "commands": commands,
        "service_states": service_states,
        "service_stopped": (data / "service.stopped").exists(),
        "cache_removed": not cache.exists(),
        "data_retained": (data / "retained.txt").exists(),
        "manifest_entry_removed": "management@gate" not in manifest_plugins,
    }


def _management_service_ready(container_id: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "exec",
            container_id,
            "python",
            "-c",
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18768/', timeout=2).read()",
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _fitbit_runtime_probe(container_id: str) -> dict[str, object]:
    script = (
        "import json, urllib.request\n"
        "result = {}\n"
        "for key, path in [('data', '/api/data'), ('snapshot', '/api/tool/fitbit_health_snapshot')]:\n"
        "    with urllib.request.urlopen('http://127.0.0.1:18765' + path, timeout=3) as response:\n"
        "        result[key] = json.loads(response.read())\n"
        "print(json.dumps(result))\n"
    )
    result = subprocess.run(
        ["docker", "exec", container_id, "python", "-c", script],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        return {"error": result.stdout.strip()}
    try:
        raw: object = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"error": result.stdout.strip()}
    if not isinstance(raw, dict):
        return {"error": repr(raw)}
    return {str(key): value for key, value in raw.items()}


def _persistent_file_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "monitor.runtime.log"
    }


def _wait_process_count(
    container_id: str,
    marker: str,
    predicate: Callable[[int], bool],
) -> int:
    deadline = time.monotonic() + 8
    count = -1
    while time.monotonic() < deadline:
        count = _container_process_count(container_id, marker)
        if predicate(count):
            return count
        time.sleep(0.1)
    return count


def _control_roundtrip(path: Path, content: str) -> dict[str, object]:
    """通过正式 JSON-RPC 客户端运行 turn，并返回兼容 gate 断言的结果。"""

    async def run() -> dict[str, object]:
        # 1. 建立完成握手的控制连接和独立 thread。
        async with await ControlClient.connect(str(path)) as client:
            thread = await client.start_thread({"source": "plugin-hot-reload-gate"})

            # 2. 等待终态事件，失败也作为真实结果交给 gate 判定。
            handle = await client.start_turn(str(thread["id"]), content)
            turn = await handle.result()
            return {
                "content": turn.get("finalResponse"),
                "status": turn.get("status"),
                "error": turn.get("error"),
                "turn_id": turn.get("id"),
            }

    return asyncio.run(run())


def _wait_candidate_status(
    container_id: str,
    *,
    after: int,
    gate_status: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        statuses = _candidate_statuses(container_id)
        for status in statuses[after:]:
            if (
                status.get("plugin_id") == "candidate_reload@gate"
                and status.get("gate_status") == gate_status
            ):
                return statuses, status
        time.sleep(0.1)
    return _candidate_statuses(container_id), {}


def _wait_snapshot_status(
    container_id: str,
    *,
    after: int,
    publication_state: str,
    plugin_id: str = "candidate_reload@gate",
) -> tuple[list[dict[str, object]], dict[str, object]]:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        statuses = _snapshot_statuses(container_id)
        for status in statuses[after:]:
            if (
                status.get("plugin_id") == plugin_id
                and status.get("publication_state") == publication_state
            ):
                return statuses, status
        time.sleep(0.1)
    return _snapshot_statuses(container_id), {}


def _wait_service_version(container_id: str, expected: str) -> str:
    deadline = time.monotonic() + 10
    value = ""
    while time.monotonic() < deadline:
        value = _candidate_service_version(container_id)
        if value == expected:
            return value
        time.sleep(0.1)
    return value


def _exercise_topology_watch(
    container_id: str,
    sandbox: Path,
    state_path: Path,
) -> dict[str, object]:
    manifest = sandbox / "home/.akashic-plugin/manifest.toml"
    initial = _read_json_object(state_path)
    initial_generation = initial.get("active_generation")

    statuses = _snapshot_statuses(container_id)
    _ = manifest.write_text(
        '[plugins."candidate_reload@gate"]\nenabled = false\n',
        encoding="utf-8",
    )
    statuses, disabled = _wait_snapshot_status(
        container_id,
        after=len(statuses),
        publication_state="disabled",
    )
    disabled_processes = _wait_process_count(
        container_id,
        "candidate_service.py",
        lambda count: count == 0,
    )

    _ = manifest.write_text(
        '[plugins."candidate_reload@gate"]\nenabled = true\n',
        encoding="utf-8",
    )
    statuses, enabled = _wait_snapshot_status(
        container_id,
        after=len(statuses),
        publication_state="committed",
    )
    enabled_service = _wait_service_version(container_id, "v1")
    enabled_state = _wait_json_value(state_path, "initialized_version", "v1")
    enabled_generation = enabled.get("new_generation")

    config = state_path.parent / "config.local.toml"
    _ = config.write_text("probe = 1\n", encoding="utf-8")
    statuses, configured = _wait_snapshot_status(
        container_id,
        after=len(statuses),
        publication_state="committed",
    )

    added_root = sandbox / "home/.akashic-plugin/cache/gate/topology_added/1.0.0"
    added_root.mkdir(parents=True)
    _ = (added_root / "plugin.py").write_text(
        "from agent.plugins import Plugin\n"
        "class TopologyAddedPlugin(Plugin):\n"
        "    name = 'topology_added'\n"
        "    version = '1.0.0'\n"
        "    async def initialize(self):\n"
        "        self.context.kv_store.set('generation', self.context.generation_id)\n"
        "    async def terminate(self):\n"
        "        self.context.kv_store.set('terminated', True)\n",
        encoding="utf-8",
    )
    added_state_path = (
        sandbox / "workspace/plugin-data/topology_added-gate/.kv.json"
    )
    added_state: dict[str, object] = {}
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        added_state = _read_json_object(added_state_path)
        if isinstance(added_state.get("generation"), str):
            break
        time.sleep(0.1)
    statuses = _snapshot_statuses(container_id)
    shutil.rmtree(added_root)
    _, removed = _wait_snapshot_status(
        container_id,
        after=len(statuses),
        publication_state="disabled",
        plugin_id="topology_added@gate",
    )
    removed_state = _wait_json_value(added_state_path, "terminated", True)

    passed = (
        isinstance(initial_generation, str)
        and disabled.get("publication_state") == "disabled"
        and disabled_processes == 0
        and enabled.get("old_generation") is None
        and isinstance(enabled_generation, str)
        and enabled_generation != initial_generation
        and enabled_service == "v1"
        and enabled_state.get("initialized_version") == "v1"
        and configured.get("old_generation") == enabled_generation
        and isinstance(configured.get("new_generation"), str)
        and configured.get("new_generation") != enabled_generation
        and isinstance(added_state.get("generation"), str)
        and removed.get("publication_state") == "disabled"
        and removed_state.get("terminated") is True
    )
    return {
        "passed": passed,
        "initial_generation": initial_generation,
        "disabled": disabled,
        "disabled_service_processes": disabled_processes,
        "enabled": enabled,
        "enabled_service": enabled_service,
        "configured": configured,
        "added": added_state,
        "removed": removed,
        "removed_state": removed_state,
    }


def _exercise_snapshot_publish(
    container_id: str,
    source_path: Path,
    state_path: Path,
    socket_path: Path,
) -> dict[str, object]:
    initial = _read_json_object(state_path)
    initial_generation = initial.get("active_generation")
    initial_service_version = _candidate_service_version(container_id)
    initial_service_processes = _container_process_count(
        container_id,
        "candidate_service.py",
    )
    baseline_before = _candidate_statuses(container_id)
    _ = source_path.write_text("not valid python !!!\n", encoding="utf-8")
    baseline_signal = subprocess.run(
        ["docker", "kill", "--signal", "HUP", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _, baseline_status = _wait_candidate_status(
        container_id,
        after=len(baseline_before),
        gate_status="failed",
    )
    initial_snapshot = baseline_status.get("snapshot_id")
    _ = source_path.write_text(_candidate_reload_source("v1"), encoding="utf-8")
    turn_release = state_path.parent / "release-v1-turn"
    detached_release = state_path.parent / "release-detached-probe"
    turn_release.unlink(missing_ok=True)
    detached_release.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        old_turn = executor.submit(_control_roundtrip, socket_path, "block snapshot")
        blocked = _wait_json_value(state_path, "blocked_v1_turn", True)
        _ = source_path.write_text(_candidate_reload_source("v2"), encoding="utf-8")
        candidate_before = _candidate_statuses(container_id)
        publish_before = _snapshot_statuses(container_id)
        signal_result = subprocess.run(
            ["docker", "kill", "--signal", "HUP", container_id],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        _, candidate_status = _wait_candidate_status(
            container_id,
            after=len(candidate_before),
            gate_status="passed",
        )
        _ = turn_release.write_text("released\n", encoding="utf-8")
        old_response = old_turn.result(timeout=10)
        _, publish_status = _wait_snapshot_status(
            container_id,
            after=len(publish_before),
            publication_state="committed",
        )

    new_response = _control_roundtrip(socket_path, "snapshot after publish")
    final_service_version = _candidate_service_version(container_id)
    final_service_processes = _container_process_count(
        container_id,
        "candidate_service.py",
    )
    _ = detached_release.write_text("released\n", encoding="utf-8")
    final_state = _wait_json_value(
        state_path,
        "detached_snapshot_visible",
        False,
    )
    current_snapshot = publish_status.get("snapshot_id")
    passed = (
        signal_result.returncode == 0
        and baseline_signal.returncode == 0
        and isinstance(initial_generation, str)
        and blocked.get("blocked_v1_turn") is True
        and candidate_status.get("active_generation") == initial_generation
        and isinstance(candidate_status.get("prepared_generation"), str)
        and publish_status.get("old_generation") == initial_generation
        and publish_status.get("new_generation")
        == candidate_status.get("prepared_generation")
        and isinstance(current_snapshot, str)
        and isinstance(initial_snapshot, str)
        and current_snapshot != initial_snapshot
        and old_response.get("content") == "snapshot-v1"
        and new_response.get("content") == "snapshot-v2"
        and initial_service_version == "v1"
        and final_service_version == "v2"
        and initial_service_processes == 1
        and final_service_processes == 1
        and _integer(final_state.get("phase_runs_v1")) >= 1
        and _integer(final_state.get("phase_runs_v2")) >= 1
        and final_state.get("detached_snapshot_visible") is False
    )
    return {
        "passed": passed,
        "initial_generation": initial_generation,
        "initial_snapshot": initial_snapshot,
        "blocked": blocked,
        "candidate_status": candidate_status,
        "publish_status": publish_status,
        "old_response": old_response,
        "new_response": new_response,
        "managed_service": {
            "initial_version": initial_service_version,
            "final_version": final_service_version,
            "initial_processes": initial_service_processes,
            "final_processes": final_service_processes,
        },
        "final_state": final_state,
        "signal": signal_result.returncode,
    }


def _exercise_candidate_prepare(
    container_id: str,
    source_path: Path,
    state_path: Path,
    socket_path: Path,
) -> dict[str, object]:
    initial = _read_json_object(state_path)
    calls_path = state_path.parent / "candidate_mcp_calls.jsonl"
    initial_calls = _mcp_call_counts(calls_path)
    initial_mcp_processes = _wait_process_count(
        container_id,
        "candidate_mcp_server.py",
        lambda count: count >= 1,
    )
    initial_generation = initial.get("active_generation")
    initial_heartbeats = _integer(initial.get("heartbeats"))
    _ = source_path.write_text("not valid python !!!\n", encoding="utf-8")
    statuses_before = _candidate_statuses(container_id)
    invalid_signal = subprocess.run(
        ["docker", "kill", "--signal", "HUP", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    statuses, invalid_status = _wait_candidate_status(
        container_id,
        after=len(statuses_before),
        gate_status="failed",
    )
    after_invalid = _read_json_object(state_path)
    after_invalid_mcp_processes = _container_process_count(
        container_id,
        "candidate_mcp_server.py",
    )
    _ = source_path.write_text(_candidate_reload_source("v2"), encoding="utf-8")
    valid_signal = subprocess.run(
        ["docker", "kill", "--signal", "HUP", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    valid_statuses, valid_status = _wait_candidate_status(
        container_id,
        after=len(statuses),
        gate_status="passed",
    )
    time.sleep(2.3)
    after_valid = _read_json_object(state_path)
    after_valid_calls = _mcp_call_counts(calls_path)
    detached_release = state_path.parent / "release-detached-probe"
    detached_release.unlink(missing_ok=True)
    passive_response = _control_roundtrip(socket_path, "snapshot lease gate")
    _ = detached_release.write_text("released\n", encoding="utf-8")
    after_passive = _wait_json_value(
        state_path,
        "detached_snapshot_visible",
        False,
    )
    after_valid_mcp_processes = _wait_process_count(
        container_id,
        "candidate_mcp_server.py",
        lambda count: count >= initial_mcp_processes + 1,
    )
    _ = source_path.write_text(_candidate_reload_source("v1"), encoding="utf-8")
    return_signal = subprocess.run(
        ["docker", "kill", "--signal", "HUP", container_id],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _, return_status = _wait_candidate_status(
        container_id,
        after=len(valid_statuses),
        gate_status="active",
    )
    after_return_mcp_processes = _wait_process_count(
        container_id,
        "candidate_mcp_server.py",
        lambda count: count == initial_mcp_processes,
    )
    after_return = _read_json_object(state_path)
    after_return_calls = _mcp_call_counts(calls_path)
    valid_skills = valid_status.get("skills")
    valid_descriptions = valid_status.get("skill_descriptions")
    valid_drift_descriptions = valid_status.get("drift_skill_descriptions")
    valid_body_hashes = valid_status.get("skill_body_hashes")
    valid_drift_body_hashes = valid_status.get("drift_skill_body_hashes")
    return_body_hashes = return_status.get("skill_body_hashes")
    return_drift_body_hashes = return_status.get("drift_skill_body_hashes")
    valid_mcp_tools = valid_status.get("mcp_tools")
    valid_readiness_checks = valid_status.get("readiness_checks")
    valid_jobs = valid_status.get("jobs")
    valid_sources = valid_status.get("proactive_sources")
    valid_description_map = (
        cast(dict[object, object], valid_descriptions)
        if isinstance(valid_descriptions, dict)
        else {}
    )
    valid_drift_description_map = (
        cast(dict[object, object], valid_drift_descriptions)
        if isinstance(valid_drift_descriptions, dict)
        else {}
    )
    hash_maps = [
        cast(dict[object, object], value) if isinstance(value, dict) else {}
        for value in (
            valid_body_hashes,
            valid_drift_body_hashes,
            return_body_hashes,
            return_drift_body_hashes,
        )
    ]
    passed = (
        invalid_signal.returncode == 0
        and valid_signal.returncode == 0
        and return_signal.returncode == 0
        and isinstance(initial_generation, str)
        and invalid_status.get("active_generation") == initial_generation
        and isinstance(invalid_status.get("snapshot_id"), str)
        and valid_status.get("snapshot_id") == invalid_status.get("snapshot_id")
        and return_status.get("snapshot_id") == invalid_status.get("snapshot_id")
        and invalid_status.get("prepared_generation") is None
        and valid_status.get("active_generation") == initial_generation
        and isinstance(valid_status.get("prepared_generation"), str)
        and isinstance(valid_skills, list)
        and isinstance(valid_mcp_tools, list)
        and valid_mcp_tools
        == [
            "mcp_candidate_feed__ack_events",
            "mcp_candidate_feed__candidate_version",
            "mcp_candidate_feed__fetch_events",
        ]
        and isinstance(valid_readiness_checks, list)
        and valid_readiness_checks
        == [
            {
                "check_id": "candidate_capabilities",
                "passed": True,
                "evidence": {
                    "mcp": "v2",
                    "job_owned": True,
                    "source": "candidate_feed",
                    "job_interval": 2,
                },
            }
        ]
        and valid_jobs == ["candidate_reload@gate:refresh"]
        and valid_sources == ["candidate_reload@gate:candidate_feed"]
        and return_status.get("jobs") == ["candidate_reload@gate:refresh"]
        and return_status.get("proactive_sources")
        == ["candidate_reload@gate:candidate_feed"]
        and valid_status.get("job_specs")
        == {"candidate_reload@gate:refresh": [{"type": "interval", "seconds": 2}]}
        and valid_status.get("proactive_source_specs")
        == {
            "candidate_reload@gate:candidate_feed": {
                "server": "candidate_feed",
                "fetch_tool": "fetch_events",
                "ack_tool": "ack_events",
                "fetch_page_size": 0,
            }
        }
        and return_status.get("job_specs")
        == {"candidate_reload@gate:refresh": [{"type": "interval", "seconds": 1}]}
        and return_status.get("proactive_source_specs")
        == {
            "candidate_reload@gate:candidate_feed": {
                "server": "candidate_feed",
                "fetch_tool": "fetch_events",
                "ack_tool": "ack_events",
                "fetch_page_size": 0,
            }
        }
        and cast(dict[str, int], after_valid_calls.get("v2", {})).get(
            "candidate_version",
            0,
        )
        == 1
        and all(
            cast(dict[str, int], after_valid_calls.get("v2", {})).get(tool, 0) == 0
            for tool in ("fetch_events", "ack_events")
        )
        and _integer(after_valid.get("job_runs_v1"))
        > _integer(initial.get("job_runs_v1"))
        and after_valid.get("job_snapshot_bound_v1") is True
        and _integer(after_valid.get("job_runs_v2")) == 0
        and after_return_calls.get("v2") == after_valid_calls.get("v2")
        and passive_response.get("content") == "snapshot-v1"
        and _integer(after_passive.get("phase_runs_v1")) >= 1
        and _integer(after_passive.get("phase_runs_v2")) == 0
        and after_passive.get("phase_tool_version_v1") == "v1"
        and after_passive.get("phase_tool_version_v2") is None
        and after_passive.get("phase_skill_body_v1") == "candidate v1"
        and after_passive.get("phase_skill_body_v2") is None
        and _integer(after_passive.get("phase_event_version_v1")) >= 1
        and _integer(after_passive.get("phase_event_version_v2")) == 0
        and _integer(after_passive.get("phase_hook_version_v1")) >= 1
        and _integer(after_passive.get("phase_hook_version_v2")) == 0
        and after_passive.get("detached_snapshot_visible") is False
        and "candidate-skill" in valid_skills
        and valid_description_map.get("candidate-skill") == "candidate v2 skill"
        and valid_drift_description_map.get("candidate-drift") == "candidate drift v2"
        and hash_maps[0].get("candidate-skill")
        == _skill_fixture_hash("v2", drift=False)
        and hash_maps[1].get("candidate-drift") == _skill_fixture_hash("v2", drift=True)
        and hash_maps[2].get("candidate-skill")
        == _skill_fixture_hash("v1", drift=False)
        and hash_maps[3].get("candidate-drift") == _skill_fixture_hash("v1", drift=True)
        and return_status.get("active_generation") == initial_generation
        and return_status.get("prepared_generation") is None
        and after_invalid.get("active_generation") == initial_generation
        and after_valid.get("active_generation") == initial_generation
        and after_valid.get("initialized_version") == "v1"
        and after_valid.get("live_mcp_version") == "v1"
        and after_return.get("live_mcp_version") == "v1"
        and _integer(after_valid.get("heartbeats")) > initial_heartbeats
        and initial_mcp_processes >= 1
        and after_invalid_mcp_processes == initial_mcp_processes
        and after_valid_mcp_processes >= initial_mcp_processes + 1
        and after_return_mcp_processes == initial_mcp_processes
    )
    return {
        "passed": passed,
        "initial": initial,
        "invalid_status": invalid_status,
        "after_invalid": after_invalid,
        "valid_status": valid_status,
        "after_valid": after_valid,
        "return_status": return_status,
        "after_return": after_return,
        "mcp_calls": {
            "initial": initial_calls,
            "after_valid": after_valid_calls,
            "after_return": after_return_calls,
        },
        "passive_response": passive_response,
        "after_passive": after_passive,
        "invalid_signal": invalid_signal.returncode,
        "valid_signal": valid_signal.returncode,
        "return_signal": return_signal.returncode,
        "mcp_processes": {
            "initial": initial_mcp_processes,
            "after_invalid": after_invalid_mcp_processes,
            "after_valid": after_valid_mcp_processes,
            "after_return": after_return_mcp_processes,
        },
    }


def _run_runtime_smoke(
    *,
    repo: Path,
    sandbox: Path,
    compose: list[str],
    env: dict[str, str],
    phase: str,
) -> tuple[bool, dict[str, object]]:
    _write_smoke_config(
        sandbox,
        proactive_enabled=phase
        in {
            "capability-hosts",
            "snapshot",
            "fitbit",
            "proactive-fetch",
        },
        fast_tick=phase == "proactive-fetch",
    )
    shutil.rmtree(
        sandbox / "home/.akashic-plugin/cache/gate",
        ignore_errors=True,
    )
    scope_state = _install_scope_plugin(sandbox) if phase == "scope" else None
    fitbit_data = (
        _install_fitbit_plugin(sandbox, Path(env["AKASHIC_PLUGIN_SOURCE"]))
        if phase == "fitbit"
        else None
    )
    management = _install_management_plugin(sandbox) if phase == "management" else None
    proactive_fetch_calls = (
        _install_proactive_fetch_plugin(sandbox) if phase == "proactive-fetch" else None
    )
    migrated_observe = (
        _install_migrated_plugins(
            sandbox,
            Path(env["AKASHIC_PLUGIN_SOURCE"]),
        )
        if phase == "plugins"
        else None
    )
    all_plugins = (
        _install_all_plugins(
            sandbox,
            Path(env["AKASHIC_PLUGIN_SOURCE"]),
        )
        if phase == "all-plugins"
        else None
    )
    candidate_states = (
        _install_candidate_plugins(sandbox)
        if phase in {"candidate", "capability-hosts", "snapshot", "topology"}
        else None
    )
    started = subprocess.run(
        [*compose, "up", "-d", "--no-build", "akashic-plugin-gate"],
        cwd=repo,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    socket = sandbox / "akashic.sock"
    container_id = ""
    control_ready = False
    dashboard_ready = False
    stable_since: float | None = None
    deadline = time.monotonic() + 30
    while started.returncode == 0 and time.monotonic() < deadline:
        container_id = subprocess.run(
            [*compose, "ps", "-a", "-q", "akashic-plugin-gate"],
            cwd=repo,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if not container_id:
            time.sleep(0.2)
            continue
        running = (
            subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
                check=False,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout.strip()
            == "true"
        )
        if not running:
            break
        control_ready = _control_ready(socket)
        dashboard_ready = _dashboard_ready(container_id)
        if control_ready and dashboard_ready:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= 1:
                break
        else:
            stable_since = None
        time.sleep(0.2)
    runtime_stable = stable_since is not None and time.monotonic() - stable_since >= 1
    process = ""
    if container_id:
        process = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "print(Path('/proc/1/cmdline').read_bytes()"
                    ".replace(b'\\0', b' ').decode())"
                ),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
    candidate_prepare: dict[str, object] = {}
    migrated_probe: dict[str, object] = {}
    all_plugins_probe: dict[str, object] = {}
    management_probe: dict[str, object] = {}
    proactive_fetch_probe: dict[str, object] = {}
    fitbit_probe: dict[str, object] = {}
    fitbit_processes = -1
    fitbit_reload: dict[str, object] = {}
    fitbit_disabled: dict[str, object] = {}
    fitbit_disabled_processes = -1
    fitbit_data_before: dict[str, str] = {}
    fitbit_data_after: dict[str, str] = {}
    if fitbit_data is not None and container_id and runtime_stable:
        fitbit_data_before = _persistent_file_hashes(fitbit_data)
        fitbit_probe = _fitbit_runtime_probe(container_id)
        fitbit_processes = _container_process_count(container_id, "monitor/server.py")
        before = _snapshot_statuses(container_id)
        fitbit_source = (
            sandbox / "home/.akashic-plugin/cache/gate/fitbit/1.1.0/plugin.py"
        )
        _ = fitbit_source.write_text(
            fitbit_source.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        _, fitbit_reload = _wait_snapshot_status(
            container_id,
            after=len(before),
            publication_state="committed",
            plugin_id="fitbit@gate",
        )
        fitbit_processes = _wait_process_count(
            container_id,
            "monitor/server.py",
            lambda count: count == 1,
        )
        fitbit_probe = _fitbit_runtime_probe(container_id)
        before = _snapshot_statuses(container_id)
        manifest = sandbox / "home/.akashic-plugin/manifest.toml"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        _ = manifest.write_text(
            '[plugins."fitbit@gate"]\nenabled = false\n',
            encoding="utf-8",
        )
        _, fitbit_disabled = _wait_snapshot_status(
            container_id,
            after=len(before),
            publication_state="disabled",
            plugin_id="fitbit@gate",
        )
        fitbit_disabled_processes = _wait_process_count(
            container_id,
            "monitor/server.py",
            lambda count: count == 0,
        )
        fitbit_data_after = _persistent_file_hashes(fitbit_data)
    if candidate_states is not None and container_id and runtime_stable:
        if phase == "snapshot":
            candidate_prepare = _exercise_snapshot_publish(
                container_id,
                candidate_states[4],
                candidate_states[5],
                socket,
            )
        elif phase == "topology":
            candidate_prepare = _exercise_topology_watch(
                container_id,
                sandbox,
                candidate_states[5],
            )
        else:
            candidate_prepare = _exercise_candidate_prepare(
                container_id,
                candidate_states[4],
                candidate_states[5],
                socket,
            )
    if migrated_observe is not None and container_id and runtime_stable:
        migrated_probe = _exercise_migrated_plugins(
            container_id,
            migrated_observe,
            sandbox,
        )
    if all_plugins is not None and container_id and runtime_stable:
        all_plugins_probe = _exercise_all_plugins(
            container_id,
            all_plugins[0],
            all_plugins[1],
        )
    if management is not None and container_id and runtime_stable:
        management_probe = _exercise_plugin_management(
            container_id,
            management[0],
            management[1],
            management[2],
        )
    if proactive_fetch_calls is not None and container_id and runtime_stable:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not proactive_fetch_calls.exists():
            time.sleep(0.1)
        calls = (
            proactive_fetch_calls.read_text(encoding="utf-8").splitlines()
            if proactive_fetch_calls.exists()
            else []
        )
        proactive_fetch_probe = {
            "passed": '"get_context"' in calls,
            "calls": calls,
        }
    _ = (
        _wait_json_value(scope_state, "event_started", True)
        if scope_state is not None and runtime_stable
        else {}
    )
    logs = subprocess.run(
        [*compose, "logs", "--no-color", "--tail", "200", "akashic-plugin-gate"],
        cwd=repo,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout
    stopped = subprocess.run(
        [*compose, "stop", "-t", "15", "akashic-plugin-gate"],
        cwd=repo,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    exit_code = -1
    if container_id:
        raw_exit_code = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.ExitCode}}", container_id],
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if raw_exit_code.isdigit():
            exit_code = int(raw_exit_code)
    phase_passed = True
    phase_evidence: object = {"phase": phase}
    if scope_state is not None:
        state: dict[str, object] = {}
        if scope_state.exists():
            raw_state: object = json.loads(scope_state.read_text(encoding="utf-8"))
            if isinstance(raw_state, dict):
                mapping = cast(dict[object, object], raw_state)
                state = {str(key): value for key, value in mapping.items()}
        state["service_started"] = (scope_state.parent / "service.started").exists()
        state["service_stopped"] = (scope_state.parent / "service.stopped").exists()
        expected: dict[str, object] = {
            "initialized": True,
            "event_started": True,
            "channel_started": True,
            "channel_stopped": True,
            "terminated": True,
            "task_cancelled": True,
            "subscription_closed": True,
            "events": 1,
            "service_started": True,
            "service_stopped": True,
        }
        phase_passed = all(state.get(key) == value for key, value in expected.items())
        generation = state.get("generation")
        phase_passed = (
            phase_passed
            and isinstance(generation, str)
            and generation.startswith("scope_gate@gate:")
        )
        expected["generation"] = "scope_gate@gate:<revision>:<sequence>"
        phase_evidence = {"phase": phase, "state": state, "expected": expected}
    if candidate_states is not None:
        (
            valid_path,
            invalid_path,
            failed_path,
            observer_path,
            _,
            _,
        ) = candidate_states
        valid_state: dict[str, object] = {}
        if valid_path.exists():
            raw_valid: object = json.loads(valid_path.read_text(encoding="utf-8"))
            if isinstance(raw_valid, dict):
                mapping = cast(dict[object, object], raw_valid)
                valid_state = {str(key): value for key, value in mapping.items()}
        observer_state: dict[str, object] = {}
        if observer_path.exists():
            raw_observer: object = json.loads(observer_path.read_text(encoding="utf-8"))
            if isinstance(raw_observer, dict):
                mapping = cast(dict[object, object], raw_observer)
                observer_state = {str(key): value for key, value in mapping.items()}
        generation = valid_state.get("generation")
        phase_passed = (
            valid_state.get("initialized") is True
            and isinstance(generation, str)
            and generation.startswith("candidate_valid@gate:")
            and not invalid_path.exists()
            and not failed_path.exists()
            and observer_state.get("failed_tool_visible") is False
            and observer_state.get("event_sent") is True
            and candidate_prepare.get("passed") is True
        )
        phase_evidence = {
            "phase": phase,
            "valid": valid_state,
            "invalid_initialized": invalid_path.exists(),
            "failed_candidate_state_exists": failed_path.exists(),
            "observer": observer_state,
            "same_id_prepare": candidate_prepare,
        }
    if fitbit_data is not None:
        runtime_log = fitbit_data / "monitor.runtime.log"
        phase_passed = (
            fitbit_processes == 1
            and fitbit_reload.get("publication_state") == "committed"
            and fitbit_reload.get("old_generation")
            != fitbit_reload.get("new_generation")
            and fitbit_disabled.get("publication_state") == "disabled"
            and fitbit_disabled_processes == 0
            and fitbit_data_before == fitbit_data_after
            and isinstance(fitbit_probe.get("data"), dict)
            and isinstance(fitbit_probe.get("snapshot"), dict)
            and runtime_log.exists()
        )
        phase_evidence = {
            "phase": phase,
            "monitor_processes": fitbit_processes,
            "disabled_monitor_processes": fitbit_disabled_processes,
            "reload": fitbit_reload,
            "disabled": fitbit_disabled,
            "persistent_data_unchanged": fitbit_data_before == fitbit_data_after,
            "probe": fitbit_probe,
            "runtime_log": str(runtime_log),
        }
    if migrated_observe is not None:
        phase_passed = migrated_probe.get("passed") is True
        phase_evidence = {"phase": phase, **migrated_probe}
    if all_plugins is not None:
        phase_passed = all_plugins_probe.get("passed") is True
        phase_evidence = {"phase": phase, **all_plugins_probe}
    if management is not None:
        phase_passed = management_probe.get("passed") is True
        phase_evidence = {"phase": phase, **management_probe}
    if proactive_fetch_calls is not None:
        phase_passed = proactive_fetch_probe.get("passed") is True
        phase_evidence = {"phase": phase, **proactive_fetch_probe}
    passed = (
        started.returncode == 0
        and control_ready
        and dashboard_ready
        and runtime_stable
        and "python main.py" in process
        and stopped.returncode == 0
        and exit_code == 0
        and phase_passed
    )
    return passed, {
        "container_id": container_id,
        "control_ready": control_ready,
        "dashboard_ready": dashboard_ready,
        "runtime_stable": runtime_stable,
        "pid1_is_main": "python main.py" in process,
        "pid1": process.strip(),
        "exit_code": exit_code,
        "start_output": started.stdout[-2000:],
        "stop_output": stopped.stdout[-2000:],
        "logs": logs[-4000:],
        "phase": phase_evidence,
    }


def _control_ready(path: Path) -> bool:
    if not path.exists():
        return False
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(0.5)
    try:
        client.connect(str(path))
        return True
    except OSError:
        return False
    finally:
        client.close()


def _dashboard_ready(container_id: str) -> bool:
    result = subprocess.run(
        [
            "docker",
            "exec",
            container_id,
            "python",
            "-c",
            (
                "import urllib.request; "
                "urllib.request.urlopen("
                "'http://127.0.0.1:2236/api/dashboard/plugins', timeout=1).read()"
            ),
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--scenario",
        choices=("sandbox-integrity", "full-runtime"),
        default="sandbox-integrity",
    )
    _ = parser.add_argument(
        "--phase",
        choices=(
            "smoke",
            "scope",
            "candidate",
            "capability-hosts",
            "snapshot",
            "topology",
            "plugins",
            "all-plugins",
            "fitbit",
            "management",
            "proactive-fetch",
        ),
        default="smoke",
    )
    _ = parser.add_argument("--inside-container", action="store_true")
    args = parser.parse_args()
    if args.inside_container:
        return 0 if _sandbox_integrity().status == "passed" else 1
    if args.scenario in ("sandbox-integrity", "full-runtime"):
        return _run_controller(scenario=args.scenario, phase=args.phase)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
