#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import cast


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
    before = {str(repo): _repository_digest(repo) for repo in repositories}

    sandbox = Path("/sandbox")
    cache = Path.home() / ".akashic-plugin" / "cache"
    test_plugin = cache / "gate" / "integrity" / "1.0.0" / "plugin.py"
    test_plugin.parent.mkdir(parents=True, exist_ok=True)
    _ = test_plugin.write_text("REVISION = 1\n", encoding="utf-8")
    _ = test_plugin.write_text("REVISION = 2\n", encoding="utf-8")

    after = {str(repo): _repository_digest(repo) for repo in repositories}
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
        CheckResult("sandbox_writable", "rw" in sandbox_options, sorted(sandbox_options)),
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
    status = "passed" if all(check.passed for check in checks) else "failed"
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
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-plugin-gate-", dir="/tmp")).resolve()
    (sandbox / "static").mkdir()
    protected = [repo.resolve(), plugin_root, host_cache]
    if any(sandbox == path or sandbox.is_relative_to(path) for path in protected):
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
    passed = (
        build_returncode == 0
        and integrity_returncode == 0
        and smoke_passed
        and cleanup_returncode == 0
        and unchanged
        and not controller_error
    )
    report: dict[str, object] = {
        "gate_id": "G-1-host" if scenario == "sandbox-integrity" else f"runtime:{phase}",
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


def _write_smoke_config(sandbox: Path) -> None:
    config = sandbox / "config.toml"
    _ = config.write_text(
        "\n".join(
            [
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
                "[channels]",
                'socket = "/sandbox/akashic.sock"',
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
                "enabled = false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _install_scope_plugin(sandbox: Path) -> Path:
    plugin_dir = (
        sandbox
        / "home/.akashic-plugin/cache/gate/scope_gate/1.0.0"
    )
    plugin_dir.mkdir(parents=True, exist_ok=True)
    _ = (plugin_dir / "plugin.py").write_text(
        "from __future__ import annotations\n"
        "import asyncio\n"
        "from agent.plugins import Plugin\n"
        "from bus.events_lifecycle import TurnCommitted\n"
        "class ScopeGatePlugin(Plugin):\n"
        "    name = 'scope_gate'\n"
        "    version = '1.0.0'\n"
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
        "        await self.context.event_bus.fanout(TurnCommitted(\n"
        "            session_key='gate:scope', channel='gate', chat_id='scope',\n"
        "            input_message='scope', persisted_user_message='scope',\n"
        "            assistant_response='scope', tools_used=[]))\n"
        "    def _check_subscription(self):\n"
        "        self.context.kv_store.set('subscription_closed', not self.subscription.active)\n"
        "    def _handle(self, event):\n"
        "        self.context.kv_store.increment('events')\n",
        encoding="utf-8",
    )
    return sandbox / "home/.akashic-plugin/data/scope_gate-gate/.kv.json"


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
        "        \"\"\"Failed candidate tool.\"\"\"\n"
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
        "    for name in ('fetch_events', 'ack_events', 'poll_events', 'candidate_version')\n"
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
    _ = reload_source.write_text(_candidate_reload_source("v1"), encoding="utf-8")
    data = sandbox / "home/.akashic-plugin/data"
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
        else "        self.context.kv_store.set('initialized_version', 'v2')\n"
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
        "from agent.plugins import (IntervalTrigger, McpServerSpec, Plugin, PluginJobSpec, "
        "PluginSemanticCheck, ProactiveSourceSpec, tool)\n"
        "class CandidateReloadPlugin(Plugin):\n"
        "    name = 'candidate_reload'\n"
        f"{skills}"
        "    @classmethod\n"
        "    def mcp_servers(cls):\n"
        f"        return [McpServerSpec(name='candidate_feed', command=('python', 'candidate_mcp_server.py'), env={{'CANDIDATE_VERSION': '{version}'}})]\n"
        "    def proactive_sources(self):\n"
        "        return [ProactiveSourceSpec(id='candidate_feed', channels=('content',), "
        "server='candidate_feed', fetch_tool='fetch_events', ack_tool='ack_events', "
        f"poll_tool='poll_events', poll_interval_seconds={1 if version == 'v1' else 2})]\n"
        "    def jobs(self):\n"
        f"        return [PluginJobSpec(id='refresh', triggers=[IntervalTrigger({1 if version == 'v1' else 2})], handler=self.refresh)]\n"
        "    async def refresh(self, context):\n"
        f"        self.context.kv_store.increment('job_runs_{version}')\n"
        "    async def readiness_semantic_checks(self, context):\n"
        "        server = context.mcp_catalog.servers['candidate_feed']\n"
        "        value = await server.client.call('candidate_version', {})\n"
        "        job = context.job_catalog.jobs['candidate_reload@gate:refresh']\n"
        "        source = context.proactive_catalog.sources['candidate_reload@gate:candidate_feed']\n"
        "        owned = getattr(job.spec.handler, '__self__', None) is self\n"
        "        job_interval = job.spec.triggers[0].seconds\n"
        "        source_interval = source.spec.poll_interval_seconds\n"
        "        evidence = {'mcp': value, 'job_owned': owned, 'source': source.spec.id, "
        "'job_interval': job_interval, 'source_interval': source_interval}\n"
        f"        return [PluginSemanticCheck('candidate_capabilities', value == '{version}' and owned and job_interval == {1 if version == 'v1' else 2} and source_interval == {1 if version == 'v1' else 2}, evidence)]\n"
        "    @tool(name='candidate_reload_tool')\n"
        "    async def run(self, event):\n"
        "        \"\"\"Candidate reload tool.\"\"\"\n"
        f"        return '{version}'\n"
        "    async def initialize(self):\n"
        f"{initialize}"
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
    marker = "plugin_candidate_status "
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


def _exercise_candidate_prepare(
    container_id: str,
    source_path: Path,
    state_path: Path,
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
            "mcp_candidate_feed__poll_events",
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
                    "source_interval": 2,
                },
            }
        ]
        and valid_jobs == ["candidate_reload@gate:refresh"]
        and valid_sources == ["candidate_reload@gate:candidate_feed"]
        and return_status.get("jobs") == ["candidate_reload@gate:refresh"]
        and return_status.get("proactive_sources")
        == ["candidate_reload@gate:candidate_feed"]
        and valid_status.get("job_specs")
        == {
            "candidate_reload@gate:refresh": [
                {"type": "interval", "seconds": 2}
            ]
        }
        and valid_status.get("proactive_source_specs")
        == {
            "candidate_reload@gate:candidate_feed": {
                "server": "candidate_feed",
                "fetch_tool": "fetch_events",
                "ack_tool": "ack_events",
                "poll_tool": "poll_events",
                "poll_interval_seconds": 2,
            }
        }
        and return_status.get("job_specs")
        == {
            "candidate_reload@gate:refresh": [
                {"type": "interval", "seconds": 1}
            ]
        }
        and return_status.get("proactive_source_specs")
        == {
            "candidate_reload@gate:candidate_feed": {
                "server": "candidate_feed",
                "fetch_tool": "fetch_events",
                "ack_tool": "ack_events",
                "poll_tool": "poll_events",
                "poll_interval_seconds": 1,
            }
        }
        and cast(dict[str, int], after_valid_calls.get("v2", {})).get(
            "candidate_version",
            0,
        )
        == 1
        and all(
            cast(dict[str, int], after_valid_calls.get("v2", {})).get(tool, 0)
            == 0
            for tool in ("fetch_events", "poll_events", "ack_events")
        )
        and _integer(after_valid.get("job_runs_v1"))
        > _integer(initial.get("job_runs_v1"))
        and _integer(after_valid.get("job_runs_v2")) == 0
        and after_return_calls.get("v2") == after_valid_calls.get("v2")
        and "candidate-skill" in valid_skills
        and valid_description_map.get("candidate-skill") == "candidate v2 skill"
        and valid_drift_description_map.get("candidate-drift") == "candidate drift v2"
        and hash_maps[0].get("candidate-skill")
        == _skill_fixture_hash("v2", drift=False)
        and hash_maps[1].get("candidate-drift")
        == _skill_fixture_hash("v2", drift=True)
        and hash_maps[2].get("candidate-skill")
        == _skill_fixture_hash("v1", drift=False)
        and hash_maps[3].get("candidate-drift")
        == _skill_fixture_hash("v1", drift=True)
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
    _write_smoke_config(sandbox)
    shutil.rmtree(
        sandbox / "home/.akashic-plugin/cache/gate",
        ignore_errors=True,
    )
    scope_state = _install_scope_plugin(sandbox) if phase == "scope" else None
    candidate_states = (
        _install_candidate_plugins(sandbox)
        if phase in {"candidate", "capability-hosts"}
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
    ipc_ready = False
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
        running = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_id],
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip() == "true"
        if not running:
            break
        ipc_ready = _ipc_ready(socket)
        dashboard_ready = _dashboard_ready(container_id)
        if ipc_ready and dashboard_ready:
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
    if candidate_states is not None and container_id and runtime_stable:
        candidate_prepare = _exercise_candidate_prepare(
            container_id,
            candidate_states[4],
            candidate_states[5],
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
        expected: dict[str, object] = {
            "initialized": True,
            "terminated": True,
            "task_cancelled": True,
            "subscription_closed": True,
            "events": 1,
        }
        phase_passed = all(state.get(key) == value for key, value in expected.items())
        generation = state.get("generation")
        phase_passed = phase_passed and isinstance(generation, str) and generation.startswith(
            "scope_gate@gate:"
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
            raw_observer: object = json.loads(
                observer_path.read_text(encoding="utf-8")
            )
            if isinstance(raw_observer, dict):
                mapping = cast(dict[object, object], raw_observer)
                observer_state = {
                    str(key): value for key, value in mapping.items()
                }
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
    passed = (
        started.returncode == 0
        and ipc_ready
        and dashboard_ready
        and runtime_stable
        and "python main.py" in process
        and stopped.returncode == 0
        and exit_code == 0
        and phase_passed
    )
    return passed, {
        "container_id": container_id,
        "ipc_ready": ipc_ready,
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


def _ipc_ready(path: Path) -> bool:
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
        choices=("smoke", "scope", "candidate", "capability-hosts"),
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
