#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path


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


def _run_controller() -> int:
    repo = Path(__file__).resolve().parents[2]
    plugin_root = Path(
        os.environ.get("AKASHIC_PLUGIN_SOURCE", "/mnt/data/coding/akashic-plugin")
    ).resolve()
    host_cache = (Path.home() / ".akashic-plugin" / "cache").resolve()
    sandbox = Path(tempfile.mkdtemp(prefix="akashic-plugin-gate-", dir="/tmp")).resolve()
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
    }
    compose = [
        "docker",
        "compose",
        "-p",
        "akashic-plugin-reload-gate",
        "-f",
        str(repo / "docker/debug/docker-compose.yml"),
        "--profile",
        "plugin-gate",
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
    integrity = subprocess.run(command, cwd=repo, env=env, check=False)
    smoke_passed, smoke_evidence = _run_runtime_smoke(
        repo=repo,
        sandbox=sandbox,
        compose=compose,
        env=env,
    )
    _ = subprocess.run(
        [*compose, "down", "--remove-orphans"],
        cwd=repo,
        env=env,
        check=False,
    )
    after = {str(path): _repository_digest(path) for path in repositories}
    unchanged = before == after
    report: dict[str, object] = {
        "gate_id": "G-1-host",
        "status": (
            "passed"
            if integrity.returncode == 0 and smoke_passed and unchanged
            else "failed"
        ),
        "checks": {
            "container_gate_passed": integrity.returncode == 0,
            "runtime_smoke_passed": smoke_passed,
            "runtime": smoke_evidence,
            "repositories_unchanged": unchanged,
            "repositories": len(repositories),
            "sandbox": str(sandbox),
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


def _run_runtime_smoke(
    *,
    repo: Path,
    sandbox: Path,
    compose: list[str],
    env: dict[str, str],
) -> tuple[bool, dict[str, object]]:
    _write_smoke_config(sandbox)
    shutil.rmtree(
        sandbox / "home/.akashic-plugin/cache/gate",
        ignore_errors=True,
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
    deadline = time.monotonic() + 30
    while started.returncode == 0 and time.monotonic() < deadline:
        container_id = subprocess.run(
            [*compose, "ps", "-q", "akashic-plugin-gate"],
            cwd=repo,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        if socket.exists() and container_id:
            break
        time.sleep(0.2)
    socket_ready = socket.exists()
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
    passed = (
        started.returncode == 0
        and socket_ready
        and "python main.py" in process
        and stopped.returncode == 0
        and exit_code == 0
    )
    return passed, {
        "container_id": container_id,
        "socket_ready": socket_ready,
        "pid1_is_main": "python main.py" in process,
        "pid1": process.strip(),
        "exit_code": exit_code,
        "start_output": started.stdout[-2000:],
        "stop_output": stopped.stdout[-2000:],
        "logs": logs[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--scenario",
        choices=("sandbox-integrity",),
        default="sandbox-integrity",
    )
    _ = parser.add_argument("--inside-container", action="store_true")
    args = parser.parse_args()
    if args.scenario == "sandbox-integrity":
        if args.inside_container:
            return 0 if _sandbox_integrity().status == "passed" else 1
        return _run_controller()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
