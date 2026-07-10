#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
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


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--scenario",
        choices=("sandbox-integrity",),
        default="sandbox-integrity",
    )
    args = parser.parse_args()
    if args.scenario == "sandbox-integrity":
        return 0 if _sandbox_integrity().status == "passed" else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
