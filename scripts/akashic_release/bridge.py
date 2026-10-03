from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from scripts.verify_host_runtime_deployment import verify_host_toolchain_deployment
from scripts.akashic_release.manifest import atomic_write
from utils.timing import measure

Run = Callable[..., subprocess.CompletedProcess[str]]
_PYPI_INDEX_URL = "https://mirrors.aliyun.com/pypi/simple"


def prepare_runtime_checkout(
    bootstrap_checkout: Path,
    commit: str,
    target: Path,
    origin: str,
) -> Path:
    """Publish a clean exact-commit runtime checkout."""

    from scripts.prepare_runtime_checkout import prepare_runtime_checkout as prepare

    return prepare(bootstrap_checkout, commit, target, origin)


def prepare_bridge_venv(
    *,
    checkout: Path,
    target: Path,
    mise: Path,
    run: Run,
    env: Mapping[str, str] | None = None,
    command_prefix: tuple[str, ...] = (),
) -> Path:
    """Bind the release to dependencies keyed by interpreter and lock bytes."""

    # 1. Resolve the exact toolchain; source commit is not a dependency input.
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Bridge venv 已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    run([*command_prefix, str(mise), "install", "--yes"], cwd=checkout, check=True, env=env)
    executable = Path(run(
        [*command_prefix, str(mise), "which", "python"], cwd=checkout,
        check=True, capture_output=True, text=True, env=env,
    ).stdout.strip()).resolve(strict=True)
    lock = checkout / "docker/host-runtime/requirements.lock"
    with executable.open("rb") as stream:
        python_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    inputs = json.dumps({"python": str(executable), "python_sha256": python_digest,
                         "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}, sort_keys=True)
    identity = hashlib.sha256(inputs.encode()).hexdigest()
    cached = target.parent / ("inputs-" + identity)
    marker = cached / ".akashic-bridge-input.json"
    # 2. Publish a ready marker only after the fixed environment is complete.
    with measure("bridge.environment", input=identity) as timing:
        if cached.exists() or cached.is_symlink():
            if cached.is_symlink() or not cached.is_dir() or marker.is_symlink() or marker.read_text() != inputs:
                raise RuntimeError(f"Bridge dependency cache 未完成或已漂移: {cached}")
            if not (cached / "bin/python").is_file():
                raise RuntimeError(f"Bridge dependency cache 缺少解释器: {cached}")
            timing["reused"] = True
        else:
            try:
                run([*command_prefix, str(mise), "exec", "--", "uv", "venv",
                     "--python", str(executable), str(cached)], cwd=checkout, check=True, env=env)
                run([*command_prefix, str(mise), "exec", "--", "uv", "pip", "install",
                     "--default-index", _PYPI_INDEX_URL, "--require-hashes", "--python",
                     str(cached / "bin/python"), "--requirement", str(lock)],
                    cwd=checkout, check=True, env=env)
                atomic_write(marker, inputs)
            except BaseException:
                if cached.exists():
                    shutil.rmtree(cached)
                raise
            timing["reused"] = False
        # The venv stays at its original path; only the release binding is new.
        target.symlink_to(cached, target_is_directory=True)
    return target / "bin/python"


def verify_bridge(
    *,
    manifest: Path,
    checkout: Path,
    mise: Path,
    bridge_python: Path,
    toolchain_digest: str,
) -> None:
    """Run the canonical identity verifier before publishing activation state."""

    verify_host_toolchain_deployment(
        manifest,
        checkout,
        mise,
        bridge_python,
        toolchain_digest,
    )
