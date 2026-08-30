from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from scripts.verify_host_runtime_deployment import verify_host_toolchain_deployment

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
) -> Path:
    """Create one commit-bound Bridge interpreter and install locked runtime deps."""

    if target.exists():
        raise FileExistsError(f"Bridge venv 已存在: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(
            [str(mise), "install", "--yes"],
            cwd=checkout,
            check=True,
        )
        python_executable = run(
            [str(mise), "which", "python"],
            cwd=checkout,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        run(
            [
                str(mise),
                "exec",
                "--",
                "uv",
                "venv",
                "--python",
                python_executable,
                str(target),
            ],
            cwd=checkout,
            check=True,
        )
        python = target / "bin" / "python"
        run(
            [
                str(mise),
                "exec",
                "--",
                "uv",
                "pip",
                "install",
                "--default-index",
                _PYPI_INDEX_URL,
                "--require-hashes",
                "--python",
                str(python),
                "--requirement",
                str(checkout / "docker/host-runtime/requirements.lock"),
            ],
            cwd=checkout,
            check=True,
        )
    except BaseException:
        if target.exists():
            shutil.rmtree(target)
        raise
    return target / "bin" / "python"


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
