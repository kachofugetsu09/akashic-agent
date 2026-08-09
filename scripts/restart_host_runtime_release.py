from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

_CORE = "akashic-core.service"
_HOME_SERVICES = "akashic-home-services.service"
_HOST_BRIDGE = "akashic-host-bridge.service"
_OPENCLI_BROWSER = "akashic-opencli-browser.service"
_DEFAULT_RUNTIME_ENV = Path("/home/huashen/.config/akashic-container/runtime.env")


def runtime_container_name(environment_file: Path) -> str:
    """Read the release container name from the staged runtime environment."""

    values: list[str] = []
    for raw_line in environment_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("AKASHIC_CONTAINER_NAME="):
            values.append(line.split("=", 1)[1])
    if (
        len(values) != 1
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", values[0]) is None
    ):
        raise RuntimeError("runtime.env 必须包含唯一合法的 AKASHIC_CONTAINER_NAME")
    return values[0]


def wait_for_core_health(container_name: str, timeout_sec: float) -> None:
    """Wait until the release-bound Core healthcheck reports healthy."""

    deadline = time.monotonic() + timeout_sec
    while True:
        result = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                container_name,
                "--format",
                "{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            status = result.stdout.strip()
        elif "No such object" in result.stderr or "No such container" in result.stderr:
            status = "starting"
        else:
            raise RuntimeError(f"Docker inspect 失败: {result.stderr.strip()}")
        if status == "healthy":
            return
        if status == "unhealthy" or status == "missing":
            raise RuntimeError(f"Core healthcheck 未通过: {status}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待 Core healthy 超时: {status}")
        time.sleep(2)


def run_release_restart(
    core_container: str,
    include_opencli_browser: bool = True,
    health_timeout_sec: float = 180,
) -> None:
    """Restart one staged Akashic release in dependency order and fail loudly."""

    # 1. Enter maintenance with every PartOf owner stopped in one transaction.
    _ = subprocess.run(
        ["systemctl", "stop", _CORE, _HOME_SERVICES, _HOST_BRIDGE], check=True
    )

    # 2. Start dependencies explicitly; PartOf does not propagate a plain start.
    _ = subprocess.run(["systemctl", "start", _HOME_SERVICES], check=True)
    _ = subprocess.run(["systemctl", "start", _HOST_BRIDGE], check=True)
    if include_opencli_browser:
        _ = subprocess.run(["systemctl", "try-restart", _OPENCLI_BROWSER], check=True)

    # 3. Start Core only after its preflight proves the live generation.
    _ = subprocess.run(["systemctl", "start", _CORE], check=True)
    for unit in (_HOME_SERVICES, _HOST_BRIDGE, _CORE):
        _ = subprocess.run(["systemctl", "is-active", "--quiet", unit], check=True)
    wait_for_core_health(core_container, health_timeout_sec)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restart a staged Akashic host runtime release"
    )
    _ = parser.add_argument(
        "--skip-opencli-browser",
        action="store_true",
        help="leave the independently managed OpenCLI browser generation unchanged",
    )
    _ = parser.add_argument("--health-timeout-sec", type=float, default=180)
    _ = parser.add_argument(
        "--runtime-env-file", type=Path, default=_DEFAULT_RUNTIME_ENV
    )
    args = parser.parse_args()
    run_release_restart(
        runtime_container_name(args.runtime_env_file),
        not args.skip_opencli_browser,
        args.health_timeout_sec,
    )


if __name__ == "__main__":
    main()
