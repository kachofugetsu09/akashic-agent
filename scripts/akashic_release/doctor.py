from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path
from typing import Callable, Mapping

from scripts.restart_host_runtime_release import wait_for_core_health
from scripts.verify_host_runtime_deployment import verify_deployment_image
from scripts.verify_host_runtime_deployment import verify_host_toolchain_deployment

Run = Callable[..., subprocess.CompletedProcess[str]]


def read_environment(path: Path) -> dict[str, str]:
    """Read one strict systemd EnvironmentFile without shell evaluation."""

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise RuntimeError(f"runtime.env 非法行 {line_number}")
        key, value = line.split("=", 1)
        if not key or key in values or any(char.isspace() for char in key):
            raise RuntimeError(f"runtime.env 非法或重复 key: {key!r}")
        values[key] = value
    return values


def probe_bridge(
    environment: Mapping[str, str],
    *,
    environment_file: Path,
    run: Run = subprocess.run,
) -> None:
    """Run the canonical Bridge probe with the generation-owned interpreter."""

    command = [
        environment["AKASHIC_BRIDGE_PYTHON"],
        "-m",
        "scripts.akashic_release.doctor",
        "--bridge-probe-environment",
        str(environment_file),
    ]
    run(
        command,
        cwd=Path(environment["AKASHIC_RUNTIME_CHECKOUT"]),
        check=True,
        capture_output=True,
        text=True,
    )


def verify_release(environment_file: Path, *, health_timeout_sec: float = 180) -> None:
    """Verify manifest, host identity, Bridge RPC, and Core health."""

    environment = read_environment(environment_file)
    for unit in ("akashic-host-bridge.service", "akashic-core.service"):
        subprocess.run(["systemctl", "is-active", "--quiet", unit], check=True)
    manifest = Path(environment["AKASHIC_RELEASE_MANIFEST"])
    checkout = Path(environment["AKASHIC_RUNTIME_CHECKOUT"])
    bridge_python = Path(environment["AKASHIC_BRIDGE_PYTHON"])
    verify_deployment_image(manifest, environment["AKASHIC_IMAGE"])
    verify_host_toolchain_deployment(
        manifest,
        checkout,
        Path(environment["AKASHIC_MISE"]),
        bridge_python,
        environment["AKASHIC_HOST_TOOLCHAIN_DIGEST"],
    )
    probe_bridge(environment, environment_file=environment_file)
    wait_for_core_health(environment["AKASHIC_CONTAINER_NAME"], health_timeout_sec)


async def _inspect_bridge(environment: Mapping[str, str]) -> dict[str, object]:
    """Inspect the Bridge inside the generation-owned Python environment."""

    from agent.host_bridge.client import HostBridgeShellProcessManager

    manager = HostBridgeShellProcessManager(
        Path(environment["AKASHIC_HOST_BRIDGE_SOCKET"]),
        "release-doctor",
        environment["AKASHIC_HOST_BRIDGE_TOKEN"],
        environment["AKASHIC_RUNTIME_COMMIT"],
        environment["AKASHIC_HOST_TOOLCHAIN_DIGEST"],
    )
    try:
        return await manager.inspect()
    finally:
        await manager.close_transport()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe one Akashic release")
    parser.add_argument("--bridge-probe-environment", type=Path, required=True)
    args = parser.parse_args()
    environment = read_environment(args.bridge_probe_environment)
    inspected = asyncio.run(_inspect_bridge(environment))
    if inspected.get("releaseCommit") != environment["AKASHIC_RUNTIME_COMMIT"]:
        raise RuntimeError("Bridge running identity 与 runtime.env 不一致")


if __name__ == "__main__":
    main()
