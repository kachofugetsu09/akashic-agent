from __future__ import annotations

import re
import subprocess
from pathlib import Path

from agent.tools.unified_exec import MAX_WRITE_STDIN_YIELD_TIME_MS

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "docker" / "host-runtime" / "systemd"


def test_core_consumes_external_services_without_owning_them() -> None:
    core_unit = (SYSTEMD / "akashic-core.service").read_text(encoding="utf-8")
    network = (ROOT / "docker/host-runtime/compose.external-services.yaml").read_text(
        encoding="utf-8"
    )

    assert "Requires=akashic-host-bridge.service" in core_unit
    assert "Wants=akashic-home-services.service" in core_unit
    assert (
        "After=docker.service akashic-host-bridge.service akashic-home-services.service"
        in core_unit
    )
    assert (
        "PartOf=akashic-host-bridge.service" in core_unit
    )
    assert "Requires=akashic-home-services.service" not in core_unit
    assert "PartOf=akashic-home-services.service" not in core_unit
    assert "home-services.env" not in core_unit
    assert "verify-running-home-services" not in core_unit
    assert "compose.external-services.yaml" in core_unit
    assert "external: true" in network
    assert not (ROOT / "docker/home-services").exists()
    assert not (SYSTEMD / "akashic-home-services.service").exists()
    assert not (SYSTEMD / "akashic-opencli-browser.service").exists()


def test_host_bridge_lease_covers_longest_write_stdin_wait() -> None:
    bridge_unit = (SYSTEMD / "akashic-host-bridge.service").read_text(
        encoding="utf-8"
    )
    match = re.search(r"--lease-timeout (\d+)", bridge_unit)

    assert match is not None
    assert int(match.group(1)) > MAX_WRITE_STDIN_YIELD_TIME_MS / 1_000 + 2


def test_release_restart_does_not_control_external_services(monkeypatch) -> None:
    from scripts.restart_host_runtime_release import run_release_restart

    calls: list[list[str]] = []

    def run(
        arguments: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        output = "healthy\n" if arguments[0] == "docker" else ""
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr("subprocess.run", run)
    run_release_restart("akashic-core")

    assert calls[:3] == [
        ["systemctl", "stop", "akashic-core.service", "akashic-host-bridge.service"],
        ["systemctl", "start", "akashic-host-bridge.service"],
        ["systemctl", "start", "akashic-core.service"],
    ]
    assert calls[3:5] == [
        ["systemctl", "is-active", "--quiet", unit]
        for unit in ("akashic-host-bridge.service", "akashic-core.service")
    ]
    assert all("akashic-home-services.service" not in call for call in calls)
    assert all("akashic-opencli-browser.service" not in call for call in calls)


def test_release_restart_rejects_unhealthy_core(monkeypatch) -> None:
    import pytest

    from scripts.restart_host_runtime_release import wait_for_core_health

    monkeypatch.setattr(
        "subprocess.run",
        lambda arguments, **_kwargs: subprocess.CompletedProcess(
            arguments, 0, "unhealthy\n", ""
        ),
    )
    with pytest.raises(RuntimeError, match="healthcheck 未通过"):
        wait_for_core_health("akashic-core", 10)


def test_release_restart_waits_for_container_creation(monkeypatch) -> None:
    from scripts.restart_host_runtime_release import wait_for_core_health

    results = iter(
        (
            subprocess.CompletedProcess(
                [], 1, "", "Error: No such object: akashic-core"
            ),
            subprocess.CompletedProcess([], 0, "healthy\n", ""),
        )
    )
    monkeypatch.setattr("subprocess.run", lambda *_args, **_kwargs: next(results))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    wait_for_core_health("akashic-core", 10)


def test_release_restart_reads_container_name_from_runtime_env(tmp_path: Path) -> None:
    from scripts.restart_host_runtime_release import runtime_container_name

    environment = tmp_path / "runtime.env"
    environment.write_text(
        "AKASHIC_CONTAINER_NAME=akashic-core-canary\n", encoding="utf-8"
    )
    assert runtime_container_name(environment) == "akashic-core-canary"
