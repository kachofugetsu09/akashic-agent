from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME_SERVICES = ROOT / "docker" / "home-services"
SYSTEMD = ROOT / "docker" / "host-runtime" / "systemd"


def test_home_service_images_are_pinned_release_inputs() -> None:
    env = (HOME_SERVICES / "env.example").read_text(encoding="utf-8")
    image_lines = [
        line for line in env.splitlines() if line.split("=", 1)[0].endswith("_IMAGE")
    ]

    assert len(image_lines) == 5
    assert "replace-me" not in env
    for line in image_lines:
        assert re.fullmatch(r"[A-Z_]+=[^\s]+@sha256:[0-9a-f]{64}", line)

    assert (HOME_SERVICES / "rsshub.env.example").is_file()


def test_systemd_is_the_only_home_service_restart_owner() -> None:
    compose = (HOME_SERVICES / "compose.yaml").read_text(encoding="utf-8")
    opencli_compose = (HOME_SERVICES / "compose.opencli.yaml").read_text(
        encoding="utf-8"
    )
    home_unit = (SYSTEMD / "akashic-home-services.service").read_text(encoding="utf-8")
    opencli_unit = (SYSTEMD / "akashic-opencli-browser.service").read_text(
        encoding="utf-8"
    )

    assert "restart: unless-stopped" not in compose + opencli_compose
    assert compose.count('restart: "no"') == 4
    assert opencli_compose.count('restart: "no"') == 1
    for unit in (home_unit, opencli_unit):
        assert " up --detach --wait --wait-timeout 180" in unit
        assert " up --no-color --no-recreate --abort-on-container-exit" in unit
        assert "Restart=always" in unit
        assert "Requires=docker.service" in unit
        assert "After=docker.service network-online.target" in unit
        assert "User=huashen" in unit
        assert "SupplementaryGroups=docker" in unit
        assert "WantedBy=multi-user.target" in unit


def test_core_waits_for_healthy_required_home_services() -> None:
    core_unit = (SYSTEMD / "akashic-core.service").read_text(encoding="utf-8")
    home_unit = (SYSTEMD / "akashic-home-services.service").read_text(encoding="utf-8")
    bridge_unit = (SYSTEMD / "akashic-host-bridge.service").read_text(encoding="utf-8")

    assert (
        "Requires=akashic-host-bridge.service akashic-home-services.service"
        in core_unit
    )
    assert "Requires=docker.service" in core_unit
    assert (
        "After=docker.service akashic-host-bridge.service akashic-home-services.service"
        in core_unit
    )
    assert (
        "PartOf=akashic-host-bridge.service akashic-home-services.service" in core_unit
    )
    assert "EnvironmentFile=%h/.config/akashic-container/home-services.env" in core_unit
    assert "--verify-running-home-services" in core_unit
    assert (
        "ExecStart=/usr/bin/env ${AKASHIC_MISE} exec -C ${AKASHIC_RUNTIME_CHECKOUT} --"
        in bridge_unit
    )
    assert "Before=akashic-core.service" in home_unit
    assert "docker/home-services/compose.yaml up --detach --wait" in home_unit


def test_release_restart_orders_sidecars_before_core(monkeypatch) -> None:
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
    assert calls[:5] == [
        [
            "systemctl",
            "stop",
            "akashic-core.service",
            "akashic-home-services.service",
            "akashic-host-bridge.service",
        ],
        ["systemctl", "start", "akashic-home-services.service"],
        ["systemctl", "start", "akashic-host-bridge.service"],
        ["systemctl", "try-restart", "akashic-opencli-browser.service"],
        ["systemctl", "start", "akashic-core.service"],
    ]
    assert calls[5:8] == [
        ["systemctl", "is-active", "--quiet", unit]
        for unit in (
            "akashic-home-services.service",
            "akashic-host-bridge.service",
            "akashic-core.service",
        )
    ]
    assert calls[8] == [
        "docker",
        "container",
        "inspect",
        "akashic-core",
        "--format",
        "{{if .State.Health}}{{.State.Health.Status}}{{else}}missing{{end}}",
    ]


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


def test_opencli_browser_remains_outside_core_dependency_graph() -> None:
    core_unit = (SYSTEMD / "akashic-core.service").read_text(encoding="utf-8")

    assert "akashic-opencli-browser.service" not in core_unit


def test_opencli_healthcheck_uses_binary_shipped_by_browser_image() -> None:
    compose = (HOME_SERVICES / "compose.opencli.yaml").read_text(encoding="utf-8")

    assert '["CMD", "wget", "-q", "-O", "/dev/null"' in compose
    assert '["CMD", "curl"' not in compose
