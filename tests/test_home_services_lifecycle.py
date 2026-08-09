from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOME_SERVICES = ROOT / "docker" / "home-services"
SYSTEMD = ROOT / "docker" / "host-runtime" / "systemd"


def test_home_service_images_are_pinned_release_inputs() -> None:
    env = (HOME_SERVICES / "env.example").read_text(encoding="utf-8")
    image_lines = [line for line in env.splitlines() if line.split("=", 1)[0].endswith("_IMAGE")]

    assert len(image_lines) == 5
    assert "replace-me" not in env
    for line in image_lines:
        assert re.fullmatch(r"[A-Z_]+=[^\s]+@sha256:[0-9a-f]{64}", line)

    assert (HOME_SERVICES / "rsshub.env.example").is_file()


def test_systemd_is_the_only_home_service_restart_owner() -> None:
    compose = (HOME_SERVICES / "compose.yaml").read_text(encoding="utf-8")
    opencli_compose = (HOME_SERVICES / "compose.opencli.yaml").read_text(encoding="utf-8")
    home_unit = (SYSTEMD / "akashic-home-services.service").read_text(encoding="utf-8")
    opencli_unit = (SYSTEMD / "akashic-opencli-browser.service").read_text(encoding="utf-8")

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

    assert "Requires=akashic-host-bridge.service akashic-home-services.service" in core_unit
    assert "Requires=docker.service" in core_unit
    assert "After=docker.service akashic-host-bridge.service akashic-home-services.service" in core_unit
    assert "PartOf=akashic-host-bridge.service akashic-home-services.service" in core_unit
    assert "ExecStart=/usr/bin/env ${AKASHIC_MISE} exec -C ${AKASHIC_RUNTIME_CHECKOUT} --" in bridge_unit
    assert "Before=akashic-core.service" in home_unit
    assert "docker/home-services/compose.yaml up --detach --wait" in home_unit


def test_opencli_browser_remains_outside_core_dependency_graph() -> None:
    core_unit = (SYSTEMD / "akashic-core.service").read_text(encoding="utf-8")

    assert "akashic-opencli-browser.service" not in core_unit
