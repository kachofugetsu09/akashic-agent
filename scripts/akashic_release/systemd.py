from __future__ import annotations

import shutil
import subprocess
import grp
import os
import pwd
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

Run = Callable[..., subprocess.CompletedProcess[str]]
_UNITS = ("akashic-host-bridge.service", "akashic-core.service")
_EXTERNAL_UNIT = "akashic-home-services.service"
_SYSTEM_UNIT_ROOT = Path("/etc/systemd/system")


def verify_external_service_contract(
    *, run: Run, unit_root: Path = _SYSTEM_UNIT_ROOT
) -> None:
    """Require the separately owned home-services lifecycle unit."""

    if unit_root != _SYSTEM_UNIT_ROOT:
        external_unit = unit_root / _EXTERNAL_UNIT
        if not external_unit.is_file():
            raise RuntimeError(f"隔离 unit root 缺少外围服务合同: {external_unit}")
        run(
            ["systemd-analyze", "verify", str(external_unit)],
            check=True,
            capture_output=True,
            text=True,
        )
        return
    run(
        ["systemctl", "cat", "--", _EXTERNAL_UNIT],
        check=True,
        capture_output=True,
        text=True,
    )


def install_units(
    *,
    checkout: Path,
    backup_root: Path,
    run: Run,
    unit_root: Path = _SYSTEM_UNIT_ROOT,
) -> bool:
    """Install changed unit templates with a recoverable pre-write backup."""

    source_root = checkout / "docker" / "host-runtime" / "systemd"
    service_account = pwd.getpwuid(os.getuid())
    service_user = service_account.pw_name
    service_home = Path(service_account.pw_dir)
    if not service_home.is_absolute():
        raise RuntimeError("service user home 必须为绝对路径")
    service_group = grp.getgrgid(os.getgid()).gr_name
    changed: list[tuple[str, bytes, Path]] = []
    for name in _UNITS:
        source = source_root / name
        target = unit_root / name
        if not source.is_file():
            raise RuntimeError(f"release 缺少 systemd unit: {source}")
        rendered = _render_unit(source, service_user, service_group, service_home)
        if not target.exists() or target.read_bytes() != rendered:
            changed.append((name, rendered, target))
    if not changed:
        if unit_root == _SYSTEM_UNIT_ROOT:
            run(["sudo", "systemctl", "enable", *_UNITS], check=True)
        return False

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_root / f"systemd-{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    for name, rendered, target in changed:
        if target.exists():
            shutil.copy2(target, backup / target.name)
        staged = backup / f".{name}.installing"
        staged.write_bytes(rendered)
        try:
            if unit_root != _SYSTEM_UNIT_ROOT:
                temporary = target.with_name(f".{target.name}.installing")
                shutil.copy2(staged, temporary)
                temporary.chmod(0o644)
                temporary.replace(target)
                continue
            run(
                ["sudo", "install", "-m", "0644", str(staged), str(target)],
                check=True,
            )
        finally:
            staged.unlink()
    if unit_root == _SYSTEM_UNIT_ROOT:
        run(["sudo", "systemctl", "daemon-reload"], check=True)
        run(["sudo", "systemctl", "enable", *_UNITS], check=True)
    return True


def _render_unit(
    source: Path,
    service_user: str,
    service_group: str,
    service_home: Path,
) -> bytes:
    text = source.read_text(encoding="utf-8")
    if text.count("User=huashen") != 1 or text.count("Group=huashen") != 1:
        raise RuntimeError(f"systemd unit 用户模板结构无效: {source}")
    rendered = (
        text.replace("User=huashen", f"User={service_user}")
        .replace("Group=huashen", f"Group={service_group}")
        .replace("%h", str(service_home))
    )
    if (
        rendered.count(f"User={service_user}") != 1
        or rendered.count(f"Group={service_group}") != 1
    ):
        raise RuntimeError(f"systemd unit 用户模板未完整渲染: {source}")
    return rendered.encode("utf-8")


def install_operator_entrypoint(
    *,
    checkout: Path,
    backup_root: Path,
    target: Path,
) -> bool:
    """Install the stable user CLI with a recoverable pre-write backup."""

    source = checkout / "scripts" / "akashic-release"
    if target.exists() and target.read_bytes() == source.read_bytes():
        return False
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_root / f"operator-cli-{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    if target.exists():
        shutil.copy2(target, backup / target.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.installing")
    shutil.copy2(source, temporary)
    temporary.chmod(0o755)
    temporary.replace(target)
    return True


def stop_runtime(*, run: Run) -> None:
    run(["sudo", "systemctl", "stop", *_UNITS[::-1]], check=True)


def start_bridge(*, run: Run) -> None:
    run(["sudo", "systemctl", "start", _UNITS[0]], check=True)
    run(["systemctl", "is-active", "--quiet", _UNITS[0]], check=True)


def start_core(*, run: Run) -> None:
    run(["sudo", "systemctl", "start", _UNITS[1]], check=True)
    run(["systemctl", "is-active", "--quiet", _UNITS[1]], check=True)
