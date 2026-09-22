from __future__ import annotations

import grp
import os
import pwd
import subprocess
from pathlib import Path

import pytest

from scripts.akashic_release.systemd import (
    environment_files,
    install_units,
    resolve_service_account,
    verify_rendered_units,
)

ACCOUNT = pwd.getpwuid(os.getuid())
GROUP = grp.getgrgid(os.getgid()).gr_name
HOME = Path(ACCOUNT.pw_dir)


def _unit(name: str, *, user: str, group: str, env: str) -> str:
    return (
        "[Unit]\nDescription=" + name + "\n\n[Service]\n"
        f"User={user}\nGroup={group}\nEnvironmentFile={env}\n"
        "ExecStart=/bin/true\n"
    )


def test_installed_unit_identity_wins_over_invoking_euid(tmp_path: Path) -> None:
    """sudo 会话里重装必须保留已安装单元的属主，而不是改用 root。"""

    installed = tmp_path / "akashic-host-bridge.service"
    installed.write_text(
        _unit("bridge", user=ACCOUNT.pw_name, group=GROUP, env=f"{HOME}/.config/akashic-container/runtime.env"),
        encoding="utf-8",
    )
    user, group, home = resolve_service_account(
        installed_unit=installed,
        runtime_env=None,
        environ={"SUDO_USER": "root"},
        fallback_uid=0,
        fallback_gid=0,
    )
    assert (user, group, home) == (ACCOUNT.pw_name, GROUP, HOME)


def test_runtime_env_owner_is_used_when_no_unit_is_installed(tmp_path: Path) -> None:
    env = tmp_path / "runtime.env"
    env.write_text("AKASHIC_ROOT=/srv/data\n", encoding="utf-8")
    user, group, home = resolve_service_account(
        installed_unit=None,
        runtime_env=env,
        environ={"SUDO_USER": "root"},
        fallback_uid=0,
        fallback_gid=0,
    )
    assert (user, group, home) == (ACCOUNT.pw_name, GROUP, HOME)


def test_sudo_caller_is_used_before_falling_back_to_euid() -> None:
    user, group, home = resolve_service_account(
        installed_unit=None,
        runtime_env=None,
        environ={"SUDO_USER": ACCOUNT.pw_name},
        fallback_uid=0,
        fallback_gid=0,
    )
    assert (user, group, home) == (ACCOUNT.pw_name, GROUP, HOME)


def test_root_caller_is_not_treated_as_the_operator() -> None:
    """只有 SUDO_USER=root 时不构成操作者证据，仍然退回当前身份。"""

    user, _, _ = resolve_service_account(
        installed_unit=None,
        runtime_env=None,
        environ={"SUDO_USER": "root"},
        fallback_uid=os.getuid(),
        fallback_gid=os.getgid(),
    )
    assert user == ACCOUNT.pw_name


def test_preflight_rejects_missing_environment_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent" / "runtime.env"
    rendered = {"akashic-core.service": _unit(
        "core", user=ACCOUNT.pw_name, group=GROUP, env=str(missing),
    ).encode("utf-8")}
    with pytest.raises(RuntimeError, match="EnvironmentFile 不存在"):
        verify_rendered_units(rendered)


def test_preflight_rejects_relative_environment_file() -> None:
    rendered = {"akashic-core.service": _unit(
        "core", user=ACCOUNT.pw_name, group=GROUP, env="config.toml",
    ).encode("utf-8")}
    with pytest.raises(RuntimeError, match="必须是绝对路径"):
        verify_rendered_units(rendered)


def test_preflight_accepts_present_environment_file(tmp_path: Path) -> None:
    env = tmp_path / "runtime.env"
    env.write_text("A=1\n", encoding="utf-8")
    rendered = {"akashic-core.service": _unit(
        "core", user=ACCOUNT.pw_name, group=GROUP, env=str(env),
    ).encode("utf-8")}
    verify_rendered_units(rendered)
    assert environment_files(rendered["akashic-core.service"]) == (str(env),)


def test_environment_files_ignores_optional_marker() -> None:
    text = "EnvironmentFile=-/etc/akashic/optional.env\n"
    assert environment_files(text) == ("/etc/akashic/optional.env",)


def test_install_units_refuses_to_change_an_installed_environment_file(tmp_path: Path) -> None:
    """已安装单元的环境文件路径不允许被重装悄悄改写。"""

    checkout = tmp_path / "checkout"
    templates = checkout / "docker" / "host-runtime" / "systemd"
    templates.mkdir(parents=True)
    unit_root = tmp_path / "units"
    unit_root.mkdir()
    other_home = tmp_path / "other-home"
    (other_home / ".config" / "akashic-container").mkdir(parents=True)
    (other_home / ".config" / "akashic-container" / "runtime.env").write_text("A=1\n", encoding="utf-8")

    for name in ("akashic-host-bridge.service", "akashic-core.service"):
        (templates / name).write_text(
            _unit(name, user="huashen", group="huashen",
                  env="%h/.config/akashic-container/runtime.env"),
            encoding="utf-8",
        )
        # 已安装单元声明了另一个环境文件；渲染会把它渲染成别的路径。
        (unit_root / name).write_text(
            _unit(name, user=ACCOUNT.pw_name, group=GROUP, env=str(tmp_path / "installed.env")),
            encoding="utf-8",
        )
    (tmp_path / "installed.env").write_text("A=1\n", encoding="utf-8")

    def run(*args: object, **kwargs: object) -> object:
        raise AssertionError("预检失败时不得执行任何 systemd 命令")

    with pytest.raises(RuntimeError, match="会改变 EnvironmentFile|EnvironmentFile 不存在"):
        install_units(
            checkout=checkout,
            backup_root=tmp_path / "backups",
            run=run,  # type: ignore[arg-type]
            unit_root=unit_root,
            runtime_env=None,
        )


def test_operator_environment_points_mise_at_the_owner_home(tmp_path: Path) -> None:
    """venv 创建必须在操作者家目录下解析运行时，而不是 sudo 的 /root。"""

    from scripts.akashic_release.ownership import operator_environment

    environment = operator_environment(HOME)
    assert environment["HOME"] == str(HOME)
    assert not environment.get("XDG_DATA_HOME", "").startswith("/root")


def test_bridge_preparation_runs_every_tool_as_the_runtime_user(tmp_path: Path) -> None:
    """sudo 安装时 mise 与 uv 都必须使用 systemd 的运行身份。"""

    from scripts.akashic_release.bridge import prepare_bridge_venv
    from scripts.akashic_release.ownership import runtime_user_prefix

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    target = tmp_path / "bridge-venv"
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[-3:] == ["which", "python"]:
            return subprocess.CompletedProcess(command, 0, stdout="/home/operator/python\n")
        if "venv" in command:
            (target / "bin").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="")

    prefix = runtime_user_prefix(user="operator", owner_uid=1000, invoking_uid=0)
    prepare_bridge_venv(
        checkout=checkout,
        target=target,
        mise=Path("/opt/mise"),
        run=run,
        command_prefix=prefix,
    )

    assert prefix == ("sudo", "-H", "-u", "operator", "--")
    assert len(commands) == 4
    assert all(tuple(command[: len(prefix)]) == prefix for command in commands)


def test_release_to_owner_keeps_exec_bits_and_grants_read(tmp_path: Path) -> None:
    from scripts.akashic_release.ownership import release_to_owner

    root = tmp_path / "artifact"
    (root / "bin").mkdir(parents=True)
    script = root / "bin" / "python"
    script.write_text("#!/bin/sh\n", encoding="utf-8")
    script.chmod(0o700)
    data = root / "manifest.json"
    data.write_text("{}\n", encoding="utf-8")
    data.chmod(0o600)

    release_to_owner(root, uid=os.getuid(), gid=os.getgid())

    assert script.stat().st_mode & 0o111, "执行位必须保留"
    assert script.stat().st_mode & 0o044, "必须补上可读位"
    assert data.stat().st_mode & 0o044


def test_resolve_runtime_owner_matches_the_unit_identity(tmp_path: Path) -> None:
    from scripts.akashic_release.ownership import resolve_runtime_owner

    env = tmp_path / "runtime.env"
    env.write_text("A=1\n", encoding="utf-8")
    user, uid, gid, home = resolve_runtime_owner(
        runtime_env=env,
        environ={"SUDO_USER": "root"},
        fallback_uid=0,
        fallback_gid=0,
    )
    assert user == ACCOUNT.pw_name
    assert (uid, gid) == (os.getuid(), os.getgid())
    assert home == HOME
