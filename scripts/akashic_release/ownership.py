"""把安装产物交还给运行时属主。

安装链可能以 root 身份被调用（`sudo akashic-release install`）。如果不显式
纠正，生成物会带上 sudo 会话的身份：Bridge venv 会指向 `/root/.local/...` 的
解释器、release manifest 与 runtime checkout 会变成 root 独占，而 systemd 单元
以操作者用户运行，于是下一轮启动才以 `Permission denied` 暴露出来。
"""
from __future__ import annotations

import grp
import os
import pwd
from collections.abc import Mapping
from pathlib import Path

from scripts.akashic_release.systemd import resolve_service_account

_READ_BITS = 0o444
_EXEC_BITS = 0o111


def resolve_runtime_owner(
    *,
    runtime_env: Path | None,
    environ: Mapping[str, str],
    fallback_uid: int,
    fallback_gid: int,
) -> tuple[str, int, int, Path]:
    """解析运行时属主，返回 (user, uid, gid, home)。

    复用 systemd 单元使用的那一套证据链，保证单元属主与产物属主一致。
    """

    user, group, home = resolve_service_account(
        installed_unit=None,
        runtime_env=runtime_env,
        environ=environ,
        fallback_uid=fallback_uid,
        fallback_gid=fallback_gid,
    )
    account = pwd.getpwnam(user)
    return user, account.pw_uid, grp.getgrnam(group).gr_gid, home


def operator_environment(owner_home: Path) -> dict[str, str]:
    """让 mise/uv 在操作者家目录下解析运行时，而不是 sudo 的 `/root`。"""

    environment = dict(os.environ)
    environment["HOME"] = str(owner_home)
    data_home = environment.get("XDG_DATA_HOME", "")
    if not data_home or data_home.startswith("/root"):
        environment["XDG_DATA_HOME"] = str(owner_home / ".local" / "share")
    return environment


def runtime_user_prefix(
    *, user: str, owner_uid: int, invoking_uid: int
) -> tuple[str, ...]:
    """Return a sudo prefix when preparation must run as the runtime user."""

    if invoking_uid == owner_uid:
        return ()
    return ("sudo", "-H", "-u", user, "--")


def release_to_owner(path: Path, *, uid: int, gid: int) -> None:
    """递归把生成物交还给运行时属主，并保证可读、目录可进入。

    只补权限、不剥夺权限：原有执行位保持不变，符号链接本身不改写。
    """

    if not path.exists() and not path.is_symlink():
        return
    for current in (path, *sorted(path.rglob("*"))):
        try:
            os.chown(current, uid, gid, follow_symlinks=False)
        except OSError:
            # 只读挂载或非特权调用：保持现有属主，由后续预检负责报错。
            continue
        if current.is_symlink():
            continue
        try:
            mode = current.stat().st_mode & 0o7777
            if current.is_dir():
                os.chmod(current, mode | 0o755)
            else:
                extra = 0o755 if mode & _EXEC_BITS else 0o644
                os.chmod(current, mode | extra | _READ_BITS)
        except OSError:
            continue


def verify_readable_by(path: Path, *, uid: int) -> None:
    """证明运行时用户确实能读到关键产物；失败即安装期报错。"""

    target = path if path.is_dir() else path.parent
    probe = target / f".akashic-owner-probe-{os.getpid()}"
    try:
        probe.write_text("", encoding="utf-8")
        os.chown(probe, uid, -1)
    except OSError as error:
        raise RuntimeError(f"运行时属主无法在 {target} 写入: {error}") from error
    finally:
        probe.unlink(missing_ok=True)
    if path.is_file() and not os.access(path, os.R_OK):
        raise RuntimeError(f"安装产物对当前进程不可读: {path}")
