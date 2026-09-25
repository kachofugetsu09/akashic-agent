from __future__ import annotations

import shutil
import tempfile
import subprocess
import grp
import os
from collections.abc import Mapping
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
    backup_root: Path | None,
    run: Run,
    unit_root: Path = _SYSTEM_UNIT_ROOT,
    runtime_env: Path | None = None,
) -> bool:
    """安装发生变化的单元；备份由部署者选择。"""

    source_root = checkout / "docker" / "host-runtime" / "systemd"
    service_user, service_group, service_home = resolve_service_account(
        installed_unit=(unit_root / _UNITS[0]) if unit_root.joinpath(_UNITS[0]).exists() else None,
        runtime_env=runtime_env,
        environ=os.environ,
        fallback_uid=os.getuid(),
        fallback_gid=os.getgid(),
    )
    if not service_home.is_absolute():
        raise RuntimeError("service user home 必须为绝对路径")
    changed: list[tuple[str, bytes, Path]] = []
    rendered_units: dict[str, bytes] = {}
    for name in _UNITS:
        source = source_root / name
        target = unit_root / name
        if not source.is_file():
            raise RuntimeError(f"release 缺少 systemd unit: {source}")
        rendered = _render_unit(source, service_user, service_group, service_home)
        _require_preserved_environment_file(target, rendered)
        rendered_units[name] = rendered
        if not target.exists() or target.read_bytes() != rendered:
            changed.append((name, rendered, target))
    # 2. 预检必须在写任何单元之前完成：环境文件缺失是启动期故障，不是安装期发现。
    verify_rendered_units(rendered_units)
    if not changed:
        if unit_root == _SYSTEM_UNIT_ROOT:
            run(["sudo", "systemctl", "enable", *_UNITS], check=True)
        return False

    backup = None
    if backup_root is not None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = backup_root / f"systemd-{timestamp}"
        backup.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="akashic-units-") as temporary_root:
        for name, rendered, target in changed:
            if target.exists() and backup is not None:
                shutil.copy2(target, backup / target.name)
            staged = Path(temporary_root) / name
            staged.write_bytes(rendered)
            if unit_root != _SYSTEM_UNIT_ROOT:
                temporary = target.with_name(f".{target.name}.installing")
                shutil.copy2(staged, temporary)
                temporary.chmod(0o644)
                temporary.replace(target)
            else:
                run(["sudo", "install", "-m", "0644", str(staged), str(target)], check=True)
    if unit_root == _SYSTEM_UNIT_ROOT:
        run(["sudo", "systemctl", "daemon-reload"], check=True)
        run(["sudo", "systemctl", "enable", *_UNITS], check=True)
    return True


def resolve_service_account(
    *,
    installed_unit: Path | None,
    runtime_env: Path | None,
    environ: Mapping[str, str],
    fallback_uid: int,
    fallback_gid: int,
) -> tuple[str, str, Path]:
    """解析运行时服务身份，绝不用 sudo 会话的 euid 覆盖已发布的属主。

    安装链可能以 root 身份被调用（`sudo akashic-release install`），此时
    `os.getuid()` 是 root，会把单元渲染成 `User=root` 和 `/root/.config/...`，
    让下一轮启动因为环境文件不存在而失败。所以按下面的优先级取身份。
    """

    # 1. 已安装单元是运行时属主的权威来源：重装必须保持同一个身份。
    installed = _installed_account(installed_unit)
    if installed is not None:
        return installed
    # 2. 操作者环境文件的所有者就是运行时属主。
    owner = _file_owner(runtime_env)
    if owner is not None:
        return owner
    # 3. sudo 调用者身份优先于 euid；root 调用者不构成"操作者是 root"的证据。
    caller = environ.get("SUDO_USER", "")
    if caller and caller != "root":
        account = pwd.getpwnam(caller)
        raw_gid = environ.get("SUDO_GID", "")
        gid = int(raw_gid) if raw_gid.isdigit() else account.pw_gid
        return account.pw_name, grp.getgrgid(gid).gr_name, Path(account.pw_dir)
    # 4. 只有没有任何操作者证据时才退回当前进程身份。
    account = pwd.getpwuid(fallback_uid)
    return account.pw_name, grp.getgrgid(fallback_gid).gr_name, Path(account.pw_dir)


def _installed_account(unit: Path | None) -> tuple[str, str, Path] | None:
    """读取已安装单元声明的属主；缺失或无效返回 None。"""

    if unit is None or not unit.is_file():
        return None
    user = group = ""
    for line in unit.read_text(encoding="utf-8").splitlines():
        if line.startswith("User="):
            user = line.split("=", 1)[1].strip()
        elif line.startswith("Group="):
            group = line.split("=", 1)[1].strip()
    if not user or not group:
        return None
    try:
        account = pwd.getpwnam(user)
    except KeyError as error:
        raise RuntimeError(f"已安装单元引用了不存在的用户: {user}") from error
    return account.pw_name, group, Path(account.pw_dir)


def _file_owner(path: Path | None) -> tuple[str, str, Path] | None:
    """以运行时环境文件的所有者作为操作者身份。"""

    if path is None:
        return None
    try:
        status = path.stat()
    except OSError:
        return None
    account = pwd.getpwuid(status.st_uid)
    return account.pw_name, grp.getgrgid(status.st_gid).gr_name, Path(account.pw_dir)


def environment_files(rendered: bytes | str) -> tuple[str, ...]:
    """列出单元声明的 EnvironmentFile；保留 systemd 的可选前缀语义。"""

    text = rendered.decode("utf-8") if isinstance(rendered, bytes) else rendered
    values: list[str] = []
    for line in text.splitlines():
        if not line.startswith("EnvironmentFile="):
            continue
        value = line.split("=", 1)[1].strip()
        values.append(value[1:] if value.startswith("-") else value)
    return tuple(values)


def verify_rendered_units(rendered: Mapping[str, bytes]) -> None:
    """写单元之前证明它们引用的环境文件真实存在且为绝对路径。"""

    problems: list[str] = []
    for name, text in sorted(rendered.items()):
        for value in environment_files(text):
            path = Path(value)
            if not path.is_absolute():
                problems.append(f"{name}: EnvironmentFile 必须是绝对路径: {value}")
            elif not path.is_file():
                problems.append(f"{name}: EnvironmentFile 不存在: {value}")
    if problems:
        raise RuntimeError("systemd unit 预检失败，未写入任何单元: " + "; ".join(problems))


def _require_preserved_environment_file(target: Path, rendered: bytes) -> None:
    """重装不得悄悄改变已安装单元的环境文件路径。"""

    if not target.is_file():
        return
    before = set(environment_files(target.read_bytes()))
    after = set(environment_files(rendered))
    if before and after and before != after:
        raise RuntimeError(
            f"systemd unit 重装会改变 EnvironmentFile: {sorted(before)} -> {sorted(after)}"
        )


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
    backup_root: Path | None,
    target: Path,
) -> bool:
    """安装固定 CLI 入口；备份由部署者选择。"""

    source = checkout / "scripts" / "akashic-release"
    if target.exists() and target.read_bytes() == source.read_bytes():
        return False
    if backup_root is not None and target.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = backup_root / f"operator-cli-{timestamp}"
        backup.mkdir(parents=True, exist_ok=False)
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
