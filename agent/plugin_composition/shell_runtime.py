from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ShellKind(str, Enum):
    ZSH = "zsh"
    BASH = "bash"
    POWERSHELL = "powershell"
    SH = "sh"
    CMD = "cmd"


@dataclass(frozen=True)
class ResolvedShell:
    kind: ShellKind
    path: Path

    def derive_argv(self, command: str, *, login: bool) -> list[str]:
        """Build the direct process argv for one shell command."""
        if self.kind in {ShellKind.ZSH, ShellKind.BASH, ShellKind.SH}:
            return [str(self.path), "-lc" if login else "-c", command]
        if self.kind is ShellKind.POWERSHELL:
            profile_args = [] if login else ["-NoProfile"]
            return [str(self.path), *profile_args, "-Command", command]
        return [str(self.path), "/c", command]


def detect_shell_kind(shell_path: str | Path) -> ShellKind | None:
    name = str(shell_path).replace("\\", "/").rsplit("/", 1)[-1].lower()
    stem = name.removesuffix(".exe")
    return {
        "zsh": ShellKind.ZSH,
        "bash": ShellKind.BASH,
        "pwsh": ShellKind.POWERSHELL,
        "powershell": ShellKind.POWERSHELL,
        "sh": ShellKind.SH,
        "cmd": ShellKind.CMD,
    }.get(stem)


def resolve_shell(requested: str | None = None) -> ResolvedShell:
    """Resolve an explicit shell or the current user's Codex-style default."""
    if requested is not None:
        return _resolve_explicit_shell(requested)
    return _resolve_default_shell()


def _resolve_explicit_shell(requested: str) -> ResolvedShell:
    """Resolve a model-selected shell without silently changing its semantics."""

    # 1. 只接受 Codex 明确定义 argv 语义的 shell 类型。
    value = requested.strip()
    if not value:
        raise ValueError("shell 不能为空")
    kind = detect_shell_kind(value)
    if kind is None:
        raise ValueError(f"不支持的 shell: {requested}")

    # 2. 显式路径必须精确存在；裸名称按 PATH 和平台常见路径查找。
    if "/" in value or "\\" in value:
        path = Path(value).expanduser()
        if not _is_executable_file(path):
            raise ValueError(f"shell 不存在或不可执行: {path}")
        return ResolvedShell(kind, path)
    resolved = _find_shell(kind, preferred_name=value)
    if resolved is None:
        raise ValueError(f"找不到 shell: {requested}")
    return resolved


def _resolve_default_shell() -> ResolvedShell:
    """Use the passwd shell first, followed by Codex's platform order."""

    # 1. Unix 默认值来自 passwd，而不是可被单次进程覆盖的 SHELL 环境变量。
    if os.name != "nt":
        user_path = _unix_user_shell_path()
        user_kind = detect_shell_kind(user_path) if user_path is not None else None
        if (
            user_path is not None
            and user_kind is not None
            and _is_executable_file(user_path)
        ):
            return ResolvedShell(user_kind, user_path)

    # 2. 采用 Codex 的平台 fallback 顺序，最终 fallback 仍必须真实可执行。
    if os.name == "nt":
        order = (ShellKind.POWERSHELL, ShellKind.CMD)
    elif sys.platform == "darwin":
        order = (ShellKind.ZSH, ShellKind.BASH, ShellKind.SH)
    else:
        order = (ShellKind.BASH, ShellKind.ZSH, ShellKind.SH)
    for kind in order:
        resolved = _find_shell(kind)
        if resolved is not None:
            return resolved
    raise RuntimeError("找不到可执行的默认 shell")


def _unix_user_shell_path() -> Path | None:
    import pwd

    try:
        value = pwd.getpwuid(os.getuid()).pw_shell
    except KeyError:
        return None
    return Path(value) if value else None


def _find_shell(
    kind: ShellKind,
    *,
    preferred_name: str | None = None,
) -> ResolvedShell | None:
    names = [preferred_name] if preferred_name is not None else _shell_names(kind)
    for name in names:
        found = shutil.which(name)
        if found is not None:
            return ResolvedShell(kind, Path(found))
    for candidate in _fallback_paths(kind):
        if _is_executable_file(candidate):
            return ResolvedShell(kind, candidate)
    return None


def _shell_names(kind: ShellKind) -> tuple[str, ...]:
    if kind is ShellKind.POWERSHELL:
        return ("pwsh", "powershell")
    if kind is ShellKind.CMD:
        return ("cmd", "cmd.exe")
    return (kind.value,)


def _fallback_paths(kind: ShellKind) -> tuple[Path, ...]:
    if kind is ShellKind.ZSH:
        return (Path("/bin/zsh"),)
    if kind is ShellKind.BASH:
        return (Path("/bin/bash"), Path("/usr/bin/bash"))
    if kind is ShellKind.SH:
        return (Path("/bin/sh"),)
    if kind is ShellKind.POWERSHELL:
        if os.name == "nt":
            return (
                Path(r"C:\Program Files\PowerShell\7\pwsh.exe"),
                Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"),
            )
        return (Path("/usr/local/bin/pwsh"),)
    return ()


def _is_executable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return os.name == "nt" or os.access(path, os.X_OK)
