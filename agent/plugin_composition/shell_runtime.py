from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from core.common.file_io import run_file_io


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

    def derive_argv(self, command: str, *, login: bool, snapshot: Path | None = None) -> list[str]:
        """Build the direct process argv for one shell command."""
        if self.kind in {ShellKind.ZSH, ShellKind.BASH, ShellKind.SH}:
            # 用户环境优先取快照：非交互 shell 只 source 一次性导出的 rc 结果。
            if login and snapshot is not None:
                script = f". {shlex.quote(str(snapshot))} || exit $?; eval {shlex.quote(command)}"
                return [str(self.path), "-c", script]
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


# ---------------------------------------------------------------------------
# 用户 shell 快照：交互式 shell 加载一次 rc，导出函数、alias、选项和环境变化；
# 之后每条命令用非交互 shell source 快照，既用到用户 zshrc 又不重复跑 profile。
# ---------------------------------------------------------------------------

_SNAPSHOT_TIMEOUT_S = 10.0
# 由工具按调用设置或只在交互 shell 有意义的变量，不进入快照。
_VOLATILE_ENV = frozenset({
    "PWD", "OLDPWD", "SHLVL", "_", "PS1", "PS2", "PS3", "PS4", "PROMPT", "RPROMPT", "RPS1",
    "TERM", "COLUMNS", "LINES", "NO_COLOR", "COLORTERM", "LANG", "LC_CTYPE", "LC_ALL",
    "PAGER", "GIT_PAGER", "GH_PAGER", "ZLE_RPROMPT_INDENT",
})
# 非交互 shell 中无效或有害的 zsh 选项。
_ZSH_SKIP_OPTIONS = frozenset({
    "interactive", "zle", "monitor", "shinstdin", "singlecommand", "login", "privileged",
    "restricted", "interactivecomments",
})
_RcSignature = tuple[tuple[str, int, int] | None, ...]
_SnapshotKey = tuple[str, _RcSignature, tuple[tuple[str, str], ...]]
_snapshots: dict[_SnapshotKey, Path | None] = {}


# rc 文件的变更签名；任何一个变化都让快照失效。
def _rc_signature(shell: ResolvedShell, env: Mapping[str, str]) -> _RcSignature:
    home = Path(env.get("HOME") or Path.home())
    if shell.kind is ShellKind.ZSH:
        zdot = Path(env.get("ZDOTDIR") or home)
        names = [zdot / ".zshenv", zdot / ".zprofile", zdot / ".zshrc", zdot / ".zlogin",
                 Path("/etc/zsh/zshenv"), Path("/etc/zsh/zprofile"), Path("/etc/zsh/zshrc"), Path("/etc/zsh/zlogin"),
                 Path("/etc/zshenv"), Path("/etc/zprofile"), Path("/etc/zshrc"), Path("/etc/zlogin")]
    else:
        names = [home / ".bashrc", home / ".bash_profile", home / ".bash_login", home / ".profile",
                 Path("/etc/bash.bashrc"), Path("/etc/profile")]
    signature: list[tuple[str, int, int] | None] = []
    for path in names:
        try:
            stat = path.stat()
        except OSError:
            signature.append(None)
            continue
        signature.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def _snapshot_key(shell: ResolvedShell, env: Mapping[str, str]) -> _SnapshotKey:
    return (str(shell.path), _rc_signature(shell, env), tuple(sorted(
        (name, value) for name, value in env.items()
        if name not in _VOLATILE_ENV and not name.startswith("AKASHIC_")
    )))


def cached_shell_snapshot(shell: ResolvedShell, env: Mapping[str, str] | None = None) -> tuple[bool, Path | None]:
    """返回 (是否已有结论, 快照路径)；只做 stat，可在事件循环内调用。"""
    if shell.kind not in {ShellKind.ZSH, ShellKind.BASH}:
        return True, None
    key = _snapshot_key(shell, os.environ if env is None else env)
    if key in _snapshots:
        return True, _snapshots[key]
    return False, None


def build_shell_snapshot(shell: ResolvedShell, env: Mapping[str, str] | None = None) -> Path | None:
    """在线程中构建快照；失败时记为 None，调用方退回 login shell。"""
    environment = dict(os.environ if env is None else env)
    key = _snapshot_key(shell, environment)
    if key in _snapshots:
        return _snapshots[key]
    try:
        path = _write_snapshot(shell, key, environment)
    except (OSError, subprocess.SubprocessError, ValueError):
        path = None
    _snapshots[key] = path
    return path


async def snapshot_shell_argv(argv: list[str], command: str, env: Mapping[str, str]) -> list[str]:
    """只在实际进程 owner 所在主机创建并复用 login shell 快照。"""
    if not argv:
        raise ValueError("shell 快照缺少执行路径")
    shell = resolve_shell(argv[0])
    if shell.kind not in {ShellKind.BASH, ShellKind.ZSH} or argv != shell.derive_argv(command, login=True):
        raise ValueError("shell 快照只支持完整的 Bash 或 Zsh login 命令")
    known, snapshot = cached_shell_snapshot(shell, env)
    if not known:
        snapshot = await run_file_io(lambda: build_shell_snapshot(shell, env))
    return shell.derive_argv(command, login=True, snapshot=snapshot)


def _write_snapshot(shell: ResolvedShell, key: _SnapshotKey, env: dict[str, str]) -> Path:
    # 1. 只在缓存创建时加载 login 与交互配置；后续命令 source 快照，不重新 login。
    root = Path(tempfile.gettempdir()) / f"akashic-shell-{os.getuid()}"
    root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(root, 0o700)
    with tempfile.TemporaryDirectory(dir=root) as work:
        q = shlex.quote(work)
        if shell.kind is ShellKind.ZSH:
            # 下划线开头也可能是普通命令的依赖，不能按名字删函数。
            dump = ("builtin typeset -f "
                    f">{q}/functions 2>/dev/null; alias -L >{q}/aliases 2>/dev/null; "
                    f"setopt >{q}/options 2>/dev/null; env -0 >{q}/env")
        else:
            dump = (f"declare -f >{q}/functions 2>/dev/null; alias -p >{q}/aliases 2>/dev/null; "
                    f"shopt -p >{q}/options 2>/dev/null; env -0 >{q}/env")
        _ = subprocess.run(
            [str(shell.path), "-lic", dump], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=_SNAPSHOT_TIMEOUT_S, cwd=env.get("HOME") or None, env=env,
            check=False, start_new_session=True,
        )
        parts = {name: (Path(work) / name) for name in ("functions", "aliases", "options", "env")}
        if not parts["env"].is_file():
            raise ValueError("shell 快照未生成环境")
        body = _assemble_snapshot(shell, {name: path.read_bytes() for name, path in parts.items() if path.is_file()}, env)
    # 2. 快照可能含用户导出的密钥：仅本用户可读，原子发布。
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    target = root / f"{shell.kind.value}-{digest}.sh"
    temporary = root / f".{target.name}.{os.getpid()}"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        _ = stream.write(body)
    os.replace(temporary, target)
    # 3. zsh 编译成 wordcode，source 时自动使用更新的 .zwc。
    if shell.kind is ShellKind.ZSH:
        _ = subprocess.run([str(shell.path), "-fc", f"zcompile {shlex.quote(str(target))}"],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=_SNAPSHOT_TIMEOUT_S, check=False)
    return target


def _assemble_snapshot(shell: ResolvedShell, parts: dict[str, bytes], env: Mapping[str, str]) -> str:
    lines = ["# akashic shell snapshot"]
    if shell.kind is ShellKind.BASH:
        lines.append("shopt -s expand_aliases")
    lines.append(parts.get("functions", b"").decode("utf-8", "replace"))
    lines.append(parts.get("aliases", b"").decode("utf-8", "replace"))
    options = parts.get("options", b"").decode("utf-8", "replace").split()
    if shell.kind is ShellKind.ZSH:
        lines.extend(f"setopt {name} 2>/dev/null" for name in options if name not in _ZSH_SKIP_OPTIONS)
    else:
        lines.append(parts.get("options", b"").decode("utf-8", "replace"))
    # 只导出 rc 新增或改动的变量，避免覆盖网关按调用设置的环境。
    for entry in parts.get("env", b"").split(b"\0"):
        name, sep, value = entry.decode("utf-8", "replace").partition("=")
        if not sep or not name.isidentifier() or name in _VOLATILE_ENV or name.startswith("AKASHIC_"):
            continue
        if env.get(name) == value:
            continue
        lines.append(f"export {name}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"
