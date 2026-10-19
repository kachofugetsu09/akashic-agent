from __future__ import annotations

import ipaddress
import os
import shlex
from pathlib import Path, PureWindowsPath
from urllib.parse import urlparse

_IS_WINDOWS = os.name == "nt"

_BANNED = frozenset(
    {
        "curlie",
        "axel",
        "aria2c",
        "nc",
        "telnet",
        "lynx",
        "w3m",
        "links",
        "http-prompt",
        "chrome",
        "firefox",
        "safari",
    }
)
_NETWORK_CMDS = frozenset({"curl", "wget", "http", "httpie", "xh"})
_NET_WRITE_FLAGS_CASE_SENSITIVE = frozenset({"-o", "-O", "-T", "-F"})
_NET_WRITE_FLAGS_LOWER = frozenset(
    {
        "--output",
        "--remote-name",
        "--upload-file",
        "--form",
        "--form-string",
        "--output-document",
        "--post-file",
        "--download",
        "--offline",
        "@",
    }
)
_RESTRICTED_META_CHARS = ("|", ";", "&", ">", "<", "`", "$(")
_RESTRICTED_SHELL_RUNNERS = frozenset(
    {
        "sh",
        "bash",
        "zsh",
        "fish",
        "python",
        "python3",
        "node",
        "perl",
        "ruby",
        "php",
        "lua",
    }
)


def validate_command(
    command: str,
    *,
    allow_network: bool,
    restricted_dir: Path | None,
    cwd: Path | None = None,
) -> str | None:
    """在 Shell 信任边界集中校验命令和路径约束。"""

    try:
        tokens = _split_command(command)
    except ValueError:
        return "命令解析失败，请检查引号是否匹配"
    if not tokens:
        return None

    command_name = _command_name(tokens[0])
    if command_name in _BANNED:
        return f"命令 '{command_name}' 不被允许（安全限制）"
    if not allow_network and command_name in _NETWORK_CMDS:
        return "当前 shell 配置禁止网络访问"

    if restricted_dir is not None:
        cwd_error = _validate_restricted_cwd(cwd, restricted_dir)
        if cwd_error:
            return cwd_error
        restricted_error = _validate_restricted_command(tokens, restricted_dir)
        if restricted_error:
            return restricted_error
    return validate_network_command(command, tokens=tokens)


def validate_network_command(
    command: str,
    *,
    tokens: list[str] | None = None,
) -> str | None:
    """仅允许网络命令访问公网 HTTP(S)，并禁止本地文件传输。"""

    if tokens is None:
        try:
            tokens = _split_command(command)
        except ValueError:
            return "命令解析失败，请检查引号是否匹配"
    if not tokens:
        return None
    command_name = _command_name(tokens[0])
    if command_name not in _NETWORK_CMDS:
        return None

    # 1. 拒绝写文件和上传参数。
    for token in tokens[1:]:
        lowered = token.lower()
        if (
            token in _NET_WRITE_FLAGS_CASE_SENSITIVE
            or lowered in _NET_WRITE_FLAGS_LOWER
        ):
            return f"网络命令参数 '{token}' 不被允许（禁止上传/写文件）"
        if any(lowered.startswith(flag + "=") for flag in _NET_WRITE_FLAGS_LOWER):
            return f"网络命令参数 '{token}' 不被允许（禁止上传/写文件）"
        if "=@" in token or token.startswith("@"):
            return f"网络命令参数 '{token}' 不被允许（禁止本地文件上传）"

    # 2. 只接受显式公网 HTTP(S) URL。
    urls = [token for token in tokens[1:] if token.startswith(("http://", "https://"))]
    if not urls:
        return "网络命令必须显式提供 http:// 或 https:// URL"
    for url in urls:
        error = _validate_url_target(url)
        if error:
            return error
    return None


def _validate_url_target(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅允许 http:// 或 https:// URL"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "URL 缺少主机名"
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return f"禁止访问内网/本地地址：{host}"
    except ValueError:
        if host.endswith(".local") or host.endswith(".localhost"):
            return f"禁止访问本地域名：{host}"
    return None


def _validate_restricted_command(tokens: list[str], restricted_dir: Path) -> str | None:
    command = " ".join(tokens)
    if any(marker in command for marker in _RESTRICTED_META_CHARS):
        return "受限 shell 禁止管道、重定向或串联命令"
    command_name = _command_name(tokens[0])
    if command_name in _RESTRICTED_SHELL_RUNNERS:
        return f"受限 shell 禁止启动解释器或二级 shell：{command_name}"
    for token in tokens[1:]:
        if token.startswith("-") or token == "--":
            continue
        error = _validate_restricted_token(token, restricted_dir)
        if error:
            return error
    return None


def _validate_restricted_cwd(cwd: Path | None, restricted_dir: Path) -> str | None:
    if cwd is None:
        return None
    try:
        resolved = cwd.resolve()
    except OSError:
        resolved = cwd
    if resolved != restricted_dir and restricted_dir not in resolved.parents:
        return f"受限 shell 禁止使用任务目录外工作目录：{cwd}"
    return None


def _validate_restricted_token(token: str, restricted_dir: Path) -> str | None:
    token = _strip_shell_quotes(token)
    if token.startswith("~"):
        return f"受限 shell 禁止访问任务目录外路径：{token}"
    if not _looks_like_path(token):
        return None
    parts = PureWindowsPath(token).parts if _IS_WINDOWS else Path(token).parts
    if any(part == ".." for part in parts):
        return f"受限 shell 禁止访问父级路径：{token}"
    windows_path = PureWindowsPath(token)
    if _IS_WINDOWS and (windows_path.drive or windows_path.root):
        return _validate_restricted_absolute_path(token, restricted_dir)
    if Path(token).is_absolute():
        return _validate_restricted_absolute_path(token, restricted_dir)
    return None


def _split_command(command: str) -> list[str]:
    return [
        _strip_shell_quotes(token)
        for token in shlex.split(command, posix=not _IS_WINDOWS)
    ]


def _command_name(token: str) -> str:
    name = PureWindowsPath(token).name if _IS_WINDOWS else Path(token).name
    return name.lower().removesuffix(".exe")


def _strip_shell_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}:
        return token[1:-1]
    return token


def _validate_restricted_absolute_path(token: str, restricted_dir: Path) -> str | None:
    if _IS_WINDOWS and os.name != "nt":
        return f"受限 shell 禁止访问任务目录外路径：{token}"
    path = Path(token)
    windows_path = PureWindowsPath(token)
    if (
        _IS_WINDOWS
        and (windows_path.drive or windows_path.root)
        and not path.is_absolute()
    ):
        return f"受限 shell 禁止访问任务目录外路径：{token}"
    if path.is_absolute():
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        try:
            restricted_resolved = restricted_dir.resolve()
        except OSError:
            restricted_resolved = restricted_dir
        if (
            resolved != restricted_resolved
            and restricted_resolved not in resolved.parents
        ):
            return f"受限 shell 禁止访问任务目录外路径：{token}"
    return None


def _looks_like_path(token: str) -> bool:
    if token in {".", ".."}:
        return True
    if _IS_WINDOWS:
        windows_path = PureWindowsPath(token)
        return (
            "\\" in token
            or "/" in token
            or bool(windows_path.drive)
            or token.startswith((".", "~"))
        )
    return "/" in token or token.startswith((".", "~"))
