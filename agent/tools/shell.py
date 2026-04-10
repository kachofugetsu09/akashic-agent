"""
Shell 工具（Bash 命令执行）
设计参考 OpenCode internal/llm/tools/bash.go：
- 禁止高风险命令黑名单（nc、telnet、浏览器等）
- 超时：默认 60s，最大 600s（10 分钟）
- 输出截断：超过 30000 字符时首尾各取一半，中间注明省略行数
- 记录执行时长
- 结构化 JSON 输出（command / exit_code / duration_ms / output）
"""

import asyncio
import json
import logging
import os
import signal
import shlex
import ipaddress
from pathlib import Path
from urllib.parse import urlparse
import time
from typing import Any

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60  # 秒（OpenCode 默认 1 分钟）
_MAX_TIMEOUT = 600  # 秒（OpenCode 最大 10 分钟）
_MAX_OUTPUT = 30_000  # 字符（与 OpenCode MaxOutputLength 一致）

# 禁止命令（对应 OpenCode bannedCommands）
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

# 对网络命令启用额外安全限制
_NETWORK_CMDS = frozenset({"curl", "wget", "http", "httpie", "xh"})
_NET_WRITE_FLAGS = frozenset(
    {
        # curl
        "-o",
        "--output",
        "-O",
        "--remote-name",
        "-T",
        "--upload-file",
        "-F",
        "--form",
        "--form-string",
        # wget
        "-O",
        "--output-document",
        "--post-file",
        # httpie/xh
        "--download",
        "--output",
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


class ShellTool(Tool):
    """在 bash 中执行命令，返回结构化结果"""

    name = "shell"

    def __init__(
        self,
        *,
        allow_network: bool = True,
        working_dir: Path | None = None,
        restricted_dir: Path | None = None,
    ) -> None:
        self._allow_network = allow_network
        self._working_dir = working_dir
        self._restricted_dir = restricted_dir.resolve() if restricted_dir else None

    @property
    def description(self) -> str:
        return (
            "在 bash 中执行命令并返回输出。\n"
            "注意：\n"
            "- 使用绝对路径，避免依赖 cd 切换目录\n"
            "- 多条命令用 ; 或 && 连接，不要用换行分隔\n"
            "- 网络命令（curl/wget/httpie/xh）仅允许访问公网 HTTP(S)，且禁止上传/写文件\n"
            "- 以下命令被禁止：nc、telnet、浏览器等高风险工具\n"
            "- 输出超过 30000 字符时自动截断\n"
            "- 超时默认 60 秒，最大 600 秒\n"
            "- 若命令是服务进程（如 python server.py、uvicorn、node app.js 等），必须用 `timeout 5 <命令> 2>&1` 包裹以快速获取启动日志，禁止直接运行导致阻塞\n"
            "禁止用途：不得用 shell 替代专用工具（read_file 读文件、web_fetch 抓网页、list_dir 列目录）。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 bash 命令",
                },
                "description": {
                    "type": "string",
                    "description": (
                        "用 5-10 字描述这条命令的作用，便于用户审查和日志追踪。"
                        "示例：'列出当前目录文件' / '安装 Python 依赖' / '查看进程状态'"
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": f"超时秒数，默认 {_DEFAULT_TIMEOUT}，最大 {_MAX_TIMEOUT}",
                    "minimum": 1,
                    "maximum": _MAX_TIMEOUT,
                },
            },
            "required": ["command", "description"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs.get("command", "").strip()
        description: str = kwargs.get("description", "")
        timeout: int = min(int(kwargs.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)

        if not command:
            return _err("命令不能为空")
        logger.info("shell [%s]: %s", description, command[:120])

        # 禁止命令检查（对应 OpenCode bannedCommands 逻辑）
        base_cmd = command.split()[0].lower()
        if base_cmd in _BANNED:
            return _err(f"命令 '{base_cmd}' 不被允许（安全限制）")
        cmd_err = _validate_command(
            command,
            allow_network=self._allow_network,
            restricted_dir=self._restricted_dir,
        )
        if cmd_err:
            return _err(cmd_err)

        start_ms = int(time.monotonic() * 1000)
        stdout, stderr, exit_code, interrupted = await _run(
            command,
            timeout,
            cwd=self._working_dir,
        )
        duration_ms = int(time.monotonic() * 1000) - start_ms

        stdout = _truncate(stdout)
        stderr = _truncate(stderr)

        # 合并输出（对应 OpenCode hasBothOutputs 逻辑）
        parts = []
        if stdout:
            parts.append(stdout)
        if stderr:
            if stdout:
                parts.append("")  # 两段之间空一行
            parts.append(stderr)
        if interrupted:
            parts.append("命令在完成前被中止")
        elif exit_code != 0:
            parts.append(f"Exit code {exit_code}")

        output = "\n".join(parts) if parts else "（无输出）"

        return json.dumps(
            {
                "command": command,
                "exit_code": exit_code,
                "interrupted": interrupted,
                "duration_ms": duration_ms,
                "output": output,
            },
            ensure_ascii=False,
        )


# ── 模块级工具函数 ────────────────────────────────────────────────


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


async def _run(
    command: str,
    timeout: int,
    cwd: Path | None = None,
) -> tuple[str, str, int, bool]:
    """执行命令，并发读取 stdout/stderr，返回 (stdout, stderr, exit_code, interrupted)"""
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd) if cwd is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # 独立 process group，便于 killpg 杀整棵进程树
    )

    def _kill_tree() -> None:
        """杀掉整棵进程树（按 pgid）。"""
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # 进程已退出或无权限

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
            proc.returncode or 0,
            False,
        )
    except asyncio.TimeoutError:
        _kill_tree()
        stdout_b, stderr_b = await proc.communicate()
        return (
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
            -1,
            True,
        )
    except asyncio.CancelledError:
        _kill_tree()
        await proc.communicate()
        raise


def _truncate(content: str) -> str:
    """超过阈值时首尾各取一半，中间注明省略行数（对应 OpenCode truncateOutput）"""
    if len(content) <= _MAX_OUTPUT:
        return content
    half = _MAX_OUTPUT // 2
    middle = content[half : len(content) - half]
    skipped_lines = middle.count("\n") + 1
    return (
        f"{content[:half]}\n\n"
        f"... [{skipped_lines} 行已省略] ...\n\n"
        f"{content[len(content) - half:]}"
    )


def _validate_command(
    command: str,
    *,
    allow_network: bool,
    restricted_dir: Path | None,
) -> str | None:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "命令解析失败，请检查引号是否匹配"
    if not tokens:
        return None

    cmd = tokens[0].lower()
    if not allow_network and cmd in _NETWORK_CMDS:
        return "当前 shell 配置禁止网络访问"

    if restricted_dir is not None:
        restricted_err = _validate_restricted_command(tokens, restricted_dir)
        if restricted_err:
            return restricted_err

    return _validate_network_command(command)


def _validate_network_command(command: str) -> str | None:
    """网络命令护栏：仅允许 HTTP(S) 且禁止内网目标与写入类参数。"""
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return "命令解析失败，请检查引号是否匹配"
    if not tokens:
        return None

    cmd = tokens[0].lower()
    if cmd not in _NETWORK_CMDS:
        return None

    # 阻止文件写入/上传相关参数
    for t in tokens[1:]:
        low = t.lower()
        if low in _NET_WRITE_FLAGS:
            return f"网络命令参数 '{t}' 不被允许（禁止上传/写文件）"
        if any(low.startswith(flag + "=") for flag in _NET_WRITE_FLAGS):
            return f"网络命令参数 '{t}' 不被允许（禁止上传/写文件）"
        # httpie/xh 支持 field=@file 语法上传文件
        if "=@" in t or t.startswith("@"):
            return f"网络命令参数 '{t}' 不被允许（禁止本地文件上传）"

    # 提取 URL 并校验
    urls = [t for t in tokens[1:] if t.startswith(("http://", "https://"))]
    if not urls:
        return "网络命令必须显式提供 http:// 或 https:// URL"

    for u in urls:
        err = _validate_url_target(u)
        if err:
            return err
    return None


def _validate_url_target(url: str) -> str | None:
    """校验 URL 目标是否为合法的公网地址。"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "仅允许 http:// 或 https:// URL"
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return "URL 缺少主机名"

    try:
        # IP 地址：禁止回环、私有、链路本地、保留地址
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return f"禁止访问内网/本地地址：{host}"
    except ValueError:
        # 域名：阻断常见本地域名后缀
        if host.endswith(".local") or host.endswith(".localhost"):
            return f"禁止访问本地域名：{host}"
    return None


def _validate_restricted_command(tokens: list[str], restricted_dir: Path) -> str | None:
    command = " ".join(tokens)
    if any(marker in command for marker in _RESTRICTED_META_CHARS):
        return "受限 shell 禁止管道、重定向或串联命令"

    base_cmd = tokens[0].lower()
    if base_cmd in _RESTRICTED_SHELL_RUNNERS:
        return f"受限 shell 禁止启动解释器或二级 shell：{base_cmd}"

    for token in tokens[1:]:
        if token.startswith("-") or token == "--":
            continue
        err = _validate_restricted_token(token, restricted_dir)
        if err:
            return err
    return None


def _validate_restricted_token(token: str, restricted_dir: Path) -> str | None:
    if token.startswith("~"):
        return f"受限 shell 禁止访问任务目录外路径：{token}"

    if not _looks_like_path(token):
        return None

    path = Path(token)
    if any(part == ".." for part in path.parts):
        return f"受限 shell 禁止访问父级路径：{token}"

    if path.is_absolute():
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved != restricted_dir and restricted_dir not in resolved.parents:
            return f"受限 shell 禁止访问任务目录外路径：{token}"
    return None


def _looks_like_path(token: str) -> bool:
    if token in {".", ".."}:
        return True
    return "/" in token or token.startswith((".", "~"))
