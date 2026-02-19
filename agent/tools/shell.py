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
import shlex
import ipaddress
from urllib.parse import urlparse
import time
from typing import Any

from agent.tools.base import Tool

_DEFAULT_TIMEOUT = 60   # 秒（OpenCode 默认 1 分钟）
_MAX_TIMEOUT = 600      # 秒（OpenCode 最大 10 分钟）
_MAX_OUTPUT = 30_000    # 字符（与 OpenCode MaxOutputLength 一致）

# 禁止命令（对应 OpenCode bannedCommands）
_BANNED = frozenset({
    "curlie", "axel", "aria2c",
    "nc", "telnet", "lynx", "w3m", "links",
    "http-prompt", "chrome", "firefox", "safari",
})

# 对网络命令启用额外安全限制
_NETWORK_CMDS = frozenset({"curl", "wget", "http", "httpie", "xh"})
_NET_WRITE_FLAGS = frozenset({
    # curl
    "-o", "--output", "-O", "--remote-name", "-T", "--upload-file",
    "-F", "--form", "--form-string",
    # wget
    "-O", "--output-document", "--post-file",
    # httpie/xh
    "--download", "--output", "--offline", "@",
})


class ShellTool(Tool):
    """在 bash 中执行命令，返回结构化结果"""

    name = "shell"
    description = (
        "在 bash 中执行命令并返回输出。\n"
        "注意：\n"
        "- 使用绝对路径，避免依赖 cd 切换目录\n"
        "- 多条命令用 ; 或 && 连接，不要用换行分隔\n"
        "- 网络命令（curl/wget/httpie/xh）仅允许访问公网 HTTP(S)，且禁止上传/写文件\n"
        "- 以下命令被禁止：nc、telnet、浏览器等高风险工具\n"
        "- 输出超过 30000 字符时自动截断\n"
        "- 超时默认 60 秒，最大 600 秒"
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令",
            },
            "timeout": {
                "type": "integer",
                "description": f"超时秒数，默认 {_DEFAULT_TIMEOUT}，最大 {_MAX_TIMEOUT}",
                "minimum": 1,
                "maximum": _MAX_TIMEOUT,
            },
        },
        "required": ["command"],
    }

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs.get("command", "").strip()
        timeout: int = min(int(kwargs.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)

        if not command:
            return _err("命令不能为空")

        # 禁止命令检查（对应 OpenCode bannedCommands 逻辑）
        base_cmd = command.split()[0].lower()
        if base_cmd in _BANNED:
            return _err(f"命令 '{base_cmd}' 不被允许（安全限制）")
        net_err = _validate_network_command(command)
        if net_err:
            return _err(net_err)

        start_ms = int(time.monotonic() * 1000)
        stdout, stderr, exit_code, interrupted = await _run(command, timeout)
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


async def _run(command: str, timeout: int) -> tuple[str, str, int, bool]:
    """执行命令，并发读取 stdout/stderr，返回 (stdout, stderr, exit_code, interrupted)"""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
            proc.returncode or 0,
            False,
        )
    except asyncio.TimeoutError:
        proc.kill()
        stdout_b, stderr_b = await proc.communicate()
        return (
            stdout_b.decode(errors="replace"),
            stderr_b.decode(errors="replace"),
            -1,
            True,
        )


def _truncate(content: str) -> str:
    """超过阈值时首尾各取一半，中间注明省略行数（对应 OpenCode truncateOutput）"""
    if len(content) <= _MAX_OUTPUT:
        return content
    half = _MAX_OUTPUT // 2
    middle = content[half: len(content) - half]
    skipped_lines = middle.count("\n") + 1
    return (
        f"{content[:half]}\n\n"
        f"... [{skipped_lines} 行已省略] ...\n\n"
        f"{content[len(content) - half:]}"
    )


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
    if host == "localhost":
        return "禁止访问 localhost"

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
