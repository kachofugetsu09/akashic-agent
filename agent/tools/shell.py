"""
Shell 工具（Bash 命令执行）
设计参考 OpenCode internal/llm/tools/bash.go：
- 禁止高风险命令黑名单（nc、telnet、浏览器等）
- 超时：默认 60s，最大 600s（10 分钟）
- 输出截断：超过 30000 字符时首尾各取一半，中间注明省略行数
- 记录执行时长
- 结构化 JSON 输出（command / exit_code / duration_ms / output）

后台任务（run_in_background=True）：
- 立即返回 background_task_id，不阻塞前台
- 输出持续写入临时日志文件
- 配合 ShellTaskOutputTool / ShellTaskStopTool 管理
"""

import asyncio
import json
import logging
import os
import signal
import shlex
import ipaddress
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4
import time
from typing import Any, Callable

from agent.tools.base import Tool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60  # 秒（OpenCode 默认 1 分钟）
_MAX_TIMEOUT = 600  # 秒（OpenCode 最大 10 分钟）
_MAX_OUTPUT = 30_000  # 字符（与 OpenCode MaxOutputLength 一致）
_STREAM_CHUNK_SIZE = 4096
_STREAM_DRAIN_GRACE_S = 0.2
_BG_TTL_S = 4 * 3600  # 后台任务最长存活时间：4 小时
_BG_EVICT_DELAY_S = 300  # 任务完成后延迟 5 分钟清理注册表和日志

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

# ── 后台任务注册表 ────────────────────────────────────────────────────

@dataclass
class _BackgroundTask:
    proc: Any  # asyncio.subprocess.Process
    log_path: str
    pump_task: asyncio.Task
    started_at: float


# 模块级单例：跨 ShellTool 实例共享
_BG_REGISTRY: dict[str, _BackgroundTask] = {}


async def _bg_pump(proc: Any, log_path: str) -> None:
    """持续从 stdout/stderr 读取并写入日志文件，直到进程退出（+ 短暂排水）。

    顺序：先等主进程退出，再尝试排水 grace 秒；超时则强制取消 drain task。
    这样即使子孙进程继承了 pipe fd，pump_task 也不会永久阻塞。
    """
    with open(log_path, "wb") as f:
        async def _drain_stream(stream) -> None:
            if stream is None:
                return
            while True:
                chunk = await stream.read(_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                f.flush()

        stdout_task = asyncio.create_task(_drain_stream(proc.stdout))
        stderr_task = asyncio.create_task(_drain_stream(proc.stderr))

        # 等主进程本体退出（不等子孙进程关 fd）
        await proc.wait()

        # 短暂排水：捕获最后几帧输出；超时后强制取消
        try:
            await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task),
                timeout=_STREAM_DRAIN_GRACE_S,
            )
        except asyncio.TimeoutError:
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)


def _schedule_eviction(task_id: str, log_path: str) -> None:
    """在当前事件循环上注册延迟清理（由 pump_task done callback 调用）。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    def _evict() -> None:
        _BG_REGISTRY.pop(task_id, None)
        try:
            os.unlink(log_path)
        except OSError:
            pass

    loop.call_later(_BG_EVICT_DELAY_S, _evict)


def _bg_kill(task_id: str) -> None:
    """杀掉后台任务、从注册表移除并立即删除日志文件。"""
    task = _BG_REGISTRY.pop(task_id, None)
    if task is None:
        return
    try:
        os.killpg(task.proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    task.pump_task.cancel()
    try:
        os.unlink(task.log_path)
    except OSError:
        pass


# ── ShellTool ────────────────────────────────────────────────────────


class ShellTool(Tool):
    """在 bash 中执行命令，返回结构化结果"""

    name = "shell"

    def __init__(
        self,
        *,
        allow_network: bool = True,
        working_dir: Path | None = None,
        restricted_dir: Path | None = None,
        spawn_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._allow_network = allow_network
        self._working_dir = working_dir
        self._restricted_dir = restricted_dir.resolve() if restricted_dir else None
        self._spawn_hook = spawn_hook

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
            "- 若命令是服务进程（如 python server.py、uvicorn、node app.js 等），"
            "请使用 run_in_background=true 后台启动，再用 task_output 获取日志\n"
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
                "run_in_background": {
                    "type": "boolean",
                    "description": (
                        "是否后台运行。设为 true 时立即返回 background_task_id，"
                        "输出写入日志文件，通过 task_output 获取、task_stop 停止。"
                        "适用于服务进程、长时间编译等不需要等待结果的场景。"
                    ),
                },
            },
            "required": ["command", "description"],
        }

    async def execute(self, **kwargs: Any) -> str:
        command: str = kwargs.get("command", "").strip()
        description: str = kwargs.get("description", "")
        timeout: int = min(int(kwargs.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)
        run_in_background: bool = bool(kwargs.get("run_in_background", False))
        on_data = kwargs.get("_on_data")

        if not command:
            return _err("命令不能为空")

        cwd = self._working_dir
        env = os.environ.copy()
        if self._spawn_hook is not None:
            hooked = self._spawn_hook(
                {
                    "command": command,
                    "cwd": str(cwd) if cwd is not None else None,
                    "env": env,
                }
            )
            command = str(hooked.get("command", command)).strip()
            cwd_val = hooked.get("cwd")
            cwd = None if cwd_val in (None, "") else Path(str(cwd_val))
            env_val = hooked.get("env")
            if isinstance(env_val, dict):
                env = {str(k): str(v) for k, v in env_val.items()}

        if self._restricted_dir is not None and cwd is None:
            cwd = self._restricted_dir

        logger.info("shell [%s]: %s", description, command[:120])

        base_cmd = command.split()[0].lower()
        if base_cmd in _BANNED:
            return _err(f"命令 '{base_cmd}' 不被允许（安全限制）")
        cmd_err = _validate_command(
            command,
            allow_network=self._allow_network,
            restricted_dir=self._restricted_dir,
            cwd=cwd,
        )
        if cmd_err:
            return _err(cmd_err)

        if run_in_background:
            return await self._execute_background(command, cwd, env)

        # ── 前台执行路径（原有逻辑）────────────────────────────────────
        start_ms = int(time.monotonic() * 1000)
        stdout, stderr, exit_code, interrupted = await _run(
            command,
            timeout,
            cwd=cwd,
            env=env,
            on_data=on_data if callable(on_data) else None,
        )
        duration_ms = int(time.monotonic() * 1000) - start_ms

        full_parts = []
        if stdout:
            full_parts.append(stdout)
        if stderr:
            if stdout:
                full_parts.append("")
            full_parts.append(stderr)
        if interrupted:
            full_parts.append("命令在完成前被中止")
        elif exit_code != 0:
            full_parts.append(f"Exit code {exit_code}")

        full_output = "\n".join(full_parts) if full_parts else "（无输出）"
        output_meta = _truncate(full_output)
        full_output_path = (
            _write_full_output(full_output) if output_meta["truncated"] else None
        )
        truncation = None
        if output_meta["truncated"]:
            truncation = {
                "strategy": output_meta["strategy"],
                "full_length": output_meta["full_length"],
                "returned_length": output_meta["returned_length"],
                "omitted_lines": output_meta["omitted_lines"],
            }

        return json.dumps(
            {
                "command": command,
                "exit_code": exit_code,
                "interrupted": interrupted,
                "duration_ms": duration_ms,
                "output": output_meta["text"],
                "truncation": truncation,
                "full_output_path": full_output_path,
            },
            ensure_ascii=False,
        )

    async def _execute_background(
        self,
        command: str,
        cwd: Path | None,
        env: dict[str, str],
    ) -> str:
        task_id = f"shell_{uuid4().hex[:12]}"
        log_fd, log_path = tempfile.mkstemp(
            prefix=f"akasic-bg-{task_id}-", suffix=".log"
        )
        os.close(log_fd)

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        pump = asyncio.create_task(_bg_pump(proc, log_path))
        pump.add_done_callback(lambda _: _schedule_eviction(task_id, log_path))
        wall_start_ms = int(time.time() * 1000)
        _BG_REGISTRY[task_id] = _BackgroundTask(
            proc=proc,
            log_path=log_path,
            pump_task=pump,
            started_at=time.monotonic(),
        )
        logger.info("shell bg started [%s] pid=%s log=%s", task_id, proc.pid, log_path)

        return json.dumps(
            {
                "command": command,
                "background_task_id": task_id,
                "status": "running",
                "output_path": log_path,
                "started_at_ms": wall_start_ms,
                "exit_code": None,
                "interrupted": False,
            },
            ensure_ascii=False,
        )


# ── ShellTaskOutputTool ──────────────────────────────────────────────


class ShellTaskOutputTool(Tool):
    """读取后台 shell 任务的当前输出，可选择阻塞等待完成。"""

    name = "task_output"

    @property
    def description(self) -> str:
        return (
            "读取后台 shell 任务的输出。\n"
            "- block=false（默认）：立即返回当前已写入的输出，不等待任务完成\n"
            "- block=true：等待任务完成或超时后再返回完整输出\n"
            "输出超过 30000 字符时自动截断，完整内容见 output_path。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "shell 工具返回的 background_task_id",
                },
                "block": {
                    "type": "boolean",
                    "description": "是否等待任务完成后再返回，默认 false",
                },
                "timeout_ms": {
                    "type": "integer",
                    "description": "block=true 时的最长等待时间（毫秒），默认 30000",
                    "minimum": 0,
                },
            },
            "required": ["task_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        task_id: str = kwargs.get("task_id", "")
        block: bool = bool(kwargs.get("block", False))
        timeout_ms: int = int(kwargs.get("timeout_ms", 30000))

        task = _BG_REGISTRY.get(task_id)
        if task is None:
            return _err(f"任务 {task_id!r} 不存在或已清理")

        if time.monotonic() - task.started_at > _BG_TTL_S:
            _bg_kill(task_id)
            return _err(f"任务 {task_id!r} 已超出 TTL（{_BG_TTL_S}s），已自动终止")

        if block and not task.pump_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(task.pump_task), timeout=timeout_ms / 1000
                )
            except asyncio.TimeoutError:
                pass

        done = task.pump_task.done()
        exit_code = task.proc.returncode if done else None
        status = "done" if done else "running"

        try:
            content = Path(task.log_path).read_bytes().decode(errors="replace")
        except OSError:
            content = ""

        output_meta = _truncate(content)
        truncation = None
        if output_meta["truncated"]:
            truncation = {
                "strategy": output_meta["strategy"],
                "full_length": output_meta["full_length"],
                "returned_length": output_meta["returned_length"],
                "omitted_lines": output_meta["omitted_lines"],
            }

        return json.dumps(
            {
                "task_id": task_id,
                "status": status,
                "exit_code": exit_code,
                "output": output_meta["text"],
                "truncation": truncation,
                "output_path": task.log_path,
            },
            ensure_ascii=False,
        )


# ── ShellTaskStopTool ────────────────────────────────────────────────


class ShellTaskStopTool(Tool):
    """停止并清理一个后台 shell 任务。"""

    name = "task_stop"

    @property
    def description(self) -> str:
        return "停止后台 shell 任务（SIGKILL 整棵进程树）并从注册表移除。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "要停止的后台任务 ID（background_task_id）",
                },
            },
            "required": ["task_id"],
        }

    async def execute(self, **kwargs: Any) -> str:
        task_id: str = kwargs.get("task_id", "")
        if task_id not in _BG_REGISTRY:
            return json.dumps(
                {"task_id": task_id, "status": "not_found"}, ensure_ascii=False
            )
        _bg_kill(task_id)
        return json.dumps({"task_id": task_id, "status": "stopped"}, ensure_ascii=False)


# ── 模块级工具函数 ────────────────────────────────────────────────


def _err(msg: str) -> str:
    return json.dumps({"error": msg}, ensure_ascii=False)


async def _run(
    command: str,
    timeout: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_data: Callable[[str], None] | None = None,
) -> tuple[str, str, int, bool]:
    """执行命令，并发读取 stdout/stderr，返回 (stdout, stderr, exit_code, interrupted)"""
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
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

    async def _pump(stream, chunks: list[str]) -> None:
        if stream is None:
            return
        while True:
            data = await stream.read(_STREAM_CHUNK_SIZE)
            if not data:
                break
            text = data.decode(errors="replace")
            chunks.append(text)
            if on_data is not None:
                on_data(text)

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_task = asyncio.create_task(_pump(proc.stdout, stdout_chunks))
    stderr_task = asyncio.create_task(_pump(proc.stderr, stderr_chunks))

    async def _finish_pumps() -> None:
        try:
            await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task),
                timeout=_STREAM_DRAIN_GRACE_S,
            )
        except asyncio.TimeoutError:
            stdout_task.cancel()
            stderr_task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    async def _wait_proc() -> int:
        if hasattr(proc, "wait"):
            return await proc.wait()
        await proc.communicate()
        return proc.returncode or 0

    try:
        await asyncio.wait_for(_wait_proc(), timeout=timeout)
        await _finish_pumps()
        return (
            "".join(stdout_chunks),
            "".join(stderr_chunks),
            proc.returncode or 0,
            False,
        )
    except asyncio.TimeoutError:
        _kill_tree()
        await _finish_pumps()
        return (
            "".join(stdout_chunks),
            "".join(stderr_chunks),
            -1,
            True,
        )
    except asyncio.CancelledError:
        _kill_tree()
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        raise


def _truncate(content: str) -> dict[str, Any]:
    """超过阈值时优先保留尾部，便于看到命令结果与错误摘要。"""
    if len(content) <= _MAX_OUTPUT:
        return {
            "text": content,
            "truncated": False,
            "strategy": "tail",
            "full_length": len(content),
            "returned_length": len(content),
            "omitted_lines": 0,
        }

    omitted = content[: len(content) - _MAX_OUTPUT]
    omitted_lines = omitted.count("\n")
    prefix = f"... [{omitted_lines} 行已省略] ...\n\n"
    tail_budget = max(0, _MAX_OUTPUT - len(prefix))
    tail = content[-tail_budget:] if tail_budget > 0 else ""
    text = prefix + tail
    return {
        "text": text,
        "truncated": True,
        "strategy": "tail",
        "full_length": len(content),
        "returned_length": len(text),
        "omitted_lines": omitted_lines,
    }


def _write_full_output(content: str) -> str:
    fd, path = tempfile.mkstemp(prefix="akasic-shell-", suffix=".log")
    os.close(fd)
    Path(path).write_text(content, encoding="utf-8")
    return path


def _validate_command(
    command: str,
    *,
    allow_network: bool,
    restricted_dir: Path | None,
    cwd: Path | None = None,
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
        cwd_err = _validate_restricted_cwd(cwd, restricted_dir)
        if cwd_err:
            return cwd_err
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
