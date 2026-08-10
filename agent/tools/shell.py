from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from agent.control.context import mint_plugin_child_capability, running_turn_id
from agent.host_bridge.factory import ShellProcessManagerProtocol
from agent.tools.base import Tool
from agent.tools.shell_security import validate_command
from agent.tools.shell_security import validate_network_command
from agent.tools.shell_command import resolve_shell
from agent.tools.unified_exec import DEFAULT_HARD_TIMEOUT_S
from agent.tools.unified_exec import DEFAULT_INITIAL_YIELD_TIME_MS
from agent.tools.unified_exec import DEFAULT_MAX_OUTPUT_TOKENS
from agent.tools.unified_exec import ExecutionCleanupReport
from agent.tools.unified_exec import MAX_HARD_TIMEOUT_S
from agent.tools.unified_exec import ShellProcessManager
from agent.tools.unified_exec import format_execution_result
from core.common.diagnostic_log import diagnostic_line
from core.error_context import current_session_key

logger = logging.getLogger(__name__)

_MAX_OUTPUT = 30_000
_LOCAL_OWNER_PREFIX = "local-shell"
_PLUGIN_ROLLOUT_OWNER_TURN_ENV = "AKASHIC_PLUGIN_ROLLOUT_OWNER_TURN"
_PLUGIN_ROLLOUT_CAPABILITY_ENV = "AKASHIC_PLUGIN_ROLLOUT_CAPABILITY"
_REMOVED_SHELL_ARGUMENTS = frozenset({"run_in_background", "auto_promote"})
_UNIFIED_EXEC_ENV = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "LANG": "C.UTF-8",
    "LC_CTYPE": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "COLORTERM": "",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GH_PAGER": "cat",
}


def _cleanup_diagnostic(
    action: str,
    owner_session_key: str,
    report: ExecutionCleanupReport,
) -> str:
    failures = ";".join(
        f"{failure.execution_id}:{failure.error_type}:{failure.message}"
        for failure in report.failures
    )
    return diagnostic_line(
        "ShellCleanup",
        event="cleanup_degraded",
        flow="runtime",
        phase="cleanup",
        session=owner_session_key,
        action=action,
        reason="execution_cleanup_unconfirmed",
        counts=(
            f"attempted:{len(report.attempted_execution_ids)},"
            f"failed:{len(report.failures)}"
        ),
        note=failures,
    )


class ShellTool(Tool):
    """启动命令，并返回终态或可续接的 execution_id。"""

    name = "shell"

    def __init__(
        self,
        manager: ShellProcessManagerProtocol | None = None,
        *,
        allow_network: bool = True,
        working_dir: Path | None = None,
        restricted_dir: Path | None = None,
        spawn_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.manager = manager or ShellProcessManager()
        self._allow_network = allow_network
        self._working_dir = working_dir
        self._restricted_dir = restricted_dir.resolve() if restricted_dir else None
        self._spawn_hook = spawn_hook

    @property
    def description(self) -> str:
        return (
            "在 shell 中执行命令。命令在短等待窗口内结束时直接返回 exit_code；"
            "仍在运行时返回 execution_id，之后用 write_stdin 等待增量输出或向 PTY 输入。\n"
            "注意：\n"
            "- execution_id 只标识这次命令执行，不是 OS PID 或 Akashic 对话 session\n"
            "- shell 默认使用当前用户的默认 shell；login 默认 true，可显式关闭\n"
            "- write_stdin 每次只返回上次读取后的新增输出，空 chars 可等待最长 300 秒\n"
            "- 需要交互输入时设置 tty=true；非 PTY 只允许用 Ctrl-C 中断\n"
            "- 放弃仍在运行的命令前调用 task_stop\n"
            "- pipeline 需要上游失败生效时，由命令显式启用 pipefail\n"
            "- 使用绝对路径，避免依赖 cd 切换目录\n"
            "- 网络命令仅允许公网 HTTP(S)，且禁止上传或写文件\n"
            "- 禁止 nc、telnet、浏览器等高风险命令\n"
            "禁止用 shell 替代 read_file、web_fetch 或 list_dir 等专用工具。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "要执行的 shell 命令；长任务需要观察进度时，"
                        "不要以 tail -n 等等待 EOF 的过滤器收尾。"
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "用 5-10 字描述命令作用，供用户审查和日志追踪。",
                },
                "cwd": {
                    "type": "string",
                    "description": "可选工作目录；相对路径按当前进程工作目录解析。",
                },
                "shell": {
                    "type": "string",
                    "description": "要启动的 shell binary；默认使用当前用户的默认 shell。",
                },
                "login": {
                    "type": "boolean",
                    "description": "true 使用 login shell 语义，false 关闭；默认 true。",
                },
                "tty": {
                    "type": "boolean",
                    "description": (
                        "命令可能在运行中请求输入（如密码或确认）时设为 true，"
                        "并在启动前确认输入可获得；默认 false。"
                    ),
                },
                "yield_time_ms": {
                    "type": "integer",
                    "description": (
                        "首次等待毫秒数，默认 10000；实际钳制在 250 到 30000。"
                    ),
                },
                "max_output_tokens": {
                    "type": "integer",
                    "minimum": 0,
                    "description": (
                        "本次工具结果的近似输出 token 预算，默认 10000；"
                        "不影响 provider 生成上限或完整诊断日志。"
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_HARD_TIMEOUT_S,
                    "description": (
                        f"执行进程组硬超时秒数，默认 {DEFAULT_HARD_TIMEOUT_S}，"
                        f"最大 {MAX_HARD_TIMEOUT_S}。"
                    ),
                },
            },
            "required": ["command", "description"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> str:
        """校验边界参数，注册执行，并返回统一结果。"""

        # 1. 解析工具边界参数。
        removed = sorted(_REMOVED_SHELL_ARGUMENTS.intersection(kwargs))
        if removed:
            return _error(f"shell 已移除参数: {', '.join(removed)}")
        command = str(kwargs.get("command", "")).strip()
        description = str(kwargs.get("description", ""))
        if not command:
            return _error("命令不能为空")
        yield_time_ms = int(kwargs.get("yield_time_ms", DEFAULT_INITIAL_YIELD_TIME_MS))
        max_output_tokens = int(
            kwargs.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        )
        hard_timeout_s = int(kwargs.get("timeout", DEFAULT_HARD_TIMEOUT_S))
        if max_output_tokens < 0:
            return _error("max_output_tokens 不能为负数")
        if hard_timeout_s < 1 or hard_timeout_s > MAX_HARD_TIMEOUT_S:
            return _error(
                f"timeout 必须在 1 到 {MAX_HARD_TIMEOUT_S} 秒之间，"
                f"收到 {hard_timeout_s}"
            )
        requested_shell = kwargs.get("shell")
        try:
            selected_shell = resolve_shell(
                None if requested_shell is None else str(requested_shell)
            )
        except ValueError as exc:
            return _error(str(exc))

        # 2. 应用当前工具实例的 cwd、环境和 spawn hook。
        cwd = self._working_dir
        cwd_arg = kwargs.get("cwd")
        if cwd_arg not in (None, ""):
            cwd = Path(str(cwd_arg)).expanduser()
        env = _shell_env()
        if self._spawn_hook is not None:
            hooked = self._spawn_hook(
                {
                    "command": command,
                    "cwd": str(cwd) if cwd is not None else None,
                    "env": env,
                }
            )
            command = str(hooked.get("command", command)).strip()
            hooked_cwd = hooked.get("cwd")
            cwd = None if hooked_cwd in (None, "") else Path(str(hooked_cwd))
            hooked_env = hooked.get("env")
            if isinstance(hooked_env, dict):
                env = {str(key): str(value) for key, value in hooked_env.items()}
        if self._restricted_dir is not None and cwd is None:
            cwd = self._restricted_dir

        # 3. 在唯一安全边界校验最终将执行的命令。
        validation_error = validate_command(
            command,
            allow_network=self._allow_network,
            restricted_dir=self._restricted_dir,
            cwd=cwd,
        )
        if validation_error:
            return _error(validation_error)
        logger.info("shell [%s]: %s", description, command[:120])
        argv = selected_shell.derive_argv(
            command,
            login=bool(kwargs.get("login", True)),
        )

        # 4. manager 在等待前注册进程；取消这里只取消等待。
        result = await self.manager.exec_command(
            command=command,
            argv=argv,
            cwd=cwd,
            env=env,
            tty=bool(kwargs.get("tty", False)),
            yield_time_ms=yield_time_ms,
            max_output_tokens=max_output_tokens,
            hard_timeout_s=hard_timeout_s,
            owner_session_key=_owner_session_key(self.manager),
        )
        return format_execution_result(result, command=command)

    async def shutdown(self) -> ExecutionCleanupReport:
        report = await self.manager.shutdown()
        if report.failures:
            logger.error(_cleanup_diagnostic("shutdown", "-", report))
        return report

    async def terminate_owner(
        self,
        owner_session_key: str,
    ) -> ExecutionCleanupReport:
        return await self.manager.terminate_owner(owner_session_key)


class ShellWriteStdinTool(Tool):
    """续接一次 shell execution 并消费新增输出。"""

    name = "write_stdin"
    description = (
        "等待 shell execution 的新增输出，或向 tty=true 的 execution 写入字符。"
        "空 chars 默认等待 5 秒、最多可等 300 秒；带输入最多等待 30 秒。"
        "结果只包含上次读取后的新增输出；running 且 output 为空只表示本次没有新增输出。"
        "已知长任务应按预期耗时使用较长等待，不要反复短轮询。一次足够长的等待仍无输出时，"
        "不要原样重复调用；先根据命令语义检查进程、日志或产物，再决定继续等待或 task_stop。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "execution_id": {
                "type": "integer",
                "description": "shell 返回的 execution_id",
            },
            "chars": {
                "type": "string",
                "description": "写入 PTY 的字符；留空表示只等待和读取。",
            },
            "yield_time_ms": {
                "type": "integer",
                "description": (
                    "本次等待毫秒数。空输入默认 5000、范围 5000 到 300000；"
                    "带输入默认 250、范围 250 到 30000。"
                ),
            },
            "max_output_tokens": {
                "type": "integer",
                "minimum": 0,
                "description": "本次工具结果的近似输出 token 预算，默认 10000。",
            },
        },
        "required": ["execution_id"],
        "additionalProperties": False,
    }

    def __init__(self, manager: ShellProcessManagerProtocol) -> None:
        self.manager = manager

    async def execute(self, **kwargs: Any) -> str:
        chars = str(kwargs.get("chars", ""))
        default_yield = 5_000 if not chars else 250
        max_output_tokens = int(
            kwargs.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        )
        if max_output_tokens < 0:
            return _error("max_output_tokens 不能为负数")
        result = await self.manager.write_stdin(
            execution_id=int(kwargs.get("execution_id", 0)),
            chars=chars,
            yield_time_ms=int(kwargs.get("yield_time_ms", default_yield)),
            max_output_tokens=max_output_tokens,
            owner_session_key=_owner_session_key(self.manager),
        )
        return format_execution_result(result)


class ShellTaskStopTool(Tool):
    """确认终止一次 shell execution 的进程组。"""

    name = "task_stop"
    description = "确认终止 shell execution 的进程组，并释放 execution_id。"
    parameters = {
        "type": "object",
        "properties": {
            "execution_id": {
                "type": "integer",
                "description": "shell 返回的 execution_id",
            },
        },
        "required": ["execution_id"],
        "additionalProperties": False,
    }

    def __init__(self, manager: ShellProcessManagerProtocol) -> None:
        self.manager = manager

    async def execute(self, **kwargs: Any) -> str:
        if "task_id" in kwargs:
            return _error("task_stop 已移除 task_id；请使用 execution_id")
        execution_id = int(kwargs.get("execution_id", 0))
        stopped = await self.manager.terminate_execution(
            execution_id,
            owner_session_key=_owner_session_key(self.manager),
        )
        return json.dumps(
            {
                "execution_id": execution_id,
                "process_status": "stopped" if stopped else "unknown",
                "status": "stopped" if stopped else "not_found",
            },
            ensure_ascii=False,
        )


def _owner_session_key(manager: ShellProcessManagerProtocol) -> str:
    return current_session_key.get() or f"{_LOCAL_OWNER_PREFIX}:{id(manager)}"


def _shell_env() -> dict[str, str]:
    env = os.environ.copy()
    turn_id = running_turn_id.get()
    if turn_id:
        env[_PLUGIN_ROLLOUT_OWNER_TURN_ENV] = turn_id
        capability = mint_plugin_child_capability(turn_id)
        if capability:
            env[_PLUGIN_ROLLOUT_CAPABILITY_ENV] = capability
        else:
            env.pop(_PLUGIN_ROLLOUT_CAPABILITY_ENV, None)
    else:
        env.pop(_PLUGIN_ROLLOUT_OWNER_TURN_ENV, None)
        env.pop(_PLUGIN_ROLLOUT_CAPABILITY_ENV, None)
    _prepend_existing_path_entries(env, _discover_user_path_entries(env))
    env.update(_UNIFIED_EXEC_ENV)
    return env


def _discover_user_path_entries(env: dict[str, str]) -> list[Path]:
    home_text = env.get("HOME")
    if not home_text:
        return []
    home = Path(home_text).expanduser()
    nvm_dir = Path(env.get("NVM_DIR") or home / ".nvm").expanduser()
    entries = [home / ".local" / "bin"]
    nvm_bin = env.get("NVM_BIN")
    if nvm_bin:
        entries.append(Path(nvm_bin).expanduser())
    entries.extend(_discover_nvm_node_bins(nvm_dir))
    return entries


def _discover_nvm_node_bins(nvm_dir: Path) -> list[Path]:
    node_root = nvm_dir / "versions" / "node"
    try:
        version_dirs = [path for path in node_root.iterdir() if path.is_dir()]
    except OSError:
        return []
    return [
        version_dir / "bin"
        for version_dir in sorted(
            version_dirs,
            key=lambda path: _node_version_key(path.name),
            reverse=True,
        )
        if (version_dir / "bin").is_dir()
    ]


def _node_version_key(version: str) -> tuple[int, int, int]:
    parts = version.removeprefix("v").split(".")
    numbers = [int(part) if part.isdigit() else 0 for part in parts[:3]]
    numbers.extend([0] * (3 - len(numbers)))
    return (numbers[0], numbers[1], numbers[2])


def _prepend_existing_path_entries(env: dict[str, str], entries: list[Path]) -> None:
    current = [path for path in env.get("PATH", "").split(os.pathsep) if path]
    seen = set(current)
    prepend: list[str] = []
    for entry in entries:
        text = str(entry)
        if text in seen or not entry.is_dir():
            continue
        prepend.append(text)
        seen.add(text)
    env["PATH"] = os.pathsep.join([*prepend, *current])


def _truncate(content: str) -> dict[str, Any]:
    """保留旧的独立文本截断 helper，供非 unified-exec 消费者使用。"""

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
    text = prefix + (content[-tail_budget:] if tail_budget else "")
    return {
        "text": text,
        "truncated": True,
        "strategy": "tail",
        "full_length": len(content),
        "returned_length": len(text),
        "omitted_lines": omitted_lines,
    }


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


_validate_command = validate_command
_validate_network_command = validate_network_command
