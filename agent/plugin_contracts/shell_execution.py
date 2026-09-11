"""shell 执行的公开结构合同。

`plugins/standard_tools/shell` 与 `shell_backend` 需要声明 shell 执行的结果类型、
预算常量与进程管理器接口。这些是纯值/Protocol，因此由合同层拥有；具体实现
（`agent/tools/unified_exec.py` 的 `ShellProcessManager`、`agent/host_bridge/` 的
host 后端）留在原处并按结构满足 Protocol。
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# shell 执行的预算常量：调用方与实现必须使用同一组默认值。
# 下面四个是 clamp_* 函数的上/下界，随函数一起移动，否则合同层无法自洽。
MIN_YIELD_TIME_MS = 250
MIN_EMPTY_YIELD_TIME_MS = 5_000
MAX_YIELD_TIME_MS = 30_000
MAX_WRITE_STDIN_YIELD_TIME_MS = 300_000
DEFAULT_INITIAL_YIELD_TIME_MS = 10_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
DEFAULT_HARD_TIMEOUT_S = 4 * 3600
MAX_HARD_TIMEOUT_S = 4 * 3600


@dataclass
class ExecutionResult:
    output: bytes
    wall_time_ms: int
    original_token_count: int
    output_omitted_bytes: int
    execution_id: int | None
    exit_code: int | None
    output_path: str | None
    finish_reason: str


@dataclass(frozen=True)
class ExecutionCleanupFailure:
    execution_id: int
    error_type: str
    message: str


@dataclass(frozen=True)
class ExecutionCleanupReport:
    attempted_execution_ids: tuple[int, ...]
    cleaned_execution_ids: tuple[int, ...]
    failures: tuple[ExecutionCleanupFailure, ...]

    @property
    def failed_execution_ids(self) -> tuple[int, ...]:
        return tuple(failure.execution_id for failure in self.failures)


class ShellProcessManagerProtocol(Protocol):
    async def exec_command(
        self,
        *,
        command: str,
        argv: list[str],
        cwd: Path | None,
        env: dict[str, str],
        tty: bool,
        yield_time_ms: int,
        max_output_tokens: int,
        hard_timeout_s: int,
        owner_session_key: str,
    ) -> ExecutionResult: ...
    async def write_stdin(
        self,
        *,
        execution_id: int,
        chars: str,
        yield_time_ms: int,
        max_output_tokens: int,
        owner_session_key: str,
    ) -> ExecutionResult: ...
    async def terminate_execution(
        self, execution_id: int, *, owner_session_key: str
    ) -> bool: ...
    async def terminate_owner(
        self, owner_session_key: str
    ) -> ExecutionCleanupReport: ...
    async def shutdown(self) -> ExecutionCleanupReport: ...
    async def active_execution_ids(self) -> list[int]: ...


def format_execution_result(
    result: ExecutionResult,
    *,
    command: str | None = None,
) -> str:
    """把内部结果转换成稳定的工具 JSON。"""

    payload: dict[str, Any] = {
        "chunk_id": f"{random.randrange(16 ** 6):06x}",
        "wall_time_ms": result.wall_time_ms,
        "output": result.output.decode(errors="replace"),
        "original_token_count": result.original_token_count,
        "process_status": _process_status(result),
        "exit_code": result.exit_code,
    }
    if command is not None:
        payload["command"] = command
    if result.execution_id is not None:
        payload["execution_id"] = result.execution_id
    if result.output_path is not None:
        payload["output_path"] = result.output_path
    if result.output_omitted_bytes:
        payload["output_omitted_bytes"] = result.output_omitted_bytes
    if result.finish_reason != "natural":
        payload["finish_reason"] = result.finish_reason
    return json.dumps(payload, ensure_ascii=False)


def _process_status(result: ExecutionResult) -> str:
    if result.execution_id is not None:
        return "running"
    if result.finish_reason == "timeout":
        return "timed_out"
    if result.exit_code == 0:
        return "succeeded"
    return "failed"



class UnknownExecutionError(RuntimeError):
    pass


def clamp_initial_yield_time(yield_time_ms: int) -> int:
    return min(max(yield_time_ms, MIN_YIELD_TIME_MS), MAX_YIELD_TIME_MS)


def clamp_write_stdin_yield_time(
    yield_time_ms: int,
    *,
    has_input: bool,
    max_empty_ms: int = MAX_WRITE_STDIN_YIELD_TIME_MS,
) -> int:
    value = max(yield_time_ms, MIN_YIELD_TIME_MS)
    if has_input:
        return min(value, MAX_YIELD_TIME_MS)
    return min(max(value, MIN_EMPTY_YIELD_TIME_MS), max_empty_ms)

