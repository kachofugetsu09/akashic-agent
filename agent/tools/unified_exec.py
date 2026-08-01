from __future__ import annotations

import asyncio
import errno
import json
import os
import random
import signal
import subprocess
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from utils.process_group import process_group_exists

MIN_YIELD_TIME_MS = 250
MIN_EMPTY_YIELD_TIME_MS = 5_000
MAX_YIELD_TIME_MS = 30_000
MAX_WRITE_STDIN_YIELD_TIME_MS = 300_000
DEFAULT_INITIAL_YIELD_TIME_MS = 10_000
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
OUTPUT_MAX_BYTES = 1024 * 1024
MAX_EXECUTIONS = 64
PROTECTED_RECENT_EXECUTIONS = 8
DEFAULT_HARD_TIMEOUT_S = 4 * 3600
MAX_HARD_TIMEOUT_S = 4 * 3600
POST_EXIT_DRAIN_GRACE_S = 0.2
TERMINATION_CONFIRM_TIMEOUT_S = 5.0
INTERRUPT = "\x03"

_IS_WINDOWS = os.name == "nt"
_ID_MIN = 1_000
_ID_MAX = 99_999


class UnknownExecutionError(RuntimeError):
    pass


class HeadTailBuffer:
    """保留稳定首尾，并明确记录被丢弃的中间字节。"""

    def __init__(self, max_bytes: int = OUTPUT_MAX_BYTES) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes 不能为负数")
        self.max_bytes = max_bytes
        self.head_budget = max_bytes // 2
        self.tail_budget = max_bytes - self.head_budget
        self.head = bytearray()
        self.tail: deque[int] = deque()
        self.omitted_bytes = 0

    @property
    def retained_bytes(self) -> int:
        return len(self.head) + len(self.tail)

    @property
    def total_bytes(self) -> int:
        return self.retained_bytes + self.omitted_bytes

    def push_chunk(self, chunk: bytes | bytearray) -> None:
        if not chunk:
            return
        data = bytes(chunk)
        if self.max_bytes == 0:
            self.omitted_bytes += len(data)
            return

        # 1. 首段一旦填满就保持稳定。
        remaining_head = max(self.head_budget - len(self.head), 0)
        head_len = min(remaining_head, len(data))
        if head_len:
            self.head.extend(data[:head_len])

        # 2. 剩余内容只保留最新尾段。
        self._push_to_tail(data[head_len:])

    def push_buffer(self, buffer: HeadTailBuffer) -> None:
        self.push_chunk(bytes(buffer.head))
        self.push_chunk(bytes(buffer.tail))
        self.omitted_bytes += buffer.omitted_bytes

    def to_bytes(self) -> bytes:
        return bytes(self.head) + bytes(self.tail)

    def to_bytes_with_omission_marker(self) -> bytes:
        if self.omitted_bytes == 0:
            return self.to_bytes()
        marker = f"... {self.omitted_bytes} bytes omitted ...".encode()
        return bytes(self.head) + b"\n" + marker + b"\n" + bytes(self.tail)

    def drain(self) -> HeadTailBuffer:
        drained = HeadTailBuffer(self.max_bytes)
        drained.head = self.head
        drained.tail = self.tail
        drained.omitted_bytes = self.omitted_bytes
        self.head = bytearray()
        self.tail = deque()
        self.omitted_bytes = 0
        return drained

    def _push_to_tail(self, chunk: bytes) -> None:
        if not chunk:
            return
        if self.tail_budget == 0:
            self.omitted_bytes += len(chunk)
            return
        if len(chunk) >= self.tail_budget:
            kept = chunk[-self.tail_budget :]
            self.omitted_bytes += len(self.tail) + len(chunk) - len(kept)
            self.tail.clear()
            self.tail.extend(kept)
            return
        self.tail.extend(chunk)
        excess = max(len(self.tail) - self.tail_budget, 0)
        for _ in range(excess):
            self.tail.popleft()
        self.omitted_bytes += excess


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


@dataclass
class _Execution:
    execution_id: int
    owner_session_key: str
    command: str
    process: asyncio.subprocess.Process
    tty: bool
    output_path: str
    log_file: BinaryIO
    started_at: float
    last_used: float
    master_fd: int | None = None
    output_buffer: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    output_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    output_event: asyncio.Event = field(default_factory=asyncio.Event)
    exit_event: asyncio.Event = field(default_factory=asyncio.Event)
    output_closed: asyncio.Event = field(default_factory=asyncio.Event)
    pump_task: asyncio.Task[None] | None = None
    hard_timeout_task: asyncio.Task[None] | None = None
    finish_reason: str = "natural"
    failure_message: str | None = None


class ShellProcessManager:
    """统一创建、续接和回收 shell execution。"""

    def __init__(
        self,
        *,
        max_executions: int = MAX_EXECUTIONS,
        max_write_stdin_yield_time_ms: int = MAX_WRITE_STDIN_YIELD_TIME_MS,
    ) -> None:
        if max_executions < 1:
            raise ValueError("max_executions 必须大于零")
        self._max_executions = max_executions
        self._max_write_stdin_yield_time_ms = max(
            max_write_stdin_yield_time_ms,
            MIN_EMPTY_YIELD_TIME_MS,
        )
        self._executions: dict[int, _Execution] = {}
        self._quarantined_owners: dict[str, ExecutionCleanupReport] = {}
        self._lock = asyncio.Lock()
        self._spawn_lock = asyncio.Lock()
        self._rng = random.SystemRandom()

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
    ) -> ExecutionResult:
        """注册一次执行，等待首个窗口，并返回完成态或续接句柄。"""

        # 1. 串行容量回收和 spawn，保证新进程在开始等待前已注册。
        async with self._spawn_lock:
            await self._ensure_owner_admitted(owner_session_key)
            await self._prune_if_needed()
            execution_id = await self._allocate_execution_id()
            execution = await self._spawn(
                execution_id=execution_id,
                owner_session_key=owner_session_key,
                command=command,
                argv=argv,
                cwd=cwd,
                env=env,
                tty=tty,
            )
            async with self._lock:
                self._executions[execution_id] = execution
            execution.pump_task = asyncio.create_task(
                self._pump_execution(execution),
                name=f"shell-pump:{execution_id}",
            )
            execution.hard_timeout_task = asyncio.create_task(
                self._enforce_hard_timeout(execution, hard_timeout_s),
                name=f"shell-timeout:{execution_id}",
            )

        # 2. 调用方取消只中止这次等待；manager 继续持有 execution。
        started = time.monotonic()
        collected = await self._collect_until_deadline(
            execution,
            started + clamp_initial_yield_time(yield_time_ms) / 1000,
        )
        return await self._build_result(
            execution,
            collected,
            started,
            max_output_tokens,
        )

    async def write_stdin(
        self,
        *,
        execution_id: int,
        chars: str,
        yield_time_ms: int,
        max_output_tokens: int,
        owner_session_key: str,
    ) -> ExecutionResult:
        """写入 PTY 或等待执行，并只消费本次新增输出。"""

        # 1. 在 owner 边界取得当前执行。
        execution = await self._get_owned_execution(
            execution_id,
            owner_session_key,
        )

        # 2. 非 PTY 仅允许把 Ctrl-C 转换为进程组中断。
        if chars:
            if execution.tty:
                try:
                    if chars == INTERRUPT:
                        self._interrupt_process_group(execution.process)
                    else:
                        await self._write_pty(execution, chars.encode())
                        await asyncio.sleep(0.1)
                except (BrokenPipeError, OSError):
                    if (
                        execution.process.returncode is None
                        and not execution.exit_event.is_set()
                    ):
                        raise
            elif chars == INTERRUPT:
                self._interrupt_process_group(execution.process)
            else:
                raise RuntimeError(
                    f"execution_id={execution_id} 未启用 tty，stdin 已关闭"
                )

        # 3. 等到输出、退出或本次截止，再返回增量结果。
        wait_ms = clamp_write_stdin_yield_time(
            yield_time_ms,
            has_input=bool(chars),
            max_empty_ms=self._max_write_stdin_yield_time_ms,
        )
        started = time.monotonic()
        collected = await self._collect_until_deadline(
            execution,
            started + wait_ms / 1000,
        )
        return await self._build_result(
            execution,
            collected,
            started,
            max_output_tokens,
        )

    async def terminate_execution(
        self,
        execution_id: int,
        *,
        owner_session_key: str,
    ) -> bool:
        """确认终止执行进程组，成功后移除执行。"""

        try:
            execution = await self._get_owned_execution(
                execution_id,
                owner_session_key,
            )
        except UnknownExecutionError:
            return False
        execution.finish_reason = "stopped"
        await self._terminate_confirmed(execution)
        await self._remove_execution(execution, delete_log=True)
        return True

    async def terminate_owner(
        self,
        owner_session_key: str,
    ) -> ExecutionCleanupReport:
        """回收 owner 执行，并隔离 cleanup 未确认的 owner。"""

        # 1. 与 spawn 串行，避免 cleanup 与同 owner 新进程交错。
        async with self._spawn_lock:
            async with self._lock:
                executions = [
                    execution
                    for execution in self._executions.values()
                    if execution.owner_session_key == owner_session_key
                ]
            report = await self._terminate_many(executions)

            # 2. cleanup 成功解除隔离；失败保留 execution 与失败证据。
            async with self._lock:
                if report.failures:
                    self._quarantined_owners[owner_session_key] = report
                else:
                    _ = self._quarantined_owners.pop(owner_session_key, None)
            return report

    async def shutdown(self) -> ExecutionCleanupReport:
        """尽力终止全部执行，并返回未清理明细。"""

        async with self._spawn_lock:
            async with self._lock:
                executions = list(self._executions.values())
            return await self._terminate_many(executions)

    async def active_execution_ids(self) -> list[int]:
        async with self._lock:
            return sorted(self._executions)

    async def _spawn(
        self,
        *,
        execution_id: int,
        owner_session_key: str,
        command: str,
        argv: list[str],
        cwd: Path | None,
        env: dict[str, str],
        tty: bool,
    ) -> _Execution:
        """创建子进程和诊断日志，但不开始等待。"""

        log_fd, output_path = tempfile.mkstemp(
            prefix=f"akashic-exec-{execution_id}-",
            suffix=".log",
        )
        log_file = os.fdopen(log_fd, "wb", buffering=0)
        master_fd: int | None = None
        slave_fd: int | None = None
        try:
            options: dict[str, Any] = {
                "cwd": str(cwd) if cwd is not None else None,
                "env": env,
            }
            if tty:
                if _IS_WINDOWS:
                    raise RuntimeError("当前平台不支持 tty=true；需要 ConPTY 实现")
                import pty

                master_fd, slave_fd = pty.openpty()
                os.set_blocking(master_fd, False)
                options.update(
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                )
            else:
                options.update(
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                if _IS_WINDOWS:
                    options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    options["start_new_session"] = True
            process = await asyncio.create_subprocess_exec(*argv, **options)
        except BaseException:
            log_file.close()
            Path(output_path).unlink(missing_ok=True)
            if master_fd is not None:
                os.close(master_fd)
            raise
        finally:
            if slave_fd is not None:
                os.close(slave_fd)

        now = time.monotonic()
        return _Execution(
            execution_id=execution_id,
            owner_session_key=owner_session_key,
            command=command,
            process=process,
            tty=tty,
            output_path=output_path,
            log_file=log_file,
            master_fd=master_fd,
            started_at=now,
            last_used=now,
        )

    async def _pump_execution(self, execution: _Execution) -> None:
        """持续写入内存增量缓冲和完整诊断日志，退出后有界排水。"""

        async def _read_output() -> None:
            if execution.tty:
                assert execution.master_fd is not None
                while True:
                    chunk = await _read_fd(execution.master_fd, 4096)
                    if not chunk:
                        return
                    await self._append_output(execution, chunk)
            else:
                assert execution.process.stdout is not None
                while True:
                    chunk = await execution.process.stdout.read(4096)
                    if not chunk:
                        return
                    await self._append_output(execution, chunk)

        reader = asyncio.create_task(
            _read_output(),
            name=f"shell-reader:{execution.execution_id}",
        )
        try:
            await execution.process.wait()
            execution.exit_event.set()
            execution.output_event.set()
            try:
                await asyncio.wait_for(
                    asyncio.shield(reader),
                    timeout=POST_EXIT_DRAIN_GRACE_S,
                )
            except asyncio.TimeoutError:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
        except asyncio.CancelledError:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            raise
        except Exception as exc:
            execution.failure_message = f"shell 输出读取失败: {exc}"
            try:
                await self._terminate_confirmed(execution)
            except Exception as terminate_exc:
                execution.failure_message += f"；进程终止失败: {terminate_exc}"
        finally:
            execution.exit_event.set()
            execution.output_closed.set()
            execution.output_event.set()

    async def _append_output(self, execution: _Execution, chunk: bytes) -> None:
        # 1. 完整日志写失败属于不可恢复的观察链路错误。
        execution.log_file.write(chunk)

        # 2. 单次消费缓冲与日志使用相同原始字节。
        async with execution.output_lock:
            execution.output_buffer.push_chunk(chunk)
            execution.output_event.set()

    async def _collect_until_deadline(
        self,
        execution: _Execution,
        deadline: float,
    ) -> HeadTailBuffer:
        """按 Codex 的 drain 语义收集新增输出直到截止或关闭。"""

        collected = HeadTailBuffer()
        post_exit_deadline: float | None = None
        while True:
            async with execution.output_lock:
                drained = execution.output_buffer.drain()
                execution.output_event.clear()
            if drained.total_bytes:
                collected.push_buffer(drained)

            if execution.exit_event.is_set():
                if execution.output_closed.is_set():
                    break
                now = time.monotonic()
                if post_exit_deadline is None:
                    post_exit_deadline = min(
                        deadline,
                        now + POST_EXIT_DRAIN_GRACE_S,
                    )
                remaining = post_exit_deadline - now
            else:
                remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            try:
                await asyncio.wait_for(execution.output_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                break

        # 最后再 drain 一次，覆盖截止点与 append 同时发生的竞争。
        async with execution.output_lock:
            final = execution.output_buffer.drain()
            execution.output_event.clear()
        if final.total_bytes:
            collected.push_buffer(final)
        return collected

    async def _build_result(
        self,
        execution: _Execution,
        collected: HeadTailBuffer,
        call_started_at: float,
        max_output_tokens: int,
    ) -> ExecutionResult:
        """刷新执行状态，并在终态移除 manager 条目。"""

        if execution.failure_message is not None:
            await self._remove_execution(execution, delete_log=False)
            raise RuntimeError(execution.failure_message)

        alive = (
            execution.process.returncode is None and not execution.exit_event.is_set()
        )
        output = _limit_output(collected, max_output_tokens)
        result = ExecutionResult(
            output=output.to_bytes_with_omission_marker(),
            wall_time_ms=int((time.monotonic() - call_started_at) * 1000),
            original_token_count=(collected.total_bytes + 3) // 4,
            output_omitted_bytes=output.omitted_bytes,
            execution_id=execution.execution_id if alive else None,
            exit_code=None if alive else execution.process.returncode,
            output_path=(
                execution.output_path
                if alive or output.omitted_bytes > 0 or collected.omitted_bytes > 0
                else None
            ),
            finish_reason=execution.finish_reason,
        )
        if alive:
            execution.last_used = time.monotonic()
            return result

        keep_log = result.output_path is not None
        await self._remove_execution(execution, delete_log=not keep_log)
        return result

    async def _get_owned_execution(
        self,
        execution_id: int,
        owner_session_key: str,
    ) -> _Execution:
        async with self._lock:
            execution = self._executions.get(execution_id)
            if execution is None or execution.owner_session_key != owner_session_key:
                raise UnknownExecutionError(f"未知 execution_id: {execution_id}")
            execution.last_used = time.monotonic()
            return execution

    async def _remove_execution(
        self,
        execution: _Execution,
        *,
        delete_log: bool,
    ) -> None:
        async with self._lock:
            current = self._executions.get(execution.execution_id)
            if current is not execution:
                return
        current_task = asyncio.current_task()
        tasks_to_join: list[asyncio.Task[None]] = []
        for task in (execution.hard_timeout_task, execution.pump_task):
            if task is not None and task is not current_task and not task.done():
                task.cancel()
                tasks_to_join.append(task)
        if tasks_to_join:
            await asyncio.gather(*tasks_to_join, return_exceptions=True)
        _kill_remaining_process_group(execution.process)
        async with self._lock:
            current = self._executions.get(execution.execution_id)
            if current is not execution:
                return
            del self._executions[execution.execution_id]
        execution.log_file.close()
        if execution.master_fd is not None:
            try:
                os.close(execution.master_fd)
            except OSError:
                pass
            execution.master_fd = None
        if delete_log:
            Path(execution.output_path).unlink(missing_ok=True)

    async def _enforce_hard_timeout(
        self,
        execution: _Execution,
        hard_timeout_s: int,
    ) -> None:
        try:
            await asyncio.sleep(hard_timeout_s)
            if execution.process.returncode is not None:
                return
            execution.finish_reason = "timeout"
            await self._terminate_confirmed(execution)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            execution.failure_message = f"shell 硬超时终止失败: {exc}"
            execution.output_event.set()

    async def _terminate_many(
        self,
        executions: list[_Execution],
    ) -> ExecutionCleanupReport:
        """逐个清理 execution，并保留每个失败的 ownership。"""

        cleaned: list[int] = []
        failures: list[ExecutionCleanupFailure] = []
        for execution in executions:
            try:
                if execution.process.returncode is None:
                    execution.finish_reason = "shutdown"
                    await self._terminate_confirmed(execution)
                await self._remove_execution(execution, delete_log=True)
                cleaned.append(execution.execution_id)
            except Exception as exc:
                failures.append(
                    ExecutionCleanupFailure(
                        execution_id=execution.execution_id,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        return ExecutionCleanupReport(
            attempted_execution_ids=tuple(
                execution.execution_id for execution in executions
            ),
            cleaned_execution_ids=tuple(cleaned),
            failures=tuple(failures),
        )

    async def _ensure_owner_admitted(self, owner_session_key: str) -> None:
        async with self._lock:
            report = self._quarantined_owners.get(owner_session_key)
        if report is None:
            return
        failed = ",".join(str(value) for value in report.failed_execution_ids)
        raise RuntimeError(
            f"owner={owner_session_key} 的 shell cleanup 未确认；"
            f"failed_execution_ids={failed}，拒绝创建新 execution"
        )

    async def _terminate_confirmed(self, execution: _Execution) -> None:
        if execution.process.returncode is None:
            try:
                _kill_process_group(execution.process)
            except ProcessLookupError:
                # 进程组可能刚刚自然退出；下面的 wait 仍必须确认终态。
                pass
        try:
            await asyncio.wait_for(
                execution.process.wait(),
                timeout=TERMINATION_CONFIRM_TIMEOUT_S,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"execution_id={execution.execution_id} 进程组终止未确认"
            ) from exc
        execution.exit_event.set()
        execution.output_event.set()

    async def _prune_if_needed(self) -> None:
        async with self._lock:
            if len(self._executions) < self._max_executions:
                return
            execution = self._select_prune_candidate(list(self._executions.values()))
        if execution is None:
            raise RuntimeError("shell execution 容量已满且没有可回收项")
        if execution.process.returncode is None:
            execution.finish_reason = "capacity_pruned"
            await self._terminate_confirmed(execution)
        await self._remove_execution(execution, delete_log=True)

    @staticmethod
    def _select_prune_candidate(
        executions: list[_Execution],
    ) -> _Execution | None:
        if not executions:
            return None
        by_recency = sorted(executions, key=lambda item: item.last_used, reverse=True)
        protected_count = min(
            PROTECTED_RECENT_EXECUTIONS,
            max(len(by_recency) - 1, 0),
        )
        protected = {item.execution_id for item in by_recency[:protected_count]}
        lru = sorted(executions, key=lambda item: item.last_used)
        for execution in lru:
            if (
                execution.execution_id not in protected
                and execution.process.returncode is not None
            ):
                return execution
        return next(
            (execution for execution in lru if execution.execution_id not in protected),
            None,
        )

    async def _allocate_execution_id(self) -> int:
        while True:
            execution_id = self._rng.randrange(_ID_MIN, _ID_MAX + 1)
            async with self._lock:
                if execution_id not in self._executions:
                    return execution_id

    async def _write_pty(self, execution: _Execution, data: bytes) -> None:
        if execution.master_fd is None:
            raise RuntimeError(f"execution_id={execution.execution_id} 的 PTY 已关闭")
        await _write_fd(execution.master_fd, data)

    @staticmethod
    def _interrupt_process_group(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if _IS_WINDOWS:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)


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


def _limit_output(buffer: HeadTailBuffer, max_output_tokens: int) -> HeadTailBuffer:
    max_bytes = max(max_output_tokens, 0) * 4
    limited = HeadTailBuffer(max_bytes)
    limited.push_buffer(buffer)
    return limited


def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    if _IS_WINDOWS:
        completed = subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0 and process.returncode is None:
            process.kill()
        return
    os.killpg(process.pid, signal.SIGKILL)


def _kill_remaining_process_group(process: asyncio.subprocess.Process) -> None:
    """与 Codex ProcessHandle.drop 一致，清理 leader 退出后残留的子孙。"""

    if _IS_WINDOWS:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Linux killpg 会因组内只剩其他 UID 的 zombie 返回 EPERM；zombie 已不再执行，
        # 由 subreaper wait 回收，不能把已完成命令误判为仍有活进程。
        if not process_group_exists(process.pid):
            return
        raise


async def _read_fd(fd: int, size: int) -> bytes:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[bytes] = loop.create_future()

    def _on_readable() -> None:
        try:
            data = os.read(fd, size)
        except BlockingIOError:
            return
        except OSError as exc:
            if exc.errno == errno.EIO:
                data = b""
            else:
                future.set_exception(exc)
                loop.remove_reader(fd)
                return
        future.set_result(data)
        loop.remove_reader(fd)

    loop.add_reader(fd, _on_readable)
    try:
        return await future
    finally:
        loop.remove_reader(fd)


async def _write_fd(fd: int, data: bytes) -> None:
    loop = asyncio.get_running_loop()
    offset = 0
    while offset < len(data):
        try:
            written = os.write(fd, data[offset:])
            offset += written
            continue
        except BlockingIOError:
            pass
        future: asyncio.Future[None] = loop.create_future()

        def _on_writable() -> None:
            if not future.done():
                future.set_result(None)
            loop.remove_writer(fd)

        loop.add_writer(fd, _on_writable)
        try:
            await future
        finally:
            loop.remove_writer(fd)
