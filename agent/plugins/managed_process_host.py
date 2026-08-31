"""Generation-scoped host for Core-owned managed processes.

The declaration registry remains the plugin-facing boundary.  This module only
accepts already-normalized declarations and exposes endpoint, health and bounded
diagnostic views; it never exposes a subprocess handle to a plugin.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import socket
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from agent.plugin_composition.process_slots import ManagedProcessDefinition
from utils.process_group import (
    OwnedProcessGroup,
    owned_process_env,
    process_group_spawn_kwargs,
)

logger = logging.getLogger(__name__)

ProcessMode = Literal["candidate", "formal"]
GenerationState = Literal[
    "starting",
    "ready",
    "degraded",
    "stopping",
    "cleanup_failed",
]

HealthCallback = Callable[[str, str, bool, str], Awaitable[None] | None]
IncidentCallback = Callable[[str, str, str, str], Awaitable[None] | None]

_DEFAULT_STOP_TIMEOUT_SECONDS = 5.0
_DEFAULT_LOG_BYTES = 64 * 1024
_DEFAULT_LOG_LINES = 512
_DEFAULT_RECOVERY_BACKOFF_SECONDS = (0.25, 1.0, 3.0)
_DEFAULT_RECOVERY_STABLE_SECONDS = 60.0
_READ_CHUNK_BYTES = 4096


class HealthReporter(Protocol):
    def __call__(self, generation_id: str, process_name: str, healthy: bool, reason: str) -> Any: ...


class IncidentReporter(Protocol):
    def __call__(self, generation_id: str, process_name: str, kind: str, message: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class ManagedProcessEndpoint:
    """Core-generated loopback endpoint for one current process epoch."""

    generation_id: str
    process_name: str
    mode: ProcessMode
    epoch: int
    port: int
    readiness_url: str


@dataclass(frozen=True, slots=True)
class ManagedProcessLogView:
    """Bounded stdout/stderr snapshot for one process generation."""

    stdout: tuple[str, ...]
    stderr: tuple[str, ...]

    @property
    def lines(self) -> tuple[str, ...]:
        """Return stdout and stderr lines in stream-local order groups."""

        return self.stdout + self.stderr


@dataclass(frozen=True, slots=True)
class GenerationCleanupTombstone:
    """Retained ownership proof when a generation could not be fully cleaned."""

    generation_id: str
    state: Literal["cleanup_failed", "degraded"]
    action: Literal["retry_generation_cleanup", "retry_runtime_recovery"]
    resource_names: tuple[str, ...]
    error: str
    attempt_count: int


FailureCallback = Callable[[GenerationCleanupTombstone], None]


class _LogRing:
    """Keep a bounded line ring."""

    def __init__(self, *, max_bytes: int, max_lines: int) -> None:
        if max_bytes <= 0 or max_lines <= 0:
            raise ValueError("managed process log ring limits must be positive")
        self._max_bytes = max_bytes
        self._max_lines = max_lines
        self._lines: deque[str] = deque()
        self._bytes = 0

    def append(self, chunk: bytes) -> None:
        text = chunk.decode("utf-8", errors="replace")
        for line in text.splitlines(keepends=True):
            self._append_line(line)
        if text and not text.splitlines(keepends=True):
            self._append_line(text)

    def _append_line(self, line: str) -> None:
        encoded_size = len(line.encode("utf-8", errors="replace"))
        if encoded_size > self._max_bytes:
            line = _truncate_utf8_tail(line, self._max_bytes)
            encoded_size = len(line.encode("utf-8", errors="replace"))
        while self._lines and (
            self._bytes + encoded_size > self._max_bytes
            or len(self._lines) >= self._max_lines
        ):
            removed = self._lines.popleft()
            self._bytes -= len(removed.encode("utf-8", errors="replace"))
        self._lines.append(line)
        self._bytes += encoded_size

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self._lines)


@dataclass
class _ProcessEpoch:
    generation_id: str
    definition: ManagedProcessDefinition
    mode: ProcessMode
    artifact_root: Path | None
    epoch: int = 0
    process: asyncio.subprocess.Process | None = None
    process_group: OwnedProcessGroup | None = None
    endpoint: ManagedProcessEndpoint | None = None
    stdout_ring: _LogRing | None = None
    stderr_ring: _LogRing | None = None
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None
    watch_task: asyncio.Task[None] | None = None
    recovery_task: asyncio.Task[None] | None = None
    stopping: bool = False
    ready: bool = False
    recovery_attempts: int = 0
    ready_at: float | None = None


@dataclass
class _Generation:
    generation_id: str
    mode: ProcessMode
    artifact_root: Path | None
    entries: dict[str, _ProcessEpoch]
    state: GenerationState = "starting"
    cleanup_attempts: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ManagedProcessGeneration:
    """Read-only endpoint facade returned after a generation becomes ready."""

    def __init__(self, host: ManagedProcessGenerationHost, generation_id: str) -> None:
        self._host = host
        self.generation_id = generation_id

    @property
    def endpoints(self) -> Mapping[str, ManagedProcessEndpoint]:
        return MappingProxyType(self._host.endpoints(self.generation_id))

    def endpoint(self, process_name: str) -> ManagedProcessEndpoint:
        return self._host.endpoint(self.generation_id, process_name)

    def logs(self, process_name: str) -> ManagedProcessLogView:
        return self._host.logs(self.generation_id, process_name)


class ManagedProcessGenerationHost:
    """Own, validate, recover and clean managed processes by generation."""

    def __init__(
        self,
        *,
        on_health: HealthReporter | None = None,
        on_incident: IncidentReporter | None = None,
        on_failure: FailureCallback | None = None,
        log_max_bytes: int = _DEFAULT_LOG_BYTES,
        log_max_lines: int = _DEFAULT_LOG_LINES,
        stop_timeout_seconds: float = _DEFAULT_STOP_TIMEOUT_SECONDS,
        recovery_backoff_seconds: tuple[float, ...] = _DEFAULT_RECOVERY_BACKOFF_SECONDS,
        recovery_stable_seconds: float = _DEFAULT_RECOVERY_STABLE_SECONDS,
    ) -> None:
        if stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        if any(delay < 0 for delay in recovery_backoff_seconds):
            raise ValueError("recovery backoff values must be non-negative")
        if recovery_stable_seconds <= 0:
            raise ValueError("recovery_stable_seconds must be positive")
        self._on_health = on_health
        self._on_incident = on_incident
        self._on_failure = on_failure
        self._log_max_bytes = log_max_bytes
        self._log_max_lines = log_max_lines
        self._stop_timeout_seconds = stop_timeout_seconds
        self._recovery_backoff_seconds = recovery_backoff_seconds
        self._recovery_stable_seconds = recovery_stable_seconds
        self._generations: dict[str, _Generation] = {}
        self._tombstones: dict[str, GenerationCleanupTombstone] = {}
        self._next_epoch = 0

    async def start_generation(
        self,
        generation_id: str,
        definitions: Mapping[str, ManagedProcessDefinition],
        *,
        mode: ProcessMode = "candidate",
        artifact_root: Path | None = None,
    ) -> ManagedProcessGeneration:
        """Start and readiness-check one isolated candidate or formal generation."""

        if not isinstance(generation_id, str) or not generation_id.strip():
            raise ValueError("managed process generation_id must be non-empty")
        if mode not in {"candidate", "formal"}:
            raise ValueError(f"unknown managed process mode: {mode!r}")
        if generation_id in self._generations or generation_id in self._tombstones:
            raise RuntimeError(f"managed process generation already exists: {generation_id}")
        normalized = self._normalize_definitions(definitions)
        generation = _Generation(
            generation_id=generation_id,
            mode=mode,
            artifact_root=artifact_root.resolve() if artifact_root is not None else None,
            entries={
                name: _ProcessEpoch(
                    generation_id=generation_id,
                    definition=definition,
                    mode=mode,
                    artifact_root=artifact_root.resolve() if artifact_root is not None else None,
                )
                for name, definition in normalized.items()
            },
        )
        self._generations[generation_id] = generation
        try:
            for entry in generation.entries.values():
                await self._start_entry(generation, entry)
            generation.state = "ready"
            return ManagedProcessGeneration(self, generation_id)
        except BaseException as error:
            cleanup_task = asyncio.create_task(
                self._cleanup_generation(generation, cause=error),
                name=f"managed_process_cleanup:{generation_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError as cleanup_cancelled:
                if cleanup_task.done() and cleanup_task.exception() is None:
                    _ = self._generations.pop(generation_id, None)
                else:
                    self._retain_cleanup_tombstone(
                        generation,
                        _task_error(cleanup_task, error),
                    )
                if isinstance(error, asyncio.CancelledError):
                    raise error
                raise cleanup_cancelled from error
            except BaseException as cleanup_failure:
                self._retain_cleanup_tombstone(generation, cleanup_failure)
                if isinstance(error, asyncio.CancelledError):
                    raise error from cleanup_failure
            else:
                _ = self._generations.pop(generation_id, None)
            raise

    async def stop_generation(self, generation_id: str) -> None:
        """Stop a generation; retain a cleanup tombstone if any owner remains."""

        generation = self._generations.get(generation_id)
        if generation is None:
            tombstone = self._tombstones.get(generation_id)
            if tombstone is not None:
                raise RuntimeError(
                    f"managed process generation cleanup 未完成: {tombstone.error}"
                )
            return
        async with generation.lock:
            generation.state = "stopping"
            cleanup_task = asyncio.create_task(
                self._cleanup_generation(generation),
                name=f"managed_process_stop:{generation_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if cleanup_task.done() and cleanup_task.exception() is None:
                    _ = self._generations.pop(generation_id, None)
                    _ = self._tombstones.pop(generation_id, None)
                else:
                    self._retain_cleanup_tombstone(
                        generation,
                        _task_error(cleanup_task, asyncio.CancelledError()),
                    )
                raise
            except BaseException as error:
                self._retain_cleanup_tombstone(generation, error)
                raise
            _ = self._generations.pop(generation_id, None)
            _ = self._tombstones.pop(generation_id, None)

    async def retry_generation_cleanup(self, generation_id: str) -> None:
        """Retry retained process ownership and remove its tombstone only on success."""

        generation = self._generations.get(generation_id)
        if generation is None:
            if generation_id not in self._tombstones:
                return
            raise RuntimeError(f"managed process cleanup owner missing: {generation_id}")
        async with generation.lock:
            generation.state = "stopping"
            cleanup_task = asyncio.create_task(
                self._cleanup_generation(generation),
                name=f"managed_process_retry:{generation_id}",
            )
            try:
                await _await_task_after_cancellation(cleanup_task)
            except asyncio.CancelledError:
                if cleanup_task.done() and cleanup_task.exception() is None:
                    _ = self._generations.pop(generation_id, None)
                    _ = self._tombstones.pop(generation_id, None)
                else:
                    self._retain_cleanup_tombstone(
                        generation,
                        _task_error(cleanup_task, asyncio.CancelledError()),
                    )
                raise
            except BaseException as error:
                self._retain_cleanup_tombstone(generation, error)
                raise
            _ = self._generations.pop(generation_id, None)
            _ = self._tombstones.pop(generation_id, None)

    async def retry_runtime_recovery(self, generation_id: str) -> None:
        """Retry a degraded runtime owner, then finish its generation cleanup."""

        tombstone = self._tombstones.get(generation_id)
        if tombstone is None or tombstone.state != "degraded":
            raise RuntimeError(
                f"managed process generation 没有待恢复 degraded owner: {generation_id}"
            )
        generation = self._require_generation(generation_id)
        async with generation.lock:
            generation.state = "stopping"
            errors: list[BaseException] = []
            for entry in generation.entries.values():
                try:
                    await self._terminate_runtime(entry)
                except BaseException as error:
                    errors.append(error)
            if errors:
                first_entry = next(iter(generation.entries.values()))
                self._retain_runtime_tombstone(generation, first_entry, errors[0])
                raise RuntimeError(
                    "managed process runtime recovery failed: "
                    + "; ".join(_error_text(error) for error in errors)
                ) from errors[0]
            _ = self._tombstones.pop(generation_id, None)
        await self.retry_generation_cleanup(generation_id)

    def get(self, generation_id: str) -> ManagedProcessGeneration | None:
        """Return a read-only generation facade when the host still owns it."""

        if generation_id not in self._generations:
            return None
        return ManagedProcessGeneration(self, generation_id)

    def tombstone(self, generation_id: str) -> GenerationCleanupTombstone | None:
        return self._tombstones.get(generation_id)

    def generation_state(self, generation_id: str) -> GenerationState:
        generation = self._generations.get(generation_id)
        if generation is not None:
            return generation.state
        tombstone = self._tombstones.get(generation_id)
        if tombstone is not None:
            return cast(GenerationState, tombstone.state)
        raise KeyError(f"unknown managed process generation: {generation_id}")

    def endpoints(self, generation_id: str) -> dict[str, ManagedProcessEndpoint]:
        generation = self._require_generation(generation_id)
        return {
            name: entry.endpoint
            for name, entry in generation.entries.items()
            if entry.endpoint is not None
        }

    def endpoint(self, generation_id: str, process_name: str) -> ManagedProcessEndpoint:
        generation = self._require_generation(generation_id)
        entry = generation.entries.get(process_name)
        if entry is None or entry.endpoint is None or not entry.ready:
            raise RuntimeError(
                f"managed process endpoint 当前不可用: {generation_id}:{process_name}"
            )
        return entry.endpoint

    def logs(self, generation_id: str, process_name: str) -> ManagedProcessLogView:
        generation = self._require_generation(generation_id)
        entry = generation.entries.get(process_name)
        if entry is None:
            raise KeyError(f"unknown managed process: {generation_id}:{process_name}")
        stdout = (
            entry.stdout_ring.snapshot()
            if entry.stdout_ring is not None
            else ()
        )
        stderr = (
            entry.stderr_ring.snapshot()
            if entry.stderr_ring is not None
            else ()
        )
        return ManagedProcessLogView(stdout=stdout, stderr=stderr)

    def health(self, generation_id: str, process_name: str) -> bool:
        generation = self._require_generation(generation_id)
        entry = generation.entries.get(process_name)
        if entry is None:
            raise KeyError(f"unknown managed process: {generation_id}:{process_name}")
        return entry.ready and not entry.stopping and entry.process is not None and entry.process.returncode is None

    async def close(self) -> None:
        """Drain every owned generation and preserve the first cleanup failure."""

        errors: list[BaseException] = []
        cancelled = False
        for generation_id in tuple(self._generations):
            try:
                await self.stop_generation(generation_id)
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as error:
                errors.append(error)
        if cancelled:
            raise asyncio.CancelledError
        if errors:
            raise RuntimeError(
                "managed process host cleanup failed: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]

    async def _start_entry(self, generation: _Generation, entry: _ProcessEpoch) -> None:
        """Spawn one process epoch, wait for Core-owned readiness, then publish it."""

        definition = entry.definition
        port = self._allocate_port(definition.formal_port if generation.mode == "formal" else None)
        command = self._resolve_command(definition.command, entry.artifact_root)
        cwd = self._resolve_cwd(definition.cwd, entry.artifact_root)
        env = self._process_env(definition.env, definition.port_env, port)
        if entry.stdout_ring is None:
            entry.stdout_ring = _LogRing(
                max_bytes=self._log_max_bytes,
                max_lines=self._log_max_lines,
            )
        if entry.stderr_ring is None:
            entry.stderr_ring = _LogRing(
                max_bytes=self._log_max_bytes,
                max_lines=self._log_max_lines,
            )
        self._next_epoch += 1
        entry.epoch = self._next_epoch
        entry.stopping = False
        entry.ready = False
        await self._emit_health(
            generation.generation_id,
            definition.name,
            False,
            "starting",
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd),
                env=owned_process_env(env),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_group_spawn_kwargs(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._emit_incident(
                generation.generation_id,
                definition.name,
                "spawn_failed",
                _error_text(error),
            )
            raise
        entry.process = process
        entry.process_group = OwnedProcessGroup.from_process(process)
        entry.stdout_task = asyncio.create_task(
            self._drain_stream(process.stdout, entry.stdout_ring),
            name=f"managed_process_stdout:{generation.generation_id}:{definition.name}:{entry.epoch}",
        )
        entry.stderr_task = asyncio.create_task(
            self._drain_stream(process.stderr, entry.stderr_ring),
            name=f"managed_process_stderr:{generation.generation_id}:{definition.name}:{entry.epoch}",
        )
        try:
            await self._wait_ready(generation, entry, port)
        except BaseException as error:
            await self._emit_incident(
                generation.generation_id,
                definition.name,
                "readiness_failed",
                _error_text(error),
            )
            raise
        entry.endpoint = ManagedProcessEndpoint(
            generation_id=generation.generation_id,
            process_name=definition.name,
            mode=generation.mode,
            epoch=entry.epoch,
            port=port,
            readiness_url=f"http://127.0.0.1:{port}{definition.readiness_path}",
        )
        entry.ready = True
        entry.ready_at = asyncio.get_running_loop().time()
        entry.watch_task = asyncio.create_task(
            self._watch_process(generation, entry, entry.epoch),
            name=f"managed_process_watch:{generation.generation_id}:{definition.name}:{entry.epoch}",
        )
        await self._emit_health(
            generation.generation_id,
            definition.name,
            True,
            "ready",
        )

    async def _wait_ready(
        self,
        generation: _Generation,
        entry: _ProcessEpoch,
        port: int,
    ) -> None:
        """Poll only the Core-generated loopback readiness URL until ready or timeout."""

        process = entry.process
        if process is None:
            raise RuntimeError("managed process has no process after spawn")
        url = f"http://127.0.0.1:{port}{entry.definition.readiness_path}"
        deadline = asyncio.get_running_loop().time() + entry.definition.startup_timeout_seconds
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"managed process 启动失败: exit={process.returncode}")
            remaining = deadline - asyncio.get_running_loop().time()
            if await asyncio.to_thread(_url_ready, url, min(0.5, remaining)):
                await asyncio.sleep(0)
                if process.returncode is not None:
                    raise RuntimeError(f"managed process 启动失败: exit={process.returncode}")
                return
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(0.05, remaining))
        raise TimeoutError(f"managed process readiness timeout: {generation.generation_id}:{entry.definition.name}")

    async def _watch_process(
        self,
        generation: _Generation,
        entry: _ProcessEpoch,
        observed_epoch: int,
    ) -> None:
        process = entry.process
        if process is None:
            return
        exit_code = await process.wait()
        if entry.stopping or entry.epoch != observed_epoch or generation.state == "stopping":
            return
        try:
            await self._emit_health(
                generation.generation_id,
                entry.definition.name,
                False,
                f"process exited: {exit_code}",
            )
            await self._emit_incident(
                generation.generation_id,
                entry.definition.name,
                "process_exit",
                f"managed process exited unexpectedly: exit={exit_code}, epoch={observed_epoch}",
            )
        except asyncio.CancelledError as error:
            if entry.stopping or generation.state == "stopping":
                raise
            await self._retain_runtime_callback_failure(generation, entry, error)
            return
        except BaseException as error:
            await self._retain_runtime_callback_failure(generation, entry, error)
            return
        try:
            await self._terminate_runtime(entry)
        except BaseException as error:
            generation.state = "degraded"
            self._retain_runtime_tombstone(generation, entry, error)
            return
        await self._recover_entry(generation, entry, observed_epoch)

    async def _recover_entry(
        self,
        generation: _Generation,
        entry: _ProcessEpoch,
        observed_epoch: int,
    ) -> None:
        """Recover only a still-current epoch and never resurrect a stopping generation."""

        if entry.stopping or entry.epoch != observed_epoch or generation.state == "stopping":
            return
        if entry.recovery_task is not None and entry.recovery_task is not asyncio.current_task():
            return
        entry.recovery_task = asyncio.current_task()
        try:
            while not entry.stopping and generation.state != "stopping":
                now = asyncio.get_running_loop().time()
                if entry.ready_at is not None and now - entry.ready_at >= self._recovery_stable_seconds:
                    entry.recovery_attempts = 0
                if entry.recovery_attempts >= len(self._recovery_backoff_seconds):
                    exhaustion = RuntimeError(
                        "managed process recovery exhausted at "
                        f"epoch={observed_epoch}"
                    )
                    generation.state = "degraded"
                    self._retain_runtime_tombstone(generation, entry, exhaustion)
                    try:
                        await self._emit_incident(
                            generation.generation_id,
                            entry.definition.name,
                            "recovery_exhausted",
                            str(exhaustion),
                        )
                    except asyncio.CancelledError:
                        if entry.stopping or generation.state == "stopping":
                            raise
                    except BaseException as callback_error:
                        self._retain_runtime_tombstone(
                            generation,
                            entry,
                            callback_error,
                        )
                    return
                delay = self._recovery_backoff_seconds[entry.recovery_attempts]
                entry.recovery_attempts += 1
                await asyncio.sleep(delay)
                if entry.stopping or generation.state == "stopping":
                    return
                try:
                    await self._start_entry(generation, entry)
                except asyncio.CancelledError as error:
                    if entry.stopping or generation.state == "stopping":
                        raise
                    await self._retain_runtime_callback_failure(
                        generation,
                        entry,
                        error,
                    )
                    return
                except BaseException as error:
                    try:
                        await self._emit_incident(
                            generation.generation_id,
                            entry.definition.name,
                            "recovery_failed",
                            _error_text(error),
                        )
                    except asyncio.CancelledError as callback_error:
                        if entry.stopping or generation.state == "stopping":
                            raise
                        await self._retain_runtime_callback_failure(
                            generation,
                            entry,
                            callback_error,
                        )
                        return
                    except BaseException as callback_error:
                        await self._retain_runtime_callback_failure(
                            generation,
                            entry,
                            callback_error,
                        )
                        return
                    try:
                        await self._terminate_runtime(entry)
                    except BaseException as cleanup_error:
                        generation.state = "degraded"
                        self._retain_runtime_tombstone(
                            generation,
                            entry,
                            cleanup_error,
                        )
                        return
                    continue
                if generation.state == "degraded":
                    generation.state = "ready"
                entry.recovery_attempts = 0
                return
        finally:
            if entry.recovery_task is asyncio.current_task():
                entry.recovery_task = None

    async def _cleanup_generation(
        self,
        generation: _Generation,
        *,
        cause: BaseException | None = None,
    ) -> None:
        """Terminate every owned process and drain both output streams before release."""

        errors: list[BaseException] = []
        for entry in reversed(tuple(generation.entries.values())):
            try:
                await self._cleanup_entry(entry)
            except BaseException as error:
                errors.append(error)
        if errors:
            detail = "; ".join(_error_text(error) for error in errors)
            raise RuntimeError(
                f"managed process generation cleanup failed: {generation.generation_id}: {detail}"
            ) from (cause or errors[0])

    async def _cleanup_entry(self, entry: _ProcessEpoch) -> None:
        entry.stopping = True
        recovery_task = entry.recovery_task
        if recovery_task is not None and recovery_task is not asyncio.current_task():
            if not recovery_task.done():
                _ = recovery_task.cancel()
            await _await_task_after_cancellation(recovery_task)
            entry.recovery_task = None
        watch_task = entry.watch_task
        if watch_task is not None and watch_task is not asyncio.current_task():
            if not watch_task.done():
                _ = watch_task.cancel()
            await _await_task_after_cancellation(watch_task)
            entry.watch_task = None
        if entry.process_group is not None:
            await entry.process_group.terminate(timeout_s=self._stop_timeout_seconds)
        if entry.process is not None:
            _ = await _await_process(entry.process)
        reader_errors: list[BaseException] = []
        for task in (entry.stdout_task, entry.stderr_task):
            if task is None:
                continue
            try:
                await _await_task_after_cancellation(task)
            except BaseException as error:
                reader_errors.append(error)
        if reader_errors:
            raise RuntimeError(
                "managed process log drain failed: "
                + "; ".join(_error_text(error) for error in reader_errors)
            ) from reader_errors[0]
        # 资源释放一旦完成，失效的 Root observer 不能把零残留伪装成
        # cleanup_failed；retry 也不能依赖已经卸载的插件回调。
        entry.process = None
        entry.process_group = None
        entry.endpoint = None
        entry.ready = False
        entry.stdout_task = None
        entry.stderr_task = None
        entry.watch_task = None
        try:
            await self._emit_health(
                entry.generation_id,
                entry.definition.name,
                False,
                "stopped",
            )
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
            logger.error(
                "managed process stopped observer 已失效: generation=%s process=%s",
                entry.generation_id,
                entry.definition.name,
            )
        except Exception as error:
            logger.error(
                "managed process stopped observer 已失效: generation=%s process=%s error=%s",
                entry.generation_id,
                entry.definition.name,
                _error_text(error),
            )

    async def _terminate_runtime(self, entry: _ProcessEpoch) -> None:
        """Kill an exited leader's complete process group before recovery."""

        watch_task = entry.watch_task
        if watch_task is not None and watch_task is not asyncio.current_task():
            if not watch_task.done():
                _ = watch_task.cancel()
            await _await_task_after_cancellation(watch_task)
            entry.watch_task = None
        if entry.process_group is not None:
            await entry.process_group.terminate(timeout_s=self._stop_timeout_seconds)
        if entry.process is not None:
            _ = await _await_process(entry.process)
        for task in (entry.stdout_task, entry.stderr_task):
            if task is not None:
                await _await_task_after_cancellation(task)
        entry.process = None
        entry.process_group = None
        entry.endpoint = None
        entry.ready = False
        entry.stdout_task = None
        entry.stderr_task = None
        entry.watch_task = None

    async def _retain_runtime_callback_failure(
        self,
        generation: _Generation,
        entry: _ProcessEpoch,
        error: BaseException,
    ) -> None:
        """Stop a runtime after a health/incident bridge failure and retain ownership."""

        generation.state = "degraded"
        try:
            await self._terminate_runtime(entry)
        except BaseException as cleanup_error:
            self._retain_runtime_tombstone(generation, entry, cleanup_error)
            return
        self._retain_runtime_tombstone(generation, entry, error)

    async def _drain_stream(
        self,
        stream: asyncio.StreamReader | None,
        ring: _LogRing | None,
    ) -> None:
        if stream is None or ring is None:
            return
        while True:
            chunk = await stream.read(_READ_CHUNK_BYTES)
            if not chunk:
                return
            ring.append(chunk)

    def _retain_cleanup_tombstone(
        self,
        generation: _Generation,
        error: BaseException,
    ) -> None:
        generation.cleanup_attempts += 1
        generation.state = "cleanup_failed"
        tombstone = GenerationCleanupTombstone(
            generation_id=generation.generation_id,
            state="cleanup_failed",
            action="retry_generation_cleanup",
            resource_names=tuple(generation.entries),
            error=_error_text(error),
            attempt_count=generation.cleanup_attempts,
        )
        self._tombstones[generation.generation_id] = tombstone
        if self._on_failure is not None:
            self._on_failure(tombstone)

    def _retain_runtime_tombstone(
        self,
        generation: _Generation,
        entry: _ProcessEpoch,
        error: BaseException,
    ) -> None:
        generation.cleanup_attempts += 1
        tombstone = GenerationCleanupTombstone(
            generation_id=generation.generation_id,
            state="degraded",
            action="retry_runtime_recovery",
            resource_names=(entry.definition.name,),
            error=_error_text(error),
            attempt_count=generation.cleanup_attempts,
        )
        self._tombstones[generation.generation_id] = tombstone
        if self._on_failure is not None:
            self._on_failure(tombstone)

    async def _emit_health(
        self,
        generation_id: str,
        process_name: str,
        healthy: bool,
        reason: str,
    ) -> None:
        if self._on_health is None:
            return
        try:
            result = self._on_health(generation_id, process_name, healthy, reason)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "managed process health callback failed: generation=%s process=%s error=%s",
                generation_id,
                process_name,
                _error_text(error),
            )
            raise

    async def _emit_incident(
        self,
        generation_id: str,
        process_name: str,
        kind: str,
        message: str,
    ) -> None:
        if self._on_incident is None:
            return
        try:
            result = self._on_incident(generation_id, process_name, kind, message)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error(
                "managed process incident callback failed: generation=%s process=%s error=%s",
                generation_id,
                process_name,
                _error_text(error),
            )
            raise

    @staticmethod
    def _normalize_definitions(
        definitions: Mapping[str, ManagedProcessDefinition],
    ) -> dict[str, ManagedProcessDefinition]:
        if not isinstance(definitions, Mapping):
            raise TypeError("managed process definitions must be a mapping")
        normalized: dict[str, ManagedProcessDefinition] = {}
        for name, definition in definitions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("managed process name must be non-empty")
            if name != definition.name:
                raise ValueError(f"managed process mapping key/name mismatch: {name}")
            if definition.formal_port < 1 or definition.formal_port > 65535:
                raise ValueError(f"managed process formal port invalid: {name}")
            if definition.port_env in definition.env:
                raise ValueError(f"managed process port_env is already declared: {name}")
            normalized[name] = definition
        return dict(sorted(normalized.items()))

    @staticmethod
    def _resolve_command(command: tuple[str, ...], artifact_root: Path | None) -> tuple[str, ...]:
        result: list[str] = []
        for item in command:
            path = Path(item)
            if artifact_root is not None and not path.is_absolute() and (
                "/" in item or "\\" in item or item.startswith(".") or item.endswith(".py")
            ):
                result.append(str((artifact_root / path).resolve()))
            else:
                result.append(item)
        return tuple(result)

    @staticmethod
    def _resolve_cwd(cwd: str, artifact_root: Path | None) -> Path:
        path = Path(cwd)
        if path.is_absolute():
            return path
        if artifact_root is not None:
            return (artifact_root / path).resolve()
        return Path.cwd() / path

    @staticmethod
    def _process_env(
        env: Mapping[str, str],
        port_env: str,
        port: int,
    ) -> dict[str, str]:
        values = dict(env)
        if port_env in values:
            raise ValueError(f"managed process port env collision: {port_env}")
        values[port_env] = str(port)
        return values

    @staticmethod
    def _allocate_port(formal_port: int | None) -> int:
        if formal_port is not None:
            _assert_port_free(formal_port)
            return formal_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _require_generation(self, generation_id: str) -> _Generation:
        generation = self._generations.get(generation_id)
        if generation is None:
            raise KeyError(f"unknown managed process generation: {generation_id}")
        return generation


async def _await_task_after_cancellation(task: asyncio.Task[Any]) -> Any:
    """Await a critical task to completion and restore caller cancellation."""

    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                break
            cancelled = True
            continue
    try:
        result = task.result()
    except asyncio.CancelledError:
        result = None
    if cancelled:
        raise asyncio.CancelledError
    return result


def _task_error(task: asyncio.Task[Any], fallback: BaseException) -> BaseException:
    """Return a task failure when available, otherwise preserve the source error."""

    if not task.done():
        return fallback
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return fallback
    return error if error is not None else fallback


async def _await_process(process: asyncio.subprocess.Process) -> int | None:
    if process.returncode is not None:
        return process.returncode
    return await process.wait()


def _assert_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"managed process formal port already occupied: {port}") from error


class _RejectReadinessRedirect(HTTPRedirectHandler):
    """Reject every readiness redirect instead of following an untrusted URL."""

    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request:
        raise URLError(f"managed process readiness redirect rejected: {newurl}")


def _url_ready(url: str, timeout: float) -> bool:
    try:
        opener = build_opener(ProxyHandler({}), _RejectReadinessRedirect())
        request = Request(url, method="GET")
        with opener.open(request, timeout=timeout) as response:
            return 200 <= response.status < 300 and response.geturl() == url
    except HTTPError as error:
        error.close()
        return False
    except (OSError, URLError):
        return False


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return message or type(error).__name__


def _truncate_utf8_tail(value: str, max_bytes: int) -> str:
    """Keep the newest complete UTF-8 code points within a hard byte budget."""

    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    return encoded[-max_bytes:].decode("utf-8", errors="ignore")
