"""Owner-scoped binary stdio for protocol clients; never truncate protocol bytes."""
from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.plugin_composition.execution import process_group_exists
from plugins.host_execution.contract import RawProcessOutput

MAX_RUNS = 32
MAX_OWNER_RUNS = 8
MAX_STDIN_FRAME = 1024 * 1024
STDIN_HIGH_WATER = 1024 * 1024
EXITED_TTL_SECONDS = 300.0
# Children see this host baseline plus the caller's explicit env; Core and bridge
# secrets (API keys, AKASHIC_*) never cross implicitly.
BASE_ENV = frozenset({"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TMPDIR", "TZ"})


def child_env(env: dict[str, str]) -> dict[str, str]:
    base = {k: v for k, v in os.environ.items() if k in BASE_ENV or k.startswith("LC_")}
    return {**base, **env}


@dataclass
class _Run:
    owner: str
    process: asyncio.subprocess.Process
    pending: dict[str, bytes] = field(default_factory=dict)
    offsets: dict[str, int] = field(default_factory=lambda: {"stdout": 0, "stderr": 0})
    locks: dict[str, asyncio.Lock] = field(default_factory=lambda: {
        "stdout": asyncio.Lock(), "stderr": asyncio.Lock(), "stdin": asyncio.Lock(),
    })
    eof: set[str] = field(default_factory=lambda: set[str]())
    closing: bool = False
    stdin_broken: bool = False
    group_gone: bool = False
    exited_at: float | None = None


# Manager owns children until confirmed whole-group teardown. Reads acknowledge the
# previous chunk by its next offset, so a lost RPC reply can be retried without losing
# bytes. Only one chunk per stream is retained; backpressure stays in the pipe.
class RawProcesses:
    def __init__(self, admit: Callable[[str], AbstractAsyncContextManager[None]]) -> None:
        self._admit = admit
        self._runs: dict[str, _Run] = {}
        self._closed = False
        self._starting = 0
        self._idle = asyncio.Event()
        self._idle.set()

    # Spawn under the owner's lock and quarantine check so owner cleanup cannot race it.
    async def open(self, *, owner: str, argv: tuple[str, ...], cwd: str,
                   env: dict[str, str]) -> str:
        # 1. Reject malformed requests before touching owner state.
        if not owner or not argv or not argv[0] or not all(isinstance(x, str) for x in argv):
            raise ValueError("owner and argv required")
        if not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise ValueError("raw process cwd must be an existing absolute directory")
        async with self._admit(owner):
            # 2. Admission may have waited on shutdown; recheck, then release exited capacity.
            if self._closed:
                raise RuntimeError("raw process manager closed")
            await self._reclaim()
            if sum(run.owner == owner for run in self._runs.values()) >= MAX_OWNER_RUNS:
                raise RuntimeError("raw process owner quota reached")
            if len(self._runs) + self._starting >= MAX_RUNS:
                raise RuntimeError("raw process capacity reached")
            # 3. Shutdown waits for in-flight spawns, then sees the registered child.
            self._starting += 1
            self._idle.clear()
            try:
                child = await self._spawn(argv, cwd, env)
                if self._closed:
                    await self._stop(_Run(owner, child))
                    raise RuntimeError("raw process manager closed during spawn")
                identity = uuid4().hex
                self._runs[identity] = _Run(owner, child)
                return identity
            finally:
                self._starting -= 1
                if not self._starting:
                    self._idle.set()

    # A cancelled caller must not orphan a child whose spawn already started.
    async def _spawn(self, argv: tuple[str, ...], cwd: str, env: dict[str, str]) -> asyncio.subprocess.Process:
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env=child_env(env), start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=256 * 1024,
        ))
        try:
            return await asyncio.shield(spawning)
        except asyncio.CancelledError:
            await self._stop(_Run("", await spawning))
            raise

    def _run(self, owner: str, identity: str) -> _Run:
        run = self._runs.get(identity)
        if run is None or run.owner != owner:
            raise LookupError("raw process not found for owner")
        return run

    async def read(self, *, owner: str, identity: str, stream: str,
                   offset: int) -> RawProcessOutput:
        run = self._run(owner, identity)
        if stream not in {"stdout", "stderr"} or offset < 0:
            raise ValueError("invalid raw stream cursor")
        async with run.locks[stream]:
            current = run.offsets[stream]
            pending = run.pending.get(stream, b"")
            if offset == current + len(pending) and pending:
                run.offsets[stream] = offset
                run.pending.pop(stream, None)
            elif offset != current:
                raise ValueError("raw stream offset does not acknowledge retained chunk")
            if stream not in run.pending:
                reader = run.process.stdout if stream == "stdout" else run.process.stderr
                assert reader is not None
                # Bounded wait makes stop and manager-lease shutdown independent
                # of a client holding a read forever.
                try:
                    data = await asyncio.wait_for(reader.read(64 * 1024), timeout=1)
                except TimeoutError:
                    return RawProcessOutput(b"", offset, False, run.process.returncode)
                run.pending[stream] = data
                if not data:
                    run.eof.add(stream)
            data = run.pending[stream]
            return RawProcessOutput(data, run.offsets[stream] + len(data),
                                    not data, run.process.returncode)

    # A stalled frame stays queued in the transport; later bytes would interleave with it.
    async def write(self, *, owner: str, identity: str, data: bytes) -> None:
        if len(data) > MAX_STDIN_FRAME:
            raise ValueError("raw stdin frame exceeds 1 MiB")
        run = self._run(owner, identity)
        async with run.locks["stdin"]:
            stdin = run.process.stdin
            if run.closing or run.stdin_broken or stdin is None:
                raise RuntimeError("raw process stdin is closed")
            if stdin.transport.get_write_buffer_size() >= STDIN_HIGH_WATER:
                raise RuntimeError("raw stdin backlog is full")
            stdin.write(data)
            try:
                await asyncio.wait_for(stdin.drain(), timeout=10)
            except (TimeoutError, ConnectionError) as error:
                # 1. Drop the partial frame and fail every later write instead of corrupting framing.
                run.stdin_broken = True
                stdin.transport.abort()
                raise RuntimeError("raw stdin stalled; stream closed") from error

    async def close(self, *, owner: str, identity: str) -> bool:
        run = self._runs.get(identity)
        if run is None or run.owner != owner:
            # Exited, drained runs may already be reclaimed; close stays idempotent.
            return False
        run.closing = True
        await self._stop(run)
        self._runs.pop(identity, None)
        return True

    # Exited runs release capacity once both streams hit EOF, or after a TTL if never drained.
    async def _reclaim(self) -> None:
        now = time.monotonic()
        for identity, run in tuple(self._runs.items()):
            if run.closing or run.process.returncode is None:
                continue
            run.exited_at = run.exited_at or now
            if run.eof != {"stdout", "stderr"} and now - run.exited_at < EXITED_TTL_SECONDS:
                continue
            if not run.group_gone and process_group_exists(run.process.pid):
                continue
            run.group_gone = run.closing = True
            await self._stop(run)
            self._runs.pop(identity, None)

    # Confirm whole-group exit, then release pipes and reap; never signal a group seen gone.
    async def _stop(self, run: _Run) -> None:
        process = run.process
        if process.stdin is not None:
            process.stdin.close()
        # 1. Signal only while the group is alive: a reaped leader's pgid may be reused later.
        for sig, seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 3.0)):
            if run.group_gone or not process_group_exists(process.pid):
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = asyncio.get_running_loop().time() + seconds
            while process_group_exists(process.pid) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.025)
        if not run.group_gone and process_group_exists(process.pid):
            raise RuntimeError(f"raw process group remains alive: {process.pid}")
        run.group_gone = True
        # 2. Let the child watcher record the exit first so closing pipes never re-reaps it.
        with suppress(TimeoutError):
            async with asyncio.timeout(2):
                while process.returncode is None:
                    await asyncio.sleep(0.01)
        # 3. Unread pipe data is discarded; closing the transport frees fds and buffers now.
        transport = cast(asyncio.SubprocessTransport, getattr(process, "_transport"))
        transport.close()
        with suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=1)

    async def close_owner(self, owner: str) -> None:
        errors: list[Exception] = []
        for identity, run in tuple(self._runs.items()):
            if run.owner == owner:
                try:
                    await self.close(owner=owner, identity=identity)
                except Exception as error:
                    errors.append(error)
        if errors:
            raise ExceptionGroup("raw process owner cleanup failed", errors)

    async def shutdown(self) -> None:
        self._closed = True
        await self._idle.wait()
        errors: list[Exception] = []
        for owner in {run.owner for run in self._runs.values()}:
            try:
                await self.close_owner(owner)
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("raw process cleanup failed", errors)
