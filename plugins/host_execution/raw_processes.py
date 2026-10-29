"""Owner-scoped binary stdio for protocol clients; never truncate protocol bytes."""
from __future__ import annotations

import asyncio
import os
import signal
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from agent.plugin_composition.execution import process_group_exists
from plugins.host_execution.contract import RawProcessOutput


@dataclass
class _Run:
    owner: str
    process: asyncio.subprocess.Process
    pending: dict[str, bytes] = field(default_factory=dict)
    offsets: dict[str, int] = field(default_factory=lambda: {"stdout": 0, "stderr": 0})
    locks: dict[str, asyncio.Lock] = field(default_factory=lambda: {
        "stdout": asyncio.Lock(), "stderr": asyncio.Lock(), "stdin": asyncio.Lock(),
    })
    closing: bool = False


class RawProcesses:
    """Manager owns children until confirmed whole-group teardown.

    Reads acknowledge the previous chunk by its next offset. A lost RPC reply
    can therefore be retried without losing bytes. Only one chunk per stream is
    retained; backpressure stays in the pipe rather than an unbounded queue.
    """

    def __init__(self) -> None:
        self._runs: dict[str, _Run] = {}
        self._closed = False
        self._starting = 0
        self._idle = asyncio.Event()
        self._idle.set()

    async def open(self, *, owner: str, argv: tuple[str, ...], cwd: str,
                   env: dict[str, str]) -> str:
        if self._closed:
            raise RuntimeError("raw process manager closed")
        if not owner or not argv or not argv[0] or not all(isinstance(x, str) for x in argv):
            raise ValueError("owner and argv required")
        if not Path(cwd).is_absolute() or not Path(cwd).is_dir():
            raise ValueError("raw process cwd must be an existing absolute directory")
        if len(self._runs) + self._starting >= 32:
            raise RuntimeError("raw process capacity reached")
        self._starting += 1
        self._idle.clear()
        child: asyncio.subprocess.Process | None = None
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env={**os.environ, **env}, start_new_session=True,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=256 * 1024,
        ))
        try:
            try:
                child = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                child = await spawning
                await self._stop(child)
                raise
            assert child is not None
            if self._closed:
                await self._stop(child)
                raise RuntimeError("raw process manager closed during spawn")
            identity = uuid4().hex
            self._runs[identity] = _Run(owner, child)
            return identity
        finally:
            self._starting -= 1
            if not self._starting:
                self._idle.set()

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
            data = run.pending[stream]
            return RawProcessOutput(data, run.offsets[stream] + len(data),
                                    not data, run.process.returncode)

    async def write(self, *, owner: str, identity: str, data: bytes) -> None:
        if len(data) > 1024 * 1024:
            raise ValueError("raw stdin frame exceeds 1 MiB")
        run = self._run(owner, identity)
        async with run.locks["stdin"]:
            if run.closing:
                raise RuntimeError("raw process is closing")
            assert run.process.stdin is not None
            run.process.stdin.write(data)
            await asyncio.wait_for(run.process.stdin.drain(), timeout=10)

    async def close(self, *, owner: str, identity: str) -> bool:
        run = self._run(owner, identity)
        run.closing = True
        await self._stop(run.process)
        self._runs.pop(identity, None)
        return True

    @staticmethod
    async def _stop(process: asyncio.subprocess.Process) -> None:
        # Drain both pipes while waiting: Process.wait otherwise can be held by
        # unread buffered output even after the OS has reaped the child.
        if process.stdin is not None:
            process.stdin.close()
        for sig, seconds in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 3.0)):
            if not process_group_exists(process.pid):
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            deadline = asyncio.get_running_loop().time() + seconds
            while process_group_exists(process.pid) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.025)
        if process_group_exists(process.pid):
            raise RuntimeError(f"raw process group remains alive: {process.pid}")
        # Readers may still have a protocol read in flight. OS exit and group
        # quiescence are authoritative, not whether buffered pipes were read.
        if process.returncode is None:
            await asyncio.sleep(0)

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
