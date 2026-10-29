"""JSON-lines transport shared by external harness adapters, not their semantics."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESS_STREAMS


class JsonLines:
    def __init__(self, ctx: Context, owner: str, identity: str):
        self.ctx, self.owner, self.identity = ctx, owner, identity
        self.streams = ctx.require(PROCESS_STREAMS)
        self.offset = 0
        self.buffer = bytearray()
        self.stderr = asyncio.create_task(ctx.entrypoint(self._drain_stderr)())

    async def _drain_stderr(self):
        offset = 0
        while True:
            value = await self.streams.read(self.ctx, self.owner, identity=self.identity,
                                            stream="stderr", offset=offset)
            offset = value.next_offset
            if value.eof:
                return

    async def send(self, value: Mapping[str, object]):
        data = json.dumps(dict(value), ensure_ascii=False).encode() + b"\n"
        await self.streams.write(self.ctx, self.owner, identity=self.identity, data=data)

    async def receive(self) -> dict[str, object]:
        while b"\n" not in self.buffer:
            value = await self.streams.read(self.ctx, self.owner, identity=self.identity,
                                            stream="stdout", offset=self.offset)
            self.offset = value.next_offset
            self.buffer.extend(value.data)
            if len(self.buffer) > 4 * 1024 * 1024:
                raise ValueError("agent protocol frame exceeds 4 MiB")
            if value.eof:
                raise EOFError("agent exited before terminal result")
        line, _, tail = self.buffer.partition(b"\n")
        self.buffer = bytearray(tail)
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("agent protocol frame must be an object")
        return value


@asynccontextmanager
async def open_json_lines(ctx: Context, owner: str, argv: tuple[str, ...], cwd: str,
                          env: dict[str, str]):
    """Close the actual process even when the consumer is cancelled."""
    streams = ctx.require(PROCESS_STREAMS)
    identity = await streams.open(ctx, owner, argv=argv, cwd=cwd, env=env)
    wire = JsonLines(ctx, owner, identity)
    try:
        yield wire
    finally:
        async def close():
            return await streams.close(ctx, owner, identity=identity)
        cleanup = asyncio.create_task(ctx.entrypoint(close)())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        finally:
            wire.stderr.cancel()
            await asyncio.gather(wire.stderr, return_exceptions=True)
