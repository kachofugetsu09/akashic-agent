"""真实 UDS、二进制管道和进程组：游标重读、背压、隔离与关闭。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.host_execution.bridge import host_bridge_pb2 as pb, transport
from plugins.host_execution.bridge.server import HostBridgeService


async def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True)
    service = HostBridgeService("fixture", 60, output, release_commit="a" * 40,
                                toolchain_digest="b" * 64, runtime_checkout=ROOT,
                                bridge_python=Path(sys.executable))
    server = transport.Server(service)
    await server.start(output / "bridge.sock")
    channel = transport.Channel(output / "bridge.sock", "fixture")
    context = pb.RequestContext(boot_id="raw", manager_id="raw", request_id="raw",
                                expected_release_commit="a" * 40, expected_toolchain_digest="b" * 64)

    async def call(operation: str, owner: str = "a", **fields):
        return await channel.call("RawProcess", pb.RawProcessRequest(
            context=context, owner_session_key=owner, operation=operation, **fields))

    try:
        await channel.call("ClaimBoot", pb.ContextRequest(context=context))
        await channel.call("OpenManager", pb.ContextRequest(context=context))
        # Deliberately exceed the shell's historical 1 MiB buffer. Nothing may
        # be dropped, merged with stderr, or corrupted by text decoding.
        code = "import os,sys; os.write(2,b'ERR\\x00'); os.write(1,b'HEAD\\x00'); data=sys.stdin.buffer.read(6); sys.stdout.buffer.write(data+b'x'*1500000); sys.stdout.buffer.flush()"
        opened = await call("open", argv=[sys.executable, "-c", code], cwd=str(output), env={})
        identity = opened.identity
        await call("write", identity=identity, data=b"hello\x00")
        cursor, chunks = 0, []
        async with asyncio.timeout(15):
            while True:
                reply = await call("read", identity=identity, stream="stdout", offset=cursor)
                if reply.data:
                    repeated = await call("read", identity=identity, stream="stdout", offset=cursor)
                    assert repeated.data == reply.data and repeated.next_offset == reply.next_offset
                    chunks.append(reply.data)
                    cursor = reply.next_offset
                if reply.eof:
                    break
        assert b"".join(chunks) == b"HEAD\x00hello\x00" + b"x" * 1500000
        err = await call("read", identity=identity, stream="stderr", offset=0)
        assert err.data == b"ERR\x00"
        try:
            await call("read", owner="other", identity=identity, stream="stdout", offset=0)
        except transport.RpcError:
            pass
        else:
            raise AssertionError("cross-owner read accepted")
        await call("close", identity=identity)
        # Stop a process with unread, full pipes and a sleeping descendant.
        code = "import os,subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)']); print(str(os.getpid())+','+str(p.pid),flush=True); sys.stdout.buffer.write(b'z'*2000000); sys.stdout.buffer.flush(); time.sleep(90)"
        identity = (await call("open", argv=[sys.executable, "-c", code], cwd=str(output), env={})).identity
        first = await call("read", identity=identity, stream="stdout", offset=0)
        pids = [int(x) for x in first.data.split(b"\n", 1)[0].split(b",")]
        async with asyncio.timeout(10):
            await channel.call("ShutdownManager", pb.ContextRequest(context=context))
        from agent.plugin_composition.execution import process_group_exists
        assert not process_group_exists(pids[0]), pids
        return {"binary_bytes": cursor, "cursor_retry": True, "stderr_separate": True,
                "owner_isolation": True, "full_pipe_shutdown": True, "descendants_stopped": True}
    finally:
        await channel.close()
        await server.stop()
        await service.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run(args.output.resolve()))
    print(json.dumps(result))
