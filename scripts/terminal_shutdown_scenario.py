"""真实 PTY 与 UDS RPC：输入写满时，关闭、boot 接管和 lease 回收仍能结束整个终端。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
from plugins.host_execution.bridge import host_bridge_pb2 as pb, transport
from plugins.host_execution.bridge.server import HostBridgeService
from plugins.host_execution.contract import PROCESSES, TERMINALS, TerminalOutput
from plugins.host_execution.processes import PluginProcesses, PluginTerminals
from plugins.host_execution.process_runtime import ShellProcessManager

COMMAND = b"stty -icanon -echo; printf '\\nBLOCKED:%s\\n' \"$$\"; sleep 120\n"
MARKER = re.compile(rb"\r?\nBLOCKED:(\d+)\r?\n")


async def pending(write: asyncio.Task[object]) -> None:
    """shell 已报告不再读取输入；确认写入仍在等待 PTY 容量。"""
    done, _ = await asyncio.wait({write}, timeout=0.2)
    assert not done, "输入未形成背压，场景无效"


async def bridge_case(output: Path, mode: str) -> dict[str, object]:
    """每个场景使用独立 Bridge、socket、manager 和真实 shell。"""
    output.mkdir()
    service = HostBridgeService("fixture", 0.3 if mode == "lease" else 60, output,
                                release_commit="a" * 40, toolchain_digest="b" * 64,
                                runtime_checkout=ROOT, bridge_python=Path(sys.executable))
    server = transport.Server(service)
    await server.start(output / "bridge.sock")
    channel = transport.Channel(output / "bridge.sock", "fixture")

    def context(boot: str) -> pb.RequestContext:
        return pb.RequestContext(boot_id=boot, manager_id="manager", request_id="fixture",
                                 expected_release_commit="a" * 40, expected_toolchain_digest="b" * 64)

    old = context("old")
    write: asyncio.Task[object] | None = None
    watchdog: asyncio.Task[None] | None = None
    try:
        # 1. 通过真实 RPC 开终端，读到 shell 屏障后再写满输入缓冲。
        await channel.call("ClaimBoot", pb.ContextRequest(context=old))
        await channel.call("OpenManager", pb.ContextRequest(context=old))
        request = pb.TerminalRequest(context=old, owner_session_key="session")
        request.open.CopyFrom(pb.TerminalOpen(cols=80, rows=24, cwd=str(output)))
        terminal_id = (await channel.call("Terminal", request)).info.terminal_id

        async def send(data: bytes) -> object:
            request = pb.TerminalRequest(context=old, owner_session_key="session")
            request.write.CopyFrom(pb.TerminalWrite(terminal_id=terminal_id, data=data))
            return await channel.call("Terminal", request)

        await send(COMMAND)
        offset, seen = 0, b""
        async with asyncio.timeout(5):
            while (match := MARKER.search(seen)) is None:
                request = pb.TerminalRequest(context=old, owner_session_key="session")
                request.read.CopyFrom(pb.TerminalRead(terminal_id=terminal_id, offset=offset, wait_ms=200))
                data = (await channel.call("Terminal", request)).output
                offset, seen = data.offset + len(data.data), seen + data.data
        pid = int(match[1])
        write = asyncio.create_task(send(b"x" * 65536))
        await pending(write)

        # 2. 所有生命周期入口都必须主动解除写入，不能等 shell 的 sleep 自行结束。
        async with asyncio.timeout(10):
            if mode == "takeover":
                result = await channel.call("ClaimBoot", pb.ContextRequest(context=context("new")))
                assert result.owner_boot_id == "new" and result.cleaned_manager_count == 1
            elif mode == "manager":
                result = await channel.call("ShutdownManager", pb.ContextRequest(context=old))
                assert not result.failures
            elif mode == "server":
                await service.shutdown()
            else:
                watchdog = asyncio.create_task(service.reap_expired())
            await write
            # lease 的写入先返回，监管进程随后完成回收；服务关闭等待同一清理 owner。
            if mode == "lease":
                await service.shutdown()
        assert not Path(f"/proc/{pid}").exists(), f"旧 shell {pid} 仍在"
        return {"blocked_write_finished": True, "shell_reaped": True}
    finally:
        for task in (write, watchdog):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (write, watchdog) if task is not None), return_exceptions=True)
        await channel.close()
        await server.stop()
        await service.shutdown()


async def local_case(output: Path) -> dict[str, object]:
    """通过真实 Context 和 PluginTerminals 调用本机服务，关闭取消阻塞请求。"""
    output.mkdir()
    root = CompositionRoot("local-terminal")
    manager = ShellProcessManager(output_dir=output)
    processes = PluginProcesses(factory=lambda: manager)
    terminals = PluginTerminals(processes)

    async def provide(ctx: Context) -> None:
        await ctx.effect(lambda: processes.close)
        await ctx.provide(PROCESSES, processes)
        await ctx.provide(TERMINALS, terminals)

    fiber = await root.mount(provide, name="fixture", runtime=PluginRuntime("fixture", root.generation_id,
                            ROOT, output, output, {}))
    ctx = fiber.context
    write: asyncio.Task[object] | None = None
    try:
        async with ctx.runtime_scope():
            info = await terminals.open(ctx, "session", cwd=output, cols=80, rows=24)
            await terminals.write(ctx, "session", info.terminal_id, COMMAND)
            offset, seen = 0, b""
            async with asyncio.timeout(5):
                while (match := MARKER.search(seen)) is None:
                    data: TerminalOutput = await terminals.read(ctx, "session", info.terminal_id, offset=offset, wait_ms=200)
                    offset, seen = data.offset + len(data.data), seen + data.data
        pid = int(match[1])

        async def send() -> object:
            async with ctx.runtime_scope():
                return await terminals.write(ctx, "session", info.terminal_id, b"x" * 65536)

        write = asyncio.create_task(send())
        await pending(write)
        await asyncio.wait_for(processes.close(), 10)
        await asyncio.gather(write, return_exceptions=True)
        assert write.cancelled()
        assert not Path(f"/proc/{pid}").exists()
        return {"write_cancelled": True, "shell_reaped": True}
    finally:
        if write is not None:
            write.cancel()
            await asyncio.gather(write, return_exceptions=True)
        await root.dispose()


async def run(output: Path) -> dict[str, object]:
    report: dict[str, object] = {"local": await local_case(output / "local")}
    for mode in ("takeover", "manager", "server", "lease"):
        report[mode] = await bridge_case(output / mode, mode)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir()
    (output / "home").mkdir()
    os.environ.update(HOME=str(output / "home"), SHELL="/bin/bash")
    report = asyncio.run(run(output))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
