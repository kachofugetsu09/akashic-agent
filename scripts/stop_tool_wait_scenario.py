"""Drive stop, new input and steer against a real Shell wait; verify interrupts, next replies and process owners."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import sqlite3
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plugins.channels.contract import CHANNEL_INPUT_V2 as CHANNEL_INPUT, CHANNEL_STEER, ChannelInboundMessage
from plugins.ledger.contract import Output, ToolResult
from tests.support.delivery_sources import sources as delivery_sources
from tests.test_default_reply import application


def stored_rows(workspace: Path) -> list[tuple]:
    with sqlite3.connect(workspace / "sessions.db") as connection:
        return connection.execute("SELECT * FROM messages ORDER BY session_key, seq").fetchall()


def configure(sources: Path, directory: Path) -> None:
    """Use a controlled model with real installed Shell, Tools, ReAct and source owners."""
    delivery_sources(sources)
    shutil.copytree(ROOT / "plugins/jobs", sources / "jobs", ignore=shutil.ignore_patterns("__pycache__"))
    background = "import os,time,pathlib; pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(120)" % str(directory / "background.pid")
    foreground = ("import os,sys,time,pathlib; print(os.getpid(),flush=True); sys.stdin.readline(); "
                  "pathlib.Path(%r).write_text(str(os.getpid())); time.sleep(120)") % str(directory / "waiting.pid")
    commands = [f"exec {shlex.quote(sys.executable)} -c {shlex.quote(code)}" for code in (background, foreground)]
    provider = sources / "test_provider/plugin.py"
    text = provider.read_text()
    start, end = text.index("    class Driver:"), text.index("    descriptor =")
    text = text[:start] + '''    class Driver:
        max_tool_schemas = None
        def estimate_context_tokens(self, messages, tools):
            return 10
        async def complete(self, request):
            import re
            calls.append(request)
            step = len(calls)
            if step <= 2:
                return LLMResponse(None, [ToolCall(f"start-{step}", "shell", {
                    "command": COMMANDS[step - 1], "description": "stop scenario",
                    "run_in_background": step == 1, "tty": step == 2,
                    "login": False, "yield_time_ms": 250})])
            if step == 3:
                identity = int(re.findall(r'"execution_id": ([0-9]+)', str(request.messages))[-1])
                return LLMResponse(None, [ToolCall("wait", "write_stdin", {
                    "execution_id": identity, "chars": "wait\\n", "yield_time_ms": 30000})])
            text = str(request.messages)
            return LLMResponse("next input answered" if "answer the next input" in text else "no next input")
'''.replace("COMMANDS", repr(commands)) + text[end:]
    provider.write_text(text)


async def wait_file(path: Path) -> int:
    """Wait for the child process to acknowledge stdin, rather than guessing tool start time."""
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.01)
        return int(path.read_text())


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def run(directory: Path, *, mode: str) -> dict[str, object]:
    """stop / input / input-stop / steer-stop: only a user stop interrupts the running tool."""
    directory.mkdir()
    workspace = directory / "workspace"
    async with application(directory, replying=True, shell=True,
                           extra_sources=lambda sources: configure(sources, directory)) as (log, host):
        root = host.live_root
        assert root is not None
        accept = root.context.require(CHANNEL_INPUT)

        def message(text: str) -> ChannelInboundMessage:
            return ChannelInboundMessage("test", "user", "room", text, datetime.now(UTC), {})

        async def send(identity: str, text: str):
            return await accept("test:room", identity, message(text))

        # 1. A real stdin acknowledgement proves the long wait is running.
        await send("first", "start the work")
        foreground = await wait_file(directory / "waiting.pid")
        background = await wait_file(directory / "background.pid")
        before = stored_rows(workspace)
        started = time.monotonic()

        # 2. A new input queues behind the running tool; only /stop interrupts it.
        async with asyncio.timeout(5):
            if mode == "stop":
                await send("stop", "/stop")
                await send("next", "answer the next input")
            elif mode == "steer-stop":
                with root.context.borrow(CHANNEL_STEER) as steers:
                    assert steers is not None and await steers.stage("test:room", "next", message("answer the next input"))
                await send("stop", "/stop")
            else:
                await send("next", "answer the next input")
                await asyncio.sleep(1)
                assert alive(foreground), "new input interrupted the running tool"
                assert not any(isinstance(row.body, Output) and row.body.finish == "complete"
                               for row in log.reader("test:room").snapshot()), "replied before the tool result"
                if mode == "input-stop":
                    await send("stop", "/stop")
                else:
                    # The tool finishes on its own; its result and the new input reach the next request together.
                    os.kill(foreground, signal.SIGKILL)

        async def completed():
            async for _ in log.catalog().follow():
                rows = log.reader("test:room").snapshot()
                if any(isinstance(row.body, Output) and row.body.finish == "complete" for row in rows):
                    return rows
            raise AssertionError("message subscription closed")

        rows = await asyncio.wait_for(completed(), 5)
        elapsed = time.monotonic() - started
        assert elapsed < 5, elapsed
        answers = [row.body for row in rows if isinstance(row.body, Output) and row.body.finish == "complete"]
        assert "next input answered" in str(answers[-1]), answers[-1]
        receipts = [row.body for row in rows if isinstance(row.body, ToolResult)]
        assert len(receipts) == 3, receipts
        assert len({receipt.call_ref for receipt in receipts}) == 3
        assert (receipts[-1].outcome == "interrupted") == (mode != "input"), receipts[-1]
        after = stored_rows(workspace)
        assert after[:len(before)] == before, "existing message rows changed"

        # 3. Foreground cleanup runs before the next reply; explicit background work survives.
        assert not alive(foreground), "foreground process survived stop"
        assert alive(background), "background process was stopped"
    assert stored_rows(workspace) == after, "shutdown changed message history"
    return {"mode": mode, "next_reply_seconds": round(elapsed, 3), "receipts": len(receipts),
            "tool_outcome": receipts[-1].outcome, "next_input_answered": True,
            "foreground_stopped": True, "background_survived": True, "history_preserved": True}


@asynccontextmanager
async def bridge(directory: Path):
    """Run the real Unix-socket bridge used by deployed Shell tools."""
    from plugins.host_execution.bridge.boot import claim_host_bridge_boot

    socket = directory / "bridge.sock"
    os.environ.update(AKASHIC_EXECUTION_MODE="host-bridge", AKASHIC_HOST_BRIDGE_SOCKET=str(socket),
                      AKASHIC_HOST_BRIDGE_TOKEN="stop-scenario", AKASHIC_BOOT_ID="stop-scenario",
                      AKASHIC_RUNTIME_COMMIT="a" * 40, AKASHIC_HOST_TOOLCHAIN_DIGEST="b" * 64)
    token = directory / "bridge.token"
    token.write_text("stop-scenario")
    with (directory / "bridge.log").open("w") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "plugins.host_execution.bridge.server", "--socket", str(socket),
            "--token-file", str(token), "--lease-timeout", "60", "--artifact-root", str(directory / "artifacts"),
            "--release-commit", "a" * 40, "--toolchain-digest", "b" * 64,
            "--runtime-checkout", str(ROOT), "--bridge-python", sys.executable, stdout=log, stderr=log)
        try:
            async with asyncio.timeout(5):
                while not socket.exists():
                    assert process.returncode is None, (directory / "bridge.log").read_text()
                    await asyncio.sleep(0.01)
            assert await claim_host_bridge_boot() is not None
            yield
        finally:
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(process.wait(), 5)


async def main(directory: Path):
    """Check both execution backends with isolated homes, databases and child processes."""
    os.environ.update(HOME=str(directory / "home"), AKASHIC_PLUGIN_HOME=str(directory / "home"),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXECUTION_MODE="local", AKASHIC_WORKLOAD_SOCKET="")
    modes = ("stop", "input", "input-stop", "steer-stop")
    results = {"local": [await run(directory / mode, mode=mode) for mode in modes]}
    async with bridge(directory):
        results["host-bridge"] = [await run(directory / ("bridge-" + mode), mode=mode) for mode in modes]
    return results


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-stop-wait-") as directory:
        print(json.dumps(asyncio.run(main(Path(directory)))))
