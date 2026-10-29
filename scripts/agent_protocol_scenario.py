"""Real composition/stdio/process teardown with a deterministic external peer."""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.plugin_composition import CompositionRoot, PluginRuntime
from plugins.host_execution.contract import PROCESS_STREAMS
from plugins.host_execution.processes import PluginProcesses, PluginProcessStreams
from plugins.host_execution.process_runtime import ShellProcessManager
from plugins.subagent.protocol import open_json_lines
from plugins.subagent.contract import AgentTask
from plugins.subagent_codex.plugin import execute as codex, Config
from plugins.subagent_claude_code.plugin import execute as claude

async def run(output: Path, count: int):
    output.mkdir(parents=True)
    backend = ShellProcessManager(output_dir=output)
    processes = PluginProcesses(factory=lambda: backend)
    root = CompositionRoot("protocol-scenario")
    contexts = []
    async def provider(ctx):
        await ctx.provide(PROCESS_STREAMS, PluginProcessStreams(processes))
    async def consumer(ctx):
        contexts.append(ctx)
    await root.mount(provider, name="host_execution")
    await root.mount(consumer, name="adapter", inject=(PROCESS_STREAMS,),
        runtime=PluginRuntime("adapter", "adapter", output, output, output, {}))
    rng = random.Random(20261029)
    results = []
    try:
        for n in range(count):
            mode = rng.choice(["codex", "claude"])
            case = rng.choice(["success", "failed", "hang", "eof", "malformed", "approval", "unsupported", "mcp"]
                              if mode == "codex"
                              else ["success", "failed", "hang", "eof", "malformed", "cross-session"])
            events = []
            async def emit(event): events.append(event.text)
            ctx = contexts[0]
            try:
                async with ctx.runtime_scope():
                    async with asyncio.timeout(.3 if case == "hang" else 5):
                        async with open_json_lines(ctx, str(n), (sys.executable, str(ROOT / "scripts/agent_protocol_peer.py")),
                                str(output), {**os.environ, "PEER_MODE": mode, "PEER_CASE": case}) as wire:
                            task = AgentTask(str(n), "read only", str(output))
                            result = await (codex(wire, task, Config(), emit) if mode == "codex" else claude(wire, task, emit))
                            assert case in {"success", "approval", "failed", "unsupported"}, (mode, case, result)
                            assert result.status == ("failed" if case == "failed" else "completed")
                            assert "MUST-NOT-LEAK" not in result.text
            except (TimeoutError, EOFError, ValueError, RuntimeError) as error:
                assert case in {"hang", "eof", "malformed", "cross-session", "mcp"}, (mode, case, error)
            assert not backend.raw._runs, "child not released"
            results.append({"mode": mode, "case": case, "passed": True})
    finally:
        await root.dispose()
        await processes.close()
    return {"seed": 20261029, "iterations": count, "external_peer": "simulated", "results": results}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    args = p.parse_args()
    result = asyncio.run(run(args.output.resolve(), args.count))
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != "results"}))
