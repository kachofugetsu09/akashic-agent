"""Load actual application and adapter plugins; only external model is a fixture."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
from typing import cast
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_default_reply import application
from tests.support.delivery_sources import sources as delivery_sources
from plugins.subagent.contract import AGENT_RUNNERS

async def main(path):
    path.mkdir(parents=True)
    os.environ.update(AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXECUTION_MODE="local")
    def configure(sources):
        from agent.plugin_composition.config_input import save_config
        launcher = path / "peer"
        launcher.write_text("#!/bin/sh\nexec " + sys.executable + " " + str(ROOT / "scripts/agent_protocol_peer.py") + "\n")
        launcher.chmod(0o700)
        save_config(path / "workspace/plugin-data/subagent_codex-builtin", {"executable": str(launcher)})
        provider = sources / "test_provider/plugin.py"
        text = provider.read_text().replace('if len(calls) == 1:', 'if len(calls) == 1:\n                return LLMResponse(None, [ToolCall("load", "load_tools", {"plugin": "subagent_codex"})])\n            if len(calls) == 2:')
        text = text.replace('ToolCall("provider-call", "write_evidence", {})', 'ToolCall("provider-call", "tool_call", {"name": "delegate_codex", "arguments": {"task": "read only", "cwd": ' + repr(str(path)) + '}})')
        provider.write_text(text)
        delivery_sources(sources)
        for name in ("jobs", "subagent", "host_execution", "assets", "subagent_codex", "subagent_claude_code",
                     *(("ui", "shell_ui", "conversation_ui", "web_auth", "channels", "akashic_clients") if os.environ.get("SCENARIO_PORT") else ())):
            shutil.copytree(ROOT / "plugins" / name, sources / name, ignore=shutil.ignore_patterns("__pycache__"))
        if os.environ.get("SCENARIO_PORT"):
            shutil.copytree(ROOT / "static/dashboard", sources / "ui/static/dashboard", dirs_exist_ok=True)
            shutil.copytree(ROOT / "static/chat", sources / "akashic_clients/static/chat", dirs_exist_ok=True)
    async with application(path, replying=True, extra_sources=configure) as (_, host):
        from datetime import UTC, datetime
        from plugins.channels.contract import CHANNEL_INPUT_V2, ChannelInboundMessage
        from plugins.ledger.contract import OWNER_STATE
        root=host.live_root
        assert root is not None
        runners=root.context.require(AGENT_RUNNERS)
        names={str(row["name"]) for row in runners.catalog()}
        assert names == {"codex", "claude_code"}, names
        await root.context.require(CHANNEL_INPUT_V2)("akashic:runner-scenario", "input", ChannelInboundMessage(
            "test", "user", "room", "delegate read-only", datetime.now(UTC), {}))
        async with asyncio.timeout(20):
            while True:
                async with runners.ctx.runtime_scope():
                    rows = runners.records("akashic:runner-scenario")
                if rows and rows[0]["status"] not in {"accepted", "running"}:
                    assert rows[0]["status"] == "completed", rows
                    break
                await asyncio.sleep(.05)
        # The notice carries settlement facts only; the receipt is read through the result tool.
        from plugins.jobs.contract import JOB_NOTICES
        from plugins.subagent.runners import ReadResult, Runners
        async with runners.ctx.runtime_scope():
            receipt = await ReadResult(cast(Runners, runners), "codex").invoke("scenario", {
                "id": rows[0]["id"], "session_id": "akashic:runner-scenario"})
        assert "已完成：中文碎包" in str(receipt.parts[0].value), receipt
        jobs = await anext(root.context.require(JOB_NOTICES).follow_jobs("akashic:runner-scenario"))
        assert jobs and all("中文碎包" not in job.detail for job in jobs), jobs
        print(json.dumps({"actual_application_loaded": True, "runners": sorted(names), "delegation": rows,
                          "notice_details": [job.detail for job in jobs], "receipt_tool": True}), flush=True)
        if os.environ.get("SCENARIO_PORT"):
            from bootstrap.web_shell import create_web_shell_server
            await create_web_shell_server(path / "workspace", port=int(os.environ["SCENARIO_PORT"])).serve()
if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
