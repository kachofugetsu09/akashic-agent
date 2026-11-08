"""Real installed reply loop: unlock, discover, query a provider, persist evidence."""
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_default_reply import application
from plugins.channels.contract import CHANNEL_INPUT_V2, ChannelInboundMessage
from plugins.ledger.contract import Output, ToolResult

PROVIDER = '''from plugins.userinformation.contract import USER_INFORMATION
api_version=3
name="sample_information"
version="1.0.0"
inject=(USER_INFORMATION,)
async def apply(ctx):
    class Provider:
        async def describe(self):
            return {"queries":{"current":{"parameters":{"type":"object","properties":{}}}}}
        async def query(self, query, arguments):
            return {"observed_at":"2026-11-08T07:00:00Z","status":"unknown","evidence_ref":"fixture:observation"}
    await ctx.require(USER_INFORMATION).register(ctx, Provider())
'''


def sources(path):
    shutil.copytree(ROOT/'plugins/userinformation', path/'userinformation')
    (path/'sample_information').mkdir()
    (path/'sample_information/plugin.py').write_text(PROVIDER)
    p = path/'test_provider/plugin.py'
    text = p.read_text()
    start = text.index('            if len(calls) == 1:')
    end = text.index('            return LLMResponse("finished")', start)
    text = text[:start] + '''            steps = [
                ("load_tools", {"plugin":"userinformation"}),
                ("tool_call", {"name":"user_information_list","arguments":{}}),
                ("tool_call", {"name":"user_information_query","arguments":{"provider":"sample_information","query":"current"}}),
            ]
            if len(calls) <= len(steps):
                name, args = steps[len(calls)-1]
                return LLMResponse(None, [ToolCall(str(len(calls)), name, args)])
''' + text[end:]
    p.write_text(text)


async def main():
    with tempfile.TemporaryDirectory(prefix='userinformation-') as directory:
        async with application(Path(directory), replying=True, extra_sources=sources) as (log, host):
            root = host.live_root
            await root.context.require(CHANNEL_INPUT_V2)('test:room','u1',
                ChannelInboundMessage('test','user','room','read user information',datetime.now(UTC),{}))
            async def finished():
                async for _ in log.catalog().follow():
                    rows = log.reader('test:room').snapshot()
                    if any(isinstance(r.body,Output) and r.body.finish=='complete' for r in rows):
                        return rows
            rows = await asyncio.wait_for(finished(),30)
            results = [r.body for r in rows if isinstance(r.body,ToolResult)]
            for r in results: print(r.outcome, r.parts)
            assert len(results)==3 and all(r.outcome=='success' for r in results)
            assert 'fixture:observation' in str(results[-1].parts)
            print(json.dumps({'real_reply_loop':'passed','persisted_tool_results':len(results)}))

if __name__=='__main__': asyncio.run(main())
