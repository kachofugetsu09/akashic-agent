"""前端 loader 经真实 Shell/Models HTTP 读取临时调用账；不请求模型。"""
from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
import uvicorn
from fastapi import FastAPI

from agent.plugin_composition.models import MODEL_CALL_STATS, BoundModelDescriptor, CapabilitySources, ModelCapabilities, ModelRequest
from bootstrap.web_shell import create_web_shell_app
from plugins.models import dashboard
from plugins.models.store import ModelsStore


@asynccontextmanager
async def serve(app, *, uds=None, sockets=None):
    server = uvicorn.Server(uvicorn.Config(app, uds=uds, log_level="critical"))
    task = asyncio.create_task(server.serve(sockets=sockets))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    task.result()
                    raise AssertionError("listener 未就绪")
                await asyncio.sleep(0)
        yield
    finally:
        server.should_exit = True
        await task


async def check(workspace: Path) -> None:
    store = ModelsStore(workspace / "models.db", workspace / "backups")
    store.initialize()
    try:
        descriptor = BoundModelDescriptor("binding", "snapshot", 0, "model", "connection", "driver", "1",
                                          "private-auth-identity", "scenario-model", "agent", None,
                                          ModelCapabilities(), CapabilitySources(), "digest")
        call_id = store.start_call(descriptor, ModelRequest(messages=[]))
        store.record_first_token(call_id, 12)
        store.finish_call(call_id, usage=None, failure=None, duration_ms=45)
        before = store.read_calls("", 100)
        class StatsOnly:
            def require(self, key):
                if key != MODEL_CALL_STATS:
                    raise AssertionError("此只读场景不得操作模型配置或驱动")
                return store.read_call_stats
        models = FastAPI()
        dashboard.register(models, StatsOnly())
        shell = create_web_shell_app(workspace / "config.toml", workspace)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            base = "http://127.0.0.1:" + str(listener.getsockname()[1])
            async with serve(models, uds=str(workspace / "runtime/dashboard.sock")), serve(shell, sockets=[listener]):
                # 运行真实 TypeScript loader 和 parser，以原生 fetch 访问临时 Shell。
                script = '''
import { loadWebModelCallStats } from './frontend/chat/src/model-call-stats.ts';
const [base, id] = process.argv.slice(1);
const nativeFetch = globalThis.fetch;
globalThis.fetch = (path, options) => nativeFetch(new URL(path, base), options);
const value = await loadWebModelCallStats(id, new AbortController().signal);
if (value.call_record_id !== id || value.model !== 'scenario-model' || value.duration_ms !== 45 || value.first_token_ms !== 12) throw Error('stats mismatch');
if ('binding' in value || JSON.stringify(value).includes('private-auth-identity')) throw Error('private fields leaked');
console.log('frontend loader and parser: PASS');
'''
                process = await asyncio.create_subprocess_exec("node", "--experimental-strip-types", "--input-type=module",
                    "-e", script, base, call_id, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await process.communicate()
                assert process.returncode == 0, (out.decode(), err.decode())
                async with httpx.AsyncClient(base_url=base) as http:
                    assert (await http.get("/api/dashboard/models/calls/missing")).status_code == 404
                    retired = await http.get("/api/settings/model/calls/" + call_id)
                    assert retired.status_code == 410 and retired.json()["code"] == "model_settings_moved"
                    assert (await http.post("/api/settings/model/command", json={})).status_code == 410
                assert store.read_calls("", 100) == before
    finally:
        store.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-stats-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: real frontend fetch/parse → Shell → Models router → call ledger; 404/410; no model or state mutation")
