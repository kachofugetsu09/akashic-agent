"""启动一次性 Models/Gemini HTTP 夹具，供真实浏览器验收。"""
from __future__ import annotations

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.plugin_composition import CompositionRoot, MODEL_CATALOG, MODEL_DRIVERS
from plugins.gemini.driver import definition
from plugins.models.model_settings_http import BoundModelControl, create_model_settings_router
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.state import ModelsState
from plugins.models.store import ModelsStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """真实配置 owner 只打开临时数据库，不读取正式 workspace。"""
    # 1. 在独立 Root 注册真实 Models 服务和 Gemini 驱动。
    with TemporaryDirectory(prefix="akashic-gemini-e2e-") as directory:
        root = CompositionRoot("gemini-discovery-e2e")
        store = ModelsStore(Path(directory) / "models.sqlite3", Path(directory) / "backups")
        async def mount_models(ctx):
            store.initialize()
            state = ModelsState(store, context=ctx)
            app.state.models = state
            await ctx.effect(lambda: store.close, label="fixture-store")
            await ctx.provide(MODEL_DRIVERS, state.drivers)
            await ctx.provide(MODEL_SETTINGS, state.settings)
            await ctx.provide(MODEL_CATALOG, state.catalog)
        async def mount_gemini(ctx):
            await ctx.require(MODEL_DRIVERS).register(ctx, definition())
        async def mount_api(ctx):
            app.include_router(create_model_settings_router(BoundModelControl(ctx), prefix="/api/dashboard/models"))
        await root.mount(mount_models, name="models")
        await root.mount(mount_gemini, name="gemini", inject=(MODEL_DRIVERS,))
        await root.mount(mount_api, name="http", inject=(MODEL_SETTINGS, MODEL_CATALOG))
        # 2. 只控制外部协议夹具；被测 UI、HTTP adapter、驱动和事务均使用实际代码。
        app.state.hold = asyncio.Event()
        app.state.hold.set()
        app.state.entered = asyncio.Event()
        app.state.fail = False
        app.state.calls = []
        try:
            yield
        finally:
            app.state.hold.set()
            await root.dispose()


app = FastAPI(lifespan=lifespan)
repo = Path(__file__).resolve().parents[1]
app.mount("/files", StaticFiles(directory=repo / "frontend"), name="source")


@app.get("/provider/{version}/models")
async def models(version: str, request: Request):
    """分页返回协议目录；用 barrier 验证结果迟到和取消。"""
    app.state.calls.append(request.url.path)
    app.state.entered.set()
    await app.state.hold.wait()
    if request.headers.get("x-goog-api-key") != "fixture-key":
        return JSONResponse({"error": {"message": "fixture authentication failed"}}, status_code=401)
    names = ["gemini-good", "gemini-denied"] if not request.query_params.get("pageToken") else ["gemini-extra"]
    return {"models": [{"name": "models/" + name, "supportedGenerationMethods": ["generateContent"],
                        "inputTokenLimit": 8192, "outputTokenLimit": 1024} for name in names],
            **({"nextPageToken": "second"} if not request.query_params.get("pageToken") else {})}


@app.post("/provider/{version}/models/{model}:generateContent")
async def generate(version: str, model: str, request: Request):
    """外部服务失败由真实驱动转换，不能伪造配置保存成功。"""
    app.state.calls.append(request.url.path)
    if app.state.fail or model == "gemini-denied":
        return JSONResponse({"error": {"message": "fixture model unavailable"}}, status_code=503)
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


@app.post("/fixture/{action}")
async def control(action: str):
    """控制夹具的网络结果，不绕过被测配置事务。"""
    if action == "hold":
        app.state.hold.clear()
        app.state.entered.clear()
    elif action == "entered":
        await app.state.entered.wait()
    elif action == "release":
        app.state.hold.set()
    elif action == "fail":
        app.state.fail = True
    elif action == "recover":
        app.state.fail = False
    else:
        return JSONResponse({"detail": "unknown fixture action"}, status_code=404)
    return {"ok": True}


@app.get("/fixture/state")
async def state():
    """返回真实已提交配置的摘要，绝不把 credential 写进报告。"""
    snapshot = app.state.models.catalog_snapshot()
    return {"revision": snapshot.revision, "connections": len(snapshot.connections),
            "models": [m.model for m in snapshot.models], "calls": app.state.calls,
            "digest": hashlib.sha256(repr(snapshot).encode()).hexdigest()}


@app.get("/")
async def page():
    """挂载真实 Models 页面与 Gemini 连接类型，保留宿主提供的模型选择层。"""
    return HTMLResponse('''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="/files/chat/src/theme.css"><link rel="stylesheet" href="/files/plugins/models/src/style.css">
<style>body{margin:0;background:var(--ak-paper-canvas);color:var(--ak-ink-primary)}main{max-width:60rem;margin:auto;padding:16px;box-sizing:border-box}</style></head><body><main id="host"></main>
<script type="module">
import {activate as models} from '/files/plugins/models/src/module.js';
import {activate as gemini} from '/files/plugins/gemini/src/module.js';
let modelEntry, geminiEntry;
const register = setter => ({ui:{inject:(_id,fn)=>fn({register:setter})}});
gemini(register(entry=>{geminiEntry=entry;return()=>{};}));
models({...register(entry=>{modelEntry=entry;return()=>{};}),http:{request:(path,init)=>fetch(path,init)}});
const child={entries:[geminiEntry],render:(_id,host,props)=>geminiEntry.render(host,null,props)};
window.disposeModels=modelEntry.render(document.querySelector('#host'),{child:()=>child});
</script></body></html>''')


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("GEMINI_FIXTURE_PORT", "2317")), log_level="warning")
