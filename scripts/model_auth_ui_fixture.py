"""用真实 Models owner、驱动与 HTTP 路由提供一次性浏览器登录夹具。"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import CompositionRoot
from plugins.models.contract import (
    MODEL_CATALOG,
    MODEL_DRIVERS,
)
from plugins.codex import auth as codex_auth
from plugins.codex.driver import definition as codex_definition
from plugins.models.model_settings_http import BoundModelControl, create_model_settings_router
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.state import ModelsState
from plugins.models.store import ModelsStore
from plugins.opencode_go.driver import definition as opencode_definition
from plugins.openai_compatible.driver import definition as openai_definition
from plugins.gemini.driver import definition as gemini_definition
from plugins.models.settings import AddConnection, AddModel
from plugins.models.contract import ModelCapabilities, CapabilitySources

PAGE = """<!doctype html><meta charset="utf-8"><title>模型登录验收</title>
<link rel="stylesheet" href="/packages/akashic-theme/src/material-tokens.css">
<link rel="stylesheet" href="/packages/akashic-theme/src/brand-tokens.css">
<link rel="stylesheet" href="/packages/akashic-theme/src/paper-surface.css">
<link rel="stylesheet" href="/frontend/plugins/models/src/style.css">
<link rel="stylesheet" href="/fixture/frontend/plugins/models/src/module.css">
<style>body{background:var(--ak-paper-canvas);color:var(--ak-ink-primary);margin:0;font-family:var(--ak-type-reading)} *{box-sizing:border-box} [hidden]{display:none!important}</style>
<main id="host"></main><script type="module">
import {activate as models} from '/fixture/frontend/plugins/models/src/module.js';
import {WebDataRegistry} from '/fixture/frontend/plugins/ui/dashboard/src/webData.js';
import {initializeTheme} from '/fixture/packages/akashic-theme/src/theme-runtime.js';
initializeTheme();
const registry = new WebDataRegistry();
import {activate as codex} from '/frontend/plugins/codex/src/module.js';
import {activate as opencode} from '/frontend/plugins/opencode_go/src/module.js';
const providers = [];
const providerContext = {ui:{inject:(_, mount)=>mount({register(entry){providers.push(entry);return ()=>{};}})}};
codex(providerContext);opencode(providerContext);
import {activate as openai} from '/frontend/plugins/openai_compatible/src/module.js';
import {activate as gemini} from '/frontend/plugins/gemini/src/module.js';
openai(providerContext);gemini(providerContext);
models({data:{resource:(key, load, options)=>registry.create(key, load, options).resource},ui:{inject:(_, mount)=>mount({register(entry){
  if (entry.id !== 'models') return ()=>{};
  return entry.render(document.querySelector('#host'), {child:()=>({
    entries:providers,
    render(id, host, props){return providers.find(entry=>entry.id===id).render(host, {}, props);},
  })});
}})},http:{request:(path, init)=>fetch(path, init)}});
</script>"""


def create_app(port: int, frontend: Path, *, connection_edit: bool = False) -> FastAPI:
    """挂载真实认证实现；只把外部服务替换为本机 HTTP 协议端点。"""
    base = f"http://127.0.0.1:{port}"
    codex_auth.CODEX_AUTH_BASE = base + "/upstream"
    codex_auth.CODEX_API_BASE = base + "/upstream/codex"
    receipts: list[dict[str, object]] = []
    completion_calls = 0

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. 独立目录保存真实 SQLite 与凭据，结束后由各自 owner 关闭。
        with tempfile.TemporaryDirectory(prefix="akashic-auth-ui-") as directory:
            assets = Path(directory) / "assets"
            subprocess.run([
                str(ROOT / "node_modules/.bin/esbuild"),
                str(frontend / "plugins/models/src/module.js"),
                str(frontend / "plugins/ui/dashboard/src/webData.ts"),
                str(frontend.parent / "packages/akashic-theme/src/theme-runtime.ts"),
                "--bundle", "--format=esm", "--outdir=" + str(assets), "--outbase=" + str(frontend.parent),
            ], check=True, capture_output=True)
            app.mount("/fixture", StaticFiles(directory=assets))
            root = CompositionRoot("auth-ui-fixture")

            async def models(ctx):
                store = ModelsStore(Path(directory) / "models.sqlite3", Path(directory) / "backups")
                store.initialize()
                if connection_edit:
                    rev = store.add_connection(AddConnection(0, "magpie", "Magpie fixture", "openai-compatible", base + "/upstream/v1", "api:magpie", {"driver":"api_key", "access_token":"fixture-key"}, {"format_version":1}))
                    for model in ("old-removed", "still-valid"):
                        rev = store.add_model(AddModel(rev, model, "magpie", "chat", model, ModelCapabilities(), CapabilitySources(), driver_config={"format_version":1}))
                    rev = store.add_connection(AddConnection(rev, "gemini-gw", "Gemini 网关", "gemini", base + "/upstream/gemini/v1beta", "api:gemini-gw", {"driver":"api_key", "access_token":"gemini-key"}, {"format_version":1}))
                    rev = store.add_connection(AddConnection(rev, "opencode-gw", "OpenCode Go", "opencode-go", base + "/upstream/opencode/v1", "opencode-go", {"driver":"api_key", "access_token":"opencode-key"}, {"format_version":1}))
                state = ModelsState(store, context=ctx)
                await ctx.effect(lambda: store.close, label="models-store")
                await ctx.provide(MODEL_DRIVERS, state.drivers)
                await ctx.provide(MODEL_SETTINGS, state.settings)
                await ctx.provide(MODEL_CATALOG, state.catalog)

            async def drivers(ctx):
                async def read_version():
                    return "0.0.0"

                await ctx.require(MODEL_DRIVERS).register(ctx, codex_definition(read_version))
                await ctx.require(MODEL_DRIVERS).register(ctx, opencode_definition())
                await ctx.require(MODEL_DRIVERS).register(ctx, openai_definition())
                await ctx.require(MODEL_DRIVERS).register(ctx, gemini_definition())

            await root.mount(models, name="models")
            await root.mount(drivers, name="auth-drivers", inject=(MODEL_DRIVERS,))
            app.include_router(create_model_settings_router(
                BoundModelControl(root.context), prefix="/api/dashboard/models",
            ))
            try:
                yield
            finally:
                await root.dispose()

    app = FastAPI(lifespan=lifespan)
    app.mount("/frontend", StaticFiles(directory=frontend), name="frontend")
    app.mount("/packages", StaticFiles(directory=frontend.parent / "packages"), name="packages")

    @app.middleware("http")
    async def record_commands(request: Request, call_next):
        if request.url.path == "/api/dashboard/models/command":
            body = await request.json()
            # 2. 只保存身份、参数键与结果，不保存 API Key 或令牌。
            response = await call_next(request)
            receipts.append({
                "type": body["type"], "connection_id": body.get("connection_id"),
                "driver_id": body.get("driver_id"), "endpoint": body.get("endpoint"),
                "input_keys": sorted(body.get("input", {})), "status": response.status_code,
            })
            return response
        return await call_next(request)

    @app.get("/upstream/v1/models")
    async def models_list():
        return {"object":"list", "data":[{"id":"still-valid", "object":"model"}]}

    @app.post("/upstream/v1/chat/completions")
    async def completion(request: Request):
        nonlocal completion_calls
        completion_calls += 1
        body = await request.json()
        if request.headers.get("authorization") == "Bearer invalid-key":
            return JSONResponse({"error":{"message":"invalid fixture key"}}, status_code=401)
        if body["model"] == "old-removed":
            return JSONResponse({"error":{"message":"magpie knows no model old-removed"}}, status_code=404)
        return {"id":"fixture", "object":"chat.completion", "model":body["model"], "choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}], "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}

    @app.get("/")
    async def page():
        return HTMLResponse(PAGE)

    @app.get("/receipts")
    async def read_receipts():
        return receipts

    @app.get("/completion_calls")
    async def read_completion_calls():
        return {"count": completion_calls}

    @app.post("/upstream/api/accounts/deviceauth/usercode")
    async def device_code(request: Request):
        assert (await request.json())["client_id"] == codex_auth.CODEX_CLIENT_ID
        return {"device_auth_id": "local-device", "user_code": "LOCAL-CODE", "interval": "3"}

    @app.post("/upstream/api/accounts/deviceauth/token")
    async def device_token(request: Request):
        assert (await request.json())["device_auth_id"] == "local-device"
        return {"authorization_code": "local-code", "code_verifier": "local-verifier"}

    @app.post("/upstream/oauth/token")
    async def token():
        payload = base64.urlsafe_b64encode(json.dumps({
            "https://api.openai.com/auth": {"chatgpt_account_id": "local-account"},
        }).encode()).decode().rstrip("=")
        return {"access_token": "local-access", "refresh_token": "local-refresh",
                "id_token": f"local.{payload}.local", "expires_in": 3600}

    @app.get("/upstream/codex/models")
    async def codex_models():
        return {"models": [{"slug": name, "context_window": 8192,
                            "input_modalities": ["text"], "supported_in_api": True,
                            "use_responses_lite": lite}
                           for name, lite in (("local-codex", True), ("local-codex-standard", False))]}

    @app.post("/upstream/codex/responses")
    async def codex_response(request: Request):
        """按上游协议拒绝错误请求，让浏览器覆盖目录配置到保存的完整路径。"""
        body = await request.json()
        if "extra_body" in body:
            return JSONResponse({"detail": "Unsupported parameter: extra_body"}, status_code=400)
        if request.headers.get("x-openai-internal-codex-responses-lite") == "true":
            if body.get("reasoning", {}).get("context") != "all_turns":
                return JSONResponse({"detail": "Responses Lite requires reasoning.context=all_turns"}, status_code=400)
        events = ({"type": "response.output_text.done", "text": "OK"},
                  {"type": "response.completed", "response": {}})
        return StreamingResponse(iter("data: " + json.dumps(event) + "\n\n" for event in events),
                                 media_type="text/event-stream")

    @app.get("/upstream/opencode/v1/models")
    async def opencode_models():
        return {"data": [{"id": "local-opencode"}]}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frontend", type=Path, default=ROOT / "frontend")
    parser.add_argument("--connection-edit", action="store_true")
    args = parser.parse_args()
    uvicorn.run(create_app(args.port, args.frontend, connection_edit=args.connection_edit), host="127.0.0.1", port=args.port)
