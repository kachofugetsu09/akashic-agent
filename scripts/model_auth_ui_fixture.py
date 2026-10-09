"""用真实 Models owner、驱动与 HTTP 路由提供一次性浏览器登录夹具。"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition import CompositionRoot, MODEL_CATALOG, MODEL_DRIVERS
from plugins.codex import auth as codex_auth
from plugins.codex.driver import definition as codex_definition
from plugins.models.model_settings_http import BoundModelControl, create_model_settings_router
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.state import ModelsState
from plugins.models.store import ModelsStore
from plugins.opencode_go.driver import definition as opencode_definition

PAGE = """<!doctype html><meta charset="utf-8"><title>模型登录验收</title>
<link rel="stylesheet" href="/frontend/plugins/models/src/style.css">
<style>body{background:#f5f4ee;color:#222;margin:32px} [hidden]{display:none!important}</style>
<main id="host"></main><script type="module">
import {activate as models} from '/frontend/plugins/models/src/module.js';
import {activate as codex} from '/frontend/plugins/codex/src/module.js';
import {activate as opencode} from '/frontend/plugins/opencode_go/src/module.js';
const providers = [];
const providerContext = {ui:{inject:(_, mount)=>mount({register(entry){providers.push(entry);return ()=>{};}})}};
codex(providerContext);opencode(providerContext);
models({ui:{inject:(_, mount)=>mount({register(entry){
  return entry.render(document.querySelector('#host'), {child:()=>({
    entries:providers,
    render(id, host, props){return providers.find(entry=>entry.id===id).render(host, {}, props);},
  })});
}})},http:{request:(path, init)=>fetch(path, init)}});
</script>"""


def create_app(port: int, frontend: Path) -> FastAPI:
    """挂载真实认证实现；只把外部服务替换为本机 HTTP 协议端点。"""
    base = f"http://127.0.0.1:{port}"
    codex_auth.CODEX_AUTH_BASE = base + "/upstream"
    codex_auth.CODEX_API_BASE = base + "/upstream/codex"
    receipts: list[dict[str, object]] = []

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1. 独立目录保存真实 SQLite 与凭据，结束后由各自 owner 关闭。
        with tempfile.TemporaryDirectory(prefix="akashic-auth-ui-") as directory:
            root = CompositionRoot("auth-ui-fixture")

            async def models(ctx):
                store = ModelsStore(Path(directory) / "models.sqlite3", Path(directory) / "backups")
                store.initialize()
                state = ModelsState(store, context=ctx)
                await ctx.effect(lambda: store.close, label="models-store")
                await ctx.provide(MODEL_DRIVERS, state.drivers)
                await ctx.provide(MODEL_SETTINGS, state.settings)
                await ctx.provide(MODEL_CATALOG, state.catalog)

            async def drivers(ctx):
                await ctx.require(MODEL_DRIVERS).register(ctx, codex_definition())
                await ctx.require(MODEL_DRIVERS).register(ctx, opencode_definition())

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

    @app.middleware("http")
    async def record_commands(request: Request, call_next):
        if request.url.path == "/api/dashboard/models/command":
            body = await request.json()
            # 2. 只保存身份、参数键与结果，不保存 API Key 或令牌。
            response = await call_next(request)
            receipts.append({
                "type": body["type"], "connection_id": body.get("connection_id"),
                "driver_id": body.get("driver_id"),
                "input_keys": sorted(body.get("input", {})), "status": response.status_code,
            })
            return response
        return await call_next(request)

    @app.get("/")
    async def page():
        return HTMLResponse(PAGE)

    @app.get("/receipts")
    async def read_receipts():
        return receipts

    @app.post("/upstream/api/accounts/deviceauth/usercode")
    async def device_code(request: Request):
        assert (await request.json())["client_id"] == codex_auth.CODEX_CLIENT_ID
        return {"device_auth_id": "local-device", "user_code": "LOCAL-CODE", "interval": 3}

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
        return {"models": [{"slug": "local-codex", "context_window": 8192,
                            "input_modalities": ["text"], "supported_in_api": True}]}

    @app.get("/upstream/opencode/v1/models")
    async def opencode_models():
        return {"data": [{"id": "local-opencode"}]}

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frontend", type=Path, default=ROOT / "frontend")
    args = parser.parse_args()
    uvicorn.run(create_app(args.port, args.frontend), host="127.0.0.1", port=args.port)
