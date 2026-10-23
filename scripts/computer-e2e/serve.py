"""把真实 Computer Dashboard adapter 接到隔离容器，挂载生产 UI 模块。"""
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
import uvicorn

from agent.plugin_composition import CredentialRef, DashboardContext
from plugins.computer import stream
from plugins.computer.dashboard import register
from plugins.computer.stream import STREAM_CONFIG, StreamConfig, StreamSettings

ROOT = Path(__file__).resolve().parents[2]
TEMP = Path(os.environ["COMPUTER_E2E_ARTIFACTS"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    client.close()


app = FastAPI(lifespan=lifespan)
# 设了 COMPUTER_E2E_LAN 时画面走 WebRTC，容器需把 UDP 30000 发布到该地址。
# COMPUTER_E2E_TURN=<turn 地址>|<用户名>|<密码> 时由本服务冒充 Cloudflare 凭据接口，指向测试 TURN。
TURN = os.environ.get("COMPUTER_E2E_TURN", "")
stream.TURN_API = f"http://127.0.0.1:{os.environ['COMPUTER_E2E_PORT']}/e2e/turn/{{}}"
stream_config = StreamConfig(lan_address=os.environ.get("COMPUTER_E2E_LAN") or None,
                             turn_key_id="e2e" if TURN else None, turn_api_token=CredentialRef(("e2e",)) if TURN else None)


async def read_token() -> str:
    return "e2e"


stream_settings = StreamSettings(stream_config, read_token)
client = register(app, DashboardContext(
    plugin_id="computer", plugin_dir=ROOT / "plugins/computer", data_root=TEMP,
    validation=False, _workload_urls={
        ("computer", "gateway"): os.environ["COMPUTER_E2E_GATEWAY"],
        ("computer", "stream"): os.environ["COMPUTER_E2E_STREAM"],
    },
    _resolve=lambda key: {STREAM_CONFIG: stream_settings}[key],
))


@app.post("/e2e/turn/{key}")
def turn_credentials(key: str):
    url, username, credential = TURN.split("|")
    host = url.split("?")[0]
    return {"iceServers": {"urls": [host + "?transport=udp", host + "?transport=tcp"], "username": username, "credential": credential}}


@app.get("/")
def page():
    return HTMLResponse((ROOT / "scripts/computer-e2e/host.html").read_text())


@app.get("/shell")
def shell():
    return HTMLResponse((ROOT / "scripts/computer-e2e/shell.html").read_text())


@app.get("/shell.js")
def shell_module():
    return FileResponse(TEMP / "shell.js")


@app.get("/product-band.css")
def product_band():
    return FileResponse(ROOT / "frontend/theme/src/product-band.css")


@app.get("/theme.js")
def theme():
    return PlainTextResponse('export const currentTheme=()=>({id:"paper"}); export const subscribeTheme=()=>()=>{};', media_type="text/javascript")


@app.get("/chat-app.js")
def chat_app():
    # 这里只提供聊天挂载占位（Host SDK @akashic/chat-app-v1）；不伪造 Computer 或网络回执。
    return PlainTextResponse(
        "export function mountChat(host) {"
        " host.innerHTML = '<textarea aria-label=\"对话输入\">继续当前任务</textarea>';"
        " const nothing = () => {};"
        " return { setRailActions: nothing, setTools: nothing, setWorkbenchIntents: nothing, quote: nothing,"
        " shown: nothing, dispose: () => host.replaceChildren() }; }",
        media_type="text/javascript",
    )


@app.get("/modules/{plugin}/{name}")
def module(plugin: str, name: str):
    if plugin not in {"computer", "conversation_ui", "shell_ui"} or name not in {"web_module.js", "web_module.css"}:
        return PlainTextResponse("Unknown module", status_code=404)
    return FileResponse(ROOT / "plugins" / plugin / name)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["COMPUTER_E2E_PORT"]))
