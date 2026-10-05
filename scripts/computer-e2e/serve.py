"""把真实 Computer Dashboard adapter 接到隔离容器，挂载生产 UI 模块。"""
from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
import uvicorn

from agent.plugin_composition import DashboardContext
from plugins.computer.dashboard import register

ROOT = Path(__file__).resolve().parents[2]
TEMP = Path(os.environ["COMPUTER_E2E_ARTIFACTS"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    client.close()


app = FastAPI(lifespan=lifespan)
client = register(app, DashboardContext(
    plugin_id="computer", plugin_dir=ROOT / "plugins/computer", data_root=TEMP,
    validation=False, _workload_urls={
        ("computer", "gateway"): os.environ["COMPUTER_E2E_GATEWAY"],
        ("computer", "stream"): os.environ["COMPUTER_E2E_STREAM"],
    },
))


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


@app.get("/chat")
def chat():
    # 这里只提供聊天占位入口；不伪造 Computer 或网络回执。
    return HTMLResponse('<textarea aria-label="对话输入">继续当前任务</textarea>')


@app.get("/modules/{plugin}/{name}")
def module(plugin: str, name: str):
    if plugin not in {"computer", "conversation_ui", "shell_ui"} or name not in {"web_module.js", "web_module.css"}:
        return PlainTextResponse("Unknown module", status_code=404)
    return FileResponse(ROOT / "plugins" / plugin / name)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["COMPUTER_E2E_PORT"]))
