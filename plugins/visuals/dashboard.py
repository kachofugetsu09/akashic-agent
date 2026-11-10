"""只提供隔离文档的静态入口，不读取附件或业务状态。"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from agent.plugin_composition import DashboardContext, ServiceKey
from plugins.ui.contract import UI, Configuration

SETTINGS = ServiceKey[Configuration]("visuals.settings.v1")
inject = (UI, SETTINGS)


def register(app: FastAPI, context: DashboardContext) -> None:
    context.require(UI).register_configuration(app, context, SETTINGS, "/api/dashboard/visuals/config")
    register_frame(app, context)


def register_frame(app: FastAPI, context: DashboardContext) -> None:
    """由插件 UI 生命周期发布静态入口，响应自身强制脚本沙箱。"""
    document = (context.plugin_dir / "bootstrap.html").read_text(encoding="utf-8")

    @app.get("/visuals/frame.html")
    async def frame() -> HTMLResponse:
        return HTMLResponse(document, headers={
            "Cache-Control": "no-cache",
            "Content-Security-Policy": (
                "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; "
                "style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; "
                "frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; "
                "frame-ancestors 'self'"
            ),
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        })
