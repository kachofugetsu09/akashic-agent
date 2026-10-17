"""网站快照的静态入口：只交付隔离文档的引导页，不读取附件或业务状态。"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from agent.plugin_composition import DashboardContext

# 不给 allow-same-origin：页面是 opaque origin，拿不到 akashic 的 cookie、存储与父窗口。
# 外部 https 资源（CDN、字体、接口）放行；快照里内联的资源都是 data URI。
POLICY = (
    "sandbox allow-scripts allow-forms allow-modals allow-popups; default-src 'none'; "
    "script-src 'unsafe-inline' 'unsafe-eval' https: data: blob:; style-src 'unsafe-inline' https: data:; "
    "img-src https: data: blob:; font-src https: data:; media-src https: data: blob:; "
    "connect-src https:; worker-src blob: data:; frame-src https:; object-src 'none'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
)


def register(app: FastAPI, context: DashboardContext) -> None:
    document = (context.plugin_dir / "frame.html").read_text(encoding="utf-8")

    @app.get("/sites/frame.html")
    async def frame() -> HTMLResponse:
        return HTMLResponse(document, headers={
            "Cache-Control": "no-cache", "Content-Security-Policy": POLICY,
            "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff",
        })
