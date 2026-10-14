"""只向运维客户端提供 HTTP；读取和剖析控制分别登记。"""
from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition.requests import RequestContext
from plugins.runtime_diagnostics.contract import DIAGNOSTICS
from .profiling import PROFILES, ProfileBusy

inject = (DIAGNOSTICS, PROFILES)


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    seconds: int = Field(default=30, ge=10, le=120)


def reject_browser_request(request: Request) -> None:
    # 无前端模块不再有模块身份许可，显式拒绝浏览器跨站表单等直接请求。
    if "sec-fetch-site" in request.headers:
        raise HTTPException(403, "诊断接口仅供运维客户端访问")


def register(app: FastAPI, context: RequestContext) -> None:
    """不接受文件路径、日志内容或任意执行参数。"""
    # 宿主只接收 APIRoute；新版 FastAPI 的嵌套路由不属于这个合同。
    browser_guard = [Depends(reject_browser_request)]

    @app.get("/api/dashboard/runtime-diagnostics", dependencies=browser_guard)
    async def read(after: int = Query(default=0, ge=0)) -> JSONResponse:
        report = context.require(DIAGNOSTICS).read(after=after)
        report["profile"] = context.require(PROFILES).status()
        return JSONResponse(report, headers={"Cache-Control": "no-store"})

    @app.get("/api/dashboard/runtime-diagnostics/export", dependencies=browser_guard)
    async def export() -> JSONResponse:
        report = context.require(DIAGNOSTICS).read()
        report["profile"] = context.require(PROFILES).status()
        return JSONResponse(report, headers={"Cache-Control": "no-store",
                            "Content-Disposition": 'attachment; filename="akashic-diagnostics.json"'})

    @app.post("/api/dashboard/runtime-diagnostics/profile", dependencies=browser_guard)
    async def start(request: ProfileRequest) -> dict[str, object]:
        try:
            return await context.require(PROFILES).start(request.seconds)
        except ProfileBusy as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/dashboard/runtime-diagnostics/profile/stop", dependencies=browser_guard)
    async def stop() -> dict[str, object]:
        return await context.require(PROFILES).stop()
