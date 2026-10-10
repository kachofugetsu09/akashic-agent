"""Bridge 状态路由由执行插件注册，UI 只负责传输与请求许可。"""
from __future__ import annotations

from fastapi import FastAPI
from agent.plugin_composition.requests import RequestContext
from plugins.host_execution.contract import HOST_STATUS

inject = (HOST_STATUS,)


def register(app: FastAPI, context: RequestContext) -> None:
    @app.get("/api/runtime/host-bridge")
    async def status() -> dict[str, object]:
        return context.require(HOST_STATUS).snapshot()
