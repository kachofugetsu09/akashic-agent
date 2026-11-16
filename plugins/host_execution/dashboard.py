"""Bridge 状态路由由执行插件注册，UI 只负责传输与请求许可。"""
from __future__ import annotations

from datetime import UTC, datetime
from fastapi import FastAPI
from agent.plugin_composition.requests import RequestContext
from core.common.file_io import run_file_io
from plugins.host_execution.contract import HOST_STATUS
from .deploy_status import read_deploy_snapshot

inject = (HOST_STATUS,)
_STARTED_AT = datetime.now(UTC).isoformat()


def register(app: FastAPI, context: RequestContext) -> None:
    @app.get("/api/runtime/host-bridge")
    async def status() -> dict[str, object]:
        return context.require(HOST_STATUS).snapshot()

    @app.get("/api/dashboard/runtime-version")
    async def runtime_version() -> dict[str, object]:
        snapshot = await run_file_io(
            lambda: read_deploy_snapshot(started_at=_STARTED_AT, history_limit=10)
        )
        return {
            "bridge": context.require(HOST_STATUS).snapshot(),
            **snapshot,
        }
