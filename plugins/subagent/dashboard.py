"""Subagent Workbench 与配置 HTTP 路由：暴露会话树血缘、子流读取、Steer/续聊与取消控制。"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import DashboardContext, ServiceKey
from plugins.jobs.contract import JOB_NOTICES
from plugins.ui.contract import SETTINGS_SLOT, UI, Configuration

from .runners import Runners
from .runtime import SubagentBusy, Subagents

SETTINGS = ServiceKey[Configuration]("subagent.settings.v1")
SUBAGENT_UI_SERVICE = ServiceKey[tuple[Subagents, Runners]]("subagent.ui_service.v1")
CONVERSATION_TOOLS_V2_DIGEST = "b3f5e9aea07351d69ab1ae55d2dcf18c5e352fc61b6354d4af34dbeca16787eb"
UI_CONTRACT_DIGESTS = {
    **SETTINGS_SLOT,
    "conversation.tools.v2": CONVERSATION_TOOLS_V2_DIGEST,
}

inject = (UI, SETTINGS, SUBAGENT_UI_SERVICE, JOB_NOTICES)


class SendBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session: str = Field(min_length=1, max_length=512)
    job_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=32000)


class CancelBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session: str = Field(min_length=1, max_length=512)
    job_id: str = Field(min_length=1, max_length=64)


def register(app: FastAPI, context: DashboardContext) -> None:
    context.require(UI).register_configuration(app, context, SETTINGS, "/api/dashboard/subagent/config")

    @app.get("/api/dashboard/subagent/tree")
    async def tree(session: str = Query(min_length=1, max_length=512)) -> dict[str, object]:
        jobs, runners = context.require(SUBAGENT_UI_SERVICE)
        await jobs.ensure_index()
        return {
            "session": session,
            "subagents": jobs.session_tree(session),
            "runs": runners.records(session),
        }

    @app.get("/api/dashboard/subagent/messages")
    async def messages(
        session: str = Query(min_length=1, max_length=512),
        job_id: str = Query(min_length=1, max_length=64),
        after_seq: int = Query(default=-1, ge=-1),
        before_seq: int | None = Query(default=None, ge=0),
        limit: int = Query(default=100, ge=1, le=200),
    ) -> dict[str, object]:
        jobs, _ = context.require(SUBAGENT_UI_SERVICE)
        await jobs.ensure_index()
        root_id, _ = jobs.resolve_lineage(session)
        try:
            return await jobs.read_child_messages(root_id, job_id, after_seq=after_seq, before_seq=before_seq, limit=limit)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from None

    @app.post("/api/dashboard/subagent/send")
    async def send(body: SendBody) -> dict[str, object]:
        jobs, _ = context.require(SUBAGENT_UI_SERVICE)
        await jobs.ensure_index()
        root_id, _ = jobs.resolve_lineage(body.session)
        msg_text = body.message.strip()
        if not msg_text:
            raise HTTPException(status_code=422, detail="消息不能为空")
        try:
            return await jobs.send(root_id, body.job_id, msg_text, background=True)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from None
        except PermissionError as error:
            raise HTTPException(status_code=403, detail=str(error)) from None
        except SubagentBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from None

    @app.post("/api/dashboard/subagent/cancel")
    async def cancel(body: CancelBody) -> dict[str, object]:
        jobs, _ = context.require(SUBAGENT_UI_SERVICE)
        await jobs.ensure_index()
        root_id, _ = jobs.resolve_lineage(body.session)
        items = jobs.session_tree(body.session, limit=100)
        target = next((item for item in items if item.get("job_id") == body.job_id), None)
        if target is None or target.get("root_session_id") != root_id:
            raise HTTPException(status_code=404, detail="子任务不属于当前会话树")
        accepted = await jobs.cancel(body.job_id)
        return {"accepted": accepted}

    @app.post("/api/dashboard/subagent/runner-stop")
    async def runner_stop(body: CancelBody) -> dict[str, object]:
        _, runners = context.require(SUBAGENT_UI_SERVICE)
        if runners.receipt(body.session, body.job_id) is None:
            raise HTTPException(status_code=404, detail="外部任务不属于当前会话")
        accepted = await context.require(JOB_NOTICES).stop_job(body.session, body.job_id)
        return {"accepted": accepted}
