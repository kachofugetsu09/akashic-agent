"""Session-scoped subagent lineage, attention banner, and external activity projection."""
from __future__ import annotations

import asyncio
from agent.plugin_composition import Context
from plugins.jobs.contract import JOB_NOTICES
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest
from .runners import Runners
from .runtime import SubagentBusy, Subagents


async def register_ui(ctx: Context, runners: Runners, jobs: Subagents) -> None:
    async def query(method: str, payload: dict[str, object], *, session_id: str | None,
                    turn_id: str | None):
        if not session_id:
            raise PluginUiRpcInvalidRequest("需要会话")
        if method == "runners.list" and not payload:
            await jobs.ensure_index()
            return {
                "runs": runners.records(session_id),
                "subagents": jobs.session_tree(session_id),
                "catalog": runners.catalog(),
            }
        if method == "runners.stop" and set(payload) == {"id"}:
            identity = payload["id"]
            if not isinstance(identity, str) or runners.receipt(session_id, identity) is None:
                raise PluginUiRpcInvalidRequest("任务不属于本会话")
            return {"accepted": await ctx.require(JOB_NOTICES).stop_job(session_id, identity)}
        if method == "subagent.cancel" and set(payload) == {"job_id"}:
            job_id = payload["job_id"]
            if not isinstance(job_id, str) or not job_id:
                raise PluginUiRpcInvalidRequest("缺少 job_id")
            await jobs.ensure_index()
            root_id, _ = jobs.resolve_lineage(session_id)
            if not any(item.get("job_id") == job_id and item.get("root_session_id") == root_id for item in jobs.session_tree(session_id, limit=100)):
                raise PluginUiRpcInvalidRequest("子任务不属于本会话")
            return {"accepted": await jobs.cancel(job_id)}
        if method == "subagent.read" and {"job_id"}.issubset(payload):
            job_id = payload["job_id"]
            after_seq = payload.get("after_seq", -1)
            if not isinstance(job_id, str) or not isinstance(after_seq, int):
                raise PluginUiRpcInvalidRequest("参数无效")
            await jobs.ensure_index()
            root_id, _ = jobs.resolve_lineage(session_id)
            try:
                return await jobs.read_child_messages(root_id, job_id, after_seq=after_seq)
            except (KeyError, PermissionError) as error:
                raise PluginUiRpcInvalidRequest(str(error)) from None
        if method == "subagent.send" and set(payload) == {"job_id", "message"}:
            job_id, msg = payload["job_id"], payload["message"]
            if not isinstance(job_id, str) or not isinstance(msg, str) or not msg.strip():
                raise PluginUiRpcInvalidRequest("发送参数无效")
            await jobs.ensure_index()
            root_id, _ = jobs.resolve_lineage(session_id)
            try:
                return await jobs.send(root_id, job_id, msg, background=True)
            except (KeyError, PermissionError, SubagentBusy) as error:
                raise PluginUiRpcInvalidRequest(str(error)) from None
        raise PluginUiRpcInvalidRequest("未知任务操作")

    async def follow(session_id: str):
        previous, seen = None, (-1, -1)
        while True:
            rev = (runners.revisions.get(session_id, 0), jobs.revision)
            if rev != seen:
                seen = rev
                await jobs.ensure_index()
                value = {
                    "runs": runners.records(session_id),
                    "subagents": jobs.session_tree(session_id),
                }
                if value != previous:
                    yield value
                    previous = value
            await asyncio.sleep(0.5)

    await ctx.require(UI_SLOTS).register_plugin_ui(
        ctx,
        PluginUiDefinition(
            module="runner_ui.js",
            stylesheet="runner_ui.css",
            slots=("header.meta", "tool.result", "tool.summary", "chat.notice"),
        ),
        query=query,
        follow=follow,
    )
