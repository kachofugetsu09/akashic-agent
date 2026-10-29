"""Session-scoped external activity projection; never added to model context."""
from __future__ import annotations

import asyncio
from agent.plugin_composition import Context
from plugins.jobs.contract import JOB_NOTICES
from plugins.ui.contract import UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest
from .runners import Runners


async def register_ui(ctx: Context, runners: Runners) -> None:
    async def query(method: str, payload: dict[str, object], *, session_id: str | None,
                    turn_id: str | None):
        if not session_id:
            raise PluginUiRpcInvalidRequest("需要会话")
        if method == "runners.list" and not payload:
            return {"runs": runners.records(session_id), "catalog": runners.catalog()}
        if method == "runners.stop" and set(payload) == {"id"}:
            identity = payload["id"]
            if not isinstance(identity, str) or not any(row["id"] == identity for row in runners.records(session_id)):
                raise PluginUiRpcInvalidRequest("任务不属于本会话")
            return {"accepted": await ctx.require(JOB_NOTICES).stop_job(session_id, identity)}
        raise PluginUiRpcInvalidRequest("未知任务操作")

    async def follow(session_id: str):
        previous = None
        while True:
            value = {"runs": runners.records(session_id)}
            if value != previous:
                yield value
                previous = value
            await asyncio.sleep(0.5)

    await ctx.require(UI_SLOTS).register_plugin_ui(ctx,
        PluginUiDefinition(module="runner_ui.js", stylesheet="runner_ui.css", slots=("header.end",)),
        query=query, follow=follow)
