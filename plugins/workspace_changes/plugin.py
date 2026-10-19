"""会话工作目录的改动：按范围（未提交、本分支、单个提交）查看，每轮记下摘要；只读，不改用户的 index、HEAD 与 refs。"""
from typing import cast

from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESSES
from plugins.ledger.contract import MESSAGE_CATALOG, OWNER_STATE
from plugins.sources.contract import SOURCE_CHANGED_V3
from plugins.working_directory.contract import WORKING_DIRECTORY
from plugins.turn_projection.contract import TURN_PROJECTION
from plugins.ui.contract import UI, UI_SLOTS, PluginUiDefinition, PluginUiRpcInvalidRequest

from . import dashboard
from .changes import WORKSPACE_CHANGES, WorkspaceChanges

api_version = 3
name = "workspace_changes"
version = "1.0.0"
desc = "查看会话工作目录（git 仓库）的改动，回复下方给出每轮的摘要"
author = "Akashic Core"
inject = (PROCESSES, WORKING_DIRECTORY, MESSAGE_CATALOG, OWNER_STATE, TURN_PROJECTION)
workspace_roots = ()
workspace_files = ()


async def apply(ctx: Context) -> None:
    changes = WorkspaceChanges(ctx)
    _ = await ctx.provide(WORKSPACE_CHANGES, changes)
    # 插件停用时不采集：窗口任务与监听随本插件的生命周期。
    _ = await ctx.spawn(changes.windows.run(), name="workspace-changes-turns")
    _ = await ctx.on(SOURCE_CHANGED_V3, changes.windows.changed)

    # UI 缺席时只是没有浏览器入口，统计照常进行。
    async def register_ui(child: Context) -> None:
        _ = await child.require(UI).register(
            child, web="web_module.js", dashboard=lambda: dashboard,
            requires=("conversation.tools.v2",),
            contract_digests={
                "conversation.tools.v2": "dcbf5db629d4fa6fed4019ad91663f9af94c5b9a02f9e67c94b0745e40f19049",
            },
        )

    _ = await ctx.inject((UI, WORKSPACE_CHANGES), register_ui, name="workspace-changes-ui")

    # 回复下方的摘要只要计数与第一个文件（点开时定位），不把整份文件列表送进聊天界面。
    async def message_ui(child: Context) -> None:
        async def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
            message_id = payload.get("message_id")
            if method != "turn.message" or set(payload) != {"message_id"} or not session_id \
                    or not isinstance(message_id, str) or not message_id:
                raise PluginUiRpcInvalidRequest("改动摘要需要会话与消息")
            return summarize(await changes.windows.by_message(session_id, message_id))

        _ = await child.require(UI_SLOTS).register_plugin_ui(
            child, PluginUiDefinition(module="message_ui.js", stylesheet="message_ui.css", slots=("turn.after_answer",)),
            query=query,
        )

    _ = await ctx.inject((UI_SLOTS,), message_ui, name="workspace-changes-message-ui")


def summarize(turn: dict[str, object]) -> dict[str, object]:
    files = cast(list[dict[str, object]], turn.get("files") or [])
    return {
        "phase": turn["phase"], "turn_id": turn.get("turn_id"), "error": turn.get("error"),
        "path": files[0]["path"] if files else None,
        "files": len(files), "added": sum(cast(int, item["added"] or 0) for item in files),
        "removed": sum(cast(int, item["removed"] or 0) for item in files),
        "uncertain": bool(turn.get("baseline_late") or turn.get("end_late") or turn.get("truncated")),
    }
