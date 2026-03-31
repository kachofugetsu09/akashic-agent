from __future__ import annotations

from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.memorize import MemorizeTool
from agent.tools.message_lookup import FetchMessagesTool, SearchMessagesTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.shell import ShellTool
from agent.tools.tool_search import ToolSearchTool


def register_common_meta_tools(
    tools: ToolRegistry,
    readonly_tools: dict[str, object],
    session_store,
    push_tool: MessagePushTool | None = None,
) -> MessagePushTool:
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(
        ShellTool(),
        always_on=True,
        risk="external-side-effect",
        search_hint="终端 脚本 bash 命令",
    )
    tools.register(
        readonly_tools["web_search"],
        always_on=True,
        risk="read-only",
        search_hint="谷歌 Bing 查资料",
    )
    tools.register(
        readonly_tools["web_fetch"],
        always_on=True,
        risk="read-only",
        search_hint="读取网址 浏览网页",
    )
    tools.register(
        readonly_tools["read_file"],
        always_on=True,
        risk="read-only",
    )
    tools.register(
        readonly_tools["list_dir"],
        always_on=True,
        risk="read-only",
        search_hint="ls 查看目录",
    )
    tools.register(
        FetchMessagesTool(session_store),
        always_on=True,
        risk="read-only",
        search_hint="消息回溯 按ID查对话原文 source_ref",
    )
    tools.register(
        SearchMessagesTool(session_store),
        always_on=True,
        risk="read-only",
        search_hint="你之前说 聊过什么 历史对话",
    )
    resolved_push_tool = push_tool or MessagePushTool()
    tools.register(
        resolved_push_tool,
        always_on=True,
        risk="external-side-effect",
    )
    return resolved_push_tool


def register_memory_meta_tools(
    tools: ToolRegistry,
    memorize_tool: MemorizeTool | None = None,
    write_file_tool: WriteFileTool | None = None,
    edit_file_tool: EditFileTool | None = None,
) -> None:
    if memorize_tool is not None:
        tools.register(
            memorize_tool,
            always_on=True,
            risk="write",
        )
    if write_file_tool is not None:
        tools.register(
            write_file_tool,
            always_on=True,
            risk="write",
        )
    if edit_file_tool is not None:
        tools.register(
            edit_file_tool,
            always_on=True,
            risk="write",
        )
