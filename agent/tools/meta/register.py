from __future__ import annotations

from typing import Any, cast

from agent.tools.base import Tool
from agent.host_bridge.factory import build_shell_process_manager
from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.message_lookup import FetchMessagesTool, SearchMessagesTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.shell import ShellTaskStopTool, ShellTool, ShellWriteStdinTool
from agent.tools.tool_search import ToolSearchTool


def register_common_meta_tools(
    tools: ToolRegistry,
    readonly_tools: dict[str, Tool],
    session_store: Any,
    push_tool: MessagePushTool | None = None,
) -> MessagePushTool:
    shell_manager = build_shell_process_manager()
    tools.register(ToolSearchTool(tools), always_on=True, risk="read-only")
    tools.register(
        ShellTool(shell_manager),
        always_on=True,
        risk="external-side-effect",
        search_hint="终端 脚本 bash 命令",
    )
    tools.register(
        ShellWriteStdinTool(shell_manager),
        always_on=True,
        risk="external-side-effect",
        search_hint="等待命令 增量输出 stdin 交互",
    )
    tools.register(
        ShellTaskStopTool(shell_manager),
        always_on=True,
        risk="external-side-effect",
        search_hint="停止命令 task_stop 终止执行进程组",
    )
    tools.register(
        cast(Tool, readonly_tools["web_search"]),
        always_on=True,
        risk="read-only",
        search_hint="谷歌 Bing 查资料",
    )
    tools.register(
        cast(Tool, readonly_tools["web_fetch"]),
        always_on=True,
        risk="read-only",
        search_hint="读取网址 浏览网页",
    )
    tools.register(
        cast(Tool, readonly_tools["read_file"]),
        always_on=True,
        risk="read-only",
    )
    tools.register(
        cast(Tool, readonly_tools["list_dir"]),
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
    tools.register(
        WriteFileTool(),
        always_on=True,
        risk="write",
    )
    tools.register(
        EditFileTool(),
        always_on=True,
        risk="write",
    )
    return resolved_push_tool
