from __future__ import annotations

from typing import cast

from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.forget_memory import ForgetMemoryTool
from agent.tools.memorize import MemorizeTool
from agent.tools.message_lookup import FetchMessagesTool, SearchMessagesTool
from agent.tools.recall_memory import RecallMemoryTool
from agent.tools.message_push import MessagePushTool
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from agent.tools.shell import ShellTool, ShellTaskOutputTool, ShellTaskStopTool
from agent.tools.tool_search import ToolSearchTool

_MEMORY_TOOL_NAMES = {"recall_memory", "memorize", "forget_memory"}


def register_common_meta_tools(
    tools: ToolRegistry,
    readonly_tools: dict[str, Tool],
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
        ShellTaskOutputTool(),
        always_on=True,
        risk="read-only",
        search_hint="后台任务输出 task_output 进程日志",
    )
    tools.register(
        ShellTaskStopTool(),
        always_on=True,
        risk="external-side-effect",
        search_hint="停止后台任务 task_stop 杀进程",
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


def _register_memory_tool(
    tools: ToolRegistry,
    tool: Tool,
    *,
    risk: str,
    search_hint: str | None = None,
) -> None:
    if tool.name not in _MEMORY_TOOL_NAMES:
        raise ValueError(f"未知 memory 工具: {tool.name}")
    if tools.has_tool(tool.name):
        raise ValueError(f"memory 工具重复注册: {tool.name}")
    tools.register(
        tool,
        always_on=True,
        risk=risk,
        search_hint=search_hint,
    )


def register_memory_meta_tools(
    tools: ToolRegistry,
    memorize_tool: MemorizeTool | Tool | None = None,
    forget_tool: ForgetMemoryTool | Tool | None = None,
    recall_tool: RecallMemoryTool | Tool | None = None,
) -> None:
    if memorize_tool is not None:
        _register_memory_tool(
            tools,
            memorize_tool,
            risk="write",
        )
    if forget_tool is not None:
        _register_memory_tool(
            tools,
            forget_tool,
            risk="write",
            search_hint="记错了 删除记忆 撤销错误记忆 失效记忆",
        )
    if recall_tool is not None:
        _register_memory_tool(
            tools,
            recall_tool,
            risk="read-only",
            search_hint="记得 以前 历史 做过什么 有没有 重构 记忆查询",
        )
