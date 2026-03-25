from __future__ import annotations

from agent.tools.filesystem import EditFileTool, WriteFileTool
from agent.tools.list_tools import ListToolsTool
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
    tools.register(ToolSearchTool(tools), always_on=True, tags=["meta"], risk="read-only")
    tools.register(ListToolsTool(tools), always_on=True, tags=["meta"], risk="read-only")
    tools.register(
        ShellTool(),
        always_on=True,
        tags=["system"],
        risk="external-side-effect",
        search_keywords=["终端", "命令", "bash", "运行命令", "执行脚本", "shell"],
    )
    tools.register(
        readonly_tools["web_search"],
        always_on=True,
        tags=["web"],
        risk="read-only",
        search_keywords=["搜索", "网络搜索", "谷歌", "bing", "查资料"],
    )
    tools.register(
        readonly_tools["web_fetch"],
        always_on=True,
        tags=["web"],
        risk="read-only",
        search_keywords=["网页", "抓取网页", "读取网址", "fetch", "浏览网页"],
    )
    tools.register(
        readonly_tools["read_file"],
        always_on=True,
        tags=["filesystem"],
        risk="read-only",
        search_keywords=["读文件", "查看文件", "文件内容", "read"],
    )
    tools.register(
        readonly_tools["list_dir"],
        always_on=True,
        tags=["filesystem"],
        risk="read-only",
        search_keywords=["查看目录", "列出文件", "ls", "目录内容", "浏览目录", "dir"],
    )
    tools.register(
        FetchMessagesTool(session_store),
        always_on=True,
        tags=["memory", "session"],
        risk="read-only",
        search_keywords=["消息回溯", "按ID查消息", "fetch messages", "source_ref"],
    )
    tools.register(
        SearchMessagesTool(session_store),
        always_on=True,
        tags=["memory", "session"],
        risk="read-only",
        search_keywords=["搜索历史消息", "全文检索消息", "search messages", "原始对话", "你之前说", "聊过什么", "具体内容"],
    )
    resolved_push_tool = push_tool or MessagePushTool()
    tools.register(
        resolved_push_tool,
        always_on=True,
        tags=["message"],
        risk="external-side-effect",
        search_keywords=["推送消息", "发送消息", "通知用户", "给用户发消息", "push"],
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
            tags=["memory"],
            risk="write",
            search_keywords=["记忆", "存储知识", "记录信息", "备忘", "memorize"],
        )
    if write_file_tool is not None:
        tools.register(
            write_file_tool,
            always_on=True,
            tags=["filesystem", "memory"],
            risk="write",
            search_keywords=["写文件", "保存文件", "创建文件", "写入文件", "新建文件"],
        )
    if edit_file_tool is not None:
        tools.register(
            edit_file_tool,
            always_on=True,
            tags=["filesystem", "memory"],
            risk="write",
            search_keywords=["编辑文件", "修改文件", "更新文件", "patch文件"],
        )
