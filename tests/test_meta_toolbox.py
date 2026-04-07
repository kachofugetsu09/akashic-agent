from agent.tools.base import Tool
from agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from agent.tools.meta import (
    META_TOOLBOX_NAMES,
    build_meta_toolbox_prompt,
    register_common_meta_tools,
    register_memory_meta_tools,
)
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool


class _RecallMemoryToolStub(Tool):
    name = "recall_memory"
    description = "test"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return ""


class _ForgetMemoryToolStub(Tool):
    name = "forget_memory"
    description = "test"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return ""


def test_meta_toolbox_prompt_contains_grouped_overview():
    prompt = build_meta_toolbox_prompt()

    assert "MetaToolBox" in prompt
    assert "[Read]" in prompt
    assert "recall_memory" in prompt
    assert "message_push" in prompt
    assert "write_file" in prompt


def test_register_meta_tool_helpers_mark_expected_tools_always_on():
    tools = ToolRegistry()
    readonly_tools = {
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(requester=object()),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }

    push_tool = register_common_meta_tools(
        tools,
        readonly_tools,
        session_store=object(),
    )
    register_memory_meta_tools(
        tools,
        forget_tool=_ForgetMemoryToolStub(),
        recall_tool=_RecallMemoryToolStub(),
        write_file_tool=WriteFileTool(),
        edit_file_tool=EditFileTool(),
    )

    always_on = tools.get_always_on_names()
    assert isinstance(push_tool, MessagePushTool)
    assert set(META_TOOLBOX_NAMES) - {"memorize"} <= always_on
