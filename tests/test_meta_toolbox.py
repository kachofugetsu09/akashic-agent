from typing import Any, cast
import pytest
from agent.tools.base import Tool
from agent.tools.filesystem import ListDirTool, ReadFileTool
from agent.tools.meta.catalog import build_meta_toolbox_prompt
from agent.tools.meta.register import register_common_meta_tools
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolRegistry
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from bootstrap.toolsets.meta import CommonMetaToolsetProvider
from bootstrap.toolsets.protocol import ToolsetDeps
from session.store import SessionStore


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
        "web_fetch": WebFetchTool(requester=cast(Any, object())),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }

    push_tool = register_common_meta_tools(
        tools,
        readonly_tools,
        session_store=object(),
    )
    always_on = tools.get_always_on_names()
    assert isinstance(push_tool, MessagePushTool)
    assert {
        "tool_search",
        "shell",
        "write_stdin",
        "task_stop",
        "web_search",
        "web_fetch",
        "read_file",
        "list_dir",
        "fetch_messages",
        "search_messages",
        "message_push",
        "write_file",
        "edit_file",
    } <= always_on
    assert "request_user_confirmation" not in always_on


def test_common_meta_toolset_registers_load_skill(tmp_path):
    tools = ToolRegistry()
    session_store = SessionStore(tmp_path / "sessions.db")
    readonly_tools = {
        "web_search": WebSearchTool(),
        "web_fetch": WebFetchTool(requester=cast(Any, object())),
        "read_file": ReadFileTool(),
        "list_dir": ListDirTool(),
    }

    result = CommonMetaToolsetProvider(readonly_tools).register(
        tools,
        ToolsetDeps(
            config=None,
            workspace=tmp_path,
            session_store=session_store,
        ),
    )
    session_store.close()

    assert tools.has_tool("load_skill")
    assert "load_skill" in result.always_on_names


def test_common_meta_toolset_rejects_missing_required_dependencies(tmp_path):
    readonly_tools = {
        "web_search": cast(Any, object()),
        "web_fetch": cast(Any, object()),
        "read_file": cast(Any, object()),
        "list_dir": cast(Any, object()),
    }

    with pytest.raises(ValueError, match="session_store"):
        CommonMetaToolsetProvider(readonly_tools).register(
            ToolRegistry(),
            ToolsetDeps(config=None, workspace=tmp_path),
        )

    with pytest.raises(ValueError, match="web_search"):
        CommonMetaToolsetProvider({}).register(
            ToolRegistry(),
            ToolsetDeps(
                config=None,
                workspace=tmp_path,
                session_store=cast(Any, object()),
            ),
        )
