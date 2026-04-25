"""
Baseline tests for the core-decoupling refactor.

Tests are written against BEHAVIOR contracts, not implementation details:
- Do NOT assert on ContextVar names or import paths.
- DO assert on what callers observe.

These tests must stay green throughout the refactor, including while
compatibility shims are in place.

Covers:
  Item 1 — tool_search exclusion behavior
    Currently implemented via _excluded_names_ctx ContextVar in tool_search.py.
    After refactoring: same exclusion, achieved through explicit interface.

  Item 2a — session metadata update behavior
    Currently implemented via _update_session_runtime_metadata in
    agent/looping/memory_gate.py.
    After refactoring: same function, lives in agent/core/.

  Item 2b — history type conversion
    Canonical types live in agent/core/types.py.
"""

from __future__ import annotations

import inspect
import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock


# ── Item 1: tool_search exclusion behavior ────────────────────────────────────


def _make_tool_search(search_results: list[dict], doc_results: list[dict] | None = None):
    """Helper: build a ToolSearchTool backed by a minimal registry stub."""
    from agent.tools.tool_search import ToolSearchTool

    registry = MagicMock()
    registry.search.return_value = search_results
    registry.get_schemas_as_doc_results.return_value = doc_results or []
    registry.has_tool.side_effect = lambda name: name in {r["name"] for r in (doc_results or [])}
    registry._documents = {}
    return ToolSearchTool(registry)


@pytest.mark.asyncio
async def test_tool_search_keyword_no_exclusion():
    """Keyword search with no visible-names exclusion returns all matches."""
    tool = _make_tool_search([{"name": "shell", "summary": "run shell"}])
    result = json.loads(await tool.execute(query="shell"))
    names = [m["name"] for m in result["matched"]]
    assert "shell" in names


@pytest.mark.asyncio
async def test_tool_search_keyword_excludes_visible_names():
    """Keyword search excludes tools that are already visible to the LLM.

    The Reasoner calls set_excluded_names() before dispatching tool_search.
    """
    from agent.tools.tool_search import ToolSearchTool

    registry = MagicMock()
    registry.search.side_effect = lambda query, top_k, allowed_risk, excluded_names: (
        [] if excluded_names and "shell" in excluded_names
        else [{"name": "shell", "summary": "run shell"}]
    )
    tool = ToolSearchTool(registry)

    # Without exclusion — shell appears
    raw_no_excl = await tool.execute(query="shell")
    assert any(m["name"] == "shell" for m in json.loads(raw_no_excl)["matched"])

    # With exclusion via set_excluded_names — shell must NOT appear
    tool.set_excluded_names({"shell"})
    raw_excl = await tool.execute(query="shell")
    assert not any(m["name"] == "shell" for m in json.loads(raw_excl)["matched"]), (
        "Already-visible tools must be excluded from tool_search results"
    )


@pytest.mark.asyncio
async def test_tool_search_select_excluded_reports_already_loaded():
    """select: path reports already-visible tools as 'already loaded', not as matched."""
    from agent.tools.tool_search import ToolSearchTool

    registry = MagicMock()
    registry.has_tool.return_value = True
    registry.get_schemas_as_doc_results.return_value = []
    registry._documents = {}
    tool = ToolSearchTool(registry)

    tool.set_excluded_names({"shell"})
    raw = await tool.execute(query="select:shell")
    result = json.loads(raw)
    assert result["matched"] == []
    assert "shell" in result.get("tip", ""), (
        "select: on an already-visible tool must produce a tip, not a match entry"
    )


@pytest.mark.asyncio
async def test_tool_search_select_not_excluded_returns_match():
    """select: path returns tool schemas when tool is NOT already visible."""
    from agent.tools.tool_search import ToolSearchTool

    registry = MagicMock()
    registry.has_tool.return_value = True
    registry.get_schemas_as_doc_results.return_value = [
        {"name": "web_search", "summary": "search the web"}
    ]
    registry._documents = {"web_search": MagicMock(risk="read-only")}
    tool = ToolSearchTool(registry)

    tool.set_excluded_names({"shell"})  # shell excluded, not web_search
    raw = await tool.execute(query="select:web_search")
    assert any(m["name"] == "web_search" for m in json.loads(raw)["matched"])


# ── Item 3: _unlock_from_tool_search behavior ────────────────────────────────
#
# Canonical behavior: parse a tool_search JSON result and return the set of
# tool names in "matched". After refactoring this logic lives in
# ToolDiscoveryState.unlock_from_result(); the shim (_unlock_from_tool_search
# in reasoner.py) is deleted entirely.


def _make_discovery():
    from agent.core.runtime_support import ToolDiscoveryState
    return ToolDiscoveryState()


def test_unlock_extracts_matched_names():
    """Names in 'matched' list are returned as a set."""
    d = _make_discovery()
    result = d.unlock_from_result(
        '{"matched": [{"name": "shell"}, {"name": "web_search"}]}'
    )
    assert result == {"shell", "web_search"}


def test_unlock_empty_matched():
    """Empty matched list → empty set."""
    d = _make_discovery()
    assert d.unlock_from_result('{"matched": []}') == set()


def test_unlock_invalid_json_is_silent():
    """Invalid JSON must not raise; returns empty set."""
    d = _make_discovery()
    assert d.unlock_from_result("not-json") == set()


def test_unlock_item_missing_name_skipped():
    """Items without a 'name' key are silently skipped."""
    d = _make_discovery()
    assert d.unlock_from_result('{"matched": [{"summary": "no name here"}]}') == set()


def test_unlock_blank_name_skipped():
    """Items with empty string name are silently skipped."""
    d = _make_discovery()
    assert d.unlock_from_result('{"matched": [{"name": ""}]}') == set()


# ── Item 2a: session metadata update behavior ─────────────────────────────────


def _make_session(metadata: dict | None = None) -> SimpleNamespace:
    s = SimpleNamespace()
    s.metadata = metadata or {}
    return s


def test_session_metadata_update_counts_tool_calls():
    """_update_session_runtime_metadata records total tool call count for the turn."""
    from agent.looping.memory_gate import _update_session_runtime_metadata

    session = _make_session()
    tool_chain = [
        {"calls": [{"name": "shell"}, {"name": "web_search"}]},
        {"calls": [{"name": "read_file"}]},
    ]
    _update_session_runtime_metadata(session, tools_used=["shell", "web_search", "read_file"], tool_chain=tool_chain)
    assert session.metadata["last_turn_tool_calls_count"] == 3


def test_session_metadata_update_has_no_task_tools_after_update_now_removal():
    """_update_session_runtime_metadata no longer tracks removed task tools."""
    from agent.looping.memory_gate import _update_session_runtime_metadata

    session = _make_session()
    _update_session_runtime_metadata(
        session,
        tools_used=["shell"],
        tool_chain=[{"calls": [{"name": "shell"}]}],
    )
    assert session.metadata["last_turn_had_task_tool"] is False
    assert session.metadata["recent_task_tools"] == []


def test_session_metadata_update_no_task_tools():
    """Turn without task tools: last_turn_had_task_tool is False."""
    from agent.looping.memory_gate import _update_session_runtime_metadata

    session = _make_session()
    _update_session_runtime_metadata(
        session,
        tools_used=["shell"],
        tool_chain=[{"calls": [{"name": "shell"}]}],
    )
    assert session.metadata["last_turn_had_task_tool"] is False
    assert session.metadata["recent_task_tools"] == []


def test_session_metadata_update_task_tools_turns_rolling_window():
    """_task_tools_turns keeps only the last 2 turns."""
    from agent.looping.memory_gate import _update_session_runtime_metadata

    session = _make_session()
    _update_session_runtime_metadata(session, tools_used=["shell"], tool_chain=[])
    _update_session_runtime_metadata(session, tools_used=["web_search"], tool_chain=[])
    _update_session_runtime_metadata(session, tools_used=["shell"], tool_chain=[])
    turns = session.metadata["_task_tools_turns"]
    assert len(turns) == 2, "_task_tools_turns must be capped at 2 turns"


def test_session_metadata_update_recent_task_tools_dedup():
    """recent_task_tools flattens and deduplicates across the rolling window."""
    from agent.looping.memory_gate import _update_session_runtime_metadata

    session = _make_session()
    _update_session_runtime_metadata(session, tools_used=["shell"], tool_chain=[])
    _update_session_runtime_metadata(session, tools_used=["shell"], tool_chain=[])
    recent = session.metadata["recent_task_tools"]
    assert recent == []


def test_session_metadata_update_sets_last_turn_ts():
    """last_turn_ts is set to a non-empty ISO timestamp."""
    from agent.looping.memory_gate import _update_session_runtime_metadata

    session = _make_session()
    _update_session_runtime_metadata(session, tools_used=[], tool_chain=[])
    ts = session.metadata.get("last_turn_ts", "")
    assert ts and "T" in ts, "last_turn_ts must be an ISO-format timestamp"


# ── Item 2b: history type conversion ──────────────────────────────────────────


def test_to_tool_call_groups_empty():
    """to_tool_call_groups([]) returns []."""
    from agent.core.types import to_tool_call_groups
    assert to_tool_call_groups([]) == []


def test_to_tool_call_groups_basic():
    """to_tool_call_groups converts raw chain dicts to ToolCallGroup objects."""
    from agent.core.types import to_tool_call_groups, ToolCallGroup, ToolCall

    raw = [
        {
            "text": "I'll call shell",
            "calls": [
                {"call_id": "c1", "name": "shell", "arguments": {"cmd": "ls"}, "result": "ok"},
            ],
        }
    ]
    groups = to_tool_call_groups(raw)
    assert len(groups) == 1
    assert isinstance(groups[0], ToolCallGroup)
    assert groups[0].text == "I'll call shell"
    assert len(groups[0].calls) == 1
    call = groups[0].calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "shell"
    assert call.arguments == {"cmd": "ls"}
    assert call.result == "ok"


def test_to_tool_call_groups_non_dict_arguments_coerced():
    """to_tool_call_groups coerces non-dict arguments to empty dict."""
    from agent.core.types import to_tool_call_groups

    raw = [{"text": "", "calls": [{"call_id": "c1", "name": "x", "arguments": "bad", "result": ""}]}]
    groups = to_tool_call_groups(raw)
    assert groups[0].calls[0].arguments == {}


def test_history_message_fields():
    """HistoryMessage holds role, content, tools_used, tool_chain."""
    from agent.core.types import HistoryMessage

    msg = HistoryMessage(role="user", content="hello", tools_used=["shell"])
    assert msg.role == "user"
    assert msg.content == "hello"
    assert msg.tools_used == ["shell"]
    assert msg.tool_chain == []


def test_turn_types_shim_is_removed():
    """looping.turn_types 已移除，核心类型统一从 agent.core.types 导入。"""
    import importlib.util

    assert importlib.util.find_spec("agent.looping.turn_types") is None


def test_core_boundary_modules_do_not_import_looping_turn_types():
    """Canonical core/retrieval/postturn contracts must not depend on looping.turn_types."""
    from agent.core import context_store
    from agent.postturn import protocol as postturn_protocol
    from agent.retrieval import protocol as retrieval_protocol

    assert "agent.looping.turn_types" not in inspect.getsource(context_store)
    assert "agent.looping.turn_types" not in inspect.getsource(retrieval_protocol)
    assert "agent.looping.turn_types" not in inspect.getsource(postturn_protocol)
