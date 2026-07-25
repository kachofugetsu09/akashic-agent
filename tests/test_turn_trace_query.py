from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.core.runtime_support import ToolDiscoveryState
from agent.tracing.turn_trace_query import TurnTraceQueryService
from plugins.observe.db import open_db


def _make_observe_db(path: Path) -> None:
    conn = open_db(path)
    conn.close()


def _insert_turn(
    path: Path,
    *,
    session_key: str,
    user_msg: str,
    tool_chain: list[dict],
    tool_calls: list[dict] | None = None,
    react_iteration_count: int = 1,
) -> int:
    conn = sqlite3.connect(path)
    cur = conn.execute(
        """
        INSERT INTO turns (
            ts, source, session_key, user_msg, llm_output,
            tool_calls, tool_chain_json, react_iteration_count, error
        )
        VALUES (?, 'agent', ?, ?, 'ok', ?, ?, ?, NULL)
        """,
        (
            "2026-07-13T11:11:03+00:00",
            session_key,
            user_msg,
            json.dumps(tool_calls or [], ensure_ascii=False),
            json.dumps(tool_chain, ensure_ascii=False),
            react_iteration_count,
        ),
    )
    conn.commit()
    assert cur.lastrowid is not None
    turn_id = int(cur.lastrowid)
    conn.close()
    return turn_id


def test_get_turn_returns_real_tool_summary(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    turn_id = _insert_turn(
        db,
        session_key="cli:s1",
        user_msg="根据项目文档和源码回答",
        tool_chain=[
            {"text": "", "calls": [{"name": "read_file", "status": "success", "result": "a"}]},
            {"text": "", "calls": [{"name": "read_file", "status": "success", "result": "b"}]},
            {"text": "", "calls": [{"name": "read_file", "status": "success", "result": "c"}]},
        ],
        react_iteration_count=4,
    )

    result = TurnTraceQueryService(db).get_turn("cli:s1", turn_id)

    assert result.ok is True
    assert result.turn is not None
    assert result.turn.id == turn_id
    assert result.turn.real_tool_counts == {"read_file": 3}
    assert [tool.name for tool in result.turn.tools] == ["read_file", "read_file", "read_file"]
    assert all(tool.real_executed for tool in result.turn.tools)


def test_get_turn_is_current_session_only(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    turn_id = _insert_turn(
        db,
        session_key="cli:other",
        user_msg="private",
        tool_chain=[],
    )

    result = TurnTraceQueryService(db).get_turn("cli:current", turn_id)

    assert result.ok is False
    assert result.error_code == "turn_not_found"


def test_recent_nth_completed_selector_uses_current_session_reverse_order(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    first = _insert_turn(db, session_key="cli:s1", user_msg="first", tool_chain=[])
    second = _insert_turn(db, session_key="cli:s1", user_msg="second", tool_chain=[])
    _insert_turn(db, session_key="cli:other", user_msg="other", tool_chain=[])

    result = TurnTraceQueryService(db).resolve("cli:s1", selector="recent_nth_completed", n=2)

    assert result.ok is True
    assert result.turn is not None
    assert result.turn.id == first
    assert result.turn.id != second


def test_nth_user_question_in_window_uses_chronological_recent_window(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    first = _insert_turn(db, session_key="cli:s1", user_msg="first", tool_chain=[])
    second = _insert_turn(db, session_key="cli:s1", user_msg="second", tool_chain=[])
    _insert_turn(db, session_key="cli:s1", user_msg="third", tool_chain=[])

    result = TurnTraceQueryService(db).resolve("cli:s1", selector="nth_user_question_in_window", n=2)

    assert result.ok is True
    assert result.turn is not None
    assert result.turn.id == second
    assert result.turn.id != first


def test_nth_user_question_in_window_returns_candidates_when_outside_recent_window(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    _insert_turn(db, session_key="cli:s1", user_msg="only one", tool_chain=[])

    result = TurnTraceQueryService(db).resolve("cli:s1", selector="nth_user_question_in_window", n=2)

    assert result.ok is False
    assert result.error_code == "ambiguous_selector"
    assert result.candidates


def test_batch_skipped_tool_is_not_real_executed(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    turn_id = _insert_turn(
        db,
        session_key="cli:s1",
        user_msg="doc evidence",
        tool_chain=[
            {
                "text": "",
                "calls": [
                    {"name": "search_docs", "status": "success", "result": "{}"},
                    {"name": "fetch_doc_chunk", "status": "success", "result": "{}"},
                    {
                        "name": "fetch_doc_chunk",
                        "status": "batch_skipped_by_react_boundary",
                        "result": '{"ok": false, "error_code": "react_boundary_batch_skip"}',
                    },
                ],
            }
        ],
    )

    result = TurnTraceQueryService(db).get_turn("cli:s1", turn_id)

    assert result.ok is True
    assert result.turn is not None
    assert result.turn.real_tool_counts == {"search_docs": 1, "fetch_doc_chunk": 1}
    assert result.turn.skipped_tool_counts == {"fetch_doc_chunk": 1}


def test_access_blocked_tool_is_not_real_executed_even_without_status(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    turn_id = _insert_turn(
        db,
        session_key="cli:s1",
        user_msg="doc question tried local file",
        tool_chain=[
            {
                "text": "",
                "calls": [
                    {
                        "name": "read_file",
                        "result": '{"ok": false, "error_code": "blocked_by_tool_access_gateway"}',
                    },
                ],
            }
        ],
    )

    result = TurnTraceQueryService(db).get_turn("cli:s1", turn_id)

    assert result.ok is True
    assert result.turn is not None
    assert result.turn.real_tool_counts == {}
    assert result.turn.skipped_tool_counts == {"read_file": 1}


def test_runtime_boundary_blocked_tool_is_not_real_executed(tmp_path: Path) -> None:
    db = tmp_path / "observe.db"
    _make_observe_db(db)
    turn_id = _insert_turn(
        db,
        session_key="cli:s1",
        user_msg="doc question tried local file after gateway",
        tool_chain=[
            {
                "text": "",
                "calls": [
                    {
                        "name": "read_file",
                        "status": "blocked_by_tool_boundary",
                        "boundary_action": "block",
                        "boundary_reason": "tool_blocked_by_doc_rag_policy",
                        "result": (
                            '{"ok": false, '
                            '"error_code": "tool_blocked_by_doc_rag_policy"}'
                        ),
                    },
                ],
            }
        ],
    )

    result = TurnTraceQueryService(db).get_turn("cli:s1", turn_id)

    assert result.ok is True
    assert result.turn is not None
    assert result.turn.real_tool_counts == {}
    assert result.turn.skipped_tool_counts == {"read_file": 1}
    tool = result.turn.tools[0]
    assert tool.status == "blocked_by_tool_boundary"
    assert tool.error_code == "tool_blocked_by_doc_rag_policy"


def test_inspect_turn_trace_is_not_added_to_lru() -> None:
    state = ToolDiscoveryState()

    state.update(
        "cli:s1",
        ["inspect_turn_trace", "read_file"],
        always_on=set(),
    )

    assert state.get_preloaded("cli:s1") == {"read_file"}
