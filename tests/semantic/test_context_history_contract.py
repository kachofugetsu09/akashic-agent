from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from agent.core.passive_turn import DefaultReasoner
from agent.core.runtime_support import LLMServices, ToolDiscoveryState
from agent.core.types import ContextRenderResult, ContextRequest, ReasonerResult
from agent.looping.ports import LLMConfig
from agent.provider import ContextLengthError
from session.manager import SessionManager
from tests_scenarios.contracts.oracles import (
    assert_no_forbidden_writes,
    assert_rows_unchanged,
)


def _snapshot(
    connection: sqlite3.Connection,
    query: str,
    parameters: tuple[object, ...] = (),
) -> list[tuple[object, ...]]:
    return [tuple(row) for row in connection.execute(query, parameters).fetchall()]


def _seed_embeddings(manager: SessionManager, message_ids: list[str]) -> None:
    connection = manager._store._conn
    connection.execute("""
        CREATE TABLE message_embeddings (
            message_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (message_id, model)
        )
        """)
    connection.executemany(
        """
        INSERT INTO message_embeddings
            (message_id, content_hash, model, embedding, dim, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                message_id,
                f"hash:{message_id}",
                "gate-model",
                f"vector:{message_id}".encode(),
                1,
                "before",
                "before",
            )
            for message_id in message_ids
        ],
    )
    connection.commit()


def _messages_snapshot(
    manager: SessionManager, session_key: str
) -> list[tuple[object, ...]]:
    return _snapshot(
        manager._store._conn,
        """
        SELECT id, session_key, seq, role, content, tool_chain, extra, ts
        FROM messages
        WHERE session_key = ?
        ORDER BY seq
        """,
        (session_key,),
    )


def _embeddings_snapshot(manager: SessionManager) -> list[tuple[object, ...]]:
    return _snapshot(
        manager._store._conn,
        """
        SELECT message_id, content_hash, model, embedding, dim, created_at, updated_at
        FROM message_embeddings
        ORDER BY message_id, model
        """,
    )


def _seed_session(workspace: Path) -> tuple[SessionManager, str]:
    manager = SessionManager(workspace)
    session_key = "semantic:context-retry"
    session = manager.get_or_create(session_key)
    for index in range(6):
        role = "user" if index % 2 == 0 else "assistant"
        session.add_message(role, f"message-{index}")
    manager.save(session)
    _seed_embeddings(
        manager,
        [cast(str, message["id"]) for message in session.messages],
    )
    return manager, session_key


def _reasoner(
    manager: SessionManager,
    history_windows: list[int],
) -> DefaultReasoner:
    def render(request: ContextRequest, **_kwargs: object) -> ContextRenderResult:
        history_windows.append(len(request.history))
        return ContextRenderResult(
            system_prompt="",
            messages=[
                *request.history,
                {"role": "user", "content": request.current_message},
            ],
        )

    tools = SimpleNamespace(
        get_always_on_names=lambda: set(),
        get_deferred_names=lambda visible=None: {"builtin": [], "mcp": {}},
        get_schemas=lambda names=None: [],
        get_tool=lambda name: None,
    )
    return DefaultReasoner(
        llm=cast(
            Any,
            LLMServices(
                provider=SimpleNamespace(chat=AsyncMock()),
                light_provider=SimpleNamespace(),
            ),
        ),
        llm_config=LLMConfig(model="semantic-gate", max_iterations=1, max_tokens=128),
        tools=cast(Any, tools),
        discovery=ToolDiscoveryState(),
        tool_search_enabled=False,
        memory_window=40,
        context=cast(Any, SimpleNamespace(render=render)),
        session_manager=manager,
    )


def _message() -> SimpleNamespace:
    return SimpleNamespace(
        content="continue",
        media=[],
        channel="semantic",
        chat_id="context-retry",
        timestamp=datetime.now(timezone.utc),
        metadata={},
    )


def test_context_retry_is_projection_over_append_only_history(tmp_path: Path) -> None:
    manager, session_key = _seed_session(tmp_path)
    session = manager.get_or_create(session_key)
    before_messages = _messages_snapshot(manager, session_key)
    before_embeddings = _embeddings_snapshot(manager)
    before_highwater = max(cast(int, row[2]) for row in before_messages)
    statements: list[str] = []
    manager._store._conn.set_trace_callback(statements.append)
    windows: list[int] = []
    reasoner = _reasoner(manager, windows)
    reasoner.run = AsyncMock(
        side_effect=[
            ContextLengthError("too long"),
            ContextLengthError("too long"),
            ContextLengthError("too long"),
            ContextLengthError("too long"),
            ContextLengthError("too long"),
            ReasonerResult(reply="ok", metadata={"tools_used": [], "tool_chain": []}),
        ]
    )

    result = asyncio.run(reasoner.run_turn(msg=_message(), session=session))

    assert result.reply == "ok"
    assert windows[:5] == [6, 6, 6, 6, 6]
    assert windows[5] == 3
    assert len(session.messages) == 3
    assert_rows_unchanged(
        before_messages,
        _messages_snapshot(manager, session_key),
        state_name="sessions.db/messages",
    )
    assert_rows_unchanged(
        before_embeddings,
        _embeddings_snapshot(manager),
        state_name="message_embeddings",
    )
    assert_no_forbidden_writes(
        statements,
        tables=("messages", "message_embeddings"),
    )

    manager._store._conn.set_trace_callback(None)
    manager.close()
    reloaded_manager = SessionManager(tmp_path)
    reloaded = reloaded_manager.get_or_create(session_key)
    assert [message["content"] for message in reloaded.messages] == [
        f"message-{index}" for index in range(6)
    ]
    reloaded.add_message("assistant", "message-6")
    reloaded_manager.save(reloaded)
    assert reloaded.messages[-1]["seq"] == before_highwater + 1
    reloaded_manager.close()


def test_history_oracle_rejects_historical_delete_mutant(tmp_path: Path) -> None:
    manager, session_key = _seed_session(tmp_path)
    before_messages = _messages_snapshot(manager, session_key)
    before_embeddings = _embeddings_snapshot(manager)
    statements: list[str] = []
    connection = manager._store._conn
    connection.set_trace_callback(statements.append)

    connection.execute(
        "DELETE FROM message_embeddings WHERE message_id IN (?, ?, ?)",
        tuple(str(row[0]) for row in before_messages[:3]),
    )
    connection.execute(
        "DELETE FROM messages WHERE session_key = ? AND seq < ?",
        (session_key, 3),
    )
    connection.commit()

    with pytest.raises(AssertionError, match="既有行发生删改"):
        assert_rows_unchanged(
            before_messages,
            _messages_snapshot(manager, session_key),
            state_name="sessions.db/messages",
        )
    with pytest.raises(AssertionError, match="既有行发生删改"):
        assert_rows_unchanged(
            before_embeddings,
            _embeddings_snapshot(manager),
            state_name="message_embeddings",
        )
    with pytest.raises(AssertionError, match="受保护状态删改"):
        assert_no_forbidden_writes(
            statements,
            tables=("messages", "message_embeddings"),
        )
    manager.close()
