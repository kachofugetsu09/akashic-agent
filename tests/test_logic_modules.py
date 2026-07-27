from __future__ import annotations
from typing import Any, cast

import asyncio
import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.prompting import is_context_frame
from agent.provider import LLMResponse
from proactive_v2.loop import ProactiveLoop
from proactive_v2.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizerLoop,
)
from session.manager import (
    Session,
    SessionManager,
    _TOOL_RESULT_CHAR_BUDGET,
)
from session.store import SessionStore


def _seed_message_embeddings(store: SessionStore, message_ids: list[str]) -> None:
    store._conn.execute(
        """
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
        """
    )
    store._conn.executemany(
        """
        INSERT INTO message_embeddings
            (message_id, content_hash, model, embedding, dim, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                message_id,
                f"hash:{message_id}",
                "test-model",
                f"embedding:{message_id}".encode(),
                1,
                "before",
                "before",
            )
            for message_id in message_ids
        ],
    )
    store._conn.commit()


@pytest.mark.asyncio
async def test_memory_optimizer_loop_and_memory_port_cover_paths(tmp_path: Path):
    memory = MagicMock()
    memory.snapshot_pending.return_value = "- [identity] x"
    memory.read_long_term.return_value = "MEM"
    memory.read_self.return_value = "# Akashic 的自我认知\n## 人格与形象\n- x"
    memory.get_memory_context.return_value = "ctx"
    memory.write_long_term = MagicMock()
    memory.commit_pending_snapshot = MagicMock()
    memory.rollback_pending_snapshot = MagicMock()
    memory.write_self = MagicMock()
    provider = MagicMock()
    provider.chat = AsyncMock(
        side_effect=[
            LLMResponse(
                content=(
                    "# 用户长期记忆\n\n"
                    "## 用户事实\n- x\n\n"
                    "## 用户偏好\n- y\n\n"
                    "## 用户明确要求长期记住的关键内容\n- z"
                )
            ),
            LLMResponse(
                content=(
                    "# Akashic 的自我认知\n\n"
                    "## 人格与形象\n- x\n\n"
                    "## 我对当前用户的理解\n- y\n\n"
                    "## 我们关系的定义\n- z"
                )
            ),
        ]
    )
    opt = MemoryOptimizer(memory, provider, "m", max_tokens=100)
    opt._STEP_DELAY_SECONDS = 0
    await opt.optimize()
    memory.write_long_term.assert_called_once()
    memory.write_self.assert_called_once()

    loop = MemoryOptimizerLoop(opt, interval_seconds=10, _now_fn=lambda: datetime(2025, 1, 1, 0, 0, 1))
    assert loop._seconds_until_next_tick() >= 1.0
    loop.stop()


@pytest.mark.asyncio
async def test_session_manager_and_proactive_loop_cover_paths(tmp_path: Path):
    session = Session("telegram:1")
    session.add_message("user", "hi", media=["/tmp/a.png"])
    session.add_message(
        "assistant",
        "reply",
        proactive=True,
        state_summary_tag="tag",
        source_refs=[{"source_name": "Feed", "title": "T", "url": "https://x"}],
    )
    session.messages[-1]["tool_chain"] = [
        {"calls": [{"call_id": "1", "name": "tool", "arguments": {}, "result": "ok"}]}
    ]
    history = session.get_history()
    assert len(history) == 3
    assert history[0]["role"] == "user"
    assert history[1] == {"role": "assistant", "content": "[主动推送] reply"}
    assert history[2]["role"] == "user"
    assert is_context_frame(str(history[2]["content"]))
    manager = SessionManager(tmp_path)
    manager.save(session)
    loaded = manager.get_or_create("telegram:1")
    assert loaded.key == "telegram:1"
    await manager.append_messages(session, [{"role": "user", "content": "next"}])
    assert manager.list_sessions()
    assert manager.get_channel_metadata("telegram")[0]["chat_id"] == "1"
    manager.invalidate("telegram:1")

    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._cfg = SimpleNamespace(
        interval_seconds=10,
        score_weight_energy=0.5,
        tick_interval_s1=3,
        tick_interval_s0=4,
        tick_jitter=0.0,
    )
    loop._presence = None
    loop._trace_proactive_rate_decision = MagicMock()
    loop._scheduler = SimpleNamespace(next_interval=lambda base_score: 10)
    assert loop._next_interval() == 10
    loop._presence = SimpleNamespace(
        get_last_user_at=lambda session_key: datetime.now(timezone.utc)
    )
    loop._sense = SimpleNamespace(
        target_session_key=lambda: "telegram:1",
    )
    loop._rng = None
    loop._memory = SimpleNamespace(get_memory_context=lambda: "ctx")
    loop._sessions = SimpleNamespace(workspace=tmp_path)
    (tmp_path / "AGENTS.md").write_text("guide", encoding="utf-8")
    loop._sender = SimpleNamespace(send=AsyncMock(return_value=True))
    loop._engine = SimpleNamespace(tick=AsyncMock(return_value=0.2))


@pytest.mark.parametrize(
    "payload",
    ["{broken", "[]", "", sqlite3.Binary(b"\xff"), sqlite3.Binary(b"")],
)
def test_session_metadata_corruption_fails_at_database_boundary(
    tmp_path: Path,
    payload: object,
) -> None:
    manager = SessionManager(tmp_path)
    session_key = "telegram:broken"
    manager.get_or_create(session_key)
    manager._store._conn.execute(
        "UPDATE sessions SET metadata = ? WHERE key = ?",
        (payload, session_key),
    )
    manager._store._conn.commit()

    with pytest.raises(ValueError, match=session_key):
        manager.get_channel_metadata("telegram")
    with pytest.raises(ValueError, match=session_key):
        manager._store.get_session_meta(session_key)
    with pytest.raises(ValueError, match=session_key):
        manager._store.list_sessions_for_dashboard()


def test_session_manager_rejects_orphan_messages_without_metadata(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    try:
        manager._store.insert_message(
            "telegram:orphan",
            role="user",
            content="孤立消息",
            ts="2026-07-13T00:00:00+00:00",
            seq=0,
        )

        with pytest.raises(ValueError, match="session metadata 缺失"):
            manager.get_or_create("telegram:orphan")
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_session_batch_persistence_rolls_back_all_messages_on_failure(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:atomic")
    session.add_message("user", "第一条")
    session.add_message("assistant", "第二条")
    manager._store._conn.execute(
        """
        CREATE TRIGGER reject_assistant_message
        BEFORE INSERT ON messages
        WHEN NEW.role = 'assistant'
        BEGIN
            SELECT RAISE(ABORT, '测试写入失败');
        END
        """
    )
    manager._store._conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="测试写入失败"):
        await manager.append_messages(session, session.messages)

    assert manager._store.count_messages(session.key) == 0
    assert all("id" not in message for message in session.messages)


@pytest.mark.asyncio
async def test_session_batch_persistence_uses_one_commit(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:batch")
    session.add_message("user", "第一条")
    session.add_message("assistant", "第二条")
    statements: list[str] = []
    manager._store._conn.set_trace_callback(statements.append)

    await manager.append_messages(session, session.messages)

    assert statements.count("COMMIT") == 1


def test_session_manager_preserves_message_extra_payload_order(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("telegram:extra-order")
    session.messages.append(
        {
            "role": "user",
            "custom_first": "先",
            "session_key": "must-skip",
            "content": "正文",
            "custom_second": {"nested": "值"},
            "seq": 999,
            "timestamp": "2026-07-23T00:00:00+00:00",
            "tool_chain": None,
        }
    )

    manager.save(session)

    row = manager._store._conn.execute(
        "SELECT role, content, seq, extra FROM messages WHERE session_key = ?",
        (session.key,),
    ).fetchone()
    assert row is not None
    assert tuple(row[:3]) == ("user", "正文", 0)
    assert row[3] == json.dumps(
        {"custom_first": "先", "custom_second": {"nested": "值"}},
        ensure_ascii=False,
    )


def test_session_manager_preserves_message_field_evaluation_order(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class TraceMessage(dict[str, object]):
        def get(self, key: str, default: object = None) -> object:
            events.append(f"get:{key}")
            return super().get(key, default)

        def items(self):
            events.append("items")
            return super().items()

    manager = SessionManager(tmp_path)
    manager._store.persist_session = lambda *args, **kwargs: []
    fixed = datetime(2026, 7, 23, tzinfo=timezone.utc)
    session = Session("telegram:extra-order", created_at=fixed, updated_at=fixed)
    message = TraceMessage(
        role="user",
        content="正文",
        timestamp=fixed.isoformat(),
        custom="extra",
    )

    manager._persist_session(session, [message], updated_at=fixed)

    assert events == [
        "get:id",
        "get:timestamp",
        "get:content",
        "get:role",
        "get:tool_chain",
        "items",
    ]


def test_session_persistence_allocates_sequences_inside_transaction(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    store_a = SessionStore(db_path)
    store_b = SessionStore(db_path)
    key = "telegram:concurrent"

    def persist(store: SessionStore, content: str) -> list[dict[str, Any]]:
        return store.persist_session(
            key,
            created_at="2026-07-13T00:00:00+00:00",
            updated_at="2026-07-13T00:00:01+00:00",
            last_consolidated=0,
            metadata={},
            messages=[
                {
                    "role": "user",
                    "content": content,
                    "timestamp": "2026-07-13T00:00:01+00:00",
                    "extra": {},
                }
            ],
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(persist, store_a, "来自 A"),
                pool.submit(persist, store_b, "来自 B"),
            ]
            rows = [future.result(timeout=5) for future in futures]
        messages = store_a.fetch_session_messages(key)
    finally:
        store_a.close()
        store_b.close()

    assert {str(row[0]["id"]) for row in rows} == {
        f"{key}:0",
        f"{key}:1",
    }
    assert [str(message["id"]) for message in messages] == [
        f"{key}:0",
        f"{key}:1",
    ]


def test_session_store_reuses_existing_fts_without_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.persist_session(
        "telegram:fts",
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:01+00:00",
        last_consolidated=0,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "全文索引消息",
                "timestamp": "2026-07-13T00:00:01+00:00",
                "extra": {},
            }
        ],
    )
    store.close()

    statements: list[str] = []

    original_ensure_fts = SessionStore._ensure_fts

    def trace_constructor_fts(instance: SessionStore) -> None:
        instance._conn.set_trace_callback(statements.append)
        original_ensure_fts(instance)

    monkeypatch.setattr(SessionStore, "_ensure_fts", trace_constructor_fts)
    reopened = SessionStore(db_path)

    assert reopened._has_fts is True
    assert not any("VALUES('rebuild')" in statement for statement in statements)
    assert reopened.search_messages("全文索引")[1] == 1
    reopened.close()


def test_session_store_rebuilds_fts_when_trigger_is_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.persist_session(
        "telegram:fts-trigger",
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:01+00:00",
        last_consolidated=0,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "触发器缺失后仍要检索",
                "timestamp": "2026-07-13T00:00:01+00:00",
                "extra": {},
            }
        ],
    )
    store._conn.execute("DROP TRIGGER messages_ai")
    store._conn.commit()
    statements: list[str] = []
    store._conn.set_trace_callback(statements.append)

    store._ensure_fts()

    assert any("VALUES('rebuild')" in statement for statement in statements)
    assert store.search_messages("触发器缺失")[1] == 1
    store.close()


def test_session_store_disables_fts_only_when_capability_is_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _MissingFtsConnection:
        def execute(self, sql: str, _params: object = ()) -> object:
            if sql.startswith("SELECT name, sql"):
                return SimpleNamespace(fetchone=lambda: None)
            raise sqlite3.OperationalError("no such module: fts5")

    store = SessionStore.__new__(SessionStore)
    store._conn = _MissingFtsConnection()  # type: ignore[assignment]
    store._has_fts = True
    store._closed = True

    with caplog.at_level(logging.WARNING, logger="session.store"):
        store._ensure_fts()

    assert store._has_fts is False
    assert "FTS5/trigram" in caplog.text


def test_session_store_reraises_non_capability_fts_errors() -> None:
    class _BrokenFtsConnection:
        def execute(self, _sql: str, _params: object = ()) -> object:
            raise sqlite3.OperationalError("database is locked")

    store = SessionStore.__new__(SessionStore)
    store._conn = _BrokenFtsConnection()  # type: ignore[assignment]
    store._has_fts = True
    store._closed = True

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        store._ensure_fts()

    assert store._has_fts is True


@pytest.mark.parametrize(
    ("column", "payload", "field"),
    [
        ("extra", '[["role", "spoofed"]]', "message extra"),
        ("extra", '{"role": "spoofed"}', "不得覆盖消息列字段"),
        ("tool_chain", '{"call": "invalid"}', "message tool_chain"),
        ("extra", "", "message extra"),
        ("tool_chain", "", "message tool_chain"),
        ("extra", '{"media": "path"}', "message media"),
        ("extra", '{"media": null}', "message media"),
        ("extra", '{"source_refs": ["bad"]}', "message source_refs"),
        ("tool_chain", '[{"calls": null}]', "message tool_chain"),
        ("tool_chain", '[null]', "message tool_chain"),
        ("tool_chain", '[{"calls": [null]}]', "message tool_chain"),
    ],
)
def test_session_store_rejects_invalid_message_json(
    tmp_path: Path,
    column: str,
    payload: str,
    field: str,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    store.persist_session(
        "telegram:json",
        created_at="2026-07-13T00:00:00+00:00",
        updated_at="2026-07-13T00:00:01+00:00",
        last_consolidated=0,
        metadata={},
        messages=[
            {
                "role": "user",
                "content": "消息",
                "timestamp": "2026-07-13T00:00:01+00:00",
                "extra": {},
            }
        ],
    )
    store._conn.execute(
        f"UPDATE messages SET {column} = ? WHERE id = ?",
        (payload, "telegram:json:0"),
    )
    store._conn.commit()

    with pytest.raises(ValueError, match=field):
        store.fetch_session_messages("telegram:json")
    store.close()


@pytest.mark.parametrize("payload", ['{"media": [}', '{"media": "path"}'])
def test_session_store_media_lookup_rejects_invalid_extra(
    tmp_path: Path,
    payload: str,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    try:
        store.persist_session(
            "telegram:media",
            created_at="2026-07-13T00:00:00+00:00",
            updated_at="2026-07-13T00:00:01+00:00",
            last_consolidated=0,
            metadata={},
            messages=[
                {
                    "role": "user",
                    "content": "消息",
                    "timestamp": "2026-07-13T00:00:01+00:00",
                    "extra": {},
                }
            ],
        )
        store._conn.execute(
            "UPDATE messages SET extra = ? WHERE id = ?",
            (payload, "telegram:media:0"),
        )
        store._conn.commit()

        with pytest.raises(ValueError, match="telegram:media:0"):
            store.media_path_exists(tmp_path / "path")
    finally:
        store.close()


@pytest.mark.parametrize(
    ("column", "payload", "field"),
    [("role", "system", "message role"), ("content", None, "message content")],
)
def test_session_store_rejects_invalid_message_columns(
    tmp_path: Path,
    column: str,
    payload: object,
    field: str,
) -> None:
    store = SessionStore(tmp_path / "sessions.db")
    try:
        store.persist_session(
            "telegram:columns",
            created_at="2026-07-13T00:00:00+00:00",
            updated_at="2026-07-13T00:00:01+00:00",
            last_consolidated=0,
            metadata={},
            messages=[
                {
                    "role": "user",
                    "content": "消息",
                    "timestamp": "2026-07-13T00:00:01+00:00",
                    "extra": {},
                }
            ],
        )
        store._conn.execute(
            f"UPDATE messages SET {column} = ? WHERE id = ?",
            (payload, "telegram:columns:0"),
        )
        store._conn.commit()

        with pytest.raises(ValueError, match=field):
            store.fetch_session_messages("telegram:columns")
    finally:
        store.close()


def test_session_get_history_returns_empty_when_window_is_zero():
    session = Session("cli:1")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")

    assert session.get_history(max_messages=0) == []


def test_session_get_history_skips_cached_llm_frame_by_default():
    session = Session("cli:1")
    session.add_message("user", "old")
    session.add_message("assistant", "old reply")
    session.last_consolidated = 2
    user_content = "[当前消息时间: x]\nhello"
    session.add_message(
        "user",
        "hello",
        llm_context_frame="<system-reminder data-system-context-frame=\"true\">\n\n## retrieved_memory\n旧记忆",
        llm_user_content=user_content,
    )
    session.add_message("assistant", "world")

    history = session.get_history(start_index=session.last_consolidated)

    assert history == [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "world"},
    ]


def test_session_get_history_replays_proactive_as_short_assistant_with_meta_frame():
    session = Session("cli:1")
    session.add_message(
        "assistant",
        "这是一条主动消息",
        proactive=True,
        source_refs=[
            {
                "source_name": "feed",
                "title": "标题",
                "url": "https://example.com/a",
            }
        ],
    )

    history = session.get_history()

    assert len(history) == 2
    assert history[0] == {"role": "assistant", "content": "[主动推送] 这是一条主动消息"}
    assert history[1]["role"] == "user"
    content = str(history[1]["content"])
    assert is_context_frame(content)
    assert "recent_proactive_message_meta" in content
    assert "proactive_meta" in content


def test_session_get_history_allows_proactive_assistant_boundary():
    session = Session("cli:1")
    session.add_message("user", "old")
    session.add_message("assistant", "old reply")
    session.add_message("assistant", "主动消息", proactive=True)
    session.add_message("user", "刚才那个")
    session.last_consolidated = 2

    history = session.get_history(start_index=session.last_consolidated)

    assert history == [
        {"role": "assistant", "content": "[主动推送] 主动消息"},
        {"role": "user", "content": "刚才那个"},
    ]


def test_session_get_history_rewinds_consolidated_index_to_user_boundary():
    session = Session("cli:1")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")
    session.last_consolidated = 1

    history = session.get_history(start_index=session.last_consolidated)

    assert history[0] == {"role": "user", "content": "hello"}


def test_session_get_history_keeps_full_consolidated_tail():
    session = Session("cli:1")
    for i in range(5):
        session.add_message("user", f"u{i}")

    history = session.get_history(max_messages=2, start_index=0)

    assert session.consolidation_requested is False
    assert history == [
        {"role": "user", "content": "u0"},
        {"role": "user", "content": "u1"},
        {"role": "user", "content": "u2"},
        {"role": "user", "content": "u3"},
        {"role": "user", "content": "u4"},
    ]


def test_session_get_history_assistant_only_returns_empty():
    session = Session("cli:1")
    session.add_message("assistant", "a1")
    session.add_message("assistant", "a2")

    assert session.get_history(start_index=0) == []


def test_session_get_history_skips_legacy_context_frame_by_default():
    session = Session("cli:1")
    session.add_message(
        "user",
        "hello",
        llm_context_frame="[SYSTEM_CONTEXT_FRAME]\n\n## recent_context\n旧内容",
        llm_user_content="hello",
    )

    history = session.get_history(start_index=0)

    assert history == [{"role": "user", "content": "hello"}]


def test_session_get_history_does_not_inject_inference_tag():
    session = Session("cli:1")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")

    history = session.get_history()

    assert history[-1] == {"role": "assistant", "content": "world"}


def test_session_get_history_keeps_reasoning_content():
    session = Session("cli:1")
    session.add_message("user", "hello")
    session.add_message(
        "assistant",
        "world",
        reasoning_content="先想一下",
    )
    session.messages[-1]["tool_chain"] = [
        {
            "text": "",
            "reasoning_content": "准备调用工具",
            "calls": [
                {
                    "call_id": "call-1",
                    "name": "dummy",
                    "arguments": {},
                    "result": "ok",
                }
            ],
        }
    ]

    history = session.get_history()

    assert history[1]["reasoning_content"] == "准备调用工具"
    assert history[-1]["reasoning_content"] == "先想一下"


def test_session_get_history_keeps_short_tool_results_after_consolidation_tail():
    session = Session("cli:1")
    session.last_consolidated = 0
    for i in range(3):
        session.add_message("user", f"u{i}")
        session.add_message("assistant", f"a{i}")
        session.messages[-1]["tool_chain"] = [
            {
                "text": "",
                "calls": [
                    {
                        "call_id": f"call-{i}",
                        "name": "dummy",
                        "arguments": {},
                        "result": f"result-{i}",
                    }
                ],
            }
        ]

    history = session.get_history(start_index=session.last_consolidated)
    tool_contents = [m["content"] for m in history if m.get("role") == "tool"]

    assert tool_contents == ["result-0", "result-1", "result-2"]


def test_session_get_history_truncates_long_tool_results_in_middle():
    session = Session("cli:1")
    long_result = "head-" + "x" * (_TOOL_RESULT_CHAR_BUDGET + 200) + "-tail"
    session.add_message("user", "u")
    session.add_message("assistant", "a")
    session.messages[-1]["tool_chain"] = [
        {
            "text": "",
            "calls": [
                {
                    "call_id": "call-1",
                    "name": "dummy",
                    "arguments": {},
                    "result": long_result,
                }
            ],
        }
    ]

    history = session.get_history()
    tool_content = cast(
        str,
        next(m["content"] for m in history if m.get("role") == "tool"),
    )

    assert tool_content.startswith("Total output lines: 1\n\nhead-")
    assert "chars truncated" in tool_content
    assert tool_content.endswith("-tail")
    assert len(tool_content) < len(long_result)


def test_session_history_does_not_mask_non_oserror_media_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"image")

    def fail_read(_path: Path) -> bytes:
        raise ValueError("损坏的媒体读取状态")

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    session = Session("cli:media")
    session.add_message("user", "查看图片", media=[str(image)])

    with pytest.raises(ValueError, match="损坏的媒体读取状态"):
        session.get_history()


@pytest.mark.asyncio
async def test_proactive_loop_wrapper_methods_cover_paths(tmp_path: Path):
    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._cfg = SimpleNamespace(
        interval_seconds=10,
        score_weight_energy=0.5,
        tick_interval_s1=3,
        tick_interval_s0=4,
        tick_jitter=0.0,
        lifecycle="default",
        default_channel="telegram",
        default_chat_id="42",
    )
    loop._running = False
    loop._state_store_owned = False
    loop._state_closed = False
    loop._runtime_snapshot_store = None
    loop._stopped = asyncio.Event()
    loop._wake = asyncio.Event()
    loop._reload_lock = asyncio.Lock()
    loop._kernel_started = False
    loop._active_kernel_lease = None
    loop._active_snapshot_id = None
    loop._trace_proactive_rate_decision = MagicMock()
    loop._presence = SimpleNamespace(
        get_last_user_at=lambda session_key: datetime.now(timezone.utc)
    )
    loop._sense = SimpleNamespace(
        target_session_key=lambda: "telegram:1",
    )
    loop._rng = None
    loop._memory = SimpleNamespace(get_memory_context=lambda: "ctx")
    loop._sessions = SimpleNamespace(workspace=tmp_path)
    (tmp_path / "AGENTS.md").write_text("guide", encoding="utf-8")
    loop._sender = SimpleNamespace(send=AsyncMock(return_value=True))
    loop._proactive_kernel = SimpleNamespace(
        run_tick=AsyncMock(return_value=0.2),
        start=AsyncMock(return_value=None),
        stop=AsyncMock(return_value=None),
    )
    loop._run_loop = AsyncMock(return_value=None)

    assert await loop._tick() == 0.2
    loop._scheduler = SimpleNamespace(next_interval=lambda base_score: 7)
    assert loop._next_interval() == 7
    await loop.run()
    loop._proactive_kernel.start.assert_awaited_once()
    loop._run_loop.assert_awaited_once()
    loop._proactive_kernel.stop.assert_awaited_once()
