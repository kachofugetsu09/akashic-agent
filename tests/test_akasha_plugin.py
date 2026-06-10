from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import numpy as np

from bus.events_lifecycle import TurnCommitted
from core.memory.engine import MemoryQuery, MemoryQueryIntent, MemoryScope
from agent.plugins.context import PluginContext, PluginKVStore
from plugins.akasha.config import AkashaConfig
from plugins.akasha.engine import (
    ActivationTrace,
    AkashaCandidate,
    AkashaMemoryEngine,
    PendingActivation,
    _AkashaRetrieval,
    _compute_candidates,
    _load_turn_card,
)
from plugins.akasha.core import (
    AkashaNode,
    activation_edge_updates,
    build_dense_message_index,
    dense_message_candidates,
    reinforce_boost_from_payload,
)
from plugins.akasha.plugin import AkashaPlugin
from plugins.akasha.replay import AkashaReplayRuntime, ReplayMessage, _turn_messages
from plugins.akasha.store import (
    ActivationEventRow,
    AkashaStore,
    EdgeUpdate,
    SourceMessage,
)
from scripts.build_akasha_db import _iter_replay_turns, _load_embeddings_from_cache, _skip_message


QUERY_TS = datetime.fromtimestamp(1_700_000_000.0, timezone.utc)


def _init_sessions_db(path: Path) -> None:
    with closing(sqlite3.connect(str(path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
                ("s:1", "s", 1, "assistant", "第一条助手回复会被截断展示并保留引用", "2026-01-01T00:00:01+00:00"),
                ("s:2", "s", 2, "user", "第二条用户消息只在联想块", "2026-01-01T00:00:02+00:00"),
                ("s:3", "s", 3, "assistant", "第二条助手回复也会被截断", "2026-01-01T00:00:03+00:00"),
            ],
        )
        db.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
        db.execute("INSERT INTO messages_fts(rowid, content) SELECT rowid, content FROM messages")
        db.commit()


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        _ = text
        return [1.0, 0.0]


def _candidate(key: str, score: float) -> AkashaCandidate:
    return AkashaCandidate(
        key=key,
        source="Dense",
        ripple=0.0,
        direct=score,
        state=0.0,
        edge=0.0,
        long=0.0,
        resource=1.0,
        fan=0,
        score=score,
    )


def test_reinforce_boost_payload_uses_exact_tool_chain_call_name() -> None:
    wrong_chain = [{"calls": [{"name": "not_reinforce_memory"}]}]
    reinforce_chain = [{"calls": [{"name": "reinforce_memory"}]}]

    assert reinforce_boost_from_payload({}, wrong_chain) == 1.0
    assert reinforce_boost_from_payload({}, json.dumps(reinforce_chain)) == 3.0
    assert reinforce_boost_from_payload({"akasha_reinforce": {"boost": "4"}}, []) == 4.0


def test_dense_message_candidates_vectorized_preserves_turn_ranking() -> None:
    nodes = {
        "s:0": AkashaNode(
            key="s:0",
            anchor_id="m0",
            session_key="s",
            turn_seq=0,
            first_ts_unix=QUERY_TS.timestamp(),
            salience=0.0,
            strength=0.0,
            resource=1.0,
            recall_count=0,
            last_activated_ts=0.0,
            last_strength_ts=QUERY_TS.timestamp(),
            last_resource_ts=QUERY_TS.timestamp(),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            emb_count=1,
        ),
        "s:2": AkashaNode(
            key="s:2",
            anchor_id="m2",
            session_key="s",
            turn_seq=2,
            first_ts_unix=QUERY_TS.timestamp(),
            salience=0.0,
            strength=0.0,
            resource=1.0,
            recall_count=0,
            last_activated_ts=0.0,
            last_strength_ts=QUERY_TS.timestamp(),
            last_resource_ts=QUERY_TS.timestamp(),
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            emb_count=1,
        ),
        "s:4": AkashaNode(
            key="s:4",
            anchor_id="m4",
            session_key="s",
            turn_seq=4,
            first_ts_unix=QUERY_TS.timestamp(),
            salience=0.0,
            strength=0.0,
            resource=1.0,
            recall_count=0,
            last_activated_ts=0.0,
            last_strength_ts=QUERY_TS.timestamp(),
            last_resource_ts=QUERY_TS.timestamp(),
            embedding=np.array([0.0, 0.0], dtype=np.float32),
            emb_count=1,
        ),
    }
    message_embeddings = {
        "m0": np.array([1.0, 0.0], dtype=np.float32),
        "m2": np.array([0.8, 0.6], dtype=np.float32),
        "m3": np.array([0.9, 0.1], dtype=np.float32),
        "bad-dim": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "m4": np.array([0.0, 0.0], dtype=np.float32),
    }
    message_turn_keys = {
        "m0": "s:0",
        "m2": "s:2",
        "m3": "s:2",
        "bad-dim": "s:0",
        "m4": "s:4",
    }
    loop_result = dense_message_candidates(
        np.array([1.0, 0.0], dtype=np.float32),
        nodes,
        message_embeddings,
        message_turn_keys,
        limit=3,
    )
    indexed_result = dense_message_candidates(
        np.array([1.0, 0.0], dtype=np.float32),
        nodes,
        message_embeddings,
        message_turn_keys,
        limit=3,
        message_index=build_dense_message_index(message_embeddings),
    )

    assert [item.key for item in loop_result] == ["s:0", "s:2", "s:4"]
    assert [item.key for item in indexed_result] == [item.key for item in loop_result]
    assert [item.score for item in loop_result] == pytest.approx([
        1.0,
        0.9 / ((0.9 ** 2 + 0.1 ** 2) ** 0.5),
        0.0,
    ])
    assert [item.score for item in indexed_result] == pytest.approx(
        [item.score for item in loop_result]
    )


def test_store_merges_user_and_assistant_into_turn_node(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        store.upsert_message_node(
            SourceMessage("s:0", "s", 0, "user", "用户消息", "2026-01-01T00:00:00+00:00"),
            [1.0, 0.0],
        )
        store.upsert_message_node(
            SourceMessage("s:1", "s", 1, "assistant", "助手消息", "2026-01-01T00:00:01+00:00"),
            [0.0, 1.0],
        )

        nodes = store.list_nodes()
    finally:
        store.close()

    assert len(nodes) == 1
    assert nodes[0].key == "s:0"
    assert nodes[0].anchor_id == "s:0"
    assert nodes[0].emb_count == 2


def test_reset_schema_keeps_embedding_cache(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    message = SourceMessage(
        "s:0",
        "s",
        0,
        "user",
        "用户消息",
        "2026-01-01T00:00:00+00:00",
    )
    try:
        store.upsert_cached_embedding(message=message, model="m", embedding=[1.0, 2.0])
        _ = store.upsert_message_node(message, [1.0, 0.0])
        store.reset_schema()

        cached = store.get_cached_embedding(message=message, model="m")
        nodes = store.list_nodes()
    finally:
        store.close()

    assert cached == [1.0, 2.0]
    assert nodes == []


def test_load_embeddings_from_cache_counts_hits_and_misses(
    tmp_path: Path,
) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    messages = [
        SourceMessage("s:0", "s", 0, "user", "已缓存", "2026-01-01T00:00:00+00:00"),
        SourceMessage("s:1", "s", 1, "assistant", "新消息", "2026-01-01T00:00:01+00:00"),
    ]
    try:
        store.upsert_cached_embedding(
            message=messages[0],
            model="m",
            embedding=[1.0, 0.0],
        )

        embeddings, hits, misses = _load_embeddings_from_cache(
            store=store,
            model="m",
            messages=messages,
        )
    finally:
        store.close()

    assert hits == 1
    assert misses == 1
    assert embeddings == {"s:0": [1.0, 0.0]}


def test_replay_and_runtime_use_same_directional_stdp_edges(tmp_path: Path) -> None:
    candidate = _candidate("s:0", 0.8)
    ts = QUERY_TS.timestamp()
    expected = {
        (item.src_key, item.dst_key): 0.12 * item.strength
        for item in activation_edge_updates("s:2", [candidate], ts)
    }
    replay_store = AkashaStore(tmp_path / "replay.db")
    runtime_store = AkashaStore(tmp_path / "runtime.db")
    try:
        with closing(sqlite3.connect(":memory:")) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(),
                source_db_path=tmp_path / "sessions.db",
                source_cursor=source_db.cursor(),
                message_embeddings={},
                message_turn_keys={},
            )
            replay.commit_turn(
                [
                    ReplayMessage(
                        SourceMessage(
                            "m2",
                            "s",
                            2,
                            "user",
                            "当前消息",
                            QUERY_TS.isoformat(),
                        ),
                        [1.0, 0.0],
                    )
                ],
                [candidate],
            )

        engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
        engine._store = runtime_store
        engine._graph_lock = threading.RLock()
        engine._edges = {}
        engine._edges_meta = {}
        engine._edges_by_src = {}
        engine._fan = {}
        engine._commit_pending_activation(
            "s:2",
            PendingActivation(query_id="q", seq=2, ts=ts, items=[candidate]),
        )

        assert replay_store.load_edges() == pytest.approx(expected)
        assert runtime_store.load_edges() == pytest.approx(expected)
        assert expected[("s:0", "s:2")] > expected[("s:2", "s:0")]
    finally:
        replay_store.close()
        runtime_store.close()


def test_replay_writes_query_log_with_activation_items(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    monkeypatch.setattr("plugins.akasha.core.get_jieba_keywords", lambda _: "")
    replay_store = AkashaStore(tmp_path / "replay.db")
    old_messages = [
        SourceMessage("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
        SourceMessage("s:2", "s", 2, "user", "第二条用户消息只在联想块", "2026-01-01T00:00:02+00:00"),
    ]
    try:
        replay_store.upsert_message_node(old_messages[0], [1.0, 0.0])
        replay_store.upsert_message_node(old_messages[1], [0.98, 0.02])
        with closing(sqlite3.connect(str(db_path))) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(dense_seed_threshold=0.1, nearby_dense_threshold=0.0),
                source_db_path=db_path,
                source_cursor=source_db.cursor(),
                message_embeddings={
                    "s:0": np.array([1.0, 0.0], dtype=np.float32),
                    "s:2": np.array([0.98, 0.02], dtype=np.float32),
                },
                message_turn_keys={"s:0": "s:0", "s:2": "s:2"},
            )
            result = replay.replay_turn([
                ReplayMessage(
                    SourceMessage("s:4", "s", 4, "user", "第一条", QUERY_TS.isoformat()),
                    [1.0, 0.0],
                )
            ])

        rows, total = replay_store.list_query_logs(session_key="s", page=1, page_size=10)
        assert total == 1
        raw = replay_store.get_query_log(str(rows[0]["query_id"]))
        assert raw is not None
        activation_items = json.loads(str(raw["activation_items_json"]))
        dense_items = json.loads(str(raw["dense_items_json"]))
        ripple_items = json.loads(str(raw["ripple_items_json"]))
        assert str(rows[0]["query_id"]).startswith("s:4:context:")
        assert rows[0]["intent"] == "context"
        assert rows[0]["activated_count"] == len(result.activation_items)
        assert rows[0]["dense_count"] == len(dense_items)
        assert rows[0]["ripple_count"] == len(ripple_items)
        assert raw["text_block_preview"]
        assert activation_items
        assert dense_items
        assert isinstance(ripple_items, list)
        assert activation_items[0]["user_message"] in {
            "第一条用户消息需要完整展示",
            "第二条用户消息只在联想块",
        }
    finally:
        replay_store.close()


def test_replay_empty_query_commits_without_activation_or_query_log(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    replay_store = AkashaStore(tmp_path / "replay.db")
    replay_store.upsert_message_node(
        SourceMessage("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
        [1.0, 0.0],
    )
    try:
        with closing(sqlite3.connect(str(db_path))) as source_db:
            replay = AkashaReplayRuntime(
                store=replay_store,
                config=AkashaConfig(dense_seed_threshold=0.1, nearby_dense_threshold=0.0),
                source_db_path=db_path,
                source_cursor=source_db.cursor(),
                message_embeddings={"s:0": np.array([1.0, 0.0], dtype=np.float32)},
                message_turn_keys={"s:0": "s:0"},
            )
            result = replay.replay_turn([
                ReplayMessage(
                    SourceMessage("s:4", "s", 4, "user", "", QUERY_TS.isoformat()),
                    [0.0, 0.0],
                )
            ])

        rows, total = replay_store.list_query_logs(session_key="s", page=1, page_size=10)
        assert result.current_key == "s:4"
        assert result.activation_items == []
        assert total == 0
        assert rows == []
        with closing(sqlite3.connect(str(tmp_path / "replay.db"))) as db:
            assert db.execute("SELECT COUNT(*) FROM akasha_activation_events").fetchone()[0] == 0
    finally:
        replay_store.close()


def test_query_log_content_loader_allows_empty_user_message(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("m0", "s", 0, "user", "", QUERY_TS.isoformat()),
                ("m1", "s", 1, "assistant", "assistant preview", QUERY_TS.isoformat()),
            ],
        )
        db.commit()
        user_message, assistant_preview = _turn_messages(
            db.cursor(),
            "s:0",
            assistant_preview_chars=9,
        )

    assert user_message == ""
    assert assistant_preview == "assistant..."


def test_akasha_rebuild_skips_scheduler_messages() -> None:
    scheduler_user = SourceMessage(
        "scheduler:job:0",
        "scheduler:job",
        0,
        "user",
        "查询北京天气",
        "2026-01-01T00:00:00+00:00",
    )
    normal_user = SourceMessage(
        "telegram:1:0",
        "telegram:1",
        0,
        "user",
        "今天聊 Akasha",
        "2026-01-01T00:00:01+00:00",
    )

    assert _skip_message(scheduler_user, set()) is True
    assert _skip_message(normal_user, set()) is False
    assert list(_iter_replay_turns([scheduler_user, normal_user], set())) == [[normal_user]]


@pytest.mark.asyncio
async def test_runtime_skips_scheduler_turn_even_without_extra_flag(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._session_db_path = db_path
    engine._embedder = SimpleNamespace(embed_batch=AsyncMock(side_effect=AssertionError("should skip")))

    await engine._on_turn_committed(
        TurnCommitted(
            session_key="scheduler:job",
            channel="telegram",
            chat_id="1",
            input_message="查询天气",
            persisted_user_message="查询天气",
            assistant_response="天气回复",
            tools_used=[],
        )
    )

    engine._embedder.embed_batch.assert_not_awaited()


def test_load_turn_card_uses_full_user_and_short_assistant(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    card = _load_turn_card(
        db_path,
        "s:0",
        assistant_preview_chars=15,
        score=0.8,
        lane="dense",
        signals={},
    )

    assert card is not None
    assert card.user_message == "第一条用户消息需要完整展示"
    assert card.assistant_preview == "第一条助手回复会被截断展示并保..."
    assert card.source_ref == '["s:0", "s:1"]'
    assert card.happened_at == "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_query_places_overlap_in_dense_and_ripple_only_in_ripple(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(assistant_preview_chars=15)
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    engine._remember_pending_activation = lambda request, items, **_: None
    engine._retrieve = lambda query, query_vec, request, *, now_ts, update_state: _AkashaRetrieval(
        dense_items=[
            AkashaCandidate(
                key="s:0",
                source="Dense",
                ripple=0.0,
                direct=0.9,
                state=0.0,
                edge=0.0,
                long=0.0,
                resource=1.0,
                fan=0,
                score=0.9,
            )
        ],
        ripple_items=[
            AkashaCandidate(
                key="s:0",
                source="Dense",
                ripple=0.6,
                direct=0.9,
                state=1.0,
                edge=0.0,
                long=0.0,
                resource=1.0,
                fan=0,
                score=0.8,
            ),
            AkashaCandidate(
                key="s:2",
                source="Graph",
                ripple=0.5,
                direct=0.4,
                state=0.8,
                edge=0.2,
                long=0.0,
                resource=1.0,
                fan=1,
                score=0.7,
            ),
        ],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=2),
        seq=4,
    )

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="context",
            scope=MemoryScope(session_key="s"),
            timestamp=QUERY_TS,
        )
    )

    assert "## 左脑记忆：精确回忆" in result.text_block
    assert "## 右脑联想：潜意识第一反应" in result.text_block
    assert "# Akasha memory now=" in result.text_block
    assert '- user="第一条用户消息需要完整展示" assistant=' in result.text_block
    assert " t=01-01 source_ref=" in result.text_block
    assert " score=" not in result.text_block
    dense_block, ripple_block = result.text_block.split("## 右脑联想：潜意识第一反应", 1)
    assert 'source_ref=["s:0", "s:1"]' in dense_block
    assert 'source_ref=["s:0", "s:1"]' not in ripple_block
    assert 'source_ref=["s:2", "s:3"]' in ripple_block


@pytest.mark.asyncio
async def test_context_block_sorts_injected_cards_by_time_desc(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    def candidate(key: str, score: float) -> AkashaCandidate:
        return AkashaCandidate(
            key=key,
            source="Dense",
            ripple=0.0,
            direct=score,
            state=0.0,
            edge=0.0,
            long=0.0,
            resource=1.0,
            fan=0,
            score=score,
        )

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(dense_top_k=10, ripple_top_k=10)
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    engine._remember_pending_activation = lambda request, items, **_: None
    engine._retrieve = lambda query, query_vec, request, *, now_ts, update_state: _AkashaRetrieval(
        dense_items=[candidate("s:0", 0.9), candidate("s:2", 0.8)],
        ripple_items=[],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=2),
        seq=4,
    )

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="context",
            scope=MemoryScope(session_key="s"),
            timestamp=QUERY_TS,
        )
    )

    assert result.text_block.index('source_ref=["s:2", "s:3"]') < result.text_block.index(
        'source_ref=["s:0", "s:1"]'
    )


def test_cards_from_keys_deduplicates_same_user_assistant_pair(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        db.executemany(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("s:0", "s", 0, "user", "我现在健康状态怎么样呢", "2026-01-01T00:00:00+00:00"),
                ("s:1", "s", 1, "assistant", "健康状态的话……我这边真的没有更多信息", "2026-01-01T00:00:01+00:00"),
                ("s:2", "s", 2, "user", "我现在健康状态怎么样呢", "2026-01-01T00:00:02+00:00"),
                ("s:3", "s", 3, "assistant", "健康状态的话……我这边真的没有更多信息", "2026-01-01T00:00:03+00:00"),
                ("s:4", "s", 4, "user", "另一个问题", "2026-01-01T00:00:04+00:00"),
                ("s:5", "s", 5, "assistant", "第三次回复", "2026-01-01T00:00:05+00:00"),
            ],
        )
        db.commit()

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(assistant_preview_chars=15)
    engine._session_db_path = db_path

    cards = engine._cards_from_keys(
        [
            ("s:0", 0.9, "ripple", {}),
            ("s:2", 0.8, "ripple", {}),
            ("s:4", 0.7, "ripple", {}),
        ],
        limit=10,
    )

    assert [card.source_ref for card in cards] == [
        '["s:0", "s:1"]',
        '["s:4", "s:5"]',
    ]


@pytest.mark.asyncio
async def test_context_query_uses_akasha_top_k_over_default_query_limit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "sessions.db"
    with closing(sqlite3.connect(str(db_path))) as db:
        db.execute(
            """
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                session_key TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                ts TEXT NOT NULL
            )
            """
        )
        rows = []
        for turn in range(24):
            user_seq = turn * 2
            rows.append((f"s:{user_seq}", "s", user_seq, "user", f"用户消息{turn}", "2026-01-01T00:00:00+00:00"))
            rows.append((f"s:{user_seq + 1}", "s", user_seq + 1, "assistant", f"助手回复{turn}", "2026-01-01T00:00:01+00:00"))
        db.executemany("INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)", rows)
        db.commit()

    def candidate(key: str, score: float) -> AkashaCandidate:
        return AkashaCandidate(
            key=key,
            source="Dense",
            ripple=0.0,
            direct=score,
            state=0.0,
            edge=0.0,
            long=0.0,
            resource=1.0,
            fan=0,
            score=score,
        )

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig(dense_top_k=10, ripple_top_k=10, inject_max_chars=20000)
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    engine._remember_pending_activation = lambda request, items, **_: None
    engine._retrieve = lambda query, query_vec, request, *, now_ts, update_state: _AkashaRetrieval(
        dense_items=[candidate(f"s:{turn * 2}", 1.0 - turn * 0.01) for turn in range(12)],
        ripple_items=[candidate(f"s:{24 + turn * 2}", 0.8 - turn * 0.01) for turn in range(12)],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=24),
        seq=48,
    )

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="context",
            scope=MemoryScope(session_key="s"),
            limit=8,
            timestamp=QUERY_TS,
        )
    )

    assert result.trace["dense_count"] == 10
    assert result.trace["ripple_count"] == 10
    assert result.text_block.count("source_ref=") == 20


def test_compute_candidates_uses_activation_limit_for_stateful_replay(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        for seq in range(30):
            _ = store.upsert_message_node(
                SourceMessage(
                    f"s:{seq}",
                    "s",
                    seq,
                    "user",
                    f"消息 {seq}",
                    "2026-01-01T00:00:00+00:00",
                    salience=1.0,
                ),
                [1.0, 0.0],
            )
        nodes = {node.key: node for node in store.list_nodes()}
    finally:
        store.close()

    candidates, suppressed, trace = _compute_candidates(
        "消息",
        np.array([1.0, 0.0], dtype=np.float32),
        nodes,
        {},
        100,
        config=AkashaConfig(dense_top_k=30, activate_limit=8),
        fan={},
        soft_recall=False,
        return_limit=8,
    )

    assert len(candidates) == 8
    assert trace.seed_count == 30
    assert suppressed == []


def test_query_log_keeps_context_and_answer_for_same_seq(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._store = store
    engine._akasha_config = AkashaConfig()
    result = _AkashaRetrieval(
        dense_items=[],
        ripple_items=[],
        activation_items=[],
        trace=ActivationTrace(seed_count=1, pool_count=2),
        seq=10,
    )
    try:
        cases: list[tuple[MemoryQueryIntent, str]] = [("context", "注入文本"), ("answer", "")]
        for intent, text_block in cases:
            engine._write_query_log(
                request=MemoryQuery(
                    text="同一轮问题",
                    intent=intent,
                    scope=MemoryScope(session_key="s"),
                    timestamp=QUERY_TS,
                ),
                result=result,
                seq=10,
                dense_cards=[],
                ripple_cards=[],
                text_block=text_block,
            )

        items, total = store.list_query_logs(session_key="s", page=1, page_size=10)
    finally:
        store.close()

    assert total == 2
    assert {item["intent"] for item in items} == {"context", "answer"}


@pytest.mark.asyncio
async def test_read_only_query_skips_akasha_state_effects(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)

    engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
    engine._akasha_config = AkashaConfig()
    engine._session_db_path = db_path
    engine._embedder = FakeEmbedder()
    side_effects: list[str] = []
    update_state_values: list[bool] = []

    def fake_retrieve(
        query: str,
        query_vec: np.ndarray,
        request: MemoryQuery,
        *,
        now_ts: float,
        update_state: bool,
    ) -> _AkashaRetrieval:
        _ = (query, query_vec, request, now_ts)
        update_state_values.append(update_state)
        return _AkashaRetrieval(
            dense_items=[_candidate("s:0", 0.9)],
            ripple_items=[],
            activation_items=[_candidate("s:2", 0.8)],
            trace=ActivationTrace(seed_count=1, pool_count=1),
            seq=4,
        )

    engine._retrieve = fake_retrieve
    engine._remember_pending_activation = lambda *_, **__: side_effects.append("pending")
    engine._write_query_log = lambda *_, **__: side_effects.append("query_log")

    result = await engine.query(
        MemoryQuery(
            text="用户消息",
            intent="answer",
            effect="read_only",
            scope=MemoryScope(session_key="s"),
            timestamp=QUERY_TS,
        )
    )

    assert update_state_values == [False]
    assert side_effects == []
    assert result.trace["effect"] == "read_only"
    assert result.records


def test_undo_removes_akasha_turn_state_after_session_delete(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    _init_sessions_db(db_path)
    store = AkashaStore(tmp_path / "akasha.db")
    try:
        messages = [
            SourceMessage("s:0", "s", 0, "user", "第一条用户消息需要完整展示", "2026-01-01T00:00:00+00:00"),
            SourceMessage("s:1", "s", 1, "assistant", "第一条助手回复会被截断展示并保留引用", "2026-01-01T00:00:01+00:00"),
            SourceMessage("s:2", "s", 2, "user", "第二条用户消息只在联想块", "2026-01-01T00:00:02+00:00"),
        ]
        for index, message in enumerate(messages):
            embedding = [1.0, 0.0] if index < 2 else [0.0, 1.0]
            store.upsert_cached_embedding(message=message, model="m", embedding=embedding)
            _ = store.upsert_message_node(message, embedding)
        store.upsert_edges([
            EdgeUpdate("s:0", "s:2", 1.0, 0),
            EdgeUpdate("s:2", "s:0", 1.0, 0),
        ])
        store.insert_activation_events([
            ActivationEventRow(
                seq=0,
                query_id="s:0",
                activated_key="s:2",
                source="Dense",
                score=0.8,
                direct_score=0.8,
                state_score=0.0,
                edge_score=0.0,
                long_score=0.0,
                resource=1.0,
                fan=0,
            )
        ])
        store.insert_query_log(
            query_id="s:0:context:abc",
            session_key="s",
            seq=0,
            query_text="第一条用户消息",
            intent="context",
            ts="2026-01-01T00:00:00+00:00",
            seed_count=1,
            pool_count=2,
            activated_count=1,
            activation_threshold=0.2,
            dense_count=1,
            ripple_count=1,
            inject_chars=10,
            source_ref_count=2,
            activation_items_json="[]",
            dense_items_json="[]",
            ripple_items_json="[]",
            text_block_preview="preview",
        )

        engine = cast(Any, AkashaMemoryEngine.__new__(AkashaMemoryEngine))
        engine._store = store
        engine._session_db_path = db_path
        engine._config = SimpleNamespace(
            memory=SimpleNamespace(embedding=SimpleNamespace(model="m"))
        )
        engine._graph_lock = threading.RLock()
        engine._nodes = {}
        engine._edges = {}
        engine._edges_by_src = {}
        engine._fan = {}
        engine._message_embeddings = {}
        engine._message_turn_keys = {}
        engine._load_graph_cache()

        dry_run = engine.undo_by_message_sources(["s:0", "s:1"], dry_run=True)
        with closing(sqlite3.connect(str(db_path))) as db:
            _ = db.execute("DELETE FROM messages WHERE id IN ('s:0', 's:1')")
            db.commit()
        result = engine.undo_by_message_sources(["s:0", "s:1"])

        assert dry_run["affected_ids"] == ["s:0"]
        assert result["affected_ids"] == ["s:0"]
        assert result["restored_ids"] == []
        assert result["rollback_source_ids"] == ["s:0", "s:1"]
        assert store.get_node("s:0") is None
        assert store.get_node("s:2") is not None
        assert store.load_edges() == {}
        assert store.list_query_logs(page=1, page_size=10)[1] == 0
        with closing(sqlite3.connect(str(store.db_path))) as db:
            event_count = db.execute("SELECT COUNT(1) FROM akasha_activation_events").fetchone()[0]
            cache_count = db.execute("SELECT COUNT(1) FROM akasha_embedding_cache").fetchone()[0]
        assert event_count == 0
        assert cache_count == 1
        assert "s:0" not in engine._nodes
        assert ("s:0", "s:2") not in engine._edges
        assert "s:0" not in engine._message_embeddings
        assert "s:1" not in engine._message_turn_keys
    finally:
        store.close()


def test_akashalast_command_only_registers_for_akasha_engine(tmp_path: Path) -> None:
    akasha = AkashaPlugin()
    akasha.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".akasha-kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )
    default = AkashaPlugin()
    default.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".default-kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="default")),
    )

    assert akasha.telegram_bot_commands() == [("akashalast", "查看上一轮 Akasha 检索诊断")]
    assert len(akasha.before_turn_modules()) == 1
    assert default.telegram_bot_commands() == []
    assert default.before_turn_modules() == []


def test_akashalast_renders_latest_query_log(tmp_path: Path) -> None:
    store = AkashaStore(tmp_path / "memory" / "akasha.db")
    try:
        activation_items = json.dumps([
            {
                "user_message": "这个是他转的别人的帖子而已",
                "assistant_preview": "啊你说得对，那个是转推",
                "score": 0.501,
                "source": "Dense",
                "path_type": "direct",
            }
        ], ensure_ascii=False)
        dense_items = json.dumps([
            {
                "user_message": "这个是他转的别人的帖子而已",
                "assistant_preview": "啊你说得对，那个是转推",
                "score": 0.703,
                "source": "Dense",
            }
        ], ensure_ascii=False)
        ripple_items = json.dumps([
            {
                "user_message": "我纠正过你几次有关汪远哲这个名字",
                "assistant_preview": "花月哥哥，这个错误我真是犯过",
                "score": 0.247,
                "source": "FTS",
                "path_type": "direct",
                "direct": 0.41,
                "state": 0.18,
                "edge": 0.08,
                "resource": 1.0,
                "fan": 32,
            }
        ], ensure_ascii=False)
        store.insert_query_log(
            query_id="s:2:context:abc",
            session_key="s",
            seq=2,
            query_text="这个其实不是她的 是她转发的别人的帖子",
            intent="context",
            ts="2026-05-24T22:15:00+08:00",
            seed_count=11,
            pool_count=81,
            activated_count=4,
            activation_threshold=0.22,
            dense_count=1,
            ripple_count=1,
            inject_chars=100,
            source_ref_count=2,
            activation_items_json=activation_items,
            dense_items_json=dense_items,
            ripple_items_json=ripple_items,
            text_block_preview="preview",
        )
    finally:
        store.close()

    plugin = AkashaPlugin()
    plugin.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="akasha",
        plugin_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        workspace=tmp_path,
        memory_engine=SimpleNamespace(describe=lambda: SimpleNamespace(name="akasha")),
    )

    reply = plugin.render_last_query("s")

    assert "🧠 Akasha 记忆检索诊断" in reply
    assert "📍 会话: `s` | seq `2`" in reply
    assert "• 种子节点 (Seeds): `11` 个" in reply
    assert "🔥 本轮图激活节点 (Activated Nodes):" in reply
    assert "🎯 左脑精确回忆 (Dense):" in reply
    assert "🌊 右脑联想记忆 (Ripple):" in reply
    assert "分: `0.501` | 源: `Dense` | 径: `direct`" in reply
    assert "得: `0.703` | 源: `Dense`" in reply
    assert "因: `dir:0.41 st:0.18 edg:0.08 res:1.00 fan:32`" in reply
