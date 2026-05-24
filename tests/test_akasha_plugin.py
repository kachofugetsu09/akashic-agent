from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest
import numpy as np

from core.memory.engine import MemoryQuery, MemoryScope
from plugins.akasha.config import AkashaConfig
from plugins.akasha.engine import (
    ActivationTrace,
    AkashaCandidate,
    AkashaMemoryEngine,
    _AkashaRetrieval,
    _compute_candidates,
    _load_turn_card,
)
from plugins.akasha.store import AkashaStore, SourceMessage
from scripts.build_akasha_db import _embed_batch_with_cache


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
        db.commit()


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        _ = text
        return [1.0, 0.0]


class FakeBatchEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[10.0 + index, 0.0] for index, _ in enumerate(texts)]


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


@pytest.mark.asyncio
async def test_embed_batch_with_cache_only_embeds_missing_messages(
    tmp_path: Path,
) -> None:
    store = AkashaStore(tmp_path / "akasha.db")
    messages = [
        SourceMessage("s:0", "s", 0, "user", "已缓存", "2026-01-01T00:00:00+00:00"),
        SourceMessage("s:1", "s", 1, "assistant", "新消息", "2026-01-01T00:00:01+00:00"),
    ]
    fake = FakeBatchEmbedder()
    try:
        store.upsert_cached_embedding(
            message=messages[0],
            model="m",
            embedding=[1.0, 0.0],
        )

        embeddings, hits, misses = await _embed_batch_with_cache(
            store=store,
            embedder=cast(Any, fake),
            model="m",
            batch=messages,
        )
    finally:
        store.close()

    assert hits == 1
    assert misses == 1
    assert fake.calls == [["新消息"]]
    assert embeddings == [[1.0, 0.0], [10.0, 0.0]]


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
    engine._remember_pending_activation = lambda request, items: None
    engine._retrieve = lambda query, query_vec, request: _AkashaRetrieval(
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
                source="Expanded",
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
        )
    )

    assert "## 左脑记忆：精确回忆" in result.text_block
    assert "## 右脑联想：潜意识第一反应" in result.text_block
    assert '- user="第一条用户消息需要完整展示" assistant=' in result.text_block
    dense_block, ripple_block = result.text_block.split("## 右脑联想：潜意识第一反应", 1)
    assert 'source_ref=["s:0", "s:1"]' in dense_block
    assert 'source_ref=["s:0", "s:1"]' not in ripple_block
    assert 'source_ref=["s:2", "s:3"]' in ripple_block


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
    engine._remember_pending_activation = lambda request, items: None
    engine._retrieve = lambda query, query_vec, request: _AkashaRetrieval(
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
