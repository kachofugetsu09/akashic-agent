from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta

import pytest

from agent.plugin_composition.semantic_interest import ConversationSemanticInterest
from session.embedding_store import MessageEmbeddingStore


class _EmbeddingApi:
    model_id = "fixture-embedding"

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "looks proactive": [1.0, 0.0],
            "looks passive": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


@pytest.mark.asyncio
async def test_scores_against_passive_turns_and_ignores_twenty_proactive_rows(
    tmp_path,
) -> None:
    db_path = tmp_path / "sessions.db"
    now = datetime(2026, 8, 25, 1, tzinfo=UTC)
    with closing(sqlite3.connect(db_path)) as connection, connection:
        connection.execute("""
            CREATE TABLE messages(
                id TEXT PRIMARY KEY, session_key TEXT, seq INTEGER, role TEXT,
                content TEXT, extra TEXT, ts TEXT
            )
            """)
        rows = [
            ("u-pro", "mobile", 1, "user", "proactive seed", "{}", now.isoformat()),
            *[
                (
                    f"p{index}",
                    "mobile",
                    index + 2,
                    "assistant",
                    f"push {index}",
                    '{"proactive":true}',
                    (now + timedelta(seconds=index + 1)).isoformat(),
                )
                for index in range(20)
            ],
            (
                "u-passive",
                "mobile",
                22,
                "user",
                "passive user",
                "{}",
                (now + timedelta(seconds=30)).isoformat(),
            ),
            (
                "a-passive",
                "mobile",
                23,
                "assistant",
                "passive assistant",
                "{}",
                (now + timedelta(seconds=31)).isoformat(),
            ),
        ]
        connection.executemany("INSERT INTO messages VALUES(?, ?, ?, ?, ?, ?, ?)", rows)
    embeddings = MessageEmbeddingStore(db_path)
    embeddings.upsert(
        message_id="u-pro",
        content="proactive seed",
        model=_EmbeddingApi.model_id,
        embedding=[1.0, 0.0],
    )
    for index in range(20):
        embeddings.upsert(
            message_id=f"p{index}",
            content=f"push {index}",
            model=_EmbeddingApi.model_id,
            embedding=[1.0, 0.0],
        )
    embeddings.upsert(
        message_id="u-passive",
        content="passive user",
        model=_EmbeddingApi.model_id,
        embedding=[0.0, 1.0],
    )
    embeddings.upsert(
        message_id="a-passive",
        content="passive assistant",
        model=_EmbeddingApi.model_id,
        embedding=[0.0, 1.0],
    )
    embeddings.close()

    service = ConversationSemanticInterest(db_path, _EmbeddingApi())
    scores = await service.score(
        ("looks proactive", "looks passive"),
        cutoff=(now + timedelta(minutes=1)).isoformat(),
    )

    assert scores[0] == 0.0
    assert scores[1] == pytest.approx(0.999)


@pytest.mark.asyncio
async def test_candidate_validation_cannot_read_formal_semantics() -> None:
    service = ConversationSemanticInterest.candidate_validation()

    with pytest.raises(RuntimeError, match="candidate 验证期禁止"):
        await service.score(("candidate",), cutoff=datetime.now(UTC).isoformat())
