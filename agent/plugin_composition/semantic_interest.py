from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Awaitable, Sequence
from contextlib import closing
from pathlib import Path
from typing import Protocol, cast

from agent.plugin_composition.model import ServiceKey
from session.embedding_store import MessageEmbeddingStore

_SEMANTIC_CALIBRATION_POWER = 4


class _EmbeddingApi(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed_batch(self, texts: list[str]) -> Awaitable[list[list[float]]]: ...


class ConversationSemanticInterest:
    """Score candidate text against completed non-proactive conversation turns."""

    def __init__(
        self,
        db_path: Path | None,
        embedding_api: _EmbeddingApi | None,
    ) -> None:
        self._db_path = db_path
        self._embedding_api = embedding_api

    @classmethod
    def candidate_validation(cls) -> ConversationSemanticInterest:
        """Keep candidate topology while refusing formal conversation reads."""

        return cls(None, None)

    async def score(self, texts: Sequence[str], *, cutoff: str) -> tuple[float, ...]:
        """Return the legacy calibrated maximum similarity for every candidate."""

        # 1. Candidate validation must never observe formal Session state.
        if self._db_path is None:
            raise RuntimeError("candidate 验证期禁止访问正式 conversation semantics")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("semantic interest candidate text 必须是字符串")

        # 2. A configured runtime without embeddings has no semantic evidence.
        api = self._embedding_api
        if api is None or not self._db_path.exists():
            return tuple(0.0 for _ in texts)
        model = api.model_id
        if not model:
            return tuple(0.0 for _ in texts)

        # 3. Embed candidates, then compare only with completed passive turns.
        indexed = [(index, text) for index, text in enumerate(texts) if text.strip()]
        if not indexed:
            return tuple(0.0 for _ in texts)
        vectors = await api.embed_batch([text for _, text in indexed])
        if len(vectors) != len(indexed):
            raise RuntimeError("embedding provider 返回数量与候选不一致")
        prototypes = _load_turn_prototypes(self._db_path, model=model, cutoff=cutoff)
        scores = [0.0 for _ in texts]
        for (index, _), vector in zip(indexed, vectors, strict=True):
            scores[index] = min(
                0.999,
                max(
                    0.0,
                    max((_cosine(vector, item) for item in prototypes), default=0.0),
                )
                ** _SEMANTIC_CALIBRATION_POWER,
            )
        return tuple(scores)


def _load_turn_prototypes(
    db_path: Path, *, model: str, cutoff: str
) -> tuple[list[float], ...]:
    """Rebuild the last 256 passive user-assistant turn prototypes."""

    store = MessageEmbeddingStore(db_path)
    try:
        visible = dict(store.list_until(model=model, cutoff=cutoff))
    finally:
        store.close()
    if not visible:
        return ()
    with closing(sqlite3.connect(str(db_path))) as db:
        rows = db.execute(
            """
            SELECT id, session_key, seq, role, extra, julianday(ts)
            FROM messages
            WHERE julianday(ts) <= julianday(?)
            ORDER BY session_key, seq
            """,
            (cutoff,),
        ).fetchall()
    timestamped: list[tuple[float, str, int, list[float]]] = []
    pending_user: list[float] | None = None
    pending_session = ""
    for message_id, session_key, seq, role, extra_json, ts_julian in rows:
        vector = visible.get(str(message_id))
        if role == "user":
            pending_user = vector
            pending_session = str(session_key)
            continue
        if (
            role == "assistant"
            and vector is not None
            and pending_user is not None
            and pending_session == str(session_key)
            and not _is_proactive_message(extra_json)
        ):
            timestamped.append(
                (
                    float(ts_julian),
                    str(session_key),
                    int(seq),
                    _normalize_weighted(pending_user, vector),
                )
            )
            pending_user = None
    timestamped.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(item[3] for item in timestamped[-256:])


def _normalize_weighted(user: list[float], assistant: list[float]) -> list[float]:
    if len(user) != len(assistant) or not user:
        return []
    combined = [
        0.9 * left + 0.1 * right for left, right in zip(user, assistant, strict=True)
    ]
    norm = math.sqrt(sum(value * value for value in combined))
    return [value / norm for value in combined] if norm > 0 else []


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _is_proactive_message(extra_json: object) -> bool:
    try:
        extra = json.loads(str(extra_json or "{}"))
    except json.JSONDecodeError:
        return False
    if not isinstance(extra, dict):
        return False
    return bool(cast(dict[str, object], extra).get("proactive"))


CONVERSATION_SEMANTIC_INTEREST = ServiceKey[ConversationSemanticInterest](
    "core.conversation.semantic_interest"
)

__all__ = ["CONVERSATION_SEMANTIC_INTEREST", "ConversationSemanticInterest"]
