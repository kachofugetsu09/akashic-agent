from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import tomllib
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.plugins import Plugin, tool
from bus.events_lifecycle import TurnCommitted
from memory2.embedder import Embedder

from .db import FeedbackEvent, insert_feedback, open_db
from .events import ProactiveFeedbackRecorded
from .scorer import (
    latest_turn_messages,
    parse_quote_parts,
    proactive_since_previous_user,
    recent_proactive_messages,
    score_followup,
)

logger = logging.getLogger("plugin.proactive_feedback")

_QUEUE_MAX = 100


class ProactiveFeedbackPlugin(Plugin):
    name = "proactive_feedback"

    async def initialize(self) -> None:
        workspace = self.context.workspace
        if workspace is None:
            logger.warning("proactive_feedback 插件缺少 workspace，跳过加载")
            return
        self._workspace = workspace
        self._sessions_db = workspace / "sessions.db"
        self._db_path = workspace / "proactive_feedback" / "proactive_feedback.db"
        self._queue: asyncio.Queue[TurnCommitted] = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._project_root = self.context.plugin_dir.parents[1]
        self._embedder: Embedder | None = None
        self._worker_task = asyncio.create_task(
            self._run_worker(),
            name="proactive_feedback_worker",
        )
        self.context.event_bus.on(TurnCommitted, self._on_turn_committed)

    async def terminate(self) -> None:
        task = getattr(self, "_worker_task", None)
        if task is None:
            return
        _ = task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    def _on_turn_committed(self, event: TurnCommitted) -> None:
        if event.persisted_user_message is None:
            return
        queue = getattr(self, "_queue", None)
        if queue is None:
            return
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("proactive_feedback queue full, drop session=%s", event.session_key)

    async def _run_worker(self) -> None:
        while True:
            event = await self._queue.get()
            try:
                await self._process(event)
            except Exception:
                logger.exception("proactive_feedback process failed")
            finally:
                self._queue.task_done()

    async def _process(self, event: TurnCommitted) -> None:
        user_text = event.persisted_user_message
        if not user_text or not event.assistant_response:
            return
        if not self._sessions_db.exists():
            return

        source = sqlite3.connect(self._sessions_db)
        source.row_factory = sqlite3.Row
        try:
            turn = latest_turn_messages(
                source,
                session_key=event.session_key,
                user_content=user_text,
                assistant_content=event.assistant_response,
            )
            if turn is None:
                return
            user, assistant = turn
            quote = parse_quote_parts(user.content)
            allow_pua = not bool(quote.quoted_text)
            if quote.quoted_text:
                candidates = recent_proactive_messages(
                    source,
                    session_key=event.session_key,
                    before_seq=user.seq,
                    limit=64,
                )
            else:
                candidates = proactive_since_previous_user(
                    source,
                    session_key=event.session_key,
                    before_seq=user.seq,
                    limit=8,
                )
        finally:
            source.close()
        if not candidates:
            return

        try:
            scored = await score_followup(
                embed_batch=self._get_embedder().embed_batch if allow_pua else _no_embed,
                user=user,
                assistant=assistant,
                candidates=candidates,
                allow_pua=allow_pua,
            )
        except Exception:
            logger.exception("proactive_feedback scoring failed")
            scored = None
            if candidates:
                sink = open_db(self._db_path)
                try:
                    feedback = FeedbackEvent(
                        session_key=event.session_key,
                        user_message_id=user.id,
                        assistant_message_id=assistant.id,
                        proactive_message_id=candidates[0].id,
                        feedback_type="unscored",
                        confidence="low",
                        pa_score=None,
                        pua_score=None,
                        lag_seconds=None,
                        candidate_count=len(candidates),
                        matched_by="recent_pua",
                        reason="scoring_failed",
                    )
                    event_id = insert_feedback(sink, feedback)
                finally:
                    sink.close()
                if event_id is not None:
                    await self.context.event_bus.fanout(
                        ProactiveFeedbackRecorded(event_id=event_id, feedback=feedback)
                    )
        if scored is None:
            return

        sink = open_db(self._db_path)
        try:
            feedback = FeedbackEvent(
                session_key=event.session_key,
                user_message_id=user.id,
                assistant_message_id=assistant.id,
                proactive_message_id=scored.proactive.id,
                feedback_type=scored.feedback_type,
                confidence=scored.confidence,
                pa_score=scored.pa_score,
                pua_score=scored.pua_score,
                lag_seconds=scored.lag_seconds,
                candidate_count=scored.candidate_count,
                matched_by=scored.matched_by,
                reason=scored.reason,
            )
            event_id = insert_feedback(sink, feedback)
        finally:
            sink.close()
        if event_id is not None:
            await self.context.event_bus.fanout(
                ProactiveFeedbackRecorded(event_id=event_id, feedback=feedback)
            )

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = _build_embedder(self._project_root)
        return self._embedder

    @tool(
        "get_proactive_feedback_summary",
        risk="read-only",
        search_hint="查询 proactive 主动推送反馈统计摘要",
    )
    async def get_summary(self, event: Any) -> dict[str, Any]:
        """查询 proactive 主动推送反馈统计摘要。"""
        _ = event
        db_path = getattr(self, "_db_path", None)
        if db_path is None or not Path(db_path).exists():
            return {"total": 0, "by_type": [], "by_confidence": []}
        conn = open_db(Path(db_path))
        try:
            total = conn.execute("SELECT count(*) FROM proactive_feedback_events").fetchone()[0]
            by_type = _rows(
                conn.execute(
                    """
                    SELECT feedback_type, count(*) AS count
                    FROM proactive_feedback_events
                    GROUP BY feedback_type
                    ORDER BY count DESC
                    """
                ).fetchall()
            )
            by_confidence = _rows(
                conn.execute(
                    """
                    SELECT confidence, count(*) AS count
                    FROM proactive_feedback_events
                    GROUP BY confidence
                    ORDER BY count DESC
                    """
                ).fetchall()
            )
        finally:
            conn.close()
        return {"total": total, "by_type": by_type, "by_confidence": by_confidence}


def _build_embedder(root: Path) -> Embedder:
    data = tomllib.loads((root / "config.toml").read_text())
    embedding = data["memory"]["embedding"]
    api_key = str(embedding["api_key"])
    if api_key.startswith("$"):
        api_key = os.environ[api_key[1:]]
    return Embedder(
        base_url=str(embedding["base_url"]),
        api_key=api_key,
        model=str(embedding.get("model", "text-embedding-v3")),
        output_dimensionality=embedding.get("output_dimensionality"),
    )


async def _no_embed(texts: list[str]) -> list[list[float]]:
    _ = texts
    raise RuntimeError("quoted feedback must not call embedding")


def _rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]
