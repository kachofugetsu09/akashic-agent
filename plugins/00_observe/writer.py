"""异步 TraceWriter：把 TurnTrace / RagQueryLog 写入 SQLite。

非阻塞：调用方用 emit() put_nowait，后台 task 消费队列写 DB。
Queue 满时 drop + 计数，不崩溃主循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .db import open_db
from .events import MemoryWriteTrace, RagQueryLog, TurnTrace

logger = logging.getLogger("observe.writer")

_QUEUE_MAX = 500
_ARG_MAX = 300
_RESULT_MAX = 500


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_tool_calls(tool_calls: list[dict]) -> str | None:
    if not tool_calls:
        return None
    slim = [
        {
            "name": c.get("name", ""),
            "args": str(c.get("args", c.get("arguments", "")))[:_ARG_MAX],
            "result": str(c.get("result", ""))[:_RESULT_MAX],
        }
        for c in tool_calls
    ]
    return json.dumps(slim, ensure_ascii=False)


class TraceWriter:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._queue: asyncio.Queue[
            TurnTrace | RagQueryLog | MemoryWriteTrace
        ] = asyncio.Queue(
            maxsize=_QUEUE_MAX
        )
        self._dropped = 0

    # ── 公共接口 ─────────────────────────────────

    def emit(self, event: TurnTrace | RagQueryLog | MemoryWriteTrace) -> None:
        """非阻塞 emit。Queue 满时 drop 并记录计数。"""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("observe queue full, total_dropped=%d", self._dropped)

    async def drain(self) -> None:
        """等待已入队事件写入完成。"""
        await self._queue.join()

    async def run(self) -> None:
        """后台循环，持续消费队列写 DB。作为 asyncio task 运行。"""
        conn = open_db(self._db_path)
        logger.info("observe writer started: %s", self._db_path)
        try:
            while True:
                event = await self._queue.get()
                try:
                    self._write_one(conn, event)
                except Exception:
                    logger.exception("observe write failed for %s", type(event).__name__)
                finally:
                    self._queue.task_done()
        finally:
            # flush remaining on shutdown
            while not self._queue.empty():
                try:
                    e = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    self._write_one(conn, e)
                except Exception:
                    pass
                finally:
                    self._queue.task_done()
            conn.close()
            logger.info("observe writer stopped")

    # ── 内部写入 ─────────────────────────────────

    def _write_one(
        self, conn, event: TurnTrace | RagQueryLog | MemoryWriteTrace
    ) -> None:
        ts = _now_iso()
        if isinstance(event, TurnTrace):
            _write_turn(conn, event, ts)
        elif isinstance(event, RagQueryLog):
            _write_rag(conn, event, ts)
        elif isinstance(event, MemoryWriteTrace):
            _write_memory_write(conn, event, ts)


# ── DB 写入函数 ───────────────────────────────────────────────────────────────


def _write_turn(conn, e: TurnTrace, ts: str) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO turns (
                ts, source, session_key, user_msg, llm_output,
                raw_llm_output, meme_tag, meme_media_count,
                tool_calls, tool_chain_json,
                history_window, history_messages, history_chars,
                history_tokens, prompt_tokens, next_turn_baseline_tokens,
                react_iteration_count, react_input_sum_tokens,
                react_input_peak_tokens, react_final_input_tokens,
                react_cache_prompt_tokens, react_cache_hit_tokens,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                e.source,
                e.session_key,
                e.user_msg,
                e.llm_output,
                e.raw_llm_output,
                e.meme_tag,
                e.meme_media_count,
                _serialize_tool_calls(e.tool_calls),
                e.tool_chain_json,
                e.history_window,
                e.history_messages,
                e.history_chars,
                e.history_tokens,
                e.prompt_tokens,
                e.next_turn_baseline_tokens,
                e.react_iteration_count,
                e.react_input_sum_tokens,
                e.react_input_peak_tokens,
                e.react_final_input_tokens,
                e.react_cache_prompt_tokens,
                e.react_cache_hit_tokens,
                e.error,
            ),
        )


def _write_rag(conn, e: RagQueryLog, ts: str) -> None:
    hits_json = (
        json.dumps(
            [
                {
                    "id": h.item_id,
                    "type": h.memory_type,
                    "score": round(h.score, 4),
                    "summary": h.summary,
                    "injected": h.injected,
                }
                for h in e.hits
            ],
            ensure_ascii=False,
        )
        if e.hits
        else None
    )
    with conn:
        conn.execute(
            """
            INSERT INTO rag_queries (
                ts, caller, session_key, query, orig_query,
                aux_queries, hits_json, injected_count, route_decision, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                e.caller,
                e.session_key,
                e.query,
                e.orig_query,
                json.dumps(e.aux_queries, ensure_ascii=False) if e.aux_queries else None,
                hits_json,
                e.injected_count,
                e.route_decision,
                e.error,
            ),
        )


def _write_memory_write(conn, e: MemoryWriteTrace, ts: str) -> None:
    import json as _json
    with conn:
        conn.execute(
            """
            INSERT INTO memory_writes (ts, session_key, source_ref, action, memory_type, item_id, summary, superseded_ids, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                e.session_key,
                e.source_ref,
                e.action,
                e.memory_type,
                e.item_id,
                e.summary,
                _json.dumps(e.superseded_ids, ensure_ascii=False) if e.superseded_ids else None,
                e.error,
            ),
        )
