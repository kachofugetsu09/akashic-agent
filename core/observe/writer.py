"""异步 TraceWriter：把 TurnTrace / RagTrace 写入 SQLite。

非阻塞：调用方用 emit() put_nowait，后台 task 消费队列写 DB。
Queue 满时 drop + 计数，不崩溃主循环。
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.observe.db import open_db
from core.observe.events import MemoryWriteTrace, ProactiveDecisionTrace, RagItemTrace, RagTrace, TurnTrace

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
            TurnTrace | RagTrace | ProactiveDecisionTrace | MemoryWriteTrace
        ] = asyncio.Queue(
            maxsize=_QUEUE_MAX
        )
        self._dropped = 0

    # ── 公共接口 ─────────────────────────────────

    def emit(self, event: TurnTrace | RagTrace | ProactiveDecisionTrace | MemoryWriteTrace) -> None:
        """非阻塞 emit。Queue 满时 drop 并记录计数。"""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("observe queue full, total_dropped=%d", self._dropped)

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
            # flush remaining on shutdown
            while not self._queue.empty():
                try:
                    e = self._queue.get_nowait()
                    self._write_one(conn, e)
                except Exception:
                    pass
            conn.close()
            logger.info("observe writer stopped")

    # ── 内部写入 ─────────────────────────────────

    def _write_one(
        self, conn, event: TurnTrace | RagTrace | ProactiveDecisionTrace | MemoryWriteTrace
    ) -> None:
        ts = _now_iso()
        if isinstance(event, TurnTrace):
            _write_turn(conn, event, ts)
        elif isinstance(event, RagTrace):
            _write_rag(conn, event, ts)
        elif isinstance(event, ProactiveDecisionTrace):
            _write_proactive_decision(conn, event, ts)
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
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                e.error,
            ),
        )


def _write_rag(conn, e: RagTrace, ts: str) -> None:
    with conn:
        cur = conn.execute(
            """
            INSERT INTO rag_events (
                ts, source, session_key, tick_id,
                original_query, query, gate_type,
                route_decision, route_latency_ms,
                hyde_hypothesis,
                history_scope_mode, history_gate_reason,
                injected_block, preference_block, preference_query,
                sufficiency_check_json, fallback_reason, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                e.source,
                e.session_key,
                e.tick_id or None,
                e.original_query,
                e.query,
                e.gate_type,
                e.route_decision,
                e.route_latency_ms,
                e.hyde_hypothesis,
                e.history_scope_mode,
                e.history_gate_reason,
                e.injected_block or None,
                e.preference_block or None,
                e.preference_query or None,
                e.sufficiency_check_json,
                e.fallback_reason or None,
                e.error,
            ),
        )
        rag_event_id = cur.lastrowid
        if e.items:
            conn.executemany(
                """
                INSERT INTO rag_items (
                    rag_event_id, item_id, memory_type, score, summary,
                    happened_at, extra_json, retrieval_path, injected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        rag_event_id,
                        item.item_id,
                        item.memory_type,
                        item.score,
                        item.summary,
                        item.happened_at,
                        item.extra_json,
                        item.retrieval_path,
                        1 if item.injected else 0,
                    )
                    for item in e.items
                ],
            )


def _write_proactive_decision(conn, e: ProactiveDecisionTrace, ts: str) -> None:
    # 1. 保留 stage 列里的新阶段名，同时把 stage_result_json 回填到旧列，兼容历史读侧。
    stage_json_columns_map = {
        "gate": {"gate_result_json"},
        "sense": {"sense_result_json"},
        "pre_score": {"pre_score_result_json"},
        "fetch_filter": {"fetch_filter_result_json"},
        "score": {"score_result_json"},
        "decide": {"decide_result_json"},
        "act": {"act_result_json"},
        "gate_and_sense": {
            "gate_result_json",
            "sense_result_json",
            "pre_score_result_json",
        },
        "evaluate": {
            "fetch_filter_result_json",
            "score_result_json",
        },
        "judge_and_send": {
            "decide_result_json",
            "act_result_json",
        },
    }
    stage_json_columns = stage_json_columns_map.get(e.stage, set())
    stage_json = e.stage_result_json if stage_json_columns else None
    payload = {
        "tick_id": e.tick_id,
        "ts": ts,
        "updated_ts": ts,
        "session_key": e.session_key,
        "stage": e.stage,
        "reason_code": e.reason_code,
        "should_send": None if e.should_send is None else (1 if e.should_send else 0),
        "action": e.action,
        "gate_reason": e.gate_reason,
        "pre_score": e.pre_score,
        "base_score": e.base_score,
        "draw_score": e.draw_score,
        "decision_score": e.decision_score,
        "send_threshold": e.send_threshold,
        "interruptibility": e.interruptibility,
        "candidate_count": e.candidate_count,
        "candidate_item_ids": (
            json.dumps(e.candidate_item_ids, ensure_ascii=False)
            if e.candidate_item_ids
            else None
        ),
        "sleep_state": e.sleep_state,
        "sleep_prob": e.sleep_prob,
        "sleep_available": (
            None if e.sleep_available is None else (1 if e.sleep_available else 0)
        ),
        "sleep_data_lag_min": e.sleep_data_lag_min,
        "user_replied_after_last_proactive": (
            None
            if e.user_replied_after_last_proactive is None
            else (1 if e.user_replied_after_last_proactive else 0)
        ),
        "proactive_sent_24h": e.proactive_sent_24h,
        "fresh_items_24h": e.fresh_items_24h,
        "delivery_key": e.delivery_key,
        "is_delivery_duplicate": (
            None
            if e.is_delivery_duplicate is None
            else (1 if e.is_delivery_duplicate else 0)
        ),
        "is_message_duplicate": (
            None
            if e.is_message_duplicate is None
            else (1 if e.is_message_duplicate else 0)
        ),
        "delivery_attempted": (
            None
            if e.delivery_attempted is None
            else (1 if e.delivery_attempted else 0)
        ),
        "delivery_result": e.delivery_result,
        "reasoning_preview": e.reasoning_preview,
        "reasoning": e.reasoning,
        "evidence_item_ids": (
            json.dumps(e.evidence_item_ids, ensure_ascii=False)
            if e.evidence_item_ids
            else None
        ),
        "source_refs_json": e.source_refs_json,
        "fetched_urls": (
            json.dumps(e.fetched_urls, ensure_ascii=False)
            if e.fetched_urls
            else None
        ),
        "sent_message": e.sent_message,
        "candidates_json": e.candidates_json,
        "research_status": e.research_status,
        "research_rounds_used": e.research_rounds_used,
        "research_tools_called": (
            json.dumps(e.research_tools_called, ensure_ascii=False)
            if e.research_tools_called
            else None
        ),
        "research_evidence_count": e.research_evidence_count,
        "research_reason": e.research_reason,
        "fact_claims_count": e.fact_claims_count,
        "gate_result_json": (
            stage_json if "gate_result_json" in stage_json_columns else None
        ),
        "sense_result_json": (
            stage_json if "sense_result_json" in stage_json_columns else None
        ),
        "pre_score_result_json": (
            stage_json if "pre_score_result_json" in stage_json_columns else None
        ),
        "fetch_filter_result_json": (
            stage_json if "fetch_filter_result_json" in stage_json_columns else None
        ),
        "score_result_json": (
            stage_json if "score_result_json" in stage_json_columns else None
        ),
        "decide_result_json": (
            stage_json if "decide_result_json" in stage_json_columns else None
        ),
        "act_result_json": (
            stage_json if "act_result_json" in stage_json_columns else None
        ),
        "decision_signals_json": e.decision_signals_json,
        "error": e.error,
    }
    columns = list(payload.keys())
    insert_columns = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    values = [payload[col] for col in columns]
    update_columns = [
        "updated_ts",
        "session_key",
        "stage",
        "reason_code",
        "should_send",
        "action",
        "gate_reason",
        "pre_score",
        "base_score",
        "draw_score",
        "decision_score",
        "send_threshold",
        "interruptibility",
        "candidate_count",
        "candidate_item_ids",
        "sleep_state",
        "sleep_prob",
        "sleep_available",
        "sleep_data_lag_min",
        "user_replied_after_last_proactive",
        "proactive_sent_24h",
        "fresh_items_24h",
        "delivery_key",
        "is_delivery_duplicate",
        "is_message_duplicate",
        "delivery_attempted",
        "delivery_result",
        "reasoning_preview",
        "reasoning",
        "evidence_item_ids",
        "source_refs_json",
        "fetched_urls",
        "sent_message",
        "candidates_json",
        "research_status",
        "research_rounds_used",
        "research_tools_called",
        "research_evidence_count",
        "research_reason",
        "fact_claims_count",
        "gate_result_json",
        "sense_result_json",
        "pre_score_result_json",
        "fetch_filter_result_json",
        "score_result_json",
        "decide_result_json",
        "act_result_json",
        "decision_signals_json",
        "error",
    ]
    updates = []
    for col in update_columns:
        if col == "session_key":
            updates.append(
                "session_key = CASE WHEN excluded.session_key <> '' "
                "THEN excluded.session_key ELSE proactive_decisions.session_key END"
            )
        elif col in {"updated_ts", "stage"}:
            updates.append(f"{col} = excluded.{col}")
        else:
            updates.append(f"{col} = COALESCE(excluded.{col}, proactive_decisions.{col})")
    with conn:
        conn.execute(
            f"""
            INSERT INTO proactive_decisions ({insert_columns})
            VALUES ({placeholders})
            ON CONFLICT(tick_id) DO UPDATE SET
                {", ".join(updates)}
            """,
            values,
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
