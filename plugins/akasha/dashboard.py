"""Expose real Recall and Message facts to the v3 dashboard."""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugins.snapshot import get_current_runtime_snapshot
from session.message import ContentPart, Message

from .message_plugin import AKASHA_RECORDS_VIEW
from .recalls import ContextSource, Hit, ProgramSource, Recall, RecallRecordsRead, ToolSource


def _runtime_context():
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        raise RuntimeError("Akasha Dashboard 请求缺少实际 runtime snapshot")
    return snapshot.composition_root.context


def _records() -> RecallRecordsRead:
    return _runtime_context().require(AKASHA_RECORDS_VIEW)()


def _message_text(message: Message) -> str:
    parts = getattr(message.body, "parts", ())
    return "\n".join(
        str(part.value)
        for part in parts
        if isinstance(part, ContentPart) and part.kind == "text" and isinstance(part.value, str)
    )


def _message(
    message: Message,
    *,
    lane: str,
    presented: bool,
    full_text: bool,
) -> dict[str, object]:
    text = _message_text(message)
    visible = text if full_text else text[:240]
    return {
        "message_id": message.message_id,
        "session_id": message.session_id,
        "seq": message.seq,
        "recorded_at": message.recorded_at.isoformat(),
        "author": message.author,
        "source": message.source,
        "lane": lane,
        "presented": presented,
        "text": visible,
        "text_truncated": len(visible) < len(text),
    }


def _hit_messages(
    recall: Recall,
    hit: Hit,
    *,
    full_text: bool,
) -> dict[str, object]:
    reader = _runtime_context().require(MESSAGE_CATALOG).reader(hit.session_id)
    messages: list[dict[str, object]] = []
    for message_id in hit.message_ids:
        message = reader.get(message_id)
        if message is None:
            raise ValueError(f"召回出处消息缺失: {message_id}")
        messages.append(_message(
            message,
            lane=hit.lane,
            presented=message_id in recall.presented_message_ids,
            full_text=full_text,
        ))
    return {
        "lane": hit.lane,
        "score": hit.score,
        "sources": list(hit.sources),
        "basin_ids": list(hit.basin_ids),
        "messages": messages,
    }


def _source_fields(recall: Recall) -> tuple[str, str, int, dict[str, object]]:
    source = recall.source
    source_data = source.model_dump(mode="json")
    if isinstance(source, ContextSource):
        query_text = f"上下文查询 · {source.source} · #{source.through_seq}"
        return query_text, source.session_id, source.through_seq, source_data
    if isinstance(source, ToolSource):
        query_text = f"工具查询 · {source.call_ref.message_id}:{source.call_ref.part_index}"
        return query_text, source.session_id, -1, source_data
    assert isinstance(source, ProgramSource)
    return source.query, "", -1, source_data


def _row(identity: str, recall: Recall, *, full_text: bool) -> dict[str, object]:
    query_text, session_key, seq, source = _source_fields(recall)
    hits = [_hit_messages(recall, hit, full_text=full_text) for hit in recall.hits]
    return {
        "query_id": identity,
        "session_key": session_key,
        "seq": seq,
        "ts": recall.timestamp.isoformat(),
        "query_text": query_text,
        "source": source,
        "graph_version": recall.graph_version,
        "limit": recall.limit,
        "hit_count": len(recall.hits),
        "presented_count": len(recall.presented_message_ids),
        "dense_count": sum(1 for hit in recall.hits if hit.lane == "dense"),
        "completion_count": sum(1 for hit in recall.hits if hit.lane == "completion"),
        "active_basin_count": recall.active_basin_count,
        "pushes": recall.pushes,
        "residual_l1": recall.residual_l1,
        "hits": hits,
    }


def register(app: FastAPI, context: DashboardContext) -> None:
    """Register read-only routes over Recall and the original Message log."""

    @app.get("/api/dashboard/akasha-inspector/overview")
    async def get_overview() -> dict[str, object]:
        return {"available": True, "total": len(_records().list())}

    @app.get("/api/dashboard/akasha-inspector/turns")
    async def list_turns(
        session_key: str = "",
        q: str = "",
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, object]:
        rows = [_row(identity, recall, full_text=False) for identity, recall in _records().list()]
        query = q.strip().casefold()
        rows = [
            row for row in rows
            if (not session_key or row["session_key"] == session_key)
            and (not query or query in json.dumps(row, ensure_ascii=False).casefold())
        ]
        start = (page - 1) * page_size
        return {
            "items": rows[start:start + page_size],
            "total": len(rows),
            "page": page,
            "page_size": page_size,
        }

    @app.get("/api/dashboard/akasha-inspector/turns/{query_id:path}")
    async def get_turn(query_id: str) -> dict[str, object]:
        recall = _records().read(query_id)
        if recall is None:
            raise HTTPException(status_code=404, detail="Akasha 检索记录不存在")
        return _row(query_id, recall, full_text=True)
