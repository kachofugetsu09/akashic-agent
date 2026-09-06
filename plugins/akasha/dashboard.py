"""Expose Message-backed Akasha recall records to the v3 dashboard."""

from __future__ import annotations

import json
from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugins.snapshot import get_current_runtime_snapshot

from .inspector import RecallInspector
from .message_plugin import AKASHA_RECORDS_VIEW
from .recalls import ContextSource, ProgramSource, Recall, RecallRecords, ToolSource


def _records() -> RecallRecords:
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        raise RuntimeError("Akasha Dashboard 请求缺少实际 runtime snapshot")
    return snapshot.composition_root.context.require(AKASHA_RECORDS_VIEW)()


def _inspector() -> RecallInspector:
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        raise RuntimeError("Akasha Dashboard 请求缺少实际 runtime snapshot")
    context = snapshot.composition_root.context
    records = context.require(AKASHA_RECORDS_VIEW)()
    return RecallInspector(
        read=records.read,
        list_records=records.list,
        catalog=context.require(MESSAGE_CATALOG),
    )


def _row(inspector: RecallInspector, identity: str, recall: Recall) -> dict[str, object]:
    source = recall.source
    if isinstance(source, ContextSource):
        query_text = f"上下文查询 · {source.source} · #{source.through_seq}"
        session_key = source.session_id
        seq = source.through_seq
    elif isinstance(source, ToolSource):
        query_text = f"工具查询 · {source.call_ref.message_id}:{source.call_ref.part_index}"
        session_key = source.session_id
        seq = -1
    else:
        assert isinstance(source, ProgramSource)
        query_text = source.query
        session_key = ""
        seq = -1
    hit_count = len(recall.hits)
    completion_count = sum(1 for hit in recall.hits if hit.lane == "completion")
    return {
        "query_id": identity,
        "session_key": session_key,
        "seq": seq,
        "ts": recall.timestamp.isoformat(),
        "query_text": query_text,
        "seed_count": hit_count,
        "activation_capture_available": False,
        "recall_capture_available": True,
        "activation_count": 0,
        "completion_count": completion_count,
        "pushes": recall.pushes,
        "residual_l1": recall.residual_l1,
    }


def _detail(inspector: RecallInspector, identity: str) -> dict[str, object] | None:
    value = inspector.mobile_detail(identity)
    if value is None:
        return None
    source = value.get("source")
    session_key = source.get("session_id", "") if isinstance(source, Mapping) else ""
    seq = source.get("through_seq", -1) if isinstance(source, Mapping) else -1
    precise: list[dict[str, object]] = []
    completion: list[dict[str, object]] = []
    raw_hits = value.get("hits")
    if not isinstance(raw_hits, list):
        raise ValueError("Akasha 记录的 hits 不是列表")
    for hit in raw_hits:
        if not isinstance(hit, Mapping):
            continue
        target = precise if hit.get("lane") == "dense" else completion
        for message in hit.get("messages", ()):
            if not isinstance(message, Mapping):
                continue
            target.append({
                "user_text": message.get("preview", ""),
                "assistant_preview": "",
                "ts": value["ts"],
                "score": hit.get("score"),
                "sources": hit.get("sources", ()),
                "graph_only": hit.get("lane") == "completion",
                "relation_path": (),
            })
    return {
        "query_id": identity,
        "session_key": session_key,
        "seq": seq,
        "ts": value["ts"],
        "query_text": value["query_text"],
        "assistant_text": "",
        "seed_count": len(precise) + len(completion),
        "activation_capture_available": False,
        "recall_capture_available": True,
        "activation_count": 0,
        "completion_count": len(completion),
        "graph_only_count": len(completion),
        "basin_count": 0,
        "surprise": None,
        "observed_mass": None,
        "recurrent_mass": None,
        "reactivated_mass": None,
        "potentiated_mass": None,
        "inhibited_mass": None,
        "seeds": [],
        "activation_items": [],
        "left": precise,
        "right": completion,
        "tool_left": [],
        "tool_right": [],
        "left_count": len(precise),
        "right_count": len(completion),
        "tool_left_count": 0,
        "tool_right_count": 0,
        "inject_chars": 0,
        "text_block_preview": "",
        "pushes": value["pushes"],
        "residual_l1": value["residual_l1"],
    }


def register(app: FastAPI, context: DashboardContext) -> None:
    """Register read-only routes over the runtime's Message and OwnerState services."""

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
        inspector = _inspector()
        records = _records()
        all_rows = [
            _row(inspector, identity, recall)
            for identity, recall in records.list()
        ]
        q = q.strip().casefold()
        rows = [row for row in all_rows
                if (not session_key or row["session_key"] == session_key)
                and (not q or q in json.dumps(row, ensure_ascii=False).casefold())]
        start = (page - 1) * page_size
        return {"items": rows[start:start + page_size], "total": len(rows),
                "page": page, "page_size": page_size}

    @app.get("/api/dashboard/akasha-inspector/turns/{query_id:path}")
    async def get_turn(query_id: str) -> dict[str, object]:
        item = _detail(_inspector(), query_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Akasha 检索记录不存在")
        return item
