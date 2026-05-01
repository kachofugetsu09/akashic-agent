from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException

_RAG_HIT_RE = re.compile(
    r"RagHitLog\(item_id='(?P<id>[^']*)', memory_type='(?P<memory_type>[^']*)', "
    r"score=(?P<score>[-0-9.]+), summary='(?P<summary>(?:\\'|[^'])*)', "
    r"injected=(?P<injected>True|False)\)"
)
_MEMORY_META_RE = re.compile(r"（(?P<meta>[^（）]*(?:证据|src|有印象|不确定)[^（）]*)）$")


class RecallInspectorDashboardReader:
    def __init__(self, plugin_dir: Path) -> None:
        self.plugin_dir = plugin_dir
        self.data_path = plugin_dir / ".data" / "recall_turns.jsonl"
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return (self.plugin_dir / "plugin.py").exists()

    def get_overview(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "total": 0, "latest_at": None}
        turns = self._collect_turns()
        latest = turns[0]["timestamp"] if turns else None
        return {"available": True, "total": len(turns), "latest_at": latest}

    def list_turns(
        self,
        *,
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self.available:
            return [], 0
        normalized_q = q.strip().lower()
        turns = [
            item for item in self._collect_turns()
            if _matches_recall_turn(item, session_key=session_key, q=normalized_q)
        ]
        total = len(turns)
        safe_page = max(1, page)
        safe_size = max(1, min(page_size, 200))
        start = (safe_page - 1) * safe_size
        return turns[start:start + safe_size], total

    def get_turn(self, turn_id: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        for item in self._collect_turns():
            if item["turn_id"] == turn_id:
                return item
        return None

    def _collect_turns(self) -> list[dict[str, Any]]:
        records = self._read_records()
        turns: dict[str, dict[str, Any]] = {}
        for record in records:
            turn_id = str(record.get("turn_id", "") or "")
            if not turn_id:
                continue
            turn = turns.setdefault(
                turn_id,
                {
                    "turn_id": turn_id,
                    "session_key": str(record.get("session_key", "") or ""),
                    "channel": str(record.get("channel", "") or ""),
                    "chat_id": str(record.get("chat_id", "") or ""),
                    "user_text": str(record.get("user_text", "") or ""),
                    "timestamp": str(record.get("timestamp", "") or ""),
                    "context_prepare": None,
                    "recall_memory_calls": [],
                },
            )
            for key in ("session_key", "channel", "chat_id", "user_text", "timestamp"):
                if record.get(key) and not turn.get(key):
                    turn[key] = str(record.get(key) or "")
            if record.get("kind") == "context_prepare":
                raw_context = record.get("context_prepare") or {}
                turn["context_prepare"] = _normalize_context_prepare(raw_context)
            elif record.get("kind") == "recall_memory":
                calls_obj = turn.get("recall_memory_calls")
                if isinstance(calls_obj, list):
                    cast(list[Any], calls_obj).append(record.get("recall_memory") or {})
        result = list(turns.values())
        for item in result:
            context_prepare_raw: object = item.get("context_prepare") or {}
            context_prepare = (
                cast(dict[str, Any], context_prepare_raw)
                if isinstance(context_prepare_raw, dict)
                else {}
            )
            recall_calls_raw: object = item.get("recall_memory_calls") or []
            recall_calls: list[dict[str, Any]] = []
            if isinstance(recall_calls_raw, list):
                recall_calls = [
                    cast(dict[str, Any], raw_call)
                    for raw_call in cast(list[object], recall_calls_raw)
                    if isinstance(raw_call, dict)
                ]
            item["context_prepare_count"] = int(context_prepare.get("count") or 0)
            item["recall_memory_count"] = sum(
                int(call.get("count") or 0)
                for call in recall_calls
            )
            item["recall_call_count"] = len(recall_calls)
        result.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
        return result

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.data_path.exists():
            return []
        records: list[dict[str, Any]] = []
        try:
            lines = self.data_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(cast(dict[str, Any], value))
        return records


def register(app: FastAPI, plugin_dir: Path) -> None:
    reader = RecallInspectorDashboardReader(plugin_dir)

    @app.get("/api/dashboard/recall-inspector/overview")
    def get_recall_inspector_overview() -> dict[str, Any]:
        return reader.get_overview()

    @app.get("/api/dashboard/recall-inspector/turns")
    def list_recall_inspector_turns(
        session_key: str = "",
        q: str = "",
        page: int = 1,
        page_size: int = 50,
    ) -> dict[str, Any]:
        if not reader.available:
            raise HTTPException(status_code=404, detail="recall_inspector 插件未启用")
        items, total = reader.list_turns(
            session_key=session_key,
            q=q,
            page=page,
            page_size=page_size,
        )
        return {
            "items": items,
            "total": total,
            "page": max(1, page),
            "page_size": max(1, min(page_size, 200)),
        }

    @app.get("/api/dashboard/recall-inspector/turns/{turn_id}")
    def get_recall_inspector_turn(turn_id: str) -> dict[str, Any]:
        if not reader.available:
            raise HTTPException(status_code=404, detail="recall_inspector 插件未启用")
        item = reader.get_turn(turn_id)
        if item is None:
            raise HTTPException(status_code=404, detail="召回记录不存在")
        return item


def _matches_recall_turn(
    item: dict[str, Any],
    *,
    session_key: str,
    q: str,
) -> bool:
    if session_key and item.get("session_key") != session_key:
        return False
    if not q:
        return True
    haystack = " ".join(
        [
            str(item.get("session_key") or ""),
            str(item.get("user_text") or ""),
            json.dumps(item.get("context_prepare") or {}, ensure_ascii=False),
            json.dumps(item.get("recall_memory_calls") or [], ensure_ascii=False),
        ]
    ).lower()
    return q in haystack


def _normalize_context_prepare(value: object) -> dict[str, Any]:
    context = cast(dict[str, Any], value) if isinstance(value, dict) else {}
    items = _normalize_recall_items(context.get("items"))
    injected_items = _normalize_recall_items(context.get("injected_items"))
    if not items:
        items = _items_from_rag_trace_repr(context.get("retrieval_trace_raw"))
    if not injected_items:
        injected_items = [item for item in items if item.get("injected") is True]
    result = dict(context)
    result["items"] = items
    result["injected_items"] = injected_items
    result["count"] = len(items)
    return result


def _normalize_recall_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items: list[dict[str, Any]] = []
    for raw_item in cast(list[object], value):
        if not isinstance(raw_item, dict):
            continue
        item = dict(cast(dict[str, Any], raw_item))
        summary, tags = _split_memory_summary(str(item.get("summary", "") or ""))
        item["summary"] = summary
        if tags:
            item["tags"] = tags
        items.append(item)
    return items


def _items_from_rag_trace_repr(value: object) -> list[dict[str, Any]]:
    text = str(value or "")
    items: list[dict[str, Any]] = []
    for match in _RAG_HIT_RE.finditer(text):
        summary = match.group("summary").replace("\\'", "'")
        clean_summary, tags = _split_memory_summary(summary)
        items.append(
            {
                "id": match.group("id"),
                "memory_type": match.group("memory_type"),
                "summary": clean_summary,
                "score": float(match.group("score")),
                "injected": match.group("injected") == "True",
                "tags": tags,
            }
        )
    return items


def _split_memory_summary(summary: str) -> tuple[str, list[str]]:
    text = summary.strip()
    tags: list[str] = []
    while True:
        match = _MEMORY_META_RE.search(text)
        if match is None:
            return text, tags
        meta = match.group("meta")
        for part in meta.split("；"):
            label = part.strip()
            if label.startswith("(src:") or label.startswith("src:"):
                continue
            if label == "证据: 可回源原文":
                label = "可回源原文"
            elif label == "证据: 记忆摘要":
                label = "记忆摘要"
            if label and label not in tags:
                tags.append(label)
        text = text[: match.start()].strip()
