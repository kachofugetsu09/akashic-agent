from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from typing import Any, cast

from agent.lifecycle.types import AfterToolResultCtx, BeforeTurnCtx
from agent.plugins import Plugin, on_tool_result

_CTX_SLOT = "session:ctx"
_ITEM_LINE_RE = re.compile(r"^-\s+\[([^\]]+)\]\s*(.*)$")
_META_RE = re.compile(r"（(?P<meta>[^（）]*(?:证据|src|有印象|不确定)[^（）]*)）$")


class ContextPrepareRecordModule:
    requires = (_CTX_SLOT,)

    def __init__(self, plugin: "RecallInspector") -> None:
        self._plugin = plugin

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_CTX_SLOT)
        if isinstance(ctx, BeforeTurnCtx):
            self._plugin.record_context_prepare(ctx)
        return frame


class RecallInspector(Plugin):
    name = "recall_inspector"

    async def initialize(self) -> None:
        self._lock = threading.RLock()
        self._active_turns: dict[str, str] = {}
        self._data_path = self.context.plugin_dir / ".data" / "recall_turns.jsonl"
        self._data_path.parent.mkdir(parents=True, exist_ok=True)

    def before_turn_modules_late(self) -> list[object]:
        return [ContextPrepareRecordModule(self)]

    def record_context_prepare(self, event: BeforeTurnCtx) -> None:
        turn_id = _turn_id(event.session_key, event.timestamp.isoformat(), event.content)
        self._active_turns[event.session_key] = turn_id
        block = event.retrieved_memory_block or ""
        injected_items = _items_from_block(block)
        all_hits = _hits_from_trace(event.retrieval_trace_raw)
        self._append(
            {
                "kind": "context_prepare",
                "turn_id": turn_id,
                "session_key": event.session_key,
                "channel": event.channel,
                "chat_id": event.chat_id,
                "user_text": event.content,
                "timestamp": event.timestamp.isoformat(),
                "created_at": _now_iso(),
                "context_prepare": {
                    "count": len(all_hits) if all_hits else len(injected_items),
                    "items": all_hits or injected_items,
                    "injected_items": injected_items,
                    "raw_block": block,
                    "retrieval_trace_raw": _jsonable(event.retrieval_trace_raw),
                },
            }
        )

    @on_tool_result()
    async def record_recall_memory(self, event: AfterToolResultCtx) -> None:
        if event.tool_name != "recall_memory":
            return
        turn_id = self._active_turns.get(event.session_key)
        if not turn_id:
            turn_id = _turn_id(event.session_key, _now_iso(), json.dumps(event.arguments, ensure_ascii=False))
        payload = _safe_json(event.result)
        raw_items: object = payload.get("items")
        items: list[dict[str, Any]] = []
        if isinstance(raw_items, list):
            items = [
                cast(dict[str, Any], raw_item)
                for raw_item in cast(list[object], raw_items)
                if isinstance(raw_item, dict)
            ]
        self._append(
            {
                "kind": "recall_memory",
                "turn_id": turn_id,
                "session_key": event.session_key,
                "channel": event.channel,
                "chat_id": event.chat_id,
                "timestamp": _now_iso(),
                "created_at": _now_iso(),
                "recall_memory": {
                    "arguments": dict(event.arguments),
                    "status": event.status,
                    "count": len(items),
                    "items": [_compact_item(item) for item in items],
                    "raw_result": payload,
                },
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with self._data_path.open("a", encoding="utf-8") as fh:
                _ = fh.write(line + "\n")


def _turn_id(session_key: str, timestamp: str, content: str) -> str:
    raw = f"{session_key}\n{timestamp}\n{content}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json(text: str) -> dict[str, Any]:
    try:
        value: object = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}
    return cast(dict[str, Any], value) if isinstance(value, dict) else {"raw": value}


def _items_from_block(block: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    section = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith("##"):
            section = line.lstrip("#").strip()
            continue
        match = _ITEM_LINE_RE.match(line)
        if not match:
            continue
        item_id, summary = match.groups()
        items.append(
            {
                "id": item_id.strip(),
                "summary": _split_summary_meta(summary)[0],
                "tags": _split_summary_meta(summary)[1],
                "section": section,
                "injected": True,
            }
        )
    return items


def _hits_from_trace(trace: Any) -> list[dict[str, Any]]:
    hits = getattr(trace, "hits", None)
    if not isinstance(hits, list):
        return []
    items: list[dict[str, Any]] = []
    for hit in hits:
        item_id = str(getattr(hit, "item_id", "") or "")
        if not item_id:
            continue
        confidence_label = str(getattr(hit, "confidence_label", "") or "")
        items.append(
            {
                "id": item_id,
                "summary": _split_summary_meta(str(getattr(hit, "summary", "") or ""))[0],
                "memory_type": str(getattr(hit, "memory_type", "") or ""),
                "score": getattr(hit, "score", None),
                "injected": bool(getattr(hit, "injected", False)),
                "tags": [confidence_label] if confidence_label else [],
            }
        )
    return items


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    summary, tags = _split_summary_meta(str(item.get("summary", "") or ""))
    return {
        "id": str(item.get("id", "") or ""),
        "memory_type": str(item.get("memory_type", "") or ""),
        "summary": summary,
        "tags": tags,
        "happened_at": str(item.get("happened_at", "") or ""),
        "score": item.get("score"),
        "source_ref": str(item.get("source_ref", "") or ""),
    }


def _split_summary_meta(summary: str) -> tuple[str, list[str]]:
    text = summary.strip()
    tags: list[str] = []
    while True:
        match = _META_RE.search(text)
        if match is None:
            return text, tags
        for part in match.group("meta").split("；"):
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


def _jsonable(value: Any) -> Any:
    try:
        _ = json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return repr(value)
