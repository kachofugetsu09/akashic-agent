from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

MAX_METRICS_KEYS = 8
MAX_METRICS_VALUE_STR_LEN = 60
_TIME_KEY_SUFFIXES = ("_at", "_time", "_ts")
_TIME_KEYS = {"last_seen", "updated_at", "published_at", "timestamp", "ts"}


def _trim_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _normalize_metrics(metrics: Any) -> dict[str, Any] | None:
    if not isinstance(metrics, dict) or not metrics:
        return None

    normalized: dict[str, Any] = {}
    items = list(metrics.items())
    for key, value in items[:MAX_METRICS_KEYS]:
        key_text = str(key).strip()
        if not key_text:
            continue
        if isinstance(value, str):
            normalized[key_text] = _trim_text(value, MAX_METRICS_VALUE_STR_LEN)
            continue
        if isinstance(value, (int, float, bool)) or value is None:
            normalized[key_text] = value
            continue

        text = json.dumps(value, ensure_ascii=False)
        normalized[key_text] = _trim_text(text, MAX_METRICS_VALUE_STR_LEN)

    truncated = len(items) - MAX_METRICS_KEYS
    if truncated > 0:
        normalized["_truncated_keys"] = truncated

    return normalized or None


def _looks_like_time_key(key: str) -> bool:
    return key in _TIME_KEYS or key.endswith(_TIME_KEY_SUFFIXES)


def _format_local_time(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        return None
    local_dt = dt.astimezone()
    return local_dt.strftime("%Y-%m-%d %H:%M:%S %z")


def _annotate_local_times(value: Any) -> Any:
    if isinstance(value, dict):
        annotated: dict[str, Any] = {}
        for key, item in value.items():
            annotated[key] = _annotate_local_times(item)
            if isinstance(item, str) and _looks_like_time_key(str(key)):
                local_text = _format_local_time(item)
                if local_text:
                    annotated[f"{key}_local"] = local_text
        return annotated
    if isinstance(value, list):
        return [_annotate_local_times(item) for item in value]
    return value


@dataclass(slots=True)
class AlertContract:
    item_id: str
    title: str
    content: str
    severity: str
    suggested_tone: str
    metrics: dict[str, Any] | None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_prompt_line(self, index: int) -> str:
        severity_part = f"  severity={self.severity}" if self.severity else ""
        line = f"  [{index}] id={self.item_id}{severity_part}\n       title={self.title}"
        if self.content:
            line += f"\n       内容：{self.content}"
        if self.metrics:
            line += f"\n       metrics：{json.dumps(self.metrics, ensure_ascii=False)}"
        if self.suggested_tone:
            line += f"\n       建议语气：{self.suggested_tone}"
        return line


def normalize_alert(event: dict[str, Any]) -> AlertContract:
    ack_server = str(event.get("ack_server") or "?").strip() or "?"
    event_id = str(event.get("event_id") or event.get("id") or "?").strip() or "?"
    title = str(event.get("title") or "").strip()
    content = str(event.get("content") or event.get("body") or "").strip()
    severity = str(event.get("severity") or "").strip()
    tone = str(event.get("suggested_tone") or "").strip()
    return AlertContract(
        item_id=f"{ack_server}:{event_id}",
        title=title,
        content=content,
        severity=severity,
        suggested_tone=tone,
        metrics=_normalize_metrics(event.get("metrics")),
        raw=event,
    )


@dataclass(slots=True)
class ContentContract:
    item_id: str
    title: str
    source: str
    url: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_valid_url(self) -> bool:
        return bool(self.url)

    def to_prompt_line(self, index: int, has_content: bool) -> str:
        status = "✓" if has_content else "✗(预取失败)"
        url_part = f"\n       url={self.url}" if self.has_valid_url else ""
        return (
            f"  [{index}] id={self.item_id}\n"
            f"       title={self.title}\n"
            f"       source={self.source}  正文:{status}"
            f"{url_part}"
        )


def normalize_content(item: dict[str, Any]) -> ContentContract:
    return ContentContract(
        item_id=str(item.get("id") or "").strip(),
        title=str(item.get("title") or "").strip(),
        source=str(item.get("source") or item.get("source_name") or "").strip(),
        url=str(item.get("url") or "").strip(),
        raw=item,
    )


@dataclass(slots=True)
class ContextContract:
    available: bool | None
    source: str
    raw: dict[str, Any] = field(default_factory=dict)

    def to_prompt_item(self) -> dict[str, Any]:
        payload = _annotate_local_times(dict(self.raw))
        if self.available is not None:
            payload["available"] = self.available
        if self.source:
            payload["_source"] = self.source

        if "sleep_prob" in payload and payload["sleep_prob"] is not None:
            payload["awake_prob"] = round(1.0 - float(payload["sleep_prob"]), 3)
        return payload


def normalize_context(item: dict[str, Any]) -> ContextContract:
    source = str(item.get("_source") or "").strip()
    available_raw = item.get("available")
    available = None if available_raw is None else bool(available_raw)
    return ContextContract(
        available=available,
        source=source,
        raw=item,
    )
