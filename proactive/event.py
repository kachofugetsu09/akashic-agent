"""
proactive/event.py — Proactive 信息源统一事件类型。

把 health_event（来自 Fitbit）和 FeedItem（来自订阅信息流）
包装成统一的 ProactiveEvent，通过 event.kind 显式区分，
便于引擎内部按类型分发，也为未来新信息源（日历、提醒等）预留扩展点。

设计原则：
- _raw_feed / _raw_health 只在引擎内部使用，不向 decision_signals / prompt 层暴露
- to_signal_dict() 是唯一允许进入 prompt 的序列化口
- ack_id 只返回上游提供的真实 ID；fallback hash 不可 ack
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from feeds.base import FeedItem

EventKind = Literal["feed", "health"]


@dataclass
class ProactiveEvent:
    kind: EventKind
    event_id: str          # 去重 & ack 用；health: 上游 id 或 stable hash；feed: compute_item_id
    source_type: str       # "rss" / "web" / "health_event" 等
    source_name: str       # 人类可读来源名
    title: str | None
    content: str           # health 对应 message 字段；feed 对应摘要正文
    url: str | None = None
    published_at: datetime | None = None
    severity: str | None = None        # health 专有："high" / "normal" / "low"
    # 原始对象引用，仅引擎内部使用
    _raw_feed: "FeedItem | None" = field(default=None, repr=False)
    _raw_health: dict | None = field(default=None, repr=False)
    # 标记 event_id 是否来自上游真实 id（False 表示 fallback hash，不可 ack）
    _upstream_id: str | None = field(default=None, repr=False)

    @property
    def ack_id(self) -> str | None:
        """上游提供的原始 ID；fallback hash 时返回 None（不可 ack）。"""
        return self._upstream_id

    def to_signal_dict(self) -> dict[str, object]:
        """序列化为可安全注入 decision_signals / prompt 的纯 dict。

        health 事件额外保留 "message" 键，与 prompts/proactive.py 中
        health_events[*].message 引用及 components.py 的现有逻辑保持兼容。
        """
        published = (
            self.published_at.isoformat() if self.published_at is not None else None
        )
        d: dict[str, object] = {
            "kind": self.kind,
            "event_id": self.event_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": published,
            "severity": self.severity,
        }
        if self.kind == "health":
            # 向后兼容：components.py 和 prompts 层读的是 "message"
            d["message"] = self.content
        return d

    @classmethod
    def from_health_dict(cls, event: dict) -> "ProactiveEvent":
        """从 fitbit-monitor 返回的健康事件 dict 构建 ProactiveEvent。

        event_id 策略：
        - 优先用上游 event["id"]（字符串，非空）
        - 无 id 时用 severity+message 的 stable hash 作 fallback（不可 ack）
        """
        upstream_id = str(event.get("id", "")).strip() or None
        if upstream_id:
            event_id = upstream_id
        else:
            raw = "|".join([
                str(event.get("severity", "")).strip().lower(),
                str(event.get("message", "")).strip().lower()[:200],
            ])
            event_id = "hev_" + hashlib.sha1(raw.encode()).hexdigest()[:12]
        return cls(
            kind="health",
            event_id=event_id,
            source_type="health_event",
            source_name="fitbit",
            title=None,
            content=str(event.get("message", "")).strip(),
            severity=str(event.get("severity", "")).strip().lower() or None,
            _raw_health=event,
            _upstream_id=upstream_id,
        )

    @classmethod
    def from_feed_item(cls, item: "FeedItem", item_id: str) -> "ProactiveEvent":
        """从 FeedItem 构建 ProactiveEvent。

        item_id 由调用方传入（复用 compute_item_id 的现有逻辑）。
        """
        return cls(
            kind="feed",
            event_id=item_id,
            source_type=item.source_type or "",
            source_name=item.source_name or "",
            title=item.title,
            content=item.content or "",
            url=item.url,
            published_at=item.published_at,
            severity=None,
            _raw_feed=item,
            _upstream_id=item_id,  # feed item_id 始终可信，视作"upstream"
        )
