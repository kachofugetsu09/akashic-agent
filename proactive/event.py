"""
proactive/event.py — Proactive 信息源统一事件类型。

设计：
- ProactiveEvent  — 抽象基类，定义引擎可调用的统一接口
- HealthEvent     — 来自 Fitbit 的健康事件
- FeedEvent       — 来自订阅信息流的内容条目

扩展原则：
- 新信息源只需继承 ProactiveEvent，覆盖 kind / is_urgent / ack_id / _extra_signal_fields
- 引擎层无需添加任何 kind 判断，按接口调用即可
- to_signal_dict() 是唯一允许进入 decision_signals / prompt 的序列化口
- ack_id 只返回上游提供的真实 ID；fallback hash 不可 ack
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from feeds.base import FeedItem


@dataclass
class ProactiveEvent(ABC):
    """所有 proactive 信息源事件的抽象基类。

    子类必须实现 kind 属性；可选覆盖 is_urgent / ack_id / _extra_signal_fields。
    直接实例化 ProactiveEvent 会抛出 TypeError，静态分析器和运行时均有保护。
    """

    event_id: str       # 去重 & ack 用
    source_type: str    # "rss" / "web" / "health_event" 等
    source_name: str    # 人类可读来源名
    content: str        # 正文摘要 / 健康消息
    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None

    # ------------------------------------------------------------------
    # 子类通过覆盖以下成员定义 kind-specific 行为
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def kind(self) -> str:
        """事件类型标识，子类必须返回一个稳定的字符串（"health" / "feed" / ...）。"""

    @property
    def ack_id(self) -> str | None:
        """上游真实 ID，用于 ack；None 表示不可 ack（如 fallback hash）。"""
        return None

    def is_urgent(self) -> bool:
        """是否为高优先级事件，影响 pre-score fast-path 和 force_reflect。默认 False。"""
        return False

    def _extra_signal_fields(self) -> dict[str, Any]:
        """子类覆盖此方法以注入 kind-specific 字段到 to_signal_dict()。"""
        return {}

    # ------------------------------------------------------------------
    # 公共序列化口（不应被子类整体覆盖，只覆盖 _extra_signal_fields）
    # ------------------------------------------------------------------

    def to_signal_dict(self) -> dict[str, Any]:
        """序列化为可安全注入 decision_signals / prompt 的纯 dict。

        子类通过 _extra_signal_fields() 注入 kind-specific 字段，
        不直接覆盖此方法，保证公共字段始终存在。
        """
        published = (
            self.published_at.isoformat() if self.published_at is not None else None
        )
        d: dict[str, Any] = {
            "kind": self.kind,
            "event_id": self.event_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": published,
        }
        d.update(self._extra_signal_fields())
        return d


@dataclass
class HealthEvent(ProactiveEvent):
    """来自 Fitbit 的健康事件。

    event_id 策略：
    - 优先用上游 event["id"]（字符串非空）
    - 无 id 时用 severity+message 的 stable hash 作 fallback（ack_id 返回 None）
    """

    severity: str | None = None                                        # "high" / "normal" / "low"
    _raw_health: dict = field(default_factory=dict, repr=False)        # 原始上游 dict，引擎内部用
    _upstream_id: str | None = field(default=None, repr=False)        # 上游真实 id

    @property
    def kind(self) -> str:
        return "health"

    @property
    def ack_id(self) -> str | None:
        return self._upstream_id

    def is_urgent(self) -> bool:
        return self.severity == "high"

    def _extra_signal_fields(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            # "message" 保持向后兼容：components.py 和 prompts/proactive.py 读的是这个键
            "message": self.content,
        }

    @classmethod
    def from_dict(cls, event: dict) -> "HealthEvent":
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
            event_id=event_id,
            source_type="health_event",
            source_name="fitbit",
            content=str(event.get("message", "")).strip(),
            severity=str(event.get("severity", "")).strip().lower() or None,
            _raw_health=event,
            _upstream_id=upstream_id,
        )


@dataclass
class FeedEvent(ProactiveEvent):
    """来自订阅信息流的内容条目（RSS、网页等）。

    event_id 由调用方传入（复用 compute_item_id 的现有逻辑）。
    """

    _raw_feed: "FeedItem | None" = field(default=None, repr=False)    # 原始 FeedItem，引擎内部用

    @property
    def kind(self) -> str:
        return "feed"

    @property
    def ack_id(self) -> str | None:
        # feed item_id 由 compute_item_id 确定性生成，始终可信
        return self.event_id

    @classmethod
    def from_item(cls, item: "FeedItem", item_id: str) -> "FeedEvent":
        return cls(
            event_id=item_id,
            source_type=item.source_type or "",
            source_name=item.source_name or "",
            content=item.content or "",
            title=item.title,
            url=item.url,
            published_at=item.published_at,
            _raw_feed=item,
        )
