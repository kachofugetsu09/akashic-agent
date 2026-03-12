"""
proactive/event.py — Proactive 信息源统一事件类型。

设计层次：
- ProactiveEvent  — 抽象基类，定义引擎可调用的统一接口
- AlertEvent      — 告警通道：紧急事件，bypass 内容评分，需要 ack（如健康告警、传感器报警）
- ContentEvent    — 内容流通道：参与评分、去重、pending queue（如 RSS、网页内容）
- HealthEvent     — 来自 Fitbit 的健康事件（AlertEvent）
- FeedEvent       — 来自订阅信息流的内容条目（ContentEvent）

扩展原则：
- 新告警类（湿度计、日历提醒等）继承 AlertEvent，实现 kind / ack_id / from_xxx()
- 新内容类（网页搜索、novel 等）继承 ContentEvent，实现 kind / from_xxx()
- 两条引擎通道（Stage 2 / Stage 4）与两个中间类一一对应，新类型归属明确
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

    定义引擎调用的统一接口：kind / ack_id / is_urgent / to_signal_dict。
    直接实例化会抛出 TypeError；请使用 AlertEvent 或 ContentEvent 的具体子类。
    """

    event_id: str       # 去重 & ack 用
    source_type: str    # "rss" / "web" / "health_event" 等
    source_name: str    # 人类可读来源名
    content: str        # 正文摘要 / 告警消息
    title: str | None = None
    url: str | None = None
    published_at: datetime | None = None

    @property
    @abstractmethod
    def kind(self) -> str:
        """事件类型标识，子类必须返回稳定字符串（"health" / "feed" / ...）。"""

    @property
    def ack_id(self) -> str | None:
        """上游真实 ID，用于 ack；None 表示不可 ack（fallback hash）。"""
        return None

    def is_urgent(self) -> bool:
        """是否触发 pre-score fast-path 和 force_reflect。默认 False。"""
        return False

    def _extra_signal_fields(self) -> dict[str, Any]:
        """子类覆盖此方法以向 to_signal_dict() 注入 kind-specific 字段。"""
        return {}

    def to_signal_dict(self) -> dict[str, Any]:
        """序列化为可安全注入 decision_signals / prompt 的纯 dict。

        子类通过 _extra_signal_fields() 注入额外字段，不直接覆盖此方法。
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


# ---------------------------------------------------------------------------
# 告警通道
# ---------------------------------------------------------------------------

@dataclass
class AlertEvent(ProactiveEvent, ABC):
    """告警类事件的抽象基类。

    共同特征：有 severity 等级，高优先级时 bypass 内容评分，post-send 需要 ack。
    新告警类型（湿度计、CO₂ 报警、日历提醒等）继承此类，只需实现 kind / ack_id / from_xxx()。
    """

    severity: str | None = None    # "high" / "normal" / "low"

    def is_urgent(self) -> bool:
        return self.severity == "high"

    def _extra_signal_fields(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            # "message" 保持向后兼容：components.py 和 prompts/proactive.py 读的是这个键
            "message": self.content,
        }


@dataclass
class HealthEvent(AlertEvent):
    """来自 Fitbit 的健康事件。

    event_id 策略：
    - 优先用上游 event["id"]（字符串非空）
    - 无 id 时用 severity+message 的 stable hash 作 fallback（ack_id 返回 None）
    """

    _raw_health: dict = field(default_factory=dict, repr=False)
    _upstream_id: str | None = field(default=None, repr=False)

    @property
    def kind(self) -> str:
        return "health"

    @property
    def ack_id(self) -> str | None:
        return self._upstream_id

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


# ---------------------------------------------------------------------------
# 内容流通道
# ---------------------------------------------------------------------------

@dataclass
class ContentEvent(ProactiveEvent, ABC):
    """内容流类事件的抽象基类。

    共同特征：参与 d_content 评分、进 pending queue、进 compose 候选池。
    event_id 由 compute_item_id 确定性生成，始终可作为 ack_id。
    新内容类型（网页搜索、novel-kb 等）继承此类，只需实现 kind / from_xxx()。
    """

    @property
    def ack_id(self) -> str | None:
        return self.event_id

    def is_urgent(self) -> bool:
        return False

    def to_feed_item(self) -> "FeedItem":
        """返回供下游 port 接口使用的 FeedItem 视图。

        默认从自身字段构建，确保任何 ContentEvent 子类都不会被静默丢弃。
        FeedEvent 覆盖此方法返回原始对象，以保留 dedup / pending / seen 所需的 identity。
        """
        from feeds.base import FeedItem  # 延迟 import，避免循环依赖
        return FeedItem(
            source_name=self.source_name,
            source_type=self.source_type,
            title=self.title,
            content=self.content,
            url=self.url,
            author=None,
            published_at=self.published_at,
        )


@dataclass
class FeedEvent(ContentEvent):
    """来自订阅信息流的内容条目（RSS、网页等）。

    event_id 由调用方传入（复用 compute_item_id 的现有逻辑）。
    """

    _raw_feed: "FeedItem | None" = field(default=None, repr=False)

    @property
    def kind(self) -> str:
        return "feed"

    def to_feed_item(self) -> "FeedItem":
        # 返回原始对象，保留 identity，dedup / pending / seen 逻辑依赖此一致性
        return self._raw_feed if self._raw_feed is not None else super().to_feed_item()

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
