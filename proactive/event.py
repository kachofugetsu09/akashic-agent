"""
proactive/event.py — Proactive 信息源统一事件类型。

设计层次：
- ProactiveEvent     — 抽象基类，定义引擎可调用的统一接口
- AlertEvent         — 告警通道：紧急事件，bypass 内容评分，需要 ack（如健康告警、传感器报警）
- ContentEvent       — 内容流通道：参与评分、去重、pending queue（如 RSS、网页内容）
- GenericAlertEvent  — MCP 通道接入的通用告警事件（kind 由 payload 决定）
- FeedEvent          — 来自订阅信息流的内容条目（ContentEvent）

扩展原则：
- 新告警源通过 MCP server 接入，返回标准 schema，engine 统一包装为 GenericAlertEvent
- 新内容类（网页搜索、novel 等）继承 ContentEvent，实现 kind / from_xxx()
- 两条引擎通道（Stage 2 / Stage 4）与两个中间类一一对应，新类型归属明确
- to_signal_dict() 是唯一允许进入 decision_signals / prompt 的序列化口
- ack_id 只返回上游提供的真实 ID；无 event_id 时用 fallback hash（不可 ack）
"""

from __future__ import annotations

import hashlib
import json
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
class GenericAlertEvent(AlertEvent):
    """MCP 通道接入的通用告警事件。

    kind 由 MCP payload 的 kind 字段决定，适用于任何实现了
    get_proactive_events 标准协议的 MCP server。
    """

    _kind: str = field(default="alert", repr=False)
    _upstream_id: str | None = field(default=None, repr=False)
    _ack_server: str | None = field(default=None, repr=False)

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def ack_id(self) -> str | None:
        return self._upstream_id

    @classmethod
    def from_mcp_payload(cls, payload: dict) -> "GenericAlertEvent":
        """从标准 MCP ProactiveEvent schema dict 构建实例。"""
        upstream_id = str(payload.get("event_id", "")).strip() or None
        if upstream_id:
            event_id = upstream_id
        else:
            raw = json.dumps(payload, sort_keys=True)
            event_id = "gev_" + hashlib.sha1(raw.encode()).hexdigest()[:12]

        published_at: datetime | None = None
        if payload.get("published_at"):
            try:
                published_at = datetime.fromisoformat(payload["published_at"])
            except Exception:
                pass

        ack_server = str(payload.get("ack_server", "")).strip() or None

        return cls(
            event_id=event_id,
            source_type=str(payload.get("source_type", "")).strip(),
            source_name=str(payload.get("source_name", "")).strip(),
            content=str(payload.get("content", "")).strip(),
            title=payload.get("title"),
            url=payload.get("url"),
            published_at=published_at,
            severity=str(payload.get("severity", "")).strip() or None,
            _kind=str(payload.get("kind", "alert")).strip(),
            _upstream_id=upstream_id,
            _ack_server=ack_server,
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


@dataclass
class GenericContentEvent(ContentEvent):
    """MCP 通道接入的通用内容事件。"""

    _kind: str = field(default="content", repr=False)
    _ack_server: str | None = field(default=None, repr=False)

    @property
    def kind(self) -> str:
        return self._kind

    @classmethod
    def from_mcp_payload(cls, payload: dict) -> "GenericContentEvent":
        raw = json.dumps(payload, sort_keys=True)
        event_id = str(payload.get("event_id", "")).strip() or (
            "gcv_" + hashlib.sha1(raw.encode()).hexdigest()[:12]
        )
        published_at: datetime | None = None
        if payload.get("published_at"):
            try:
                published_at = datetime.fromisoformat(payload["published_at"])
            except Exception:
                pass
        ack_server = str(payload.get("ack_server", "")).strip() or None
        return cls(
            event_id=event_id,
            source_type=str(payload.get("source_type", "")).strip(),
            source_name=str(payload.get("source_name", "")).strip(),
            content=str(payload.get("content", "")).strip(),
            title=payload.get("title"),
            url=payload.get("url"),
            published_at=published_at,
            _kind=str(payload.get("kind", "content")).strip(),
            _ack_server=ack_server,
        )
