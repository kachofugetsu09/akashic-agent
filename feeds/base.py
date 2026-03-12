"""
信息流基础类型。

FeedItem         — 单条信息流内容
FeedSubscription — 持久化的订阅记录
FeedSource       — 信息源抽象基类
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class FeedItem:
    source_name: str  # "Paul Graham's Blog"
    source_type: str  # "rss"
    title: str | None
    content: str  # 正文摘要（截断后）
    url: str | None
    author: str | None
    published_at: datetime | None


@dataclass
class FeedSubscription:
    id: str
    type: str  # "rss"
    name: str  # 人类可读名称，如 "Paul Graham"
    url: str | None = None  # RSS 地址
    note: str | None = None  # 用户备注
    enabled: bool = True
    added_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(cls, **kwargs) -> FeedSubscription:
        return cls(id=str(uuid.uuid4()), **kwargs)


class FeedSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def source_type(self) -> str: ...

    @abstractmethod
    async def fetch(self, limit: int = 5) -> list[FeedItem]: ...
