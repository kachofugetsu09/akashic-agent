from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from agent.plugin_composition import ServiceKey
from session.log import MessageCatalog, OwnerStore
from session.message import Message
from session.message_codec import json_value

from .api import Text


class Confirmation(BaseModel):
    """Delivery 首次确认送达的索引；正文仍按原 Message 读取。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    message_id: Text
    session_id: Text
    confirmed_at: AwareDatetime


def time_key(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("送达查询时间必须包含时区")
    return "confirmed-time:" + value.astimezone(timezone.utc).isoformat(timespec="microseconds") + ":"


@dataclass(frozen=True, slots=True)
class DeliveredMessage:
    message: Message
    confirmed_at: datetime


class DeliveryHistory:
    """跨来源只读真实送达历史；查询不会发送、查询 provider 或改写回执。"""

    def __init__(self, state: Callable[[], OwnerStore], catalog: MessageCatalog):
        self._state = state
        self._catalog = catalog

    def recent(self, *, since: datetime, until: datetime, limit: int,
               excluded_sources: frozenset[str] = frozenset(),
               visibility: Literal["listed", "internal"] | None = None) -> tuple[DeliveredMessage, ...]:
        """按首次本地确认时间倒序返回 [since, until) 内的不同消息。"""
        start, stop = time_key(since), time_key(until)
        if start >= stop or type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("送达查询需要递增时间范围和 1 到 1000 的 limit")
        if visibility not in {None, "listed", "internal"}:
            raise ValueError("送达查询的会话可见性无效")
        state = self._state()
        return state.snapshot(lambda: self._recent(state, start, stop, limit, excluded_sources, visibility))

    def _recent(self, state: OwnerStore, start: str, stop: str, limit: int,
                excluded_sources: frozenset[str],
                visibility: Literal["listed", "internal"] | None) -> tuple[DeliveredMessage, ...]:
        """索引分页与正文读取共享同一个数据库快照。"""
        result: list[DeliveredMessage] = []
        # 1. 先按时间索引分页，过滤来源后再计数；被动消息不会挤掉更早的通知。
        while rows := state.scan(start=start, stop=stop, limit=128):
            for key, row in rows:
                fact = Confirmation.model_validate_json(json.dumps(json_value(row.value)))
                if row.version != 0 or key != time_key(fact.confirmed_at) + fact.message_id:
                    raise ValueError("送达历史索引身份损坏")
                message = self._catalog.reader(fact.session_id).get(fact.message_id)
                if message is None:
                    raise ValueError("送达历史引用的原消息缺失")
                if message.source in excluded_sources:
                    continue
                if visibility is not None and self._catalog.attributes(message.session_id).visibility != visibility:
                    continue
                result.append(DeliveredMessage(message, fact.confirmed_at))
                if len(result) == limit:
                    return tuple(result)
            stop = rows[-1][0]
        return tuple(result)


DELIVERY_READ = ServiceKey[DeliveryHistory]("delivery.read.v1")
