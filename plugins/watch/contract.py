"""值班的事件入口：任何插件都可以把一件事交进来，之后怎么处理由值班与主 Agent 决定。"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


class WatchEvents(Protocol):
    # source 是来源的名字（如 "owntracks"）；title、text、url 都可以为空，但 title 与 text 至少要有一个。
    # key 是来源自己的稳定标识（条目 ID、邮件 ID）：同一来源同一个 key 只记一次，重复交来时返回原来那条的编号，
    # 所以来源在重启、重试时可以放心重交，不需要自己记"交过没有"。
    # state=True 表示这条是该来源的当前状态（在哪、睡没睡）：它照常是一条事件，同时顶替这个来源的上一条状态。
    async def emit(
        self, source: str, *, title: str = "", text: str = "", url: str = "", at: datetime | None = None,
        state: bool = False, key: str = "",
    ) -> str: ...


WATCH_EVENTS = ServiceKey[WatchEvents]("watch.events.v1")
