"""值班的事件入口：任何插件都可以把一件事交进来，之后怎么处理由值班与主 Agent 决定。"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


class WatchEvents(Protocol):
    # source 是来源的名字（如 "owntracks"）；title、text、url 都可以为空，但 title 与 text 至少要有一个。
    # state=True 表示这条是该来源的当前状态（在哪、睡没睡）：它照常是一条事件，同时顶替这个来源的上一条状态。
    async def emit(
        self, source: str, *, title: str = "", text: str = "", url: str = "", at: datetime | None = None,
        state: bool = False,
    ) -> str: ...


WATCH_EVENTS = ServiceKey[WatchEvents]("watch.events.v1")
