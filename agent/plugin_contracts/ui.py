"""客户端的消息展示与 Plugin UI 合同，不包含宿主实现。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from agent.plugin_composition.model import ServiceKey
from session.log import MessagePage
from agent.plugin_contracts.message import ContentPart, Message

# 工具内容 owner 提供派生展示；读取回调只允许当前消息之前的同 Session 内容。
ToolResultDisplayProvider = Callable[
    [ContentPart, Callable[[str], Awaitable[Message | None]]], Awaitable[object]
]


class MessageDisplayReader(Protocol):
    """在自己的资源作用域内投影一页，不让客户端持有插件回调。"""

    async def __call__(
        self, page: MessagePage, *, display_only: bool
    ) -> list[dict[str, object]]: ...


class PluginUiProvider(Protocol):
    async def catalog(self) -> dict[str, object]: ...

    async def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]: ...

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]: ...


MESSAGE_DISPLAY = ServiceKey[MessageDisplayReader]("core.message_display.v1")
PLUGIN_UI = ServiceKey[PluginUiProvider]("ui.plugin.v1")
