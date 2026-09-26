"""客户端的消息展示与 Mobile UI 合同，不包含宿主实现。"""

from __future__ import annotations

from typing import Protocol

from agent.plugin_composition.model import ServiceKey
from session.log import MessagePage


class MessageDisplayReader(Protocol):
    """在自己的资源作用域内投影一页，不让客户端持有插件回调。"""

    async def __call__(
        self, page: MessagePage, *, display_only: bool
    ) -> list[dict[str, object]]: ...


class MobileUiProvider(Protocol):
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
MOBILE_UI = ServiceKey[MobileUiProvider]("core.mobile_ui.v1")
