from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from agent.plugin_composition.channels import ChannelDeliveryReceipt, DeliveryStatus
from agent.plugin_composition.model import ServiceKey
if TYPE_CHECKING:
    from bus.events import ChannelMessage

DeliverySender = Callable[["ChannelMessage"], Awaitable[ChannelDeliveryReceipt]]


class PluginDeliveries:
    """Commit one complete logical message through the active Channel boundary."""

    def __init__(self, sender: DeliverySender | None) -> None:
        self._sender = sender

    @classmethod
    def candidate_validation(cls) -> PluginDeliveries:
        return cls(None)

    @property
    def formal(self) -> bool:
        """Return whether this service can commit a formal delivery."""

        return self._sender is not None

    async def send(
        self, *, channel: str, chat_id: str, content: str
    ) -> ChannelDeliveryReceipt:
        sender = self._sender
        if sender is None:
            raise RuntimeError("candidate 验证期禁止提交 delivery")
        if not channel.strip() or not chat_id.strip():
            raise ValueError("delivery route 不能为空")
        if not content:
            raise ValueError("delivery content 不能为空")
        from bus.events import ChannelMessage

        receipt = await sender(
            ChannelMessage(channel=channel, chat_id=chat_id, content=content)
        )
        if receipt.status is not DeliveryStatus.DELIVERED:
            raise RuntimeError(
                f"delivery 未成功提交: status={receipt.status.value} error={receipt.error!r}"
            )
        return receipt


DELIVERIES = ServiceKey[PluginDeliveries]("core.deliveries")
