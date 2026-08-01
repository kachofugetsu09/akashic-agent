from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from bus.events import (
    AttachmentKind,
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
)

logger = logging.getLogger(__name__)


async def deliver_message_parts(
    message: ChannelMessage,
    *,
    send_text: Callable[[str, str], Awaitable[None]],
    send_file: Callable[[str, str, str | None], Awaitable[None]],
    send_image: Callable[[str, str], Awaitable[None]],
) -> DeliveryReceipt:
    """按平台原生能力发送完整消息，并准确报告部分提交。"""

    committed = 0
    try:
        # 1. 先提交正文，再按已验证类型提交每个附件
        if message.content:
            await send_text(message.chat_id, message.content)
            committed += 1
        for attachment in message.attachments:
            if attachment.kind is AttachmentKind.FILE:
                await send_file(
                    message.chat_id,
                    attachment.source,
                    attachment.filename,
                )
            elif attachment.kind is AttachmentKind.IMAGE:
                await send_image(message.chat_id, attachment.source)
            else:
                raise ValueError(f"未知渠道附件类型: {attachment.kind}")
            committed += 1
    except Exception as error:
        # 2. 渠道边界把平台异常转换为可判定的提交终态
        status = DeliveryStatus.PARTIAL if committed else DeliveryStatus.FAILED
        logger.warning(
            "渠道消息提交失败: channel=%s chat=%s committed=%s error=%s",
            message.channel,
            message.chat_id,
            committed,
            error,
        )
        return DeliveryReceipt(status, detail=str(error))

    return DeliveryReceipt(
        DeliveryStatus.SUCCESS,
        canonical_media=tuple(item.source for item in message.attachments),
    )
