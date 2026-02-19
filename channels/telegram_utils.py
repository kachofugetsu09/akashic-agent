"""
Telegram Markdown 发送工具

将 Markdown 文本通过 telegramify-markdown 转换后发送：
- 自动分段（超出 4096 字符时）
- 长代码块以文件形式发送
- 转换失败时降级为纯文本
"""
import logging

from telegram import Bot
from telegramify_markdown import telegramify
from telegramify_markdown.content import ContentType

logger = logging.getLogger(__name__)


async def send_markdown(bot: Bot, chat_id: int | str, text: str) -> None:
    cid = int(chat_id)
    try:
        items = await telegramify(text, max_message_length=4090)
        for item in items:
            if item.content_type == ContentType.TEXT:
                await bot.send_message(
                    chat_id=cid,
                    text=item.text,
                    entities=[e.to_dict() for e in item.entities],
                )
            elif item.content_type == ContentType.FILE:
                await bot.send_document(
                    chat_id=cid,
                    document=(item.file_name, item.file_data),
                )
            elif item.content_type == ContentType.PHOTO:
                await bot.send_photo(
                    chat_id=cid,
                    photo=(item.file_name, item.file_data),
                )
    except Exception as e:
        logger.warning(f"[telegram] Markdown 转换失败，降级纯文本: {e}")
        for chunk in _split_text(text, 4090):
            await bot.send_message(chat_id=cid, text=chunk)


def _split_text(text: str, limit: int) -> list[str]:
    """按行切分文本，每段不超过 limit 字符。"""
    chunks, current = [], []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        # 单行本身超限时强制切断
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks
