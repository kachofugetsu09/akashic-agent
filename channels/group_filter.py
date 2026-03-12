"""
群聊消息过滤层

定义 GroupMessageFilter 协议，作为未来扩展的核心钩子点。
当前默认实现：检查发送者白名单 + require_at。

未来扩展示例：
    class LLMGroupFilter:
        async def should_process(self, event, group_cfg) -> bool:
            # 先用 LLM 或规则判断消息是否值得传给 agent
            ...

只需将自定义 filter 传入 QQChannel 即可替换默认行为，
无需修改 QQChannel 内部逻辑。
"""

from __future__ import annotations

import re
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agent.config_models import QQGroupConfig

logger = logging.getLogger(__name__)

# CQ 码 AT 格式：[CQ:at,qq=<qq_id>]
_CQ_AT_RE = re.compile(r"\[CQ:at,qq=(\d+)[^\]]*\]")


@runtime_checkable
class GroupMessageFilter(Protocol):
    """群消息过滤协议，返回 True 表示将消息传给 agent。"""

    async def should_process(self, event, group_cfg: QQGroupConfig) -> bool: ...


class DefaultGroupFilter:
    """
    默认过滤器：
      1. 发送者是否在 allow_from 白名单（空 = 允许所有人）
      2. require_at=True 时，消息中必须包含 @Bot
    """

    def __init__(self, bot_uin: str) -> None:
        self._bot_uin = bot_uin

    async def should_process(self, event, group_cfg: QQGroupConfig) -> bool:
        user_id = str(event.user_id)

        if group_cfg.allow_from and user_id not in group_cfg.allow_from:
            logger.debug(
                f"[group_filter] 拒绝非白名单用户  user_id={user_id}  group={group_cfg.group_id}"
            )
            return False

        if group_cfg.require_at and not _is_at_bot(event.raw_message, self._bot_uin):
            logger.debug(
                f"[group_filter] 未被 @ 忽略消息  user_id={user_id}  group={group_cfg.group_id}"
            )
            return False

        return True


def _is_at_bot(raw_message: str, bot_uin: str) -> bool:
    """检查消息中是否包含 @Bot 的 CQ 码。"""
    return any(qq == bot_uin for qq in _CQ_AT_RE.findall(raw_message))


def strip_at_segments(raw_message: str) -> str:
    """去掉消息中所有 AT CQ 码，返回干净的文本内容。"""
    return _CQ_AT_RE.sub("", raw_message).strip()
