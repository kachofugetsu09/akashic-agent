from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.tools.message_push import MessagePushTool
from proactive.presence import PresenceStore
from session.manager import SessionManager

logger = logging.getLogger(__name__)


@dataclass
class ProactiveSourceRef:
    item_id: str
    source_type: str
    source_name: str
    title: str
    url: str | None = None
    published_at: str | None = None


@dataclass
class ProactiveSendMeta:
    evidence_item_ids: list[str] = field(default_factory=list)
    source_refs: list[ProactiveSourceRef] = field(default_factory=list)
    state_summary_tag: str = "none"


class Sender:
    def __init__(
        self,
        *,
        cfg: Any,
        push_tool: MessagePushTool,
        sessions: SessionManager,
        presence: PresenceStore | None,
    ) -> None:
        self._cfg = cfg
        self._push = push_tool
        self._sessions = sessions
        self._presence = presence

    async def send(
        self,
        message: str,
        meta: ProactiveSendMeta | None = None,
    ) -> bool:
        # 1. 先校验发送目标，缺配置时直接短路。
        message = (message or "").strip()
        channel = (self._cfg.default_channel or "").strip()
        chat_id = self._cfg.default_chat_id.strip()
        if not channel or not chat_id:
            logger.warning(
                "[proactive] default_channel/default_chat_id 未配置，跳过发送"
            )
            return False
        # 2. 再调用 message_push，并仅在明确成功时继续落会话。
        logger.info(
            "[proactive] 准备发送主动消息 channel=%s chat_id=%s message_len=%d",
            channel,
            chat_id,
            len(message),
        )
        try:
            result = await self._push.execute(
                channel=channel,
                chat_id=chat_id,
                message=message,
            )
        except Exception as exc:
            logger.error("[proactive] 发送失败: %s", exc)
            return False
        if "已发送" not in result:
            logger.warning("[proactive] 发送未成功: %s", result)
            return False
        # 3. 最后把发送记录写回会话和 presence。
        await self._save_session_message(channel, chat_id, message, meta)
        logger.info("[proactive] 已发送主动消息并写入会话 → %s:%s", channel, chat_id)
        return True

    async def _save_session_message(
        self,
        channel: str,
        chat_id: str,
        message: str,
        meta: ProactiveSendMeta | None,
    ) -> None:
        key = f"{channel}:{chat_id}"
        session = self._sessions.get_or_create(key)
        session.add_message(
            "assistant",
            message,
            proactive=True,
            tools_used=["message_push"],
            evidence_item_ids=self._evidence_ids(meta),
            source_refs=self._source_refs_payload(meta),
            state_summary_tag=self._state_summary_tag(meta),
        )
        await self._sessions.save_async(session)
        if self._presence:
            self._presence.record_proactive_sent(key)

    @staticmethod
    def _evidence_ids(meta: ProactiveSendMeta | None) -> list[str]:
        if meta is None:
            return []
        return [str(item_id) for item_id in meta.evidence_item_ids if str(item_id).strip()]

    @staticmethod
    def _state_summary_tag(meta: ProactiveSendMeta | None) -> str:
        if meta is None:
            return "none"
        return str(meta.state_summary_tag or "none")

    @staticmethod
    def _source_refs_payload(
        meta: ProactiveSendMeta | None,
    ) -> list[dict[str, Any]]:
        if meta is None:
            return []
        payload: list[dict[str, Any]] = []
        for ref in meta.source_refs:
            payload.append(
                {
                    "item_id": str(ref.item_id or ""),
                    "source_type": str(ref.source_type or ""),
                    "source_name": str(ref.source_name or ""),
                    "title": str(ref.title or ""),
                    "url": ref.url,
                    "published_at": ref.published_at,
                }
            )
        return payload
