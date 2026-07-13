from __future__ import annotations

import logging

from agent.provider import LLMProvider
from proactive_v2.json_utils import extract_json_object
from proactive_v2.sensor import RecentProactiveMessage

logger = logging.getLogger(__name__)


def _format_recent_proactive_entries(
    recent_proactive: list[RecentProactiveMessage],
) -> str:
    lines: list[str] = []
    for index, message in enumerate(recent_proactive, 1):
        if not message.content:
            continue
        meta = _recent_meta(message)
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"[{index}]{suffix} {message.content}")
    return "\n---\n".join(lines)


def _recent_meta(message: RecentProactiveMessage) -> list[str]:
    meta: list[str] = []
    if message.timestamp is not None:
        meta.append(f"time={message.timestamp.isoformat()}")
    tag = message.state_summary_tag.strip()
    if tag and tag != "none":
        meta.append(f"state_tag={tag}")
    return meta


class MessageDeduper:
    def __init__(
        self,
        *,
        provider: LLMProvider,
        model: str,
        max_tokens: int,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens

    async def is_duplicate(
        self,
        new_message: str,
        recent_proactive: list[RecentProactiveMessage],
        new_state_summary_tag: str = "none",
    ) -> tuple[bool, str]:
        if not recent_proactive:
            return False, "无近期主动消息，放行"
        messages = self._build_messages(
            new_message,
            recent_proactive,
            new_state_summary_tag,
        )
        response = await self._provider.chat(
            messages=messages,
            tools=[],
            model=self._model,
            max_tokens=min(128, self._max_tokens),
        )
        payload = extract_json_object((response.content or "").strip())
        if "is_duplicate" not in payload or not isinstance(
            payload["is_duplicate"], bool
        ):
            raise ValueError("is_duplicate 必须是 boolean")
        if "reason" not in payload or not isinstance(payload["reason"], str):
            raise ValueError("reason 必须是 string")
        is_duplicate = payload["is_duplicate"]
        reason = payload["reason"]
        logger.info(
            "[proactive.deduper] is_duplicate=%s reason=%r",
            is_duplicate,
            reason[:80],
        )
        return is_duplicate, reason

    def _build_messages(
        self,
        new_message: str,
        recent_proactive: list[RecentProactiveMessage],
        new_state_summary_tag: str,
    ) -> list[dict[str, str]]:
        system_msg = (
            "你是消息重复检测器。判断【新消息】是否与【近期已发消息】在实质信息上重复。\n"
            "重复包括：同一事件重复，或同一用户状态总结/安慰框架重复。\n"
            "不重复包括：同话题但有真正新进展或明显不同角度。\n"
            "只输出 JSON。"
        )
        user_msg = (
            f"近期已发消息：\n{_format_recent_proactive_entries(recent_proactive)}\n\n"
            f"---\n新消息：{new_message}\n"
            f"新消息 state_summary_tag：{new_state_summary_tag}\n\n"
            "---\n只输出 JSON：\n"
            '{"is_duplicate": false, "reason": "简短说明"}'
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
