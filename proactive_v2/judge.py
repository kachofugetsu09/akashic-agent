from __future__ import annotations

import logging

from agent.provider import LLMProvider
from proactive_v2.json_utils import extract_json_object

logger = logging.getLogger(__name__)


def _format_recent_proactive_entries(recent_proactive: list[object]) -> str:
    lines: list[str] = []
    for index, message in enumerate(recent_proactive, 1):
        content = _field(message, "content")
        if not content:
            continue
        meta = _recent_meta(message)
        suffix = f" ({'; '.join(meta)})" if meta else ""
        lines.append(f"[{index}]{suffix} {content}")
    return "\n---\n".join(lines)


def _field(raw: object, name: str, default: str = "") -> str:
    if isinstance(raw, dict):
        return str(raw.get(name, default) or default).strip()
    return str(getattr(raw, name, default) or default).strip()


def _recent_meta(message: object) -> list[str]:
    meta: list[str] = []
    timestamp = getattr(message, "timestamp", None)
    if timestamp is not None:
        try:
            meta.append(f"time={timestamp.isoformat()}")
        except Exception:
            meta.append(f"time={timestamp}")
    tag = _field(message, "state_summary_tag", "none")
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
        recent_proactive: list[object],
        new_state_summary_tag: str = "none",
    ) -> tuple[bool, str]:
        if not recent_proactive:
            return False, "无近期主动消息，放行"
        try:
            response = await self._provider.chat(
                messages=self._build_messages(
                    new_message,
                    recent_proactive,
                    new_state_summary_tag,
                ),
                tools=[],
                model=self._model,
                max_tokens=min(128, self._max_tokens),
            )
            payload = extract_json_object((response.content or "").strip())
        except Exception as exc:
            logger.warning("[proactive.deduper] 检测失败，放行: %s", exc)
            return False, str(exc)
        is_duplicate = bool(payload.get("is_duplicate", False))
        reason = str(payload.get("reason", ""))
        logger.info(
            "[proactive.deduper] is_duplicate=%s reason=%r",
            is_duplicate,
            reason[:80],
        )
        return is_duplicate, reason

    def _build_messages(
        self,
        new_message: str,
        recent_proactive: list[object],
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
