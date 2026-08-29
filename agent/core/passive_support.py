from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from agent.core.types import HistoryMessage, to_tool_call_groups
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
)

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.runtime_support import SessionLike
    from session.manager import SessionManager

context_logger = logging.getLogger("agent.core.passive_turn.context_store")
_LOG_PREVIEW_LIMIT = 160
_REACT_CONTEXT_FIELDS = (
    "iteration_count",
    "turn_input_sum_tokens",
    "turn_input_peak_tokens",
    "final_call_input_tokens",
    "cache_prompt_tokens",
    "cache_hit_tokens",
)


def collect_skill_mentions(content: str, skill_names: list[str]) -> list[str]:
    raw_names = re.findall(r"\$([a-zA-Z0-9_:-]+)", content)
    if not raw_names:
        return []
    available = set(skill_names)
    seen: set[str] = set()
    result: list[str] = []
    for name in raw_names:
        if name in available and name not in seen:
            seen.add(name)
            result.append(name)
    if result:
        context_logger.info("检测到 $skill 提及，直接注入完整内容: %s", result)
    return result


def to_history_messages(messages: list[dict]) -> list[HistoryMessage]:
    out: list[HistoryMessage] = []
    for msg in messages:
        role = str(msg.get("role", "") or "")
        content = str(msg.get("content", "") or "")
        tools_used = [
            str(tool_name)
            for tool_name in (msg.get("tools_used") or [])
            if isinstance(tool_name, str)
        ]
        out.append(
            HistoryMessage(
                role=role,
                content=content,
                tools_used=tools_used,
                tool_chain=to_tool_call_groups(msg.get("tool_chain") or []),
            )
        )
    return out


def build_context_hint_message(section_name: str, content: str) -> dict[str, str]:
    return build_context_frame_message(
        build_context_frame_content(
            [
                PromptSectionRender(
                    name=section_name,
                    content=content,
                    is_static=False,
                )
            ]
        )
    )


def build_post_reply_context_budget(
    *,
    context: "ContextBuilder",
    history: list[dict],
) -> dict[str, int]:
    history_chars = len(json.dumps(history, ensure_ascii=False)) if history else 0
    history_tokens = max(1, history_chars // 3) if history else 0
    prompt_tokens = sum(item.est_tokens for item in context.last_debug_breakdown)
    return {
        "history_messages": len(history),
        "history_chars": history_chars,
        "history_tokens": history_tokens,
        "prompt_tokens": prompt_tokens,
        "next_turn_baseline_tokens": history_tokens + prompt_tokens,
    }


def log_post_reply_context_budget(
    *,
    session_key: str,
    budget: dict[str, int],
) -> None:
    context_logger.info(
        "post_reply_context: session_key=%s history_messages=%d history_chars=%d history_tokens~=%d prompt_tokens~=%d next_turn_baseline_tokens~=%d",
        session_key,
        budget["history_messages"],
        budget["history_chars"],
        budget["history_tokens"],
        budget["prompt_tokens"],
        budget["next_turn_baseline_tokens"],
    )


def extract_react_stats(context_retry: dict[str, object]) -> dict[str, int]:
    raw = context_retry.get("react_stats")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("reasoner react_stats 不是 dict")
    stats = cast(dict[str, int], raw)
    return {key: stats[key] for key in _REACT_CONTEXT_FIELDS if key in stats}


def log_react_context_budget(
    *,
    session_key: str,
    react_stats: dict[str, int],
) -> None:
    if not react_stats:
        return
    context_logger.info(
        "react_context: session_key=%s iteration_count=%d turn_input_sum_tokens~=%d turn_input_peak_tokens~=%d final_call_input_tokens~=%d cache_hit=%d/%d",
        session_key,
        react_stats.get("iteration_count", 0),
        react_stats.get("turn_input_sum_tokens", 0),
        react_stats.get("turn_input_peak_tokens", 0),
        react_stats.get("final_call_input_tokens", 0),
        react_stats.get("cache_hit_tokens", 0),
        react_stats.get("cache_prompt_tokens", 0),
    )


def build_session_runtime_metadata(
    metadata: Mapping[str, Any],
    *,
    tools_used: list[str],
    tool_chain: list[dict],
) -> dict[str, Any]:
    """构造当前 Turn 成功提交后应发布的 Session metadata。"""

    md = dict(metadata)
    call_count = sum(
        len(group.get("calls") or [])
        for group in tool_chain
        if isinstance(group, dict)
    )

    md["last_turn_tool_calls_count"] = call_count
    md["last_turn_ts"] = datetime.now().astimezone().isoformat()
    return md


def log_preview(value: object, limit: int = _LOG_PREVIEW_LIMIT) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def estimate_messages_tokens(messages: list[dict]) -> int:
    if not messages:
        return 0
    payload = json.dumps(messages, ensure_ascii=False)
    return max(1, len(payload) // 3)


def predict_current_user_source_ref(
    *,
    session_manager: SessionManager,
    session: SessionLike,
) -> str:
    return session_manager.peek_next_message_id(session.key)
