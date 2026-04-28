from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from agent.core.types import ChatMessage, HistoryMessage, to_tool_call_groups
from agent.prompting import (
    PromptSectionRender,
    build_context_frame_content,
    build_context_frame_message,
    is_context_frame,
)

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.core.runtime_support import SessionLike
    from session.manager import SessionManager

context_logger = logging.getLogger("agent.core.passive_turn.context_store")
_LOG_PREVIEW_LIMIT = 160


def collect_skill_mentions(content: str, skills: list[dict]) -> list[str]:
    raw_names = re.findall(r"\$([a-zA-Z0-9_-]+)", content)
    if not raw_names:
        return []
    available = {s["name"] for s in skills if isinstance(s.get("name"), str)}
    seen: set[str] = set()
    result: list[str] = []
    for name in raw_names:
        if name in available and name not in seen:
            seen.add(name)
            result.append(name)
    if result:
        context_logger.info("检测到 $skill 提及，直接注入完整内容: %s", result)
    return result


def to_chat_messages(messages: list[dict]) -> list[ChatMessage]:
    return [
        ChatMessage(
            role=str(msg.get("role", "") or ""),
            content=str(msg.get("content", "") or ""),
        )
        for msg in messages
    ]


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


def is_llm_context_frame(message: dict) -> bool:
    content = message.get("content")
    return isinstance(content, str) and is_context_frame(content)


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
    history_window: int,
) -> dict[str, int]:
    history_stats = estimate_history_budget(history)
    debug_breakdown = getattr(context, "last_debug_breakdown", []) or []
    prompt_tokens = sum(
        int(getattr(item, "est_tokens", 0) or 0)
        for item in debug_breakdown
    )
    return {
        "history_window": history_window,
        "history_messages": history_stats["messages"],
        "history_chars": history_stats["chars"],
        "history_tokens": history_stats["tokens"],
        "prompt_tokens": prompt_tokens,
        "next_turn_baseline_tokens": history_stats["tokens"] + prompt_tokens,
    }


def log_post_reply_context_budget(
    *,
    session_key: str,
    budget: dict[str, int],
) -> None:
    context_logger.info(
        "post_reply_context: session_key=%s history_window=%d history_messages=%d history_chars=%d history_tokens~=%d prompt_tokens~=%d next_turn_baseline_tokens~=%d",
        session_key,
        budget["history_window"],
        budget["history_messages"],
        budget["history_chars"],
        budget["history_tokens"],
        budget["prompt_tokens"],
        budget["next_turn_baseline_tokens"],
    )


def extract_react_stats(context_retry: dict[str, object]) -> dict[str, int]:
    raw = context_retry.get("react_stats")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key in (
        "iteration_count",
        "turn_input_sum_tokens",
        "turn_input_peak_tokens",
        "final_call_input_tokens",
        "cache_prompt_tokens",
        "cache_hit_tokens",
    ):
        value = raw.get(key)
        if value is None:
            continue
        try:
            out[key] = int(value)
        except (TypeError, ValueError):
            continue
    return out


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


def estimate_history_budget(history: list[dict]) -> dict[str, int]:
    if not history:
        return {"messages": 0, "chars": 0, "tokens": 0}
    payload = json.dumps(history, ensure_ascii=False)
    chars = len(payload)
    return {
        "messages": len(history),
        "chars": chars,
        "tokens": max(1, chars // 3),
    }


def update_session_runtime_metadata(
    session: object,
    *,
    tools_used: list[str],
    tool_chain: list[dict],
) -> None:
    md = session.metadata if isinstance(session.metadata, dict) else {}  # type: ignore[union-attr]
    call_count = sum(
        len(group.get("calls") or [])
        for group in tool_chain
        if isinstance(group, dict)
    )

    md["last_turn_tool_calls_count"] = call_count
    md["last_turn_ts"] = datetime.now().astimezone().isoformat()
    session.metadata = md  # type: ignore[union-attr]


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
    peek = getattr(session_manager, "peek_next_message_id", None)
    if callable(peek):
        return str(peek(session.key))
    if session.messages:
        last_id = str(session.messages[-1].get("id", "") or "").strip()
        if last_id:
            return last_id
    return ""


