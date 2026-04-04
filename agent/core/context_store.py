from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agent.core.types import ChatMessage, ContextBundle
from agent.looping.turn_types import HistoryMessage, to_tool_call_groups
from agent.retrieval.protocol import RetrievalRequest

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from agent.looping.ports import SessionLike
    from agent.retrieval.protocol import MemoryRetrievalPipeline
    from bus.events import InboundMessage

logger = logging.getLogger("agent.core.context_store")


class ContextStore(ABC):
    """
    ┌──────────────────────────────────────┐
    │ ContextStore                         │
    ├──────────────────────────────────────┤
    │ 1. 读取 session history              │
    │ 2. 调 retrieval pipeline             │
    │ 3. 收 skill mentions                 │
    │ 4. 输出 ContextBundle                │
    └──────────────────────────────────────┘
    """

    @abstractmethod
    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        """准备本轮对话需要的上下文。"""


class DefaultContextStore(ContextStore):
    def __init__(
        self,
        *,
        retrieval: "MemoryRetrievalPipeline",
        context: "ContextBuilder",
    ) -> None:
        self._retrieval = retrieval
        self._context = context

    async def prepare(
        self,
        *,
        msg: "InboundMessage",
        session_key: str,
        session: "SessionLike",
    ) -> ContextBundle:
        # 1. 先读取 session history，并转换成 retrieval pipeline 需要的结构。
        raw_history = session.get_history()
        history_messages = _to_history_messages(raw_history)

        # 2. 再执行 retrieval，保持当前 pipeline 行为不变。
        retrieval_result = await self._retrieval.retrieve(
            RetrievalRequest(
                message=msg.content,
                session_key=session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                history=history_messages,
                session_metadata=(
                    session.metadata if isinstance(session.metadata, dict) else {}
                ),
                timestamp=msg.timestamp,
            )
        )

        # 3. 最后补齐 ContextBundle，先把旧链路还需要的数据塞进 metadata。
        skill_mentions = _collect_skill_mentions(
            msg.content,
            self._context.skills.list_skills(filter_unavailable=False),
        )
        return ContextBundle(
            history=_to_chat_messages(raw_history),
            memory_blocks=[retrieval_result.block] if retrieval_result.block else [],
            metadata={
                "skill_mentions": skill_mentions,
                "retrieved_memory_block": retrieval_result.block,
                "retrieval_trace": retrieval_result.trace,
                "retrieval_trace_raw": (
                    retrieval_result.trace.raw
                    if retrieval_result.trace is not None
                    else None
                ),
                "retrieval_metadata": retrieval_result.metadata,
                "history_messages": history_messages,
            },
        )


def _collect_skill_mentions(content: str, skills: list[dict]) -> list[str]:
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
        logger.info("检测到 $skill 提及，直接注入完整内容: %s", result)
    return result


def _to_chat_messages(messages: list[dict]) -> list[ChatMessage]:
    return [
        ChatMessage(
            role=str(msg.get("role", "") or ""),
            content=str(msg.get("content", "") or ""),
        )
        for msg in messages
    ]


def _to_history_messages(messages: list[dict]) -> list[HistoryMessage]:
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
