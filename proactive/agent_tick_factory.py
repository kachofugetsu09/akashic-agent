from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

from agent.tools.web_fetch import WebFetchTool
from proactive import mcp_sources
from proactive_v2.agent_tick import AgentTick
from proactive_v2.tools import ToolDeps


LlmFn = Callable[[list[dict], list[dict], str | dict], Awaitable[dict | None]]
AlertFn = Callable[[], Awaitable[list[dict]]]
FeedFn = Callable[[int], Awaitable[list[dict]]]
ContextFn = Callable[[], Awaitable[list[dict]]]
RecentChatFn = Callable[[int], Awaitable[list[dict]]]
AckFn = Callable[[str, int], Awaitable[None]]
AlertAckFn = Callable[[str], Awaitable[None]]
RecentProactiveFn = Callable[[], list[dict]] | None


@dataclass
class AgentTickDeps:
    cfg: Any
    sense: Any
    presence: Any | None
    provider: Any
    model: str
    max_tokens: int
    memory: Any | None
    state_store: Any
    any_action_gate: Any
    passive_busy_fn: Any | None
    sender: Any
    deduper: Any | None
    rng: Any
    workspace_context_fn: Callable[[], str]
    observe_writer: Any | None


class AgentTickFactory:
    def __init__(self, deps: AgentTickDeps) -> None:
        self._deps = deps

    def build(self) -> AgentTick | None:
        if not self._deps.cfg.use_agent_tick:
            return None

        session_key = self._get_session_key()
        last_user_at_fn = self._build_last_user_at_fn(session_key)
        tool_deps = self._build_tool_deps()
        recent_proactive_fn = self._build_recent_proactive_fn()

        return AgentTick(
            cfg=self._deps.cfg,
            session_key=session_key,
            state_store=self._deps.state_store,
            any_action_gate=self._deps.any_action_gate,
            last_user_at_fn=last_user_at_fn,
            passive_busy_fn=self._deps.passive_busy_fn,
            sender=self._deps.sender,
            deduper=self._deps.deduper,
            tool_deps=tool_deps,
            workspace_context_fn=self._deps.workspace_context_fn,
            llm_fn=self._build_llm_fn(),
            rng=self._deps.rng,
            recent_proactive_fn=recent_proactive_fn,
        )

    def _get_session_key(self) -> str:
        try:
            return self._deps.sense.target_session_key()
        except Exception:
            return self._deps.cfg.default_chat_id or ""

    def _build_last_user_at_fn(self, session_key: str) -> Callable[[], Any | None]:
        if not self._deps.presence:
            return lambda: None
        return lambda: self._deps.presence.get_last_user_at(session_key)

    def _build_llm_fn(self) -> LlmFn:
        agent_model = self._deps.cfg.agent_tick_model or self._deps.model
        provider = self._deps.provider

        async def llm_fn(
            messages: list[dict],
            schemas: list[dict],
            tool_choice: str | dict = "auto",
        ) -> dict | None:
            resp = await provider.chat(
                messages=messages,
                tools=schemas,
                model=agent_model,
                max_tokens=self._deps.max_tokens,
                tool_choice=tool_choice,
            )
            if not resp.tool_calls:
                text = (resp.content or "").strip()
                logger.warning(
                    "[proactive_v2] llm_fn: no tool call returned (text=%r)",
                    text[:300] if text else "(empty)",
                )
                return None
            tc = resp.tool_calls[0]
            return {"id": tc.id, "name": tc.name, "input": tc.arguments}

        return llm_fn

    def _build_alert_fn(self) -> AlertFn:
        async def alert_fn() -> list[dict]:
            return await asyncio.get_running_loop().run_in_executor(
                None, mcp_sources.fetch_alert_events
            )

        return alert_fn

    def _build_feed_fn(self) -> FeedFn:
        async def feed_fn(limit: int = 5) -> list[dict]:
            events = await asyncio.get_running_loop().run_in_executor(
                None, mcp_sources.fetch_content_events
            )
            return events[:limit]

        return feed_fn

    def _build_context_fn(self) -> ContextFn:
        async def context_fn() -> list[dict]:
            return await asyncio.get_running_loop().run_in_executor(
                None, mcp_sources.fetch_context_data
            )

        return context_fn

    def _build_recent_chat_fn(self) -> RecentChatFn:
        sense = self._deps.sense

        async def recent_chat_fn(n: int = 20) -> list[dict]:
            # Sensor.collect_recent() 无参数
            return await asyncio.get_running_loop().run_in_executor(
                None, sense.collect_recent
            )

        return recent_chat_fn

    def _build_ack_fn(self) -> AckFn:
        async def ack_fn(compound_key: str, ttl_hours: int) -> None:
            """compound_key 格式："{ack_server}:{id}"，如 "feed-mcp:c1"."""
            parts = compound_key.split(":", 1)
            if len(parts) != 2:
                return
            ack_server, item_id = parts
            source_key = f"mcp:{ack_server}"
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: mcp_sources.acknowledge_content_entries(
                    [(source_key, item_id)], ttl_hours=ttl_hours
                ),
            )

        return ack_fn

    def _build_alert_ack_fn(self) -> AlertAckFn:
        async def alert_ack_fn(compound_key: str) -> None:
            """Alert 专用通道，走 acknowledge_events（非 content entries）。"""
            import types as _types
            parts = compound_key.split(":", 1)
            if len(parts) != 2:
                return
            ack_server, ack_id = parts
            event_proxy = _types.SimpleNamespace(_ack_server=ack_server, ack_id=ack_id)
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: mcp_sources.acknowledge_events([event_proxy]),
            )

        return alert_ack_fn

    def _build_tool_deps(self) -> ToolDeps:
        return ToolDeps(
            web_fetch_tool=WebFetchTool(),
            memory=self._deps.memory,
            alert_fn=self._build_alert_fn(),
            feed_fn=self._build_feed_fn(),
            context_fn=self._build_context_fn(),
            recent_chat_fn=self._build_recent_chat_fn(),
            ack_fn=self._build_ack_fn(),
            alert_ack_fn=self._build_alert_ack_fn(),
            max_chars=self._deps.cfg.agent_tick_web_fetch_max_chars,
        )

    def _build_recent_proactive_fn(self) -> RecentProactiveFn:
        recent_n = getattr(self._deps.cfg, "message_dedupe_recent_n", 5)
        if not hasattr(self._deps.sense, "collect_recent_proactive"):
            return None
        return lambda: self._deps.sense.collect_recent_proactive(recent_n)
