from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

from agent.skills import BUILTIN_SKILLS_DIR
from agent.tools.registry import ToolRegistry
from agent.tools.web_fetch import WebFetchTool
from agent.tools.web_search import WebSearchTool
from agent.turns.result import TurnOutbound, TurnResult, TurnTrace
from agent.turns.orchestrator import TurnOrchestrator
from proactive_v2 import mcp_sources
from proactive_v2.mcp_sources import McpClientPool
from proactive_v2.agent_tick import AgentTick
from proactive_v2.drift_runner import DriftRunner
from proactive_v2.drift_state import DriftStateStore
from proactive_v2.drift_tools import DriftToolDeps
from proactive_v2.gateway import GatewayDeps
from proactive_v2.tools import ToolDeps


BUILTIN_DRIFT_SKILLS_DIR = BUILTIN_SKILLS_DIR
BUILTIN_DRIFT_SKILL_NAMES = {"meme-manage", "create-drift-skill"}

LlmFn = Callable[[list[dict], list[dict], str | dict, bool], Awaitable[dict | None]]
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
    deduper: Any | None
    rng: Any
    workspace_context_fn: Callable[[], str]
    observe_writer: Any | None
    shared_tools: ToolRegistry | None = None
    turn_orchestrator: TurnOrchestrator | None = None
    pool: McpClientPool | None = None


class AgentTickFactory:
    def __init__(self, deps: AgentTickDeps) -> None:
        self._deps = deps

    def build(self) -> AgentTick:
        if self._deps.pool is None:
            raise RuntimeError("proactive_v2 依赖 MCP 连接池，pool 不能为空")

        # 1. 先确定本轮 proactive 要服务哪个 session。
        session_key = self._get_session_key()
        # 2. 再把 tick 运行期依赖逐项组装好：
        #    最近用户时间 / 工具依赖 / gateway 数据依赖 / 近期主动消息读取函数。
        last_user_at_fn = self._build_last_user_at_fn(session_key)
        tool_deps = self._build_tool_deps()
        gateway_deps = self._build_gateway_deps(tool_deps)
        recent_proactive_fn = self._build_recent_proactive_fn()
        drift_runner = self._build_drift_runner(tool_deps)

        # 3. 最后产出 AgentTick。后续每次 proactive loop 触发时都调用它的 tick()。
        return AgentTick(
            cfg=self._deps.cfg,
            session_key=session_key,
            state_store=self._deps.state_store,
            any_action_gate=self._deps.any_action_gate,
            last_user_at_fn=last_user_at_fn,
            passive_busy_fn=self._deps.passive_busy_fn,
            turn_orchestrator=self._deps.turn_orchestrator,
            deduper=self._deps.deduper,
            tool_deps=tool_deps,
            gateway_deps=gateway_deps,
            workspace_context_fn=self._deps.workspace_context_fn,
            llm_fn=self._build_llm_fn(),
            rng=self._deps.rng,
            recent_proactive_fn=recent_proactive_fn,
            drift_runner=drift_runner,
        )

    def _get_session_key(self) -> str:
        try:
            return self._deps.sense.target_session_key()
        except Exception:
            return self._deps.cfg.default_chat_id or ""

    def _build_last_user_at_fn(self, session_key: str) -> Callable[[], Any | None]:
        presence = self._deps.presence
        if presence is None:
            return lambda: None
        return lambda: presence.get_last_user_at(session_key)

    def _build_llm_fn(self) -> LlmFn:
        agent_model = self._deps.cfg.agent_tick_model or self._deps.model
        provider = self._deps.provider

        async def llm_fn(
            messages: list[dict],
            schemas: list[dict],
            tool_choice: str | dict = "auto",
            disable_thinking: bool = False,
        ) -> dict | None:
            # AgentTick 自己维护 messages 和工具 schema；
            # factory 这里只负责把 provider.chat 包成“返回首个 tool_call”的薄适配层。
            resp = await provider.chat(
                messages=messages,
                tools=schemas,
                model=agent_model,
                max_tokens=self._deps.max_tokens,
                tool_choice=tool_choice,
                extra_body={"enable_thinking": False} if disable_thinking else None,
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
        pool = self._deps.pool
        assert pool is not None

        async def alert_fn() -> list[dict]:
            return await mcp_sources.fetch_alert_events_async(pool)

        return alert_fn

    def _build_feed_fn(self) -> FeedFn:
        pool = self._deps.pool
        assert pool is not None

        async def feed_fn(limit: int = 5) -> list[dict]:
            events = await mcp_sources.fetch_content_events_async(pool)
            return events[:limit]

        return feed_fn

    def _build_context_fn(self) -> ContextFn:
        pool = self._deps.pool
        sense = self._deps.sense
        assert pool is not None

        async def context_fn() -> list[dict]:
            rows = await mcp_sources.fetch_context_data_async(pool)
            if not isinstance(rows, list):
                rows = []
            try:
                sleep_ctx = sense.sleep_context()
            except Exception:
                sleep_ctx = None
            if sleep_ctx is not None:
                fitbit_context = {
                    "_source": "fitbit_sleep",
                    "available": bool(getattr(sleep_ctx, "available", False)),
                    "sleep_state": str(getattr(sleep_ctx, "state", "unknown")),
                    "sleep_prob": getattr(sleep_ctx, "prob", None),
                    "sleep_prob_source": str(
                        getattr(sleep_ctx, "prob_source", "unavailable")
                    ),
                    "data_lag_min": getattr(sleep_ctx, "data_lag_min", None),
                    "sleep_24h": getattr(sleep_ctx, "sleep_24h", {}) or {},
                }
                rows.insert(0, fitbit_context)
            return rows

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
        pool = self._deps.pool
        assert pool is not None

        async def ack_fn(compound_key: str, ttl_hours: int) -> None:
            """compound_key 格式："{ack_server}:{id}"，如 "feed-mcp:c1"."""
            parts = compound_key.split(":", 1)
            if len(parts) != 2:
                return
            ack_server, item_id = parts
            source_key = f"mcp:{ack_server}"
            await mcp_sources.acknowledge_content_entries_async(
                pool, [(source_key, item_id)], ttl_hours=ttl_hours
            )

        return ack_fn

    def _build_alert_ack_fn(self) -> AlertAckFn:
        pool = self._deps.pool
        assert pool is not None

        async def alert_ack_fn(compound_key: str) -> None:
            """Alert 专用通道，走 acknowledge_events（非 content entries）。"""
            import types as _types
            parts = compound_key.split(":", 1)
            if len(parts) != 2:
                return
            ack_server, ack_id = parts
            event_proxy = _types.SimpleNamespace(_ack_server=ack_server, ack_id=ack_id)
            await mcp_sources.acknowledge_events_async(pool, [event_proxy])

        return alert_ack_fn

    def _build_tool_deps(self) -> ToolDeps:
        web_fetch_tool = None
        try:
            web_fetch_tool = WebFetchTool()
        except RuntimeError as e:
            logger.warning("[proactive_v2] web_fetch 不可用，已降级禁用: %s", e)
        return ToolDeps(
            web_fetch_tool=web_fetch_tool,
            memory=self._deps.memory,
            recent_chat_fn=self._build_recent_chat_fn(),
            ack_fn=self._build_ack_fn(),
            alert_ack_fn=self._build_alert_ack_fn(),
            max_chars=self._deps.cfg.agent_tick_web_fetch_max_chars,
        )

    def _build_gateway_deps(self, tool_deps: ToolDeps) -> GatewayDeps:
        return GatewayDeps(
            alert_fn=self._build_alert_fn(),
            feed_fn=self._build_feed_fn(),
            context_fn=self._build_context_fn(),
            web_fetch_tool=tool_deps.web_fetch_tool,
            max_chars=tool_deps.max_chars,
            content_limit=getattr(self._deps.cfg, "agent_tick_content_limit", 5),
        )

    def _build_recent_proactive_fn(self) -> RecentProactiveFn:
        recent_n = getattr(self._deps.cfg, "message_dedupe_recent_n", 5)
        return lambda: self._deps.sense.collect_recent_proactive(recent_n)

    def _build_drift_runner(self, tool_deps: ToolDeps) -> DriftRunner | None:
        if not getattr(self._deps.cfg, "drift_enabled", False):
            return None
        drift_dir = Path(self._deps.state_store.workspace_dir) / "drift"
        store = DriftStateStore(
            drift_dir,
            builtin_skills_dir=BUILTIN_DRIFT_SKILLS_DIR,
            include_builtin_skills=True,
            builtin_skill_names=BUILTIN_DRIFT_SKILL_NAMES,
        )
        return DriftRunner(
            store=store,
            tool_deps=DriftToolDeps(
                drift_dir=drift_dir,
                store=store,
                builtin_skills_dir=BUILTIN_DRIFT_SKILLS_DIR,
                memory=self._deps.memory,
                shared_tools=self._deps.shared_tools,
                send_message_fn=self._build_drift_send_message_fn(),
                max_web_fetch_chars=tool_deps.max_chars,
            ),
            max_steps=getattr(self._deps.cfg, "drift_max_steps", 20),
            step_recorder=self._record_drift_step,
        )

    def _record_drift_step(
        self,
        ctx,
        phase: str,
        tool_name: str,
        tool_call_id: str,
        tool_args: dict[str, Any],
        tool_result_text: str,
    ) -> None:
        tick = getattr(self, "_tick", None)
        if tick is None:
            return
        tick._record_tick_step(
            ctx,
            phase=phase,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_args=tool_args,
            tool_result_text=tool_result_text,
        )

    def _build_drift_send_message_fn(self) -> Callable[[str], Awaitable[bool]] | None:
        orchestrator = self._deps.turn_orchestrator
        session_key = self._get_session_key()
        state_store = self._deps.state_store
        if orchestrator is None:
            return None

        @dataclass
        class _SideEffect:
            callback: Callable[[], None]

            async def run(self) -> None:
                self.callback()

        async def send_message(content: str) -> bool:
            delivery_key = sha1(content[:500].encode()).hexdigest()[:16]
            result = TurnResult(
                decision="reply",
                outbound=TurnOutbound(session_key=session_key, content=content),
                trace=TurnTrace(source="proactive", extra={"source_mode": "drift"}),
                success_side_effects=[
                    _SideEffect(
                        callback=lambda: state_store.mark_delivery(
                            session_key,
                            delivery_key,
                        )
                    )
                ],
            )
            return await orchestrator.handle_proactive_turn(
                result=result,
                session_key=session_key,
                channel=str(self._deps.cfg.default_channel or "").strip(),
                chat_id=str(self._deps.cfg.default_chat_id or "").strip(),
            )

        return send_message
