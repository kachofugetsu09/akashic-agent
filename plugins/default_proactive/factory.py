from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from agent.tools.web_fetch import WebFetchTool
from plugins.default_proactive.source import McpGatewaySource
from plugins.default_proactive.runtime import (
    ProactiveFlowRuntime,
    ProactiveFlowDeps,
)
from plugins.drift_flow.factory import (
    LlmFn,
    RecentChatFn,
    build_drift_llm_fn,
    build_drift_pipeline,
    build_drift_recent_chat_fn,
)
from plugins.drift_flow.runtime import DriftTurnPipeline
from plugins.default_proactive.anyaction import AnyActionGate, QuotaStore
from plugins.default_proactive.deduper import MessageDeduper
from plugins.proactive_flow.tools import ToolDeps
from proactive_v2.runtime_scope import ProactiveRuntimeScope
from proactive_v2.sensor import RecentProactiveMessage


RecentProactiveFn = Callable[[], list[RecentProactiveMessage]] | None


class AgentTickFactory:
    def __init__(self, deps: ProactiveRuntimeScope) -> None:
        self._deps = deps

    def build_runtime(self) -> ProactiveFlowRuntime:
        return ProactiveFlowRuntime(self._build_runtime_deps())

    def _build_runtime_deps(self) -> ProactiveFlowDeps:
        # 1. 先确定本轮 proactive 要服务哪个 session。
        session_key = self._get_session_key()
        # 2. 再把 tick 运行期依赖逐项组装好：
        #    最近用户时间 / 工具依赖 / gateway 数据依赖 / 近期主动消息读取函数。
        last_user_at_fn = self._build_last_user_at_fn(session_key)
        source = self._build_mcp_source()
        tool_deps = self._build_tool_deps(source)
        gateway_deps = source.build_gateway_deps(
            web_fetch_tool=tool_deps.web_fetch_tool,
            max_chars=tool_deps.max_chars,
        )
        recent_proactive_fn = self._build_recent_proactive_fn()
        drift_pipeline = self._build_drift_pipeline()
        any_action_gate = self._deps.any_action_gate or self._build_anyaction_gate()
        deduper = self._deps.deduper
        if deduper is None:
            deduper = self._build_message_deduper()

        return ProactiveFlowDeps(
            cfg=self._deps.cfg,
            session_key=session_key,
            state_store=self._deps.state_store,
            any_action_gate=any_action_gate,
            last_user_at_fn=last_user_at_fn,
            passive_busy_fn=self._deps.passive_busy_fn,
            turn_orchestrator=self._deps.turn_orchestrator,
            deduper=deduper,
            tool_deps=tool_deps,
            gateway_deps=gateway_deps,
            workspace_context_fn=self._deps.workspace_context_fn,
            llm_fn=self._build_llm_fn(),
            rng=self._deps.rng,
            recent_proactive_fn=recent_proactive_fn,
            drift_pipeline=drift_pipeline,
            schedule_fn=self._deps.schedule_fn,
            event_bus=self._deps.event_bus,
            tool_hooks=self._deps.tool_hooks,
        )

    def _get_session_key(self) -> str:
        return self._deps.sense.target_session_key()

    def _build_last_user_at_fn(self, session_key: str) -> Callable[[], Any | None]:
        presence = self._deps.presence
        if presence is None:
            return lambda: None
        return lambda: presence.get_last_user_at(session_key)

    def _build_llm_fn(self) -> LlmFn:
        return build_drift_llm_fn(self._deps)

    def _build_mcp_source(self) -> McpGatewaySource:
        return McpGatewaySource(
            self._deps.mcp_gateway,
            self._deps.proactive_sources,
            content_limit=self._deps.cfg.agent_tick_content_limit,
        )

    def _build_anyaction_gate(self) -> AnyActionGate:
        quota_path = Path(self._deps.state_store.workspace_dir) / "proactive_quota.json"
        return AnyActionGate(
            cfg=self._deps.cfg,
            quota_store=QuotaStore(quota_path),
            rng=self._deps.rng,
        )

    def _build_message_deduper(self) -> MessageDeduper | None:
        if not self._deps.cfg.message_dedupe_enabled:
            return None
        return MessageDeduper(
            provider=self._deps.provider,
            model=self._deps.model,
            max_tokens=self._deps.max_tokens,
        )

    def _build_recent_chat_fn(self) -> RecentChatFn:
        return build_drift_recent_chat_fn(self._deps)

    def _build_tool_deps(self, source: McpGatewaySource) -> ToolDeps:
        web_fetch_tool = None
        try:
            web_fetch_tool = WebFetchTool()
        except RuntimeError as e:
            logger.warning("[proactive_v2] web_fetch 不可用，已降级禁用: %s", e)
        return ToolDeps(
            web_fetch_tool=web_fetch_tool,
            memory=self._deps.memory,
            recent_chat_fn=self._build_recent_chat_fn(),
            ack_fn=source.ack_fn,
            alert_ack_fn=source.alert_ack_fn,
            max_chars=self._deps.cfg.agent_tick_web_fetch_max_chars,
        )

    def _build_recent_proactive_fn(self) -> RecentProactiveFn:
        recent_n = self._deps.cfg.message_dedupe_recent_n
        return lambda: self._deps.sense.collect_recent_proactive(recent_n)

    def _build_drift_pipeline(self) -> DriftTurnPipeline | None:
        return build_drift_pipeline(self._deps, self._build_recent_chat_fn())
