from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from proactive_v2.context import AgentTickContext

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    blocked: bool
    reason: str
    base_score: float | None
    context_as_fallback_open: bool = False


class ProactiveGateChain:
    def __init__(
        self,
        *,
        cfg: Any,
        session_key: str,
        state_store: Any,
        any_action_gate: Any | None,
        last_user_at_fn: Callable[[], datetime | None],
        passive_busy_fn: Callable[[str], bool] | None,
        rng: Any,
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._state_store = state_store
        self._any_action_gate = any_action_gate
        self._last_user_at_fn = last_user_at_fn
        self._passive_busy_fn = passive_busy_fn
        self._rng = rng

    def check(self, ctx: AgentTickContext) -> GateResult:
        if not str(self._cfg.default_chat_id or "").strip():
            logger.debug("[proactive_v2] gate: no chat_id -> blocked")
            return GateResult(blocked=True, reason="no_target", base_score=None)

        if self._passive_busy_fn and self._passive_busy_fn(self._session_key):
            logger.debug("[proactive_v2] gate: passive_busy -> blocked")
            return GateResult(blocked=True, reason="busy", base_score=None)

        if self._state_store.count_deliveries_in_window(
            self._session_key,
            self._cfg.agent_tick_delivery_cooldown_hours,
        ) > 0:
            logger.debug("[proactive_v2] gate: delivery_cooldown -> blocked")
            return GateResult(blocked=True, reason="cooldown", base_score=None)

        if self._any_action_gate is not None:
            should_act, meta = self._any_action_gate.should_act(
                now_utc=ctx.now_utc,
                last_user_at=self._last_user_at_fn(),
            )
            if not should_act:
                logger.debug("[proactive_v2] gate: anyaction -> blocked meta=%s", meta)
                return GateResult(blocked=True, reason="presence", base_score=None)

        return GateResult(
            blocked=False,
            reason="passed",
            base_score=None,
            context_as_fallback_open=self._context_as_fallback_open(ctx),
        )

    def _context_as_fallback_open(self, ctx: AgentTickContext) -> bool:
        if self._rng.random() >= self._cfg.agent_tick_context_prob:
            return False

        last_at = self._state_store.get_last_context_only_at(self._session_key)
        count_24h = self._state_store.count_context_only_in_window(
            self._session_key,
            window_hours=24,
        )
        if last_at is not None:
            elapsed = (ctx.now_utc - last_at).total_seconds()
            if elapsed < self._cfg.context_only_min_interval_hours * 3600:
                return False
        return count_24h < self._cfg.context_only_daily_max
