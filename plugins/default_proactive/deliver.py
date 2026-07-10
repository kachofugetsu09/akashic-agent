from __future__ import annotations

from typing import Any, Callable

from proactive_v2.config import ProactiveConfig
from plugins.default_proactive.context import AgentTickContext
from plugins.default_proactive.resolve import ResolveResult


class ProactiveDeliverer:
    def __init__(
        self,
        *,
        cfg: ProactiveConfig,
        session_key: str,
        turn_orchestrator: Any,
        record_finish_fn: Callable[..., None],
    ) -> None:
        self._cfg = cfg
        self._session_key = session_key
        self._turn_orchestrator = turn_orchestrator
        self._record_finish_fn = record_finish_fn
        self.last_sent: bool | None = None

    async def deliver(
        self,
        ctx: AgentTickContext,
        decision: ResolveResult,
    ) -> float | None:
        if self._turn_orchestrator is None:
            raise RuntimeError("proactive turn_orchestrator is required")
        self.last_sent = None
        self.last_sent = await self._turn_orchestrator.handle_proactive_turn(
            result=decision.result,
            session_key=self._session_key,
            channel=str(self._cfg.default_channel or "").strip(),
            chat_id=str(self._cfg.default_chat_id or "").strip(),
        )
        if ctx.drift_entered and (ctx.draft_message or ctx.draft_media):
            ctx.drift_message_sent = bool(self.last_sent)
        self._record_finish_fn(ctx, result=decision.result)
        return 0.0
