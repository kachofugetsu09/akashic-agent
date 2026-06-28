from __future__ import annotations

from typing import Any, Callable

from proactive_v2.config import ProactiveConfig
from proactive_v2.context import AgentTickContext
from proactive_v2.modules_resolve import ResolveResult


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

    async def deliver(
        self,
        ctx: AgentTickContext,
        decision: ResolveResult,
    ) -> float | None:
        self._record_finish_fn(ctx, result=decision.result)
        if self._turn_orchestrator is None:
            raise RuntimeError("proactive turn_orchestrator is required")
        await self._turn_orchestrator.handle_proactive_turn(
            result=decision.result,
            session_key=self._session_key,
            channel=str(self._cfg.default_channel or "").strip(),
            chat_id=str(self._cfg.default_chat_id or "").strip(),
        )
        return 0.0
