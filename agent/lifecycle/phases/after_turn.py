from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import logging
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from agent.core.passive_support import (
    build_post_reply_context_budget,
    extract_react_stats,
    log_post_reply_context_budget,
    log_react_context_budget,
)
from agent.core.types import to_tool_call_groups
from agent.lifecycle.phase import PhaseFrame, PhaseModule, collect_prefixed_slots
from agent.lifecycle.types import AfterTurnCtx, TurnSnapshot
from agent.turns.outbound import OutboundDispatch, OutboundPort
from bus.event_bus import EventBus
from bus.events import OutboundMessage
from bus.events_lifecycle import TurnCommitted

if TYPE_CHECKING:
    from agent.context import ContextBuilder
    from session.manager import Session

logger = logging.getLogger(__name__)


@dataclass
class AfterTurnFrame(PhaseFrame[TurnSnapshot, OutboundMessage]):
    pass


AfterTurnModules: TypeAlias = list[PhaseModule[AfterTurnFrame]]


_BUDGET_SLOT = "turn:budget"
_REACT_STATS_SLOT = "turn:react_stats"
_TOOL_CHAIN_SLOT = "turn:tool_chain"
_OMIT_USER_TURN_SLOT = "turn:omit_user_turn"
_EXTRA_SLOT = "turn:extra"
_EXTRA_COLLECTED_SLOT = "turn:extra_collected"
_TURN_COMMITTED_SLOT = "turn:committed"
_CTX_SLOT = "turn:ctx"
_EXTRA_PREFIX = "turn:extra:"
_TELEMETRY_PREFIX = "turn:telemetry:"


class _BuildTurnWorkModule:
    def __init__(
        self,
        context: ContextBuilder,
        history_window: int = 500,
    ) -> None:
        self._context = context
        self._history_window = max(1, int(history_window))

    produces = (
        _BUDGET_SLOT,
        _REACT_STATS_SLOT,
        _TOOL_CHAIN_SLOT,
        _OMIT_USER_TURN_SLOT,
        _EXTRA_SLOT,
    )

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        snap = frame.input
        state = snap.state
        msg = state.msg
        raw_session = state.session
        if raw_session is None:
            raise RuntimeError("AfterTurn requires TurnState.session")
        session = cast("Session", raw_session)
        hw = self._history_window
        frame.slots[_BUDGET_SLOT] = build_post_reply_context_budget(
            context=self._context,
            history=session.get_history(max_messages=hw),
            history_window=hw,
        )
        frame.slots[_REACT_STATS_SLOT] = extract_react_stats(snap.ctx.context_retry)
        frame.slots[_EXTRA_SLOT] = (
            {"skip_post_memory": True}
            if (msg.metadata or {}).get("skip_post_memory")
            else {}
        )
        frame.slots[_TOOL_CHAIN_SLOT] = list(snap.ctx.tool_chain)
        frame.slots[_OMIT_USER_TURN_SLOT] = bool(
            (msg.metadata or {}).get("omit_user_turn")
        )
        return frame


class _BuildTurnCommittedModule:
    requires = (
        _BUDGET_SLOT,
        _REACT_STATS_SLOT,
        _TOOL_CHAIN_SLOT,
        _OMIT_USER_TURN_SLOT,
        _EXTRA_SLOT,
        _EXTRA_COLLECTED_SLOT,
    )
    produces = (_TURN_COMMITTED_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        snap = frame.input
        state = snap.state
        msg = state.msg
        tool_chain_list = cast(list[dict[str, Any]], frame.slots[_TOOL_CHAIN_SLOT])
        omit_user_turn = bool(frame.slots[_OMIT_USER_TURN_SLOT])
        frame.slots[_TURN_COMMITTED_SLOT] = TurnCommitted(
            session_key=state.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            input_message=msg.content,
            persisted_user_message=None if omit_user_turn else msg.content,
            assistant_response=snap.ctx.reply,
            tools_used=list(snap.ctx.tools_used),
            thinking=snap.ctx.thinking,
            raw_reply=snap.ctx.response_metadata.raw_text,
            meme_tag=snap.ctx.meme_tag,
            meme_media_count=len(snap.ctx.media),
            tool_chain_raw=copy.deepcopy(tool_chain_list),
            tool_call_groups=to_tool_call_groups(tool_chain_list),
            timestamp=msg.timestamp,
            post_reply_budget=dict(cast(dict[str, int], frame.slots[_BUDGET_SLOT])),
            react_stats=dict(cast(dict[str, int], frame.slots[_REACT_STATS_SLOT])),
            extra=dict(cast(dict[str, object], frame.slots[_EXTRA_SLOT])),
        )
        return frame


class _CollectAfterTurnExtraSlotsModule:
    requires = (_EXTRA_SLOT,)
    produces = (_EXTRA_SLOT, _EXTRA_COLLECTED_SLOT)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        extra = dict(cast(dict[str, object], frame.slots[_EXTRA_SLOT]))
        extra.update(collect_prefixed_slots(frame.slots, _EXTRA_PREFIX))
        frame.slots[_EXTRA_SLOT] = extra
        frame.slots[_EXTRA_COLLECTED_SLOT] = True
        return frame


class _FanoutTurnCommittedModule:
    requires = (_TURN_COMMITTED_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        await self._bus.fanout(cast(TurnCommitted, frame.slots[_TURN_COMMITTED_SLOT]))
        return frame


class _LogBudgetModule:
    requires = (_BUDGET_SLOT, _REACT_STATS_SLOT)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        state = frame.input.state
        log_post_reply_context_budget(
            session_key=state.session_key,
            budget=cast(dict[str, int], frame.slots[_BUDGET_SLOT]),
        )
        log_react_context_budget(
            session_key=state.session_key,
            react_stats=cast(dict[str, int], frame.slots[_REACT_STATS_SLOT]),
        )
        return frame


class _BuildAfterTurnCtxModule:
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        snap = frame.input
        state = snap.state
        frame.slots[_CTX_SLOT] = AfterTurnCtx(
            session_key=state.session_key,
            channel=snap.outbound.channel,
            chat_id=snap.outbound.chat_id,
            reply=snap.outbound.content,
            tools_used=snap.ctx.tools_used,
            thinking=snap.ctx.thinking,
            will_dispatch=state.dispatch_outbound,
        )
        return frame


class _FanoutAfterTurnCtxModule:
    requires = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        await self._bus.fanout(cast(AfterTurnCtx, frame.slots[_CTX_SLOT]))
        return frame


class _CollectAfterTurnTelemetrySlotsModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        ctx = cast(AfterTurnCtx, frame.slots[_CTX_SLOT])
        extra_metadata = dict(ctx.extra_metadata)
        extra_metadata.update(collect_prefixed_slots(frame.slots, _TELEMETRY_PREFIX))
        frame.slots[_CTX_SLOT] = replace(ctx, extra_metadata=extra_metadata)
        return frame


class _DispatchOutboundModule:
    requires = (_CTX_SLOT,)

    def __init__(self, outbound: OutboundPort) -> None:
        self._outbound = outbound

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        snap = frame.input
        outbound = snap.outbound
        if snap.state.dispatch_outbound:
            _ = await self._outbound.dispatch(
                OutboundDispatch(
                    channel=outbound.channel,
                    chat_id=outbound.chat_id,
                    content=outbound.content,
                    thinking=outbound.thinking,
                    metadata=outbound.metadata,
                    media=outbound.media,
                )
            )
        return frame


class _ReturnOutboundMessageModule:
    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        frame.output = frame.input.outbound
        return frame


def default_after_turn_modules(
    bus: EventBus,
    outbound: OutboundPort,
    context: ContextBuilder,
    history_window: int = 500,
    plugin_modules_before_commit: AfterTurnModules | None = None,
    plugin_modules_before_fanout: AfterTurnModules | None = None,
) -> AfterTurnModules:
    before_commit = plugin_modules_before_commit or []
    before_fanout = plugin_modules_before_fanout or []
    return [
        _BuildTurnWorkModule(context, history_window),
        *before_commit,
        _CollectAfterTurnExtraSlotsModule(),
        _BuildTurnCommittedModule(),
        _FanoutTurnCommittedModule(bus),
        _LogBudgetModule(),
        _BuildAfterTurnCtxModule(),
        *before_fanout,
        _CollectAfterTurnTelemetrySlotsModule(),
        _FanoutAfterTurnCtxModule(bus),
        _DispatchOutboundModule(outbound),
        _ReturnOutboundMessageModule(),
    ]
