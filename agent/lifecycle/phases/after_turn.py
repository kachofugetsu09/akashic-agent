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
from agent.control.context import running_turn_id
from agent.control.ports import InputLock
from agent.core.types import to_tool_call_groups
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    collect_prefixed_slots,
    topo_sort_modules,
)
from agent.lifecycle.types import AfterTurnCtx, TurnPersistencePolicy, TurnSnapshot
from agent.model_runtime.registry import current_model_binding
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
_PERSISTENCE_SLOT = "turn:persistence"
_EXTRA_SLOT = "turn:extra"
_EXTRA_COLLECTED_SLOT = "turn:extra_collected"
_TURN_COMMITTED_SLOT = "turn:committed"
_CTX_SLOT = "turn:ctx"
_EXTRA_PREFIX = "turn:extra:"
_TELEMETRY_PREFIX = "turn:telemetry:"


class _BuildTurnWorkModule:
    slot = "after_turn.build_work"
    requires: tuple[str, ...] = ()

    def __init__(
        self,
        context: ContextBuilder,
    ) -> None:
        self._context = context

    produces = (
        _BUDGET_SLOT,
        _REACT_STATS_SLOT,
        _TOOL_CHAIN_SLOT,
        _PERSISTENCE_SLOT,
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
        canonical_history = [
            message
            for unit in session.history_units()
            for message in unit.messages
        ]
        frame.slots[_BUDGET_SLOT] = build_post_reply_context_budget(
            context=self._context,
            history=canonical_history,
        )
        frame.slots[_REACT_STATS_SLOT] = extract_react_stats(snap.ctx.context_retry)
        extra: dict[str, object] = (
            {"skip_post_memory": True}
            if (msg.metadata or {}).get("skip_post_memory")
            else {}
        )
        binding = current_model_binding()
        if binding is not None:
            extra["model_binding"] = binding.describe("agent")
        frame.slots[_EXTRA_SLOT] = extra
        frame.slots[_TOOL_CHAIN_SLOT] = list(snap.ctx.tool_chain)
        frame.slots[_PERSISTENCE_SLOT] = state.persistence
        return frame


class _BuildTurnCommittedModule:
    requires = (
        "after_turn.collect_extras",
        _BUDGET_SLOT,
        _REACT_STATS_SLOT,
        _TOOL_CHAIN_SLOT,
        _PERSISTENCE_SLOT,
        _EXTRA_SLOT,
        _EXTRA_COLLECTED_SLOT,
    )
    slot = "after_turn.build_committed"
    produces = (_TURN_COMMITTED_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        snap = frame.input
        state = snap.state
        msg = state.msg
        tool_chain_list = cast(list[dict[str, Any]], frame.slots[_TOOL_CHAIN_SLOT])
        persistence = cast(TurnPersistencePolicy, frame.slots[_PERSISTENCE_SLOT])
        raw_react_stats = snap.ctx.context_retry.get("react_stats")
        raw_model_usage = (
            raw_react_stats.get("model_usage")
            if isinstance(raw_react_stats, dict)
            else None
        )
        raw_user_message_id = snap.outbound.metadata.get("persisted_user_message_id")
        raw_user_message_ids = snap.outbound.metadata.get(
            "persisted_user_message_ids"
        )
        persisted_user_message_ids = (
            tuple(cast(list[str], raw_user_message_ids))
            if isinstance(raw_user_message_ids, list)
            and all(isinstance(item, str) and item for item in raw_user_message_ids)
            else ()
        )
        raw_source = msg.metadata.get("_control_turn_input_source")
        input_messages = [msg.content]
        skip_post_memory = (msg.metadata or {}).get("skip_post_memory") is True
        if raw_source is not None:
            inputs = cast(InputLock, raw_source).used_inputs()
            input_messages = [item.content for item in inputs]
            skip_post_memory = any(
                item.metadata.get("skip_post_memory") is True for item in inputs
            )
        aggregate_input = "\n\n".join(input_messages)
        extra = dict(cast(dict[str, object], frame.slots[_EXTRA_SLOT]))
        if skip_post_memory:
            extra["skip_post_memory"] = True
        frame.slots[_TURN_COMMITTED_SLOT] = TurnCommitted(
            session_key=state.session_key,
            channel=msg.channel,
            chat_id=msg.chat_id,
            input_message=aggregate_input,
            persisted_user_message=(
                aggregate_input if persistence.persist_user else None
            ),
            assistant_response=snap.ctx.reply,
            tools_used=list(snap.ctx.tools_used),
            turn_id=running_turn_id.get(),
            persisted_user_message_id=(
                raw_user_message_id
                if isinstance(raw_user_message_id, str) and raw_user_message_id
                else None
            ),
            persisted_user_message_ids=persisted_user_message_ids,
            assistant_message_id=snap.outbound.session_message_id,
            thinking=snap.ctx.thinking,
            raw_reply=snap.ctx.response_metadata.raw_text,
            meme_tag=snap.ctx.meme_tag,
            meme_media_count=len(snap.ctx.media),
            tool_chain_raw=copy.deepcopy(tool_chain_list),
            tool_call_groups=to_tool_call_groups(tool_chain_list),
            timestamp=msg.timestamp,
            post_reply_budget=dict(cast(dict[str, int], frame.slots[_BUDGET_SLOT])),
            react_stats=dict(cast(dict[str, int], frame.slots[_REACT_STATS_SLOT])),
            extra=extra,
            model_usage=(
                dict(raw_model_usage) if isinstance(raw_model_usage, dict) else {}
            ),
            model_binding=_model_binding_from_extra(frame.slots[_EXTRA_SLOT]),
        )
        return frame


def _model_binding_from_extra(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("after_turn extra 不是 dict")
    raw = value.get("model_binding")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("after_turn model_binding 不是 dict")
    return {str(key): item for key, item in raw.items()}


class _CollectAfterTurnExtraSlotsModule:
    slot = "after_turn.collect_extras"
    requires = ("after_turn.build_work", _EXTRA_SLOT)
    produces = (_EXTRA_SLOT, _EXTRA_COLLECTED_SLOT)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        extra = dict(cast(dict[str, object], frame.slots[_EXTRA_SLOT]))
        extra.update(collect_prefixed_slots(frame.slots, _EXTRA_PREFIX))
        frame.slots[_EXTRA_SLOT] = extra
        frame.slots[_EXTRA_COLLECTED_SLOT] = True
        return frame


class _FanoutTurnCommittedModule:
    slot = "after_turn.fanout_committed"
    requires = ("after_turn.build_committed", _TURN_COMMITTED_SLOT)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        await self._bus.fanout(cast(TurnCommitted, frame.slots[_TURN_COMMITTED_SLOT]))
        return frame


class _LogBudgetModule:
    slot = "after_turn.log_budget"
    requires = ("after_turn.build_work", _BUDGET_SLOT, _REACT_STATS_SLOT)

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
    slot = "after_turn.build_ctx"
    requires = ("after_turn.fanout_committed",)
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
    slot = "after_turn.fanout_ctx"
    requires = ("after_turn.collect_telemetry", _CTX_SLOT)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        await self._bus.fanout(cast(AfterTurnCtx, frame.slots[_CTX_SLOT]))
        return frame


class _CollectAfterTurnTelemetrySlotsModule:
    slot = "after_turn.collect_telemetry"
    requires = ("after_turn.build_ctx", _CTX_SLOT)
    produces = (_CTX_SLOT,)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        ctx = cast(AfterTurnCtx, frame.slots[_CTX_SLOT])
        extra_metadata = dict(ctx.extra_metadata)
        extra_metadata.update(collect_prefixed_slots(frame.slots, _TELEMETRY_PREFIX))
        frame.slots[_CTX_SLOT] = replace(ctx, extra_metadata=extra_metadata)
        return frame


class _DispatchOutboundModule:
    slot = "after_turn.dispatch"
    requires = ("after_turn.fanout_ctx", _CTX_SLOT)

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
                    session_message_id=outbound.session_message_id,
                )
            )
        return frame


class _ReturnOutboundMessageModule:
    slot = "after_turn.return"
    requires = ("after_turn.dispatch",)

    async def run(self, frame: AfterTurnFrame) -> AfterTurnFrame:
        frame.output = frame.input.outbound
        return frame


def default_after_turn_modules(
    bus: EventBus,
    outbound: OutboundPort,
    context: ContextBuilder,
    plugin_modules: AfterTurnModules | None = None,
) -> AfterTurnModules:
    builtins: AfterTurnModules = [
        _BuildTurnWorkModule(context),
        _CollectAfterTurnExtraSlotsModule(),
        _BuildTurnCommittedModule(),
        _FanoutTurnCommittedModule(bus),
        _LogBudgetModule(),
        _BuildAfterTurnCtxModule(),
        _CollectAfterTurnTelemetrySlotsModule(),
        _FanoutAfterTurnCtxModule(bus),
        _DispatchOutboundModule(outbound),
        _ReturnOutboundMessageModule(),
    ]
    return cast(
        AfterTurnModules,
        topo_sort_modules(builtins + list(plugin_modules or [])),
    )
