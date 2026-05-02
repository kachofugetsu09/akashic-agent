from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from bus.event_bus import EventBus
from agent.core.runtime_support import SessionLike
from agent.core.types import ContextBundle
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    append_string_exports,
    collect_prefixed_slots,
)
from agent.lifecycle.types import BeforeTurnCtx, TurnState

if TYPE_CHECKING:
    from agent.core.passive_turn import ContextStore
    from session.manager import SessionManager


@dataclass
class BeforeTurnFrame(PhaseFrame[TurnState, BeforeTurnCtx]):
    pass


BeforeTurnModules: TypeAlias = list[PhaseModule[BeforeTurnFrame]]


_SESSION_SLOT = "session:session"
_CONTEXT_BUNDLE_SLOT = "session:context_bundle"
_CTX_SLOT = "session:ctx"
_EXTRA_HINT_PREFIX = "session:extra_hint:"
_ABORT_REPLY_SLOT = "session:abort_reply"


class _AcquireSessionModule:
    produces = (_SESSION_SLOT,)

    def __init__(self, session_manager: SessionManager) -> None:
        self._session_manager = session_manager

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        state = frame.input
        session = self._session_manager.get_or_create(state.session_key)
        state.session = session
        state.retrieval_raw = None
        frame.slots[_SESSION_SLOT] = session
        return frame


class _PrepareContextModule:
    requires = (_SESSION_SLOT,)
    produces = (_CONTEXT_BUNDLE_SLOT,)

    def __init__(self, context_store: ContextStore) -> None:
        self._context_store = context_store

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        if _CTX_SLOT in frame.slots:
            return frame
        state = frame.input
        session = cast(SessionLike, frame.slots[_SESSION_SLOT])
        bundle = await self._context_store.prepare(
            msg=state.msg,
            session_key=state.session_key,
            session=session,
        )
        state.retrieval_raw = bundle.retrieval_trace_raw
        frame.slots[_CONTEXT_BUNDLE_SLOT] = bundle
        return frame


class _BuildBeforeTurnCtxModule:
    requires = (_CONTEXT_BUNDLE_SLOT,)
    produces = (_CTX_SLOT,)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        if _CTX_SLOT in frame.slots:
            return frame
        state = frame.input
        bundle = cast(ContextBundle, frame.slots[_CONTEXT_BUNDLE_SLOT])
        frame.slots[_CTX_SLOT] = BeforeTurnCtx(
            session_key=state.session_key,
            channel=state.msg.channel,
            chat_id=state.msg.chat_id,
            content=state.msg.content,
            timestamp=state.msg.timestamp,
            skill_names=list(bundle.skill_mentions),
            retrieved_memory_block=bundle.retrieved_memory_block,
            retrieval_trace_raw=bundle.retrieval_trace_raw,
            history_messages=tuple(bundle.history_messages),
        )
        return frame


class _EmitBeforeTurnCtxModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        ctx = cast(BeforeTurnCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _ReturnBeforeTurnCtxModule:
    requires = (_CTX_SLOT,)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        frame.output = cast(BeforeTurnCtx, frame.slots[_CTX_SLOT])
        return frame


class _CollectBeforeTurnExportSlotsModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        ctx = cast(BeforeTurnCtx, frame.slots[_CTX_SLOT])
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _EXTRA_HINT_PREFIX),
        )
        abort_reply = frame.slots.get(_ABORT_REPLY_SLOT)
        if isinstance(abort_reply, str) and abort_reply:
            ctx.abort = True
            ctx.abort_reply = abort_reply
        return frame


def default_before_turn_modules(
    bus: EventBus,
    session_manager: SessionManager,
    context_store: ContextStore,
    plugin_modules_early: BeforeTurnModules | None = None,
    plugin_modules_late: BeforeTurnModules | None = None,
) -> BeforeTurnModules:
    early_modules = plugin_modules_early or []
    late_modules = plugin_modules_late or []
    return [
        _AcquireSessionModule(session_manager),
        *early_modules,
        _PrepareContextModule(context_store),
        _BuildBeforeTurnCtxModule(),
        _EmitBeforeTurnCtxModule(bus),
        *late_modules,
        _CollectBeforeTurnExportSlotsModule(),
        _ReturnBeforeTurnCtxModule(),
    ]
