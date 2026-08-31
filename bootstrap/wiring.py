from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from agent.context import ContextBuilder
from agent.lifecycle.facade import TurnLifecycle
from agent.tools.base import Tool
from bootstrap.toolsets.meta import CommonMetaToolsetProvider
from bootstrap.toolsets.protocol import ToolsetProvider

if TYPE_CHECKING:
    from agent.looping.interrupt import ActiveTurnState


ContextFactory = Callable[[Path], Any]
def wire_turn_lifecycle(
    lifecycle: TurnLifecycle,
    *,
    active_turn_states: Mapping[str, "ActiveTurnState"],
) -> None:
    from agent.lifecycle.types import AfterStepCtx

    async def _progress_reporter(ctx: AfterStepCtx) -> None:
        state = active_turn_states.get(ctx.session_key)
        if state is None:
            return
        if ctx.partial_reply:
            state.partial_reply = ctx.partial_reply
        if ctx.partial_thinking:
            state.partial_thinking = ctx.partial_thinking
        state.tools_used = list(ctx.tools_used_so_far)
        state.tool_chain_partial = list(ctx.tool_chain_partial)

    lifecycle.on_after_step(_progress_reporter)


def resolve_context_factory(name: str) -> ContextFactory:
    if name != "default":
        raise ValueError(f"未知 context wiring: {name}；可选值: default")
    return ContextBuilder


def resolve_toolset_provider(
    name: str, *, readonly_tools: dict[str, Tool] | None = None
) -> ToolsetProvider:
    if name != "meta_common":
        raise ValueError(f"未知 toolset wiring: {name}；可选值: meta_common")
    return CommonMetaToolsetProvider(readonly_tools or {})
