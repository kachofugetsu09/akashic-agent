from __future__ import annotations

from typing import TypeVar

from agent.lifecycle.types import AfterReasoningCtx, BeforeTurnCtx, PromptRenderCtx
from agent.plugin_composition import (
    CompositionError,
    CONTEXT_PREPARED_EVENT,
    EmitEventKey,
    ObserveEventKey,
    SerialEventKey,
)
from agent.plugins.snapshot import get_lifecycle_runtime_snapshot

P = TypeVar("P")

PROMPT_RENDER_EVENT = SerialEventKey[PromptRenderCtx, object]("turn.prompt_render")
AFTER_REASONING_PREPROCESS_EVENT = SerialEventKey[AfterReasoningCtx, object](
    "turn.after_reasoning.preprocess"
)
AFTER_REASONING_CLEANUP_EVENT = SerialEventKey[AfterReasoningCtx, object](
    "turn.after_reasoning.cleanup"
)


def emit_composition_lifecycle(
    key: EmitEventKey[P],
    payload: P,
) -> None:
    """Emit one synchronous lifecycle event from the request's frozen Root."""

    # 1. Bootstrap snapshots without a composition Root have no plugin listeners.
    snapshot = get_lifecycle_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        return

    # 2. The frozen Root owns listener order and propagates failures immediately.
    snapshot.composition_root.context.emit(key, payload)


async def run_composition_lifecycle(
    key: SerialEventKey[P, object],
    payload: P,
) -> None:
    """Run one lifecycle seam from the request's frozen composition Root."""

    # 1. Bootstrap snapshots without a composition Root have no plugin listeners.
    snapshot = get_lifecycle_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        return

    # 2. These domain seams order transformations but cannot terminate the turn.
    result = await snapshot.composition_root.context.serial(key, payload)
    if result is not None:
        raise CompositionError(
            "LIFECYCLE_BAIL_NOT_ALLOWED",
            f"lifecycle 接入点不接受 Bail: {key.name}",
        )


async def observe_composition_event(
    key: ObserveEventKey[P],
    payload: P,
) -> None:
    """Observe one settled fact from the request's frozen composition Root."""

    # 1. Bootstrap snapshots without a composition Root have no plugin listeners.
    snapshot = get_lifecycle_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        return

    # 2. Observe owns failure isolation for ordinary plugin listeners; binding
    #    and caller cancellation failures remain fail-loud at this boundary.
    await snapshot.composition_root.context.observe(key, payload)
