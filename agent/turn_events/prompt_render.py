from __future__ import annotations

from typing import TYPE_CHECKING

from agent.plugin_composition import SerialEventKey

if TYPE_CHECKING:
    from agent.lifecycle.types import PromptRenderCtx


PROMPT_RENDER_AFTER_EVENT_BUS: SerialEventKey["PromptRenderCtx", object] = (
    SerialEventKey("turn.prompt_render.after_event_bus")
)
