from __future__ import annotations

from typing import TYPE_CHECKING

from agent.plugin_composition import EmitEventKey

if TYPE_CHECKING:
    from bus.events_lifecycle import TurnCommitted


AFTER_TURN_COMMITTED: EmitEventKey["TurnCommitted"] = EmitEventKey(
    "turn.after_turn.committed"
)
