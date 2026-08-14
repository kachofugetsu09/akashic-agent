from __future__ import annotations

from typing import TYPE_CHECKING

from agent.plugin_composition import SerialEventKey

if TYPE_CHECKING:
    from bus.events_lifecycle import TurnCommitted


AFTER_TURN_COMMITTED: SerialEventKey["TurnCommitted", object] = SerialEventKey(
    "turn.after_turn.committed"
)
