"""Legacy import surface for the shared pure Wake ranking policy."""

from plugins.wake.hazard import (
    WAKE_ADMISSION_FLOOR,
    HazardResult,
    advance_hazard,
    rank_events,
)

__all__ = (
    "WAKE_ADMISSION_FLOOR",
    "HazardResult",
    "advance_hazard",
    "rank_events",
)
