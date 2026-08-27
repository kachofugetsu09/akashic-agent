"""Source-facing Alert and Context contracts owned by Wake."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from agent.plugin_composition import ServiceKey


class WakeAlertSource(Protocol):
    def report(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]: ...

    def status(self, *, source_id: str, event_id: str) -> str | None: ...


class WakeContextSource(Protocol):
    def report(
        self,
        *,
        source_id: str,
        event_id: str,
        payload: Mapping[str, object],
        observed_at: datetime,
        expires_at: datetime | None = None,
    ) -> Mapping[str, object]: ...


WAKE_ALERT_SOURCE = ServiceKey[WakeAlertSource]("wake.alert_source.v1")
WAKE_CONTEXT_SOURCE = ServiceKey[WakeContextSource]("wake.context_source.v1")

__all__ = [
    "WAKE_ALERT_SOURCE",
    "WAKE_CONTEXT_SOURCE",
    "WakeAlertSource",
    "WakeContextSource",
]
