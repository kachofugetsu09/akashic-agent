from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Protocol

from agent.plugin_composition import Context, EmitEventKey, ServiceKey

api_version = 3
name = "content_hint_probe"
version = "3.0.0"
desc = "Independent observer for Content's lossy changed hint"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


class ContentWakeServices(Protocol):
    def snapshot(self, now: datetime) -> Mapping[str, object]: ...


EVENTMAIL_WAKE = ServiceKey[ContentWakeServices]("eventmail.wake.v1")
EVENTMAIL_CHANGED = EmitEventKey[None]("eventmail.changed")


class ContentHintProbe:
    """Record each hint and the Content snapshot visible inside its listener."""

    def __init__(self, wake: ContentWakeServices) -> None:
        self._wake = wake
        self.snapshots: list[Mapping[str, object]] = []

    @property
    def count(self) -> int:
        return len(self.snapshots)

    def changed(self, _payload: None) -> None:
        self.snapshots.append(self._wake.snapshot(datetime.now(UTC)))


CONTENT_HINT_PROBE = ServiceKey[ContentHintProbe]("fixture.content-hint-probe.v1")
inject = (EVENTMAIL_WAKE,)


async def apply(ctx: Context, config: object) -> None:
    """Publish an in-memory hint observer without source or Timer capabilities."""

    _ = config
    probe = ContentHintProbe(ctx.require(EVENTMAIL_WAKE))
    _ = await ctx.provide(CONTENT_HINT_PROBE, probe)
    _ = await ctx.on(EVENTMAIL_CHANGED, probe.changed)
