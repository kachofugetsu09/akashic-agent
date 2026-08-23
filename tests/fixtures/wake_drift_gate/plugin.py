from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from agent.plugin_composition import RUNTIME_STARTED, Context, ServiceKey

api_version = 3
name = "wake_drift_gate"
version = "3.0.0"
desc = "Deterministic external Content and Drift proposal boundary"
author = "Akashic Core"
skill_roots = ()
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()


class BoundContentSource(Protocol):
    def submit(
        self, batch_id: str, items: Sequence[Mapping[str, object]]
    ) -> Mapping[str, object]: ...


class ContentSourceServices(Protocol):
    def bind(self, source_id: str) -> BoundContentSource: ...


class DriftProposalServices(Protocol):
    def propose(
        self,
        proposal_id: str,
        revision: str,
        payload: Mapping[str, object],
        due_at: datetime,
        *,
        next_due: datetime | None = None,
    ) -> Mapping[str, object]: ...


CONTENT_SOURCE = ServiceKey[ContentSourceServices]("content.source.v1")
DRIFT_PROPOSALS = ServiceKey[DriftProposalServices]("drift.proposals.v1")
inject = (CONTENT_SOURCE, DRIFT_PROPOSALS)


async def apply(ctx: Context, config: object) -> None:
    """Submit deterministic external facts only after formal runtime start."""

    _ = config
    content = ctx.require(CONTENT_SOURCE).bind("wake-drift-fixture")
    drift = ctx.require(DRIFT_PROPOSALS)

    def seed(_event: object) -> None:
        now = datetime.now(UTC)
        _ = content.submit(
            "fixture:1",
            (
                {
                    "item_id": "content:1",
                    "revision": "1",
                    "payload": {"kind": "fixture", "preprocess_score": 0.9},
                    "not_before": now,
                    "requires_ack": False,
                },
            ),
        )
        _ = drift.propose(
            "drift:1",
            "1",
            {"prompt": "fixture reflection"},
            now,
            next_due=now + timedelta(minutes=5),
        )

    _ = await ctx.on(RUNTIME_STARTED, seed)
