from __future__ import annotations

from typing import Any

from proactive_v2.frame import ProactiveFrame, ProactiveTickResult


class LegacyPipelineModule:
    slot = "proactive.tick.legacy_pipeline"
    phase = "proactive.tick"

    def __init__(self, pipeline: Any) -> None:
        self._pipeline = pipeline

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        frame.output = ProactiveTickResult(
            base_score=await self._pipeline.run()
        )
        return frame
