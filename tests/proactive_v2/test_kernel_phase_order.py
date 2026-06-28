from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.core.proactive_kernel import ProactiveKernel
from proactive_v2.frame import ProactiveFrame
from proactive_v2.modules_pipeline import LegacyPipelineModule
from proactive_v2.phases import ProactivePhaseRunner


class _Module:
    def __init__(
        self,
        slot: str,
        phase: str,
        calls: list[str],
        requires: tuple[str, ...] = (),
    ) -> None:
        self.slot = slot
        self.phase = phase
        self.calls = calls
        self.requires = requires

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        self.calls.append(self.slot)
        return frame


@pytest.mark.asyncio
async def test_proactive_phase_runner_orders_phase_and_requires():
    calls: list[str] = []
    runner = ProactivePhaseRunner([
        _Module("proactive.prompt.plugin", "proactive.prompt", calls),
        _Module("proactive.source.collect", "proactive.source", calls),
        _Module(
            "proactive.source.mcp_content",
            "proactive.source",
            calls,
            requires=("proactive.source.collect",),
        ),
    ])

    await runner.run(ProactiveFrame(input=object()))  # type: ignore[arg-type]

    assert calls == [
        "proactive.source.collect",
        "proactive.source.mcp_content",
        "proactive.prompt.plugin",
    ]


@pytest.mark.asyncio
async def test_proactive_kernel_runs_legacy_pipeline_module():
    pipeline = AsyncMock()
    pipeline.run = AsyncMock(return_value=0.42)
    kernel = ProactiveKernel([LegacyPipelineModule(pipeline)])

    assert await kernel.run_tick("telegram:1") == 0.42
    pipeline.run.assert_awaited_once_with()
