from __future__ import annotations

import pytest

from agent.core.proactive_kernel import ProactiveKernel
from proactive_v2.frame import ProactiveFrame, ProactiveTickResult
from proactive_v2.lifecycle import ProactiveLifecycleSpec


class _TerminalModule:
    slot = "proactive.commit"
    produces = ("run:result",)

    def __init__(self) -> None:
        self.slots: dict[str, object] | None = None
        self.run_count = 0

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        self.run_count += 1
        self.slots = frame.slots
        frame.output = ProactiveTickResult(base_score=0.42)
        return frame


@pytest.mark.asyncio
async def test_proactive_kernel_runs_compiled_lifecycle():
    terminal = _TerminalModule()
    kernel = ProactiveKernel(
        [terminal],
        lifecycle=ProactiveLifecycleSpec(
            id="test",
            terminal_slots=("run:result",),
        ),
    )

    assert await kernel.run_tick("telegram:1") == 0.42
    assert terminal.run_count == 1
    assert terminal.slots is not None
