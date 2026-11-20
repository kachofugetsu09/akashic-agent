from __future__ import annotations

from typing import cast

import pytest

from agent.plugin_composition import Context
from plugins.computer import plugin as computer_plugin
from plugins.workloads.contract import WORKLOADS, Workload, WorkloadHandle, WorkloadPort


class _FakeWorkloads:
    def __init__(self) -> None:
        self.registered: list[Workload] = []

    async def register(self, ctx: Context, workload: Workload) -> WorkloadHandle:
        _ = ctx
        self.registered.append(workload)
        return cast(WorkloadHandle, object())


class _FakeCtx:
    def __init__(self, workloads: _FakeWorkloads) -> None:
        self._workloads = workloads

    def require(self, key: object) -> _FakeWorkloads:
        assert key is WORKLOADS
        return self._workloads


@pytest.mark.asyncio
async def test_computer_workload_exposes_background_spaces_port() -> None:
    fake_workloads = _FakeWorkloads()
    _ = await computer_plugin._register_workload(cast(Context, _FakeCtx(fake_workloads)))  # pyright: ignore[reportPrivateUsage]
    assert len(fake_workloads.registered) == 1
    workload = fake_workloads.registered[0]
    assert WorkloadPort("opencli", 19826, loopback=19825) in workload.ports
    assert WorkloadPort("spaces", 19827, loopback=19827) in workload.ports
