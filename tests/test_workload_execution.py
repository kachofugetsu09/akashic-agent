"""O: unrelated plugin effects keep separate Workload admission and receipts."""
import asyncio
from dataclasses import asdict
import json
import os

import pytest

from plugins.host_execution.contract import WorkloadLease
from plugins.host_execution.controller import WorkloadControllerServer, _lease_key, _stop_key


@pytest.mark.asyncio
async def test_workload_plugin_cleanup_does_not_block_neighbor_receipt(tmp_path, monkeypatch):
    for relative in ("plugin-data", "runtime/plugin-validation"):
        (tmp_path / relative).mkdir(parents=True)
    state = tmp_path / "leases.json"
    controller = WorkloadControllerServer(workspace=tmp_path, socket_path=tmp_path / "controller.sock",
        docker_socket=tmp_path / "docker.sock", state_path=state, network="test",
        allowed_uid=os.getuid(), socket_gid=os.getgid(), workload_uid=os.getuid(), workload_gid=os.getgid())
    leases = [WorkloadLease(controller._workspace_id, name, "service", "formal", "transaction",
                            "generation", name, "a" * 64) for name in ("slow", "fast")]
    controller._leases.update({_lease_key(lease): asdict(lease) for lease in leases})
    controller._save_leases()
    entered, release = asyncio.Event(), asyncio.Event()

    async def docker(method, path, *, expected, body=None):
        assert method == "GET" and path.endswith("/json")
        if path == "/containers/slow/json":
            entered.set()
            await release.wait()
        return None  # Docker's explicit 404: the exact container is absent.

    monkeypatch.setattr(controller._engine, "request", docker)
    slow = asyncio.create_task(controller._dispatch("stop", {"lease": asdict(leases[0])}))
    await asyncio.wait_for(entered.wait(), 2)
    same = asyncio.create_task(controller._dispatch("stop", {"lease": asdict(leases[0])}))
    fast = asyncio.create_task(controller._dispatch("stop", {"lease": asdict(leases[1])}))
    try:
        receipt = await asyncio.wait_for(asyncio.shield(fast), 2)
        assert receipt["container_absent"] and receipt["mounts_released"]
        stopped = json.loads(state.with_suffix(".stopped.json").read_text())
        assert stopped[_stop_key(leases[1])]["complete"] is True
        assert _lease_key(leases[0]) in json.loads(state.read_text())
        assert not same.done()
    finally:
        release.set()
        await asyncio.gather(slow, same, fast)
