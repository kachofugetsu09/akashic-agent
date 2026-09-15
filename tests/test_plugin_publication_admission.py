"""等待普通调用不消耗提交期限；用受控时钟和调度点验证。"""
import asyncio

import pytest

from agent.plugins.snapshot import lease_runtime_snapshot
from tests.test_plugin_update_operation_lease import installed_host, update_api


@pytest.mark.asyncio
async def test_publication_wait_outlives_commit_deadline(tmp_path, monkeypatch):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        loop = asyncio.get_running_loop()
        original_time = loop.time
        elapsed = 0.0
        monkeypatch.setattr(loop, "time", lambda: original_time() + elapsed)
        before = (workspace / "runtime/plugin-stable.json").read_bytes()
        entered = asyncio.Event()
        wait_for_no_leases = host.snapshot_store.wait_for_no_leases

        async def wait(snapshot):
            entered.set()
            await wait_for_no_leases(snapshot)

        monkeypatch.setattr(host.snapshot_store, "wait_for_no_leases", wait)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "slow-caller", source=str(source), marketplace="lab")
            api.publish(ctx, "slow-caller")
            await entered.wait()
            elapsed = host.POST_PUBLISH_TIMEOUT_SECONDS + 1
            # 两次队列交接让已到期 timer 及其取消继续执行，不等待真实时间。
            for _ in range(2):
                resumed = loop.create_future()
                loop.call_soon(resumed.set_result, None)
                await resumed
            assert api.read(ctx, "slow-caller").publishing
            assert not api.read(ctx, "slow-caller").error
            assert host.current_snapshot is snapshot
            assert (workspace / "runtime/plugin-stable.json").read_bytes() == before
        await host._update_publication[1]
        assert host.read_update("slow-caller").phase == "committed"
