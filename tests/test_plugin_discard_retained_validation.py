"""普通插件更新能力可以重试所属验证资源清理，无需私有 operator 入口。"""

import pytest

from agent.plugins.snapshot import lease_runtime_snapshot
from agent.plugins.validation import ValidationHost
from tests.test_plugin_update_operation_lease import installed_host, update_api


@pytest.mark.asyncio
async def test_public_discard_retries_inactive_validation_owner(tmp_path, monkeypatch):
    """验证退出与首次 discard 清理都失败后，原 host 保留直到实际关闭成功。"""
    async with installed_host(tmp_path) as (host, source, workspace, _):
        original_close = ValidationHost.close
        attempts = []

        async def fail_then_close(owner):
            attempts.append(owner.identity)
            if len(attempts) <= 2:
                raise OSError("injected retained host close failure")
            await original_close(owner)

        monkeypatch.setattr(ValidationHost, "close", fail_then_close)
        async with lease_runtime_snapshot(host.snapshot_store) as stable:
            api, ctx = update_api(stable)
            selection_path = workspace / "runtime" / "plugin-stable.json"
            selection = selection_path.read_bytes()
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            candidate = host.latest_snapshot
            with pytest.raises(OSError, match="retained host"):
                async with api.open_validation(ctx, "request"):
                    [retained] = host._validation_hosts.values()
                    assert retained.parent_lease.snapshot is candidate
            assert not retained.active and not retained.closed
            with pytest.raises(OSError, match="retained host"):
                await api.discard(ctx, "request")
            assert host._validation_hosts[retained.identity] is retained
            assert api.read(ctx, "request").phase == "armed"
            assert host.ready_candidate is not None
            await api.discard(ctx, "request")
            assert retained.closed
            assert retained.identity not in host._validation_hosts
            assert attempts == [retained.identity] * 3
            assert host.ready_candidate is None
            assert candidate.snapshot_id not in host.snapshot_store.retained_snapshot_ids
            assert api.read(ctx, "request").phase == "rolled_back"
            assert selection_path.read_bytes() == selection
            await api.discard(ctx, "request")
            assert attempts == [retained.identity] * 3
