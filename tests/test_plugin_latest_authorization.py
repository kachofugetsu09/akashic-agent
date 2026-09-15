"""候选程序的查询、撤销与真实提交边界；用 Event 固定调度。"""
import asyncio
import sqlite3
from pathlib import Path

import pytest

from agent.plugins.snapshot import lease_runtime_snapshot
from tests.test_plugin_update_operation_lease import installed_host, update_api


@pytest.mark.asyncio
async def test_reading_latest_does_not_request_promotion(tmp_path):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        before = (workspace / "runtime/plugin-stable.json").read_bytes()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            status = await api.install(ctx, "read-only", source=str(source), marketplace="lab")
            assert status.candidate_id == host.latest_snapshot.snapshot_id
            assert api.messages(ctx, "read-only", "plugin-validation:read-only") == ()
            with pytest.raises(PermissionError, match="会话不属于"):
                api.messages(ctx, "read-only", "any-session")
            assert api.read(ctx, "read-only") == status
            assert host._update_publication is None
            assert not host._validation_hosts
            assert (workspace / "runtime/plugin-stable.json").read_bytes() == before
            await api.discard(ctx, "read-only")


@pytest.mark.asyncio
async def test_revert_while_publication_waits_for_call_lease(tmp_path):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        before = (workspace / "runtime/plugin-stable.json").read_bytes()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            candidate = host.latest_snapshot
            api.publish(ctx, "request")
            # 宿主已持有发布请求，但不能关闭当前调用仍在使用的组合。
            assert api.read(ctx, "request").publishing
            assert host.latest_snapshot is candidate
            await api.discard(ctx, "request", reason="agent reverted")
            assert api.read(ctx, "request").phase == "rolled_back"
            assert host.current_snapshot is snapshot
            assert snapshot.accepting_leases
            assert (workspace / "runtime/plugin-stable.json").read_bytes() == before


@pytest.mark.asyncio
async def test_revert_cancels_active_candidate_and_releases_real_owner(tmp_path):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        before = (workspace / "runtime/plugin-stable.json").read_bytes()
        entered, blocked = asyncio.Event(), asyncio.Event()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "request", source=str(source), marketplace="lab")

            async def call():
                async with api.open_validation(ctx, "request"):
                    entered.set()
                    await blocked.wait()

            task = asyncio.create_task(call())
            try:
                await entered.wait()
                evidence = api.read(ctx, "request").evidence
                assert evidence is not None
                await api.discard(ctx, "request", reason="agent reverted")
                assert task.cancelled()
                assert not host._validation_hosts
                assert api.read(ctx, "request").phase == "rolled_back"
                assert (workspace / "runtime/plugin-stable.json").read_bytes() == before
            finally:
                blocked.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_candidate_call_cannot_publish_or_rerun(tmp_path):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        before = (workspace / "runtime/plugin-stable.json").read_bytes()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            with pytest.raises(RuntimeError, match="program failed"):
                async with api.open_validation(ctx, "request"):
                    raise RuntimeError("program failed")
            with pytest.raises(RuntimeError, match="失败|未知"):
                api.publish(ctx, "request")
            with pytest.raises(RuntimeError, match="失败|撤销"):
                async with api.open_validation(ctx, "request"):
                    pytest.fail("failed request was replayed")
            assert (workspace / "runtime/plugin-stable.json").read_bytes() == before
            await api.discard(ctx, "request")


@pytest.mark.asyncio
async def test_revert_after_commit_reports_commit_and_keeps_selection(tmp_path):
    async with installed_host(tmp_path) as (host, source, workspace, _):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            api.publish(ctx, "request")
        await host._update_publication[1]
        selected = (workspace / "runtime/plugin-stable.json").read_bytes()
        assert host.read_update("request").phase == "committed"
        with pytest.raises(RuntimeError, match="已提交"):
            await host.discard_update("request")
        assert (workspace / "runtime/plugin-stable.json").read_bytes() == selected


@pytest.mark.asyncio
async def test_closed_evidence_missing_is_an_error_and_never_creates_a_database(tmp_path):
    async with installed_host(tmp_path) as (host, source, _, _):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "request", source=str(source), marketplace="lab")
            async with api.open_validation(ctx, "request"):
                evidence = api.read(ctx, "request").evidence
            assert not host._validation_hosts
            assert evidence is not None
            database = Path(evidence) / "sessions.db"
            backup = database.with_suffix(".saved")
            database.rename(backup)
            try:
                with pytest.raises(sqlite3.OperationalError):
                    api.messages(ctx, "request", "plugin-validation:request")
                assert not database.exists()
            finally:
                backup.rename(database)
            await api.discard(ctx, "request")


@pytest.mark.asyncio
async def test_deferred_publication_cannot_authorize_a_different_update(tmp_path):
    async with installed_host(tmp_path) as (host, source, _, _):
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            api, ctx = update_api(snapshot)
            await api.install(ctx, "first", source=str(source), marketplace="lab")
            publish = api.publication(ctx, "first")
            await api.discard(ctx, "first")
            second = await api.install(ctx, "second", source=str(source), marketplace="lab")
            with pytest.raises(RuntimeError, match="回退"):
                publish()
            with pytest.raises(RuntimeError, match="不匹配"):
                await api.discard(ctx, "first")
            assert api.read(ctx, "second") == second
            await api.discard(ctx, "second")
