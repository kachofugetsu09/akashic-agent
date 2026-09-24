"""公开安装 API 的真实 caller scope、接纳交接和当前状态投影合同。"""
from __future__ import annotations

import asyncio

import pytest

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.model import FiberState
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from tests.test_plugin_update_operation_lease import HEALTH_CONTROL, installed_host


def _caller(host):
    root = host.live_root
    assert root is not None
    return root.context.require(ServiceKey("test.context"))


@pytest.mark.asyncio
async def test_public_install_hands_off_before_old_generation_drains(tmp_path):
    """accepted 只交给宿主；旧 generation 的清理仍由同一 operation 持有。"""
    async with installed_host(tmp_path, existing=True, changed=True) as (host, source, _, _):
        caller = _caller(host)
        old = host.generation("target@lab")
        peer = host.generation("peer")
        assert old is not None
        assert peer is not None
        old_fiber = old.fiber
        assert old_fiber is not None and peer.fiber is not None
        root = host.live_root
        permit = old_fiber.context.fiber.acquire_call(old_fiber.context.fiber.activation_token)
        accepted_task = asyncio.create_task(_install_in_scope(caller, "public", source))
        operation = None
        try:
            accepted = await accepted_task
            operation = host._operation
            assert operation is not None and not operation.task.done()
            assert accepted.state == "accepted"
            assert accepted.input_ref is not None
            # The caller task has left its own scope; target disposal remains owned
            # by the host operation and is still blocked by the real OwnerCall.
            assert accepted_task.done()
            assert host.live_root is root
        finally:
            # The caller scope is gone while the real target Fiber permit blocks disposal.
            permit.release()
            await asyncio.gather(accepted_task, return_exceptions=True)
            if operation is not None:
                await asyncio.gather(operation.task, return_exceptions=True)
        assert operation is not None
        caller = _caller(host)
        async with caller.runtime_scope():
            final = caller.require(PLUGIN_UPDATES).read(caller, "public")
            assert final is not None and final.state == "active"
            assert final.input_ref == accepted.input_ref
        assert host.live_root is not None
        assert host.generation("peer") is peer


async def _install_in_scope(caller, update_id, source):
    """Run the public API from a child task that enters its own caller scope."""
    async with caller.runtime_scope():
        return await caller.require(PLUGIN_UPDATES).install(
            caller, update_id, source=str(source), marketplace="lab",
        )


@pytest.mark.asyncio
async def test_duplicate_id_is_rejected_before_new_operation_or_row_mutation(tmp_path, monkeypatch):
    """重复 ID 只读旧回执，不能启动 installer 或改写旧错误。"""
    async with installed_host(tmp_path, existing=True, changed=False) as (host, source, _, _):
        caller = _caller(host)
        async with caller.runtime_scope():
            api = caller.require(PLUGIN_UPDATES)
            first = await api.install(caller, "same", source=str(source), marketplace="lab")
            before = host._reload_journal.update("same")
            called = False

            def unexpected_install(**_kwargs):
                nonlocal called
                called = True
                raise AssertionError("duplicate install reached installer")

            monkeypatch.setattr("agent.plugins.manager.install_git_plugin", unexpected_install)
            with pytest.raises(RuntimeError, match="只能查询"):
                await api.install(caller, "same", source=str(source), marketplace="lab")
            after = host._reload_journal.update("same")
            assert after == before
            assert api.read(caller, "same") == first
            assert not called


@pytest.mark.asyncio
async def test_read_requires_real_caller_scope_and_ignores_stale_error_for_active_generation(tmp_path):
    async with installed_host(tmp_path, existing=True, changed=False) as (host, source, _, _):
        caller = _caller(host)
        api = caller.require(PLUGIN_UPDATES)
        with pytest.raises(CompositionError, match="OwnerCall"):
            api.read(caller, "missing")
        async with caller.runtime_scope():
            status = await api.install(caller, "history", source=str(source), marketplace="lab")
            assert status.state == "active"
        host._reload_journal.record_update_error("history", "old recovery diagnostic")
        async with caller.runtime_scope():
            current = api.read(caller, "history")
            assert current is not None
            assert current.state == "active"
            assert current.error == "old recovery diagnostic"


@pytest.mark.asyncio
async def test_required_health_failure_is_failed_and_retry_reuses_selection(tmp_path):
    """A real required-health failure is projected and explicitly recoverable."""
    async with installed_host(tmp_path, existing=True, changed=True) as (host, source, _, _):
        caller = _caller(host)
        root = host.live_root
        peer = host.generation("peer")
        assert root is not None
        assert peer is not None and peer.fiber is not None
        control = peer.fiber.context.require(HEALTH_CONTROL)
        async with caller.runtime_scope():
            api = caller.require(PLUGIN_UPDATES)
            accepted = await api.install(caller, "failed-activation", source=str(source), marketplace="lab")
            operation = host._operation
            assert operation is not None and accepted.state in {"accepted", "active"}
        await asyncio.gather(operation.task, return_exceptions=True)
        # This is an independent pre-existing journal diagnostic; the following
        # health failure and both retry attempts remain real runtime transitions.
        host._reload_journal.record_update_error("failed-activation", "old recovery diagnostic")
        caller = _caller(host)
        async with caller.runtime_scope():
            health = caller.require(ServiceKey("test.required-health"))
            health.degrade("controlled required health failure")
            failed = host.read_update("failed-activation")
        assert failed.state == "failed"
        assert failed.selection == "selected"
        assert "required health" in failed.error
        control["fail"] = True
        with pytest.raises(
            RuntimeError,
            match=r"目标依赖未 ACTIVE: caller state=FiberState\.FAILED",
        ):
            await host.retry_runtime_recovery("target@lab")
        control["fail"] = False
        await host.retry_runtime_recovery("target@lab")
        recovered = host.read_update("failed-activation")
        assert recovered.state == "active"
        assert recovered.selection == "selected"
        assert recovered.input_ref == failed.input_ref
        assert recovered.error == "old recovery diagnostic"
        assert host.live_root is root
        caller = _caller(host)
        async with caller.runtime_scope():
            assert caller.require(ServiceKey("test.required-health")).healthy


@pytest.mark.asyncio
async def test_install_rejects_invalid_ids_without_touching_selection(tmp_path):
    async with installed_host(tmp_path, existing=True, changed=False) as (host, source, _, _):
        caller = _caller(host)
        selection_before = host._selection.read()
        async with caller.runtime_scope():
            api = caller.require(PLUGIN_UPDATES)
            for invalid in (None, False, 0, "", " padded "):
                with pytest.raises(ValueError, match="更新 ID"):
                    await api.install(caller, invalid, source=str(source), marketplace="lab")
        assert host._selection.read() == selection_before
