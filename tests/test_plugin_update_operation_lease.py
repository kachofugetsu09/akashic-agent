"""公开安装只借用实际 caller scope，物理安装和清理由宿主持有。"""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager

import pytest

from agent.plugin_composition import CompositionError, ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
from agent.plugins._operation import OperationBusyError, OperationTimeoutError
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from session.log import MessageLog
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_install import _commit, _write_v3_plugin


HEALTH_CONTROL = ServiceKey("test.health-control")


CALLER = """
from agent.plugin_composition import ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
TARGET = ServiceKey("test.target")
CONTROL = ServiceKey("test.health-control")
api_version = 3
name = "caller"
version = "1.0.0"
inject = (PLUGIN_UPDATES, TARGET, CONTROL)
async def apply(ctx):
    control = ctx.require(CONTROL)
    health = await ctx.health("target-required")
    if control["fail"]:
        health.degrade("controlled required health failure")
    await ctx.provide(ServiceKey("test.required-health"), health)
    _ = ctx.require(TARGET)
    await ctx.provide(ServiceKey("test.context"), ctx)
"""

PEER = """
from agent.plugin_composition import ServiceKey
CONTROL = ServiceKey("test.health-control")
api_version = 3
name = "peer"
version = "1.0.0"
async def apply(ctx):
    await ctx.provide(CONTROL, {"fail": False})
    await ctx.provide(ServiceKey("test.peer"), ctx)
"""

VALUE = """
from agent.plugin_composition import ServiceKey
api_version = 3
name = "NAME"
version = "1.0.0"
async def apply(ctx):
    await ctx.provide(ServiceKey("test.NAME"), lambda: "old")
"""


@asynccontextmanager
async def installed_host(tmp_path, *, existing=True, changed=True):
    """Create one live Root with a real caller plugin and target artifact."""
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    builtin, source = tmp_path / "builtin", tmp_path / "source"
    _write_v3_plugin(builtin / "caller", name="caller", module_source=CALLER)
    _write_v3_plugin(builtin / "peer", name="peer", module_source=PEER)
    target = VALUE.replace("NAME", "target")
    _write_v3_plugin(source, name="target", module_source=target)
    _commit(source)
    initialize_plugin_workspace(workspace)
    if existing:
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    log = MessageLog(workspace / "sessions.db")
    host = PluginManager([builtin], event_bus=EventBus(), workspace=workspace,
                         message_log=log, installed_cache_root=home / "cache")
    try:
        await host.load_all()
        if changed:
            (source / "plugin.py").write_text(target.replace('lambda: "old"', 'lambda: "new"'))
            _commit(source)
        yield host, source, workspace, builtin
    finally:
        await host.terminate_all()
        log.close()


def caller_context(host):
    root = host.live_root
    assert root is not None
    return root.context.require(ServiceKey("test.context"))


async def install_in_caller_scope(caller, update_id, source):
    """Enter the real caller scope inside the child Task that invokes the API."""
    async with caller.runtime_scope():
        return await caller.require(PLUGIN_UPDATES).install(
            caller, update_id, source=str(source), marketplace="lab",
        )


def update_api(snapshot):
    """Keep the fixture helper for out-of-scope legacy tests; new tests use caller scope."""
    context = snapshot.composition_root.context
    return context.require(PLUGIN_UPDATES), context.require(ServiceKey("test.context"))


@pytest.mark.asyncio
async def test_install_requires_the_real_caller_owner_call(tmp_path):
    async with installed_host(tmp_path) as (host, source, _, _):
        caller = caller_context(host)
        api = caller.require(PLUGIN_UPDATES)
        with pytest.raises(CompositionError, match="OwnerCall"):
            await api.install(caller, "outside", source=str(source), marketplace="lab")
        async with caller.runtime_scope():
            status = await api.install(caller, "inside", source=str(source), marketplace="lab")
            assert status.update_id == "inside"


@pytest.mark.asyncio
async def test_same_input_active_is_a_noop_on_the_same_root_and_fiber(tmp_path):
    async with installed_host(tmp_path, existing=True, changed=False) as (host, source, _, _):
        caller = caller_context(host)
        root = host.live_root
        target = host.generation("target@lab")
        assert root is not None and target is not None
        async with caller.runtime_scope():
            status = await caller.require(PLUGIN_UPDATES).install(
                caller, "same-input", source=str(source), marketplace="lab",
            )
        assert status.state == "active"
        assert host.live_root is root
        assert host.generation("target@lab") is target


@pytest.mark.asyncio
async def test_caller_cancel_does_not_revoke_host_operation(tmp_path, monkeypatch):
    async with installed_host(tmp_path) as (host, source, _, _):
        entered, release, installer_returned = (
            threading.Event(), threading.Event(), threading.Event()
        )
        operation = None
        task = None
        operation_results: list[object] = []

        def blocked_install(**kwargs):
            entered.set()
            release.wait()
            result = install_git_plugin(**kwargs)
            installer_returned.set()
            return result

        monkeypatch.setattr("agent.plugins.manager.install_git_plugin", blocked_install)
        caller = caller_context(host)
        try:
            task = asyncio.create_task(
                install_in_caller_scope(caller, "cancelled-caller", source)
            )
            assert await asyncio.to_thread(entered.wait, 2)
            operation = host._operation
            assert operation is not None and not operation.task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not operation.revoked
        finally:
            release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if operation is not None:
                await asyncio.gather(operation.task, return_exceptions=True)
        assert operation is not None
        assert host._operation is operation and operation.task.done()
        assert installer_returned.is_set()
        caller = caller_context(host)
        async with caller.runtime_scope():
            assert caller.require(PLUGIN_UPDATES).read(caller, "cancelled-caller").state == "active"


@pytest.mark.asyncio
async def test_deadline_finishes_accepted_wait_but_keeps_physical_owner_busy(tmp_path, monkeypatch):
    async with installed_host(tmp_path) as (host, source, _, _):
        entered, release, installer_returned = (
            threading.Event(), threading.Event(), threading.Event()
        )
        operation = None
        task = None

        def blocked_install(**kwargs):
            entered.set()
            release.wait()
            result = install_git_plugin(**kwargs)
            installer_returned.set()
            return result

        monkeypatch.setattr("agent.plugins.manager.install_git_plugin", blocked_install)
        host.POST_PUBLISH_TIMEOUT_SECONDS = 0.05
        caller = caller_context(host)
        async with caller.runtime_scope():
            selection_before = host._selection.read()
            try:
                task = asyncio.create_task(
                    install_in_caller_scope(caller, "timed-out", source)
                )
                assert await asyncio.to_thread(entered.wait, 2)
                operation = host._operation
                assert operation is not None
                with pytest.raises(OperationTimeoutError):
                    await asyncio.wait_for(task, 1)
                assert not operation.task.done()
                assert operation.revoked
                with pytest.raises(OperationBusyError):
                    await caller.require(PLUGIN_UPDATES).install(
                        caller, "blocked-after-deadline", source=str(source), marketplace="lab",
                    )
            finally:
                release.set()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                if operation is not None:
                    operation_results = await asyncio.gather(
                        operation.task, return_exceptions=True,
                    )
            assert operation is not None
            assert installer_returned.is_set()
            assert operation.task.done() and operation.revoked
            assert any(isinstance(result, asyncio.CancelledError) for result in operation_results)
            assert not host._draining_generations
            assert host._selection.read() == selection_before
