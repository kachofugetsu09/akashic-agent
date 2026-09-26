"""公开卸载只排空目标 Fiber/硬消费者，且保留无关 live owner 与用户数据。"""
import asyncio
import sys
import pytest
from agent.plugin_composition import FiberState
from agent.plugins.install import install_git_plugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus
from tests.fixtures.plugin_workspace import initialize_plugin_workspace
from tests.test_plugin_fresh_root import module
from tests.test_plugin_install import _commit, _write_v3_plugin

CONSUMER = """
from agent.plugin_composition import ServiceKey
TARGET = ServiceKey("target.state")
STOPPING_EVENT = None
RELEASE_EVENT = None
api_version = 3
name = "consumer"
version = "1.0.0"
inject = (TARGET,)
async def apply(ctx):
    _ = ctx.require(TARGET)
    async def close():
        if STOPPING_EVENT is not None:
            STOPPING_EVENT.set()
        if RELEASE_EVENT is not None:
            await RELEASE_EVENT.wait()
    await ctx.effect(lambda: close)
    await ctx.provide(ServiceKey("consumer.state"), {"active": True})
"""

def installed_host(tmp_path, *, fail_close=False):
    """Install target, its hard consumer, and an unrelated peer for a live Manager."""
    workspace, home = tmp_path / "workspace", tmp_path / "home"
    sources = {
        "target": module("target", fail_close=fail_close),
        "consumer": CONSUMER,
        "peer": module("peer"),
    }
    for name, source_text in sources.items():
        source = tmp_path / name
        _write_v3_plugin(source, name=name, module_source=source_text)
        _commit(source)
        install_git_plugin(
            workspace=workspace,
            source=str(source),
            marketplace="lab",
            plugins_home=home,
        )
    initialize_plugin_workspace(workspace)
    host = PluginManager(
        [], event_bus=EventBus(), workspace=workspace,
        installed_cache_root=home / "cache",
    )
    return host, home / "cache/lab/target", workspace

def status_for(host, plugin_id):
    return next(item for item in host.plugin_status()["plugins"] if item["plugin_id"] == plugin_id)

def operation_status(host: PluginManager) -> dict[str, object]:
    operation = host.plugin_status()["operation"]
    assert isinstance(operation, dict)
    return operation

@pytest.mark.asyncio
async def test_uninstall_accepts_before_target_owner_drains_and_preserves_peer(
    tmp_path,
):
    """CAS accepted is returned while a real target OwnerCall still blocks cleanup."""
    host, cache, _workspace = installed_host(tmp_path)
    peer_task = None
    peer_release = asyncio.Event()
    peer_entered = asyncio.Event()
    consumer_stopping = asyncio.Event()
    consumer_release = asyncio.Event()
    try:
        await host.load_all()
        root = host.live_root
        target = host.generation("target@lab")
        consumer = host.generation("consumer@lab")
        peer = host.generation("peer@lab")
        assert root is not None and target is not None and consumer is not None and peer is not None
        assert target.fiber is not None and consumer.fiber is not None and peer.fiber is not None
        consumer.instance.module.STOPPING_EVENT = consumer_stopping
        consumer.instance.module.RELEASE_EVENT = consumer_release
        peer_context = peer.fiber.context
        peer_fiber = peer.fiber
        peer_token = peer_context.fiber.activation_token
        peer_state = dict(peer.instance.module.STATE)
        archive_ref = target.archive_ref
        assert archive_ref is not None
        archive_source = target.code_dir / "plugin.py"
        archive_source_before = archive_source.read_bytes()
        archive_descriptor_before = host._archive.read_descriptor(archive_ref)

        async def hold_peer_scope():
            async with peer_context.runtime_scope():
                peer_entered.set()
                await peer_release.wait()

        marker = target.data_dir / "keep.txt"
        marker.write_text("user data", encoding="utf-8")

        async with target.fiber.context.runtime_scope():
            accepted = await host.uninstall("target@lab")
            operation = host._operation
            assert accepted["plugin_id"] == "target@lab"
            assert accepted["state"] == "accepted"
            assert accepted["selection_ref"] == host._selection.read()
            assert operation is not None and not operation.task.done()
            assert cache.is_dir()
            assert target.module_path in sys.modules
            target_status = status_for(host, "target@lab")
            assert target_status["installed"] is True
            assert target_status["enabled"] is False
            assert target_status["selected_ref"] is None
            assert target_status["cache_exists"] is True

        await asyncio.wait_for(consumer_stopping.wait(), 5)
        peer_task = asyncio.create_task(hold_peer_scope())
        await asyncio.wait_for(peer_entered.wait(), 5)
        assert peer.fiber.state is FiberState.ACTIVE
        assert host.live_root is root
        assert host.generation("peer@lab") is peer
        assert peer.fiber is peer_fiber
        assert peer_context.fiber.activation_token is peer_token

        consumer_release.set()
        result = await asyncio.gather(operation.task, return_exceptions=True)
        assert len(result) == 1 and isinstance(result[0], dict)
        assert result[0]["plugin_id"] == "target@lab"
        assert result[0]["state"] == "removed"
        assert target.scope.closed
        assert target.module_path not in sys.modules
        assert not cache.exists()
        assert marker.read_text(encoding="utf-8") == "user data"
        assert host.live_root is root
        assert host.generation("peer@lab") is peer
        assert peer.fiber is peer_fiber
        assert peer.fiber.state is FiberState.ACTIVE
        assert peer_context.fiber.activation_token is peer_token
        assert peer.instance.module.STATE == peer_state
        assert consumer.fiber.state is FiberState.PENDING
        assert consumer.fiber.dependency_store == {}
        assert archive_source.read_bytes() == archive_source_before
        assert host._archive.read_descriptor(archive_ref) == archive_descriptor_before

        target_status = status_for(host, "target@lab")
        assert target_status["installed"] is False
        assert target_status["enabled"] is None
        assert target_status["cache_exists"] is False
        assert operation_status(host)["state"] == "done"
    finally:
        consumer_release.set()
        peer_release.set()
        if peer_task is not None:
            await asyncio.gather(peer_task, return_exceptions=True)
        await host.terminate_all()

@pytest.mark.asyncio
async def test_uninstall_cleanup_failure_keeps_owner_and_explicit_retry_finishes(tmp_path):
    """A failed close retains the same owner/cache; retry does not hit a permanent busy gate."""
    host, cache, _workspace = installed_host(tmp_path, fail_close=True)
    try:
        await host.load_all()
        target = host.generation("target@lab")
        assert target is not None
        target_fiber = target.fiber
        assert target_fiber is not None
        target_effect = target_fiber.effects[0]
        target_state = target.instance.module.STATE
        marker = target.data_dir / "keep.txt"
        marker.write_text("user data", encoding="utf-8")

        accepted = await host.uninstall("target@lab")
        operation = host._operation
        assert accepted["state"] == "accepted"
        assert operation is not None
        first_result = await asyncio.gather(operation.task, return_exceptions=True)
        assert isinstance(first_result[0], OSError)
        assert str(first_result[0]) == "connection still open"
        assert target_state["closes"] == 1
        assert host.generation("target@lab") is target
        assert target.fiber is target_fiber
        assert target_fiber.state is FiberState.UNLOADING
        assert target_effect in target_fiber.effects
        assert target.instance.module.STATE is target_state
        assert cache.is_dir()
        assert marker.read_text(encoding="utf-8") == "user data"
        failed_status = status_for(host, "target@lab")
        assert failed_status["installed"] is True
        assert failed_status["enabled"] is False
        assert failed_status["cache_exists"] is True
        assert host._draining_generations.get("target@lab") or host.generation("target@lab") is target
        pending = tuple(
            action for action in host._reload_journal.pending_recovery()
            if action.plugin_id == "target@lab" and action.generation_id == target.generation_id
        )
        assert len(pending) == 1
        assert pending[0].action == "retry_generation_cleanup"
        cleanup_tx = pending[0].tx_id
        with pytest.raises(RuntimeError, match="不匹配"):
            host._reload_journal.settle_generation_cleanup(
                tx_id=cleanup_tx, plugin_id="target@lab", generation_id="other-generation",
                receipt="forged",
            )
        assert host._reload_journal.get(cleanup_tx).phase == "cleanup_failed"

        retry_accepted = await host.uninstall("target@lab")
        retry_operation = host._operation
        assert retry_accepted["state"] == "accepted"
        assert retry_operation is not None
        retry_result = await asyncio.gather(retry_operation.task, return_exceptions=True)
        assert len(retry_result) == 1 and isinstance(retry_result[0], dict)
        assert retry_result[0]["plugin_id"] == "target@lab"
        assert retry_result[0]["state"] == "removed"
        assert target_fiber.state is FiberState.DISPOSED
        assert target_effect._closed
        assert target_state["closes"] == 2
        assert not cache.exists()
        assert marker.read_text(encoding="utf-8") == "user data"
        assert status_for(host, "target@lab")["installed"] is False
        assert host._reload_journal.get(cleanup_tx).phase == "recovered"
        assert all(action.tx_id != cleanup_tx for action in host._reload_journal.pending_recovery())
        assert host._reload_journal.events(cleanup_tx)[-1].details["cleanup_receipt"] == (
            f"generation-cleanup:{target.plugin_id}:{target.generation_id}"
        )
    finally:
        await host.terminate_all()
