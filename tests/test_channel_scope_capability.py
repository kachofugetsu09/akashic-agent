"""公开 scope lease 能力的不透明性与拒绝语义。

admission 只发放 opaque RuntimeLease；snapshot、composition_root 与任意
ServiceKey 解析都归 Core 私有，且每次解析都要核对当前 Task、active 与
exact Root。
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from agent.plugin_composition import (
    CompositionRoot, PluginRuntime, RUNTIME_STARTING, RuntimeStarting,
    SNAPSHOT_SEALING, SnapshotSealing,
)

from agent.plugin_composition.admission import SOURCE_ADMISSION, SourceAdmission
from agent.plugin_composition.channel_io import (
    INPUT_CUSTODY, CHANNEL_IDENTITY, CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ,
    ChannelIdentity, ChannelAttachmentImport, ChannelAttachmentRead,
    unavailable, unavailable_input_custody,
)
from agent.plugin_composition.channels import (
    CHANNELS, CHANNEL_INPUT, ChannelCapability, ChannelDefinition, ChannelInboundMessage,
    ChannelReady, StopReceipt,
)
from agent.plugin_composition.context import RuntimeLease, RuntimeScope
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler, RuntimeSnapshotLease, RuntimeSnapshotStore,
)
from plugins.channels import plugin


class Adapter:
    def __init__(self, context):
        self.context = context

    async def start(self):
        return ChannelReady(self.context.binding_token)

    async def deliver(self, request):
        raise AssertionError("此测试不得发送")

    async def stop(self):
        return StopReceipt(self.context.binding_token, True)


@asynccontextmanager
async def admission_root(tmp_path, *, inbound=None):
    root = CompositionRoot("capability-test")
    store = RuntimeSnapshotStore()
    admission = SourceAdmission(root.context, store, boot_id="host-boot", candidate=False)
    await root.context.provide(SOURCE_ADMISSION, admission)
    await root.context.provide(INPUT_CUSTODY, unavailable_input_custody())
    await root.context.provide(CHANNEL_IDENTITY, ChannelIdentity(unavailable, unavailable, unavailable))
    await root.context.provide(CHANNEL_ATTACHMENT_IMPORT, ChannelAttachmentImport(unavailable))
    await root.context.provide(CHANNEL_ATTACHMENT_READ, ChannelAttachmentRead(unavailable, unavailable))
    if inbound is not None:
        await root.context.provide(CHANNEL_INPUT, inbound)

    def runtime(name):
        return PluginRuntime(name, name + "-generation", tmp_path, tmp_path, tmp_path, {})

    await root.mount(plugin.apply, name="channels", inject=plugin.inject, runtime=runtime("channels"))
    async def contribute(ctx):
        await ctx.require(CHANNELS).register(ctx, ChannelDefinition(
            "probe", frozenset({ChannelCapability.OUTBOUND}), Adapter, None,
        ))

    await root.mount(contribute, name="probe", inject=(CHANNELS,), runtime=runtime("probe"))
    await root.context.serial(SNAPSHOT_SEALING, SnapshotSealing())
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    transaction = store.begin_publish(snapshot)
    lease = store.retain_publication_target(transaction)
    async with RuntimeScope(lease):
        await root.context.serial(RUNTIME_STARTING, RuntimeStarting())
    await store.commit(transaction, after_open=admission.open)
    try:
        yield root, store, admission
    finally:
        store.pause_admission()
        admission.close()
        await root.dispose()
        await store.close()


@pytest.mark.asyncio
async def test_public_lease_exposes_no_snapshot_root_or_service_surface(tmp_path):
    """admission 发放的 lease 没有 snapshot/Root 遍历，fork 也只产窄能力。"""
    async with admission_root(tmp_path) as (root, store, admission):
        lease = admission.lease(store.current.snapshot_id)
        assert type(lease) is RuntimeLease
        assert not isinstance(lease, RuntimeSnapshotLease)
        for attr in ("snapshot", "composition_root", "context", "root", "require", "get"):
            assert not hasattr(lease, attr), attr
        assert not hasattr(lease, "__dict__")
        forked = lease.fork()
        assert type(forked) is RuntimeLease
        assert forked.snapshot_id == lease.snapshot_id
        assert forked.active
        await lease.release()
        await forked.release()


@pytest.mark.asyncio
async def test_channel_input_requires_current_task_exact_root_and_active(tmp_path):
    """channel_input 正常消费 + 未绑定 Task/跨 Root/退役 release 均拒绝。"""
    accepted: list[tuple[str, str]] = []

    async def inbound(session_key, message_id, message):
        accepted.append((session_key, message_id))
        return message.content

    async with admission_root(tmp_path / "a", inbound=inbound) as (root_a, store_a, admission_a):
        lease = admission_a.lease(store_a.current.snapshot_id)

        # 1. 未绑定当前 Task 的 lease 不能解析服务。
        with pytest.raises(RuntimeError, match="未绑定在当前 Task"):
            admission_a.channel_input(lease)

        # 2. 绑定 scope 后正常消费，返回 Root 上已声明的输入端口。
        async with RuntimeScope(lease):
            accept = admission_a.channel_input(lease)
            result = await accept("s1", "m1", ChannelInboundMessage(
                channel="probe", sender="u", chat_id="r", content="hi",
                timestamp=datetime(2026, 9, 15, tzinfo=UTC), metadata={},
            ))
            assert accepted == [("s1", "m1")]
            assert result is not None

            # 3. 当前 Task 可读身份；跨 Task（含子 Task）拒绝，lease 不继承。
            assert admission_a.current_lease().snapshot_id == lease.snapshot_id

            async def child() -> None:
                admission_a.current_lease()

            child_task = asyncio.create_task(child())
            with pytest.raises(RuntimeError, match="不属于本 Root"):
                await child_task

        # 4. RuntimeScope 退出已释放 fork；同一 lease 对象不能再进 scope。
        with pytest.raises(RuntimeError):
            admission_a.channel_input(lease)

        # 5. 另一个 Root 的 admission 拒绝本 Root 的 lease（跨 Root）。
        async with admission_root(tmp_path / "b") as (root_b, store_b, admission_b):
            other = admission_b.lease(store_b.current.snapshot_id)
            async with RuntimeScope(other):
                with pytest.raises(RuntimeError, match="不属于本 Root"):
                    admission_a.channel_input(other)

        # 6. release 后 lease inactive，所有解析拒绝。
        stale = admission_a.lease(store_a.current.snapshot_id)
        await stale.release()
        assert not stale.active
        with pytest.raises(RuntimeError, match="已释放"):
            admission_a.channel_input(stale)

    # 7. 无绑定 Task 上 current_lease 拒绝。
    with pytest.raises(RuntimeError):
        admission_a.current_lease()


@pytest.mark.asyncio
async def test_scope_close_and_release_reject_further_binding(tmp_path):
    """scope 关闭/lease 释放后不能再次绑定；取消与关闭路径保持拒绝。"""
    async with admission_root(tmp_path) as (root, store, admission):
        lease = admission.lease(store.current.snapshot_id)
        scope = RuntimeScope(lease)
        async with scope:
            assert scope.is_current
        assert not scope.is_current
        with pytest.raises(RuntimeError, match="只能进入一次"):
            async with scope:
                pass
        # RuntimeScope 关闭时释放了 fork；再绑定被拒。
        with pytest.raises(RuntimeError):
            admission.channel_input(lease)
