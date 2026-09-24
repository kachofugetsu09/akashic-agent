"""Focused tests for issue-750 local Fiber lifecycle on one live Root.

覆盖 T02 内核语义：局部增删/替换只影响 changed Fiber、其 owned 子
Fiber 与实际硬依赖消费者；unrelated Root/Fiber 保持身份与生命周期计数。
协调全部使用 Event/await 屏障；无 sleep 与忙循环。

签名核对依据（未执行）：
- CompositionRoot.mount(plugin, *, name=, inject=) -> Fiber
- Context.provide(key, value) / Context.effect(setup) / Context.inject(deps, apply) 均 async
- effect 的 setup 返回 cleanup；provide 无 cleanup 参数
- FiberHandle.acquire_call(expected_activation) 同步校验 token
- CompositionError.code 为独立属性；receipt().fibers 为 FiberView tuple
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Callable, Coroutine

import pytest

from agent.plugin_composition import (
    Bail,
    CompositionError,
    CompositionRoot,
    Context,
    EmitEventKey,
    Effect,
    Fiber,
    FiberHandle,
    ObserveEventKey,
    OwnerCall,
    ParallelEventKey,
    RuntimeScope,
    SerialEventKey,
    ServiceKey,
    TransformEventKey,
)
from agent.plugin_composition.model import FiberState, PluginRuntime
from agent.plugin_composition.context import _current_runtime_scope
from agent.plugin_composition.runtime_lifecycle import (
    RUNTIME_STARTED,
    RUNTIME_STARTING,
    RUNTIME_STOPPING,
)

SVC = ServiceKey[str]("test.local.svc")
SVC_A = ServiceKey[str]("test.local.svc-a")
SVC_B = ServiceKey[str]("test.local.svc-b")


def _error_code(error: BaseException | None) -> str | None:
    return error.code if isinstance(error, CompositionError) else None


def _provider_apply(
    stats: dict[str, int],
    *,
    key: ServiceKey[str] = SVC,
    marker: str = "v",
) -> Callable[[Context], Coroutine[object, object, None]]:
    async def apply(ctx: Context) -> None:
        stats["provider_apply"] += 1
        instance = f"{marker}{stats['provider_apply']}"

        def close() -> None:
            stats["provider_close"] += 1

        await ctx.effect(lambda: close)
        await ctx.provide(key, instance)

    return apply


def _consumer_apply(
    stats: dict[str, int],
    seen: list[str],
    *,
    key: ServiceKey[str] = SVC,
) -> Callable[[Context], Coroutine[object, object, None]]:
    async def apply(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        seen.append(ctx.require(key))

    return apply


@pytest.mark.asyncio
async def test_local_replace_only_affects_changed_and_consumers() -> None:
    root = CompositionRoot("local-root")
    stats = {
        "provider_apply": 0,
        "provider_close": 0,
        "consumer_apply": 0,
        "unrelated_apply": 0,
        "unrelated_close": 0,
    }
    seen: list[str] = []

    async def unrelated(ctx: Context) -> None:
        stats["unrelated_apply"] += 1

        def close() -> None:
            stats["unrelated_close"] += 1

        await ctx.effect(lambda: close)

    provider = await root.mount(
        _provider_apply(stats), name="provider"
    )
    consumer = await root.mount(
        _consumer_apply(stats, seen), name="consumer", inject=(SVC,)
    )
    unrelated_fiber = await root.mount(unrelated, name="unrelated")
    root_token = root.instance_token

    await provider.dispose()
    provider2 = await root.mount(
        _provider_apply(stats, marker="w"), name="provider"
    )

    assert root.instance_token is root_token
    assert consumer.state == FiberState.ACTIVE
    assert seen == ["v1", "w2"], "consumer 应以新实例重新 activation，而不是整 Root"
    assert stats["consumer_apply"] == 2
    assert stats["unrelated_apply"] == 1 and stats["unrelated_close"] == 0
    assert unrelated_fiber.state == FiberState.ACTIVE
    assert stats["provider_close"] == 1
    assert provider2.context.fiber.activation_token is not None
    await root.dispose()


@pytest.mark.asyncio
async def test_optional_child_restarts_without_host_restart() -> None:
    root = CompositionRoot("host-root")
    stats = {
        "provider_apply": 0,
        "provider_close": 0,
        "host_apply": 0,
        "host_close": 0,
        "child_apply": 0,
        "other_apply": 0,
    }

    async def child_apply(ctx: Context) -> None:
        stats["child_apply"] += 1
        ctx.require(SVC)

    async def other_apply(ctx: Context) -> None:
        stats["other_apply"] += 1

    async def host(ctx: Context) -> None:
        stats["host_apply"] += 1

        def close() -> None:
            stats["host_close"] += 1

        await ctx.effect(lambda: close)
        # host 自己不 inject SVC，只挂一个注入该服务的可选子 Fiber。
        await ctx.inject((SVC,), child_apply, name="child")
        await ctx.mount(other_apply, name="other")

    provider = await root.mount(_provider_apply(stats), name="provider")
    host_fiber = await root.mount(host, name="host")
    assert stats["host_apply"] == 1 and stats["child_apply"] == 1

    await provider.dispose()
    await root.mount(_provider_apply(stats, marker="w"), name="provider")

    assert stats["host_apply"] == 1, "host 不是 changed 服务的硬消费者，不得重跑 apply"
    assert stats["host_close"] == 0, "host 的资源不得被无关替换释放"
    assert stats["child_apply"] == 2, "只有 inject changed 服务的可选子 Fiber 重载"
    assert stats["other_apply"] == 1, "宿主下无关 child 不得被重载"
    assert host_fiber.state == FiberState.ACTIVE
    await root.dispose()


@pytest.mark.asyncio
async def test_inflight_call_blocks_affected_cleanup_only() -> None:
    root = CompositionRoot("call-root")
    stats = {"provider_apply": 0, "provider_close": 0, "other_apply": 0}
    consumer_unloading = asyncio.Event()
    call_started = asyncio.Event()
    release_call = asyncio.Event()
    unrelated_done = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark_unloading() -> None:
            consumer_unloading.set()

        await ctx.effect(lambda: mark_unloading)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))

    async def other(ctx: Context) -> None:
        stats["other_apply"] += 1

    other_fiber = await root.mount(other, name="other")

    async def unrelated_call() -> None:
        # 在受影响分支已开始排空后再执行，证明无关调用不被阻塞。
        await consumer_unloading.wait()
        token = other_fiber.context.fiber.activation_token
        call = other_fiber.context.fiber.acquire_call(token)
        call.release()
        unrelated_done.set()

    async def actual_call() -> None:
        token = provider.context.fiber.activation_token
        async with provider.context.fiber.acquire_call(token):
            call_started.set()
            await release_call.wait()

    call_task = asyncio.create_task(actual_call())
    await call_started.wait()
    token = provider.context.fiber.activation_token

    unrelated_task = asyncio.create_task(unrelated_call())
    dispose_task = asyncio.create_task(provider.dispose())
    # consumer_unloading 在 provider._unload 的 _owner_became_inactive 内
    # 被设置，此时 provider.state 必已是 UNLOADING——确定性顺序，无轮询。
    await consumer_unloading.wait()

    assert provider.state == FiberState.UNLOADING
    with pytest.raises(CompositionError) as excinfo:
        provider.context.fiber.acquire_call(token)
    assert excinfo.value.code == "OWNER_UNAVAILABLE"
    assert not dispose_task.done(), "在途调用未结束前不得释放受影响资源"
    assert stats["provider_close"] == 0

    await asyncio.wait_for(unrelated_done.wait(), timeout=5)
    await unrelated_task
    assert stats["other_apply"] == 1, "无关调用在排空期间完成且未触发无关重载"

    release_call.set()
    await asyncio.gather(call_task, dispose_task)
    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_call_holder_cannot_wait_own_unload() -> None:
    root = CompositionRoot("self-wait-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    provider = await root.mount(_provider_apply(stats), name="provider")

    token = provider.context.fiber.activation_token
    call = provider.context.fiber.acquire_call(token)
    # 同一任务持有本 activation 在途调用时，等待自己的卸载必须显式失败。
    with pytest.raises(CompositionError) as excinfo:
        await provider.dispose()
    assert excinfo.value.code == "REENTRANT_CALL_WAIT"
    call.release()
    await provider.dispose()
    assert provider.state == FiberState.DISPOSED
    await root.dispose()


@pytest.mark.asyncio
async def test_stale_binding_rejected_after_owner_reload() -> None:
    """同一 consumer Fiber 因 provider 变化重启后，旧绑定 token 被拒。

    两阶段协调：旧 call 未 release 时 consumer 不能完成卸载；release 后
    dispose 完成，再挂新 provider，同 Fiber 上校验旧拒新通。
    """
    root = CompositionRoot("stale-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    seen: list[str] = []
    dependent_unloading = asyncio.Event()
    SVC2 = ServiceKey[str]("test.local.svc2")

    async def consumer_apply(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        seen.append(ctx.require(SVC))
        await ctx.provide(SVC2, f"c{stats['consumer_apply']}")

    async def dependent_apply(ctx: Context) -> None:
        ctx.require(SVC2)

        def mark() -> None:
            dependent_unloading.set()

        await ctx.effect(lambda: mark)

    provider = await root.mount(_provider_apply(stats), name="provider")
    consumer = await root.mount(
        consumer_apply, name="consumer", inject=(SVC,)
    )
    await root.mount(dependent_apply, name="dependent", inject=(SVC2,))
    old_token = consumer.context.fiber.activation_token
    old_call = consumer.context.fiber.acquire_call(old_token)

    # 独立 task 发起 provider 卸载。consumer._unload 先撤销 activation
    # 并通知其依赖方（dependent 清理置位事件），随后排空本任务持有的
    # 旧 call——事件置位即确定性证明 consumer 已 UNLOADING 且在排空。
    dispose_task = asyncio.create_task(provider.dispose())
    await dependent_unloading.wait()
    assert consumer.state == FiberState.UNLOADING
    assert stats["consumer_apply"] == 1, "旧 call 未 release，新 activation 不得出现"
    assert not dispose_task.done()

    old_call.release()
    await dispose_task
    await root.mount(_provider_apply(stats, marker="w"), name="provider")
    assert consumer.state == FiberState.ACTIVE
    new_token = consumer.context.fiber.activation_token
    assert new_token is not old_token, "重载后必须是新 activation"
    assert stats["consumer_apply"] == 2 and seen == ["v1", "w2"]

    with pytest.raises(CompositionError) as excinfo:
        consumer.context.fiber.acquire_call(old_token)
    assert excinfo.value.code == "STALE_ACTIVATION"

    new_call = consumer.context.fiber.acquire_call(new_token)
    assert new_call.activation is new_token
    new_call.release()
    await root.dispose()


@pytest.mark.asyncio
async def test_reconcile_while_holding_call_during_locked_transition() -> None:
    """本任务持 F 的 OwnerCall；另一 task 的 dispose 已持 _transition
    排空等待时，本任务的 reconcile 必须显式 REENTRANT_CALL_WAIT。"""
    root = CompositionRoot("reentrant-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    consumer_unloading = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark_unloading() -> None:
            consumer_unloading.set()

        await ctx.effect(lambda: mark_unloading)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))

    # 本任务持有 provider 的在途调用。
    token = provider.context.fiber.activation_token
    call = provider.context.fiber.acquire_call(token)

    # dispose 物理 task 持 provider._transition 并排空等待本任务的 call。
    dispose_task = asyncio.create_task(provider.dispose())
    await consumer_unloading.wait()
    assert provider.state == FiberState.UNLOADING

    with pytest.raises(CompositionError) as excinfo:
        await provider.reconcile()
    assert excinfo.value.code == "REENTRANT_CALL_WAIT"

    call.release()
    await dispose_task
    assert provider.state == FiberState.DISPOSED
    await root.dispose()


@pytest.mark.asyncio
async def test_noop_reconcile_not_rejected() -> None:
    """无锁且同 epoch 的 reconcile 不得被自等待检查误拒。"""
    root = CompositionRoot("noop-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    provider = await root.mount(_provider_apply(stats), name="provider")

    await provider.reconcile()
    assert provider.state == FiberState.ACTIVE
    assert stats["provider_apply"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_released_permit_cannot_reenter() -> None:
    """已 release 的 OwnerCall 再 async with 不得执行 body。"""
    root = CompositionRoot("permit-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    provider = await root.mount(_provider_apply(stats), name="provider")

    call = provider.context.fiber.acquire_call(
        provider.context.fiber.activation_token
    )
    call.release()
    body_ran = False
    with pytest.raises(CompositionError) as excinfo:
        async with call:
            body_ran = True
    assert excinfo.value.code == "OWNER_CALL_RELEASED"
    assert not body_ran
    await root.dispose()


@pytest.mark.asyncio
async def test_failed_new_apply_does_not_restart_old_instance() -> None:
    root = CompositionRoot("failure-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    seen: list[str] = []

    provider = await root.mount(_provider_apply(stats), name="provider")
    consumer = await root.mount(
        _consumer_apply(stats, seen), name="consumer", inject=(SVC,)
    )
    assert consumer.state == FiberState.ACTIVE

    await provider.dispose()

    async def broken(ctx: Context) -> None:
        await ctx.provide(SVC, "broken")
        raise RuntimeError("startup failed")

    replacement = await root.mount(broken, name="provider")

    assert replacement.state == FiberState.FAILED
    assert replacement.error is not None
    assert stats["provider_apply"] == 1, "旧实例不得被自动重启"
    assert stats["provider_close"] == 1
    assert root.service_value(SVC) is None, "FAILED owner 的贡献不得被消费"
    assert consumer.state == FiberState.PENDING, "消费者等待新可用实例而非回放旧实例"
    await root.dispose()


@pytest.mark.asyncio
async def test_loading_owner_not_consumable_until_active() -> None:
    root = CompositionRoot("loading-root")
    entered = asyncio.Event()
    finish = asyncio.Event()

    handles: list[object] = []

    async def slow(ctx: Context) -> None:
        # 先登记服务再阻塞：LOADING 期的"半成品"已真实存在于注册表，
        # 普通读与实际调用都必须不可见/不可接纳。
        await ctx.provide(SVC, "ready")
        handles.append(ctx.fiber)
        entered.set()
        await finish.wait()

    mount_task = asyncio.create_task(root.mount(slow, name="provider"))
    await entered.wait()

    views = {v.name: v for v in root.receipt().fibers}
    assert views["provider"].state == FiberState.LOADING
    assert root.service_value(SVC) is None, "LOADING owner 的已登记服务不得被消费"
    handle = handles[0]
    with pytest.raises(CompositionError) as excinfo:
        handle.acquire_call(handle.activation_token)
    assert excinfo.value.code == "OWNER_UNAVAILABLE"
    assert not mount_task.done()

    finish.set()
    provider = await mount_task
    assert provider.state == FiberState.ACTIVE
    call = provider.context.fiber.acquire_call(
        provider.context.fiber.activation_token
    )
    call.release()
    await root.dispose()


@pytest.mark.asyncio
async def test_cancelled_apply_cannot_publish_active_after_swallowing_cancel() -> None:
    """A swallowed cancellation still prevents the Fiber from becoming ACTIVE."""
    root = CompositionRoot("cancelled-apply-root")
    entered = asyncio.Event()
    release = asyncio.Event()
    handles: list[object] = []

    async def apply(ctx: Context) -> None:
        handles.append(ctx.fiber)
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    mount_task = asyncio.create_task(root.mount(apply, name="cancelled"))
    await entered.wait()
    mount_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await mount_task

    assert handles and handles[0].state is not FiberState.ACTIVE
    assert all(view.name != "cancelled" for view in root.receipt().fibers)
    await root.dispose()


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_owner_and_explicit_retry_succeeds() -> None:
    root = CompositionRoot("cleanup-root")
    stats = {"closes": 0}
    fail_once = {"fail": True}

    async def provider(ctx: Context) -> None:
        def cleanup() -> None:
            stats["closes"] += 1
            if fail_once["fail"]:
                fail_once["fail"] = False
                raise RuntimeError("cleanup failed")

        await ctx.effect(lambda: cleanup)
        await ctx.provide(SVC, "v")

    provider_fiber = await root.mount(provider, name="provider")
    consumer = await root.mount(
        _consumer_apply({"consumer_apply": 0}, []),
        name="consumer",
        inject=(SVC,),
    )

    token = provider_fiber.context.fiber.activation_token
    source = provider_fiber.context.fiber.acquire_call(token)
    source_scope = RuntimeScope(source)
    async with source_scope:
        captured = source_scope.capture()
    admission_wait = asyncio.create_task(captured.wait_admission_closed())
    dispose_task = asyncio.create_task(provider_fiber.dispose())
    try:
        await admission_wait
        assert provider_fiber.state is FiberState.UNLOADING
        assert stats["closes"] == 0, "撤接纳通知不得提前关闭原 registration Effect"
        await captured.close()

        with pytest.raises(RuntimeError, match="cleanup failed"):
            await dispose_task

        assert provider_fiber.state == FiberState.UNLOADING
        names = {v.name for v in root.receipt().fibers}
        assert "provider" in names, "失败 owner 必须保留句柄与证据"
        assert stats["closes"] == 1
        assert consumer.state == FiberState.PENDING

        retry_wait = asyncio.create_task(captured.wait_admission_closed())
        await retry_wait
        await provider_fiber.dispose()
        assert provider_fiber.state == FiberState.DISPOSED
        names = {v.name for v in root.receipt().fibers}
        assert "provider" not in names
        assert stats["closes"] == 2
    finally:
        await captured.close()
        if not admission_wait.done():
            admission_wait.cancel()
            try:
                await admission_wait
            except asyncio.CancelledError:
                pass
        if not dispose_task.done():
            await dispose_task
        await root.dispose()


@pytest.mark.asyncio
async def test_registration_close_keeps_exact_record_until_consumer_drain() -> None:
    """撤销先关闭接纳，排空完成后才删除唯一 registration。"""

    root = CompositionRoot("registration-drain-root")
    registration: list[Effect] = []
    consumer_context: list[Context] = []
    call_started = asyncio.Event()
    admission_closed = asyncio.Event()
    release_call = asyncio.Event()
    read_during_drain: list[str] = []
    peer_events: list[str] = []
    peer_effects: list[Effect] = []
    call_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)
        consumer_context.append(ctx)

    async def peer(ctx: Context) -> None:
        peer_effects.append(await ctx.effect(lambda: lambda: peer_events.append("close")))

    try:
        provider_fiber = await root.mount(provider, name="provider")
        consumer_fiber = await root.mount(
            consumer,
            name="consumer",
            inject=(SVC,),
        )
        peer_fiber = await root.mount(peer, name="peer")
        peer_context = peer_fiber.context
        peer_token = peer_fiber.context.fiber.activation_token

        async def admitted_call() -> None:
            token = consumer_fiber.context.fiber.activation_token
            call = consumer_fiber.context.fiber.acquire_call(token)
            async with RuntimeScope(call) as scope:
                call_started.set()
                await scope.wait_admission_closed()
                admission_closed.set()
                await release_call.wait()
                read_during_drain.append(consumer_context[0].require(SVC))

        call_task = asyncio.create_task(admitted_call())
        await call_started.wait()
        effect = registration[0]
        close_task = asyncio.create_task(effect.aclose())
        await admission_closed.wait()

        provider_record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert provider_record.revoking
        assert root._providers[SVC] is provider_record  # pyright: ignore[reportPrivateUsage]
        assert root.service_value(SVC) is None
        assert effect in provider_fiber.effects
        assert read_during_drain == []

        assert peer_fiber.context is peer_context
        assert peer_fiber.context.fiber.activation_token is peer_token
        assert peer_fiber.state == FiberState.ACTIVE
        assert peer_effects[0] in peer_fiber.effects
        async with peer_fiber.context.runtime_scope():
            pass
        assert peer_events == []

        release_call.set()
        await asyncio.gather(call_task, close_task)
        assert read_during_drain == ["v"]
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
        assert effect not in provider_fiber.effects
    finally:
        release_call.set()
        if call_task is not None and not call_task.done():
            call_task.cancel()
        tasks = [task for task in (call_task, close_task) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_registration_cleanup_failure_preserves_record_and_retry() -> None:
    """撤销失败保留同一 registration，成功 retry 后旧 close 不影响新登记。"""

    root = CompositionRoot("registration-retry-root")
    registration: list[Effect] = []
    attempts = 0
    fail_once = True

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "old"))

    async def consumer(ctx: Context) -> None:
        nonlocal attempts, fail_once
        ctx.require(SVC)

        def cleanup() -> None:
            nonlocal attempts, fail_once
            attempts += 1
            if fail_once:
                fail_once = False
                raise OSError("consumer cleanup failed")

        await ctx.effect(lambda: cleanup)

    try:
        provider_fiber = await root.mount(provider, name="provider")
        provider_context = provider_fiber.context
        provider_activation = provider_context.fiber.activation_token
        await root.mount(consumer, name="consumer", inject=(SVC,))
        effect = registration[0]
        old_record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        old_revision = old_record.revision

        with pytest.raises(BaseExceptionGroup):
            await effect.aclose()
        assert old_record.revoking
        assert root._providers[SVC] is old_record  # pyright: ignore[reportPrivateUsage]
        assert effect in old_record.owner.effects

        effects_before_duplicate = tuple(provider_fiber.effects)
        with pytest.raises(CompositionError) as duplicate_error:
            await provider_context.provide(SVC, "duplicate")
        assert duplicate_error.value.code == "DUPLICATE_SERVICE"
        assert provider_fiber.context is provider_context
        assert provider_context.fiber.activation_token is provider_activation
        assert tuple(provider_fiber.effects) == effects_before_duplicate

        await effect.aclose()
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
        new_effect = await provider_context.provide(SVC, "new")
        new_record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert new_record is not old_record
        assert new_record.owner is provider_fiber
        assert new_record.revision != old_revision
        assert provider_fiber.context is provider_context
        assert provider_context.fiber.activation_token is provider_activation
        assert new_effect in provider_fiber.effects
        await effect.aclose()
        assert root.service_value(SVC) == "new"
        assert root._providers[SVC] is new_record  # pyright: ignore[reportPrivateUsage]
        assert new_effect in provider_fiber.effects
        assert attempts == 2
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_registration_close_rejects_disposed_consumer_with_exact_record() -> None:
    """reconcile 跳过 requested dispose 时仍不能删除 exact registration。"""

    root = CompositionRoot("registration-pending-root")
    registration: list[Effect] = []
    fail_cleanup = True

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))

    async def consumer(ctx: Context) -> None:
        nonlocal fail_cleanup
        ctx.require(SVC)

        def cleanup() -> None:
            if fail_cleanup:
                raise OSError("consumer still open")

        await ctx.effect(lambda: cleanup)

    try:
        await root.mount(provider, name="provider")
        consumer_fiber = await root.mount(consumer, name="consumer", inject=(SVC,))
        effect = registration[0]
        with pytest.raises(OSError, match="consumer still open"):
            await consumer_fiber.dispose()

        with pytest.raises(CompositionError) as caught:
            await effect.aclose()
        assert caught.value.code == "DEPENDENT_CLEANUP_PENDING"
        record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert record.revoking
        assert root._providers[SVC] is record  # pyright: ignore[reportPrivateUsage]

        fail_cleanup = False
        await consumer_fiber.dispose()
        await effect.aclose()
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
    finally:
        fail_cleanup = False
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["direct", "downstream", "child"])
async def test_registration_close_guard_scans_real_wait_set(shape: str) -> None:
    """原 caller 的 direct/downstream/owned-child permit 都在变更前拒绝。"""

    root = CompositionRoot(f"registration-guard-{shape}")
    registration: list[Effect] = []
    child_handles: list[FiberHandle] = []
    call: OwnerCall | None = None
    released = False

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))

    try:
        await root.mount(provider, name="provider")

        if shape == "direct":
            async def direct(ctx: Context) -> None:
                ctx.require(SVC)

            target = await root.mount(direct, name="direct", inject=(SVC,))
            call = target.context.fiber.acquire_call(
                target.context.fiber.activation_token
            )
        elif shape == "downstream":
            async def middle(ctx: Context) -> None:
                ctx.require(SVC)
                await ctx.provide(SVC_A, "middle")

            async def downstream(ctx: Context) -> None:
                ctx.require(SVC_A)

            await root.mount(middle, name="middle", inject=(SVC,))
            target = await root.mount(
                downstream,
                name="downstream",
                inject=(SVC_A,),
            )
            call = target.context.fiber.acquire_call(
                target.context.fiber.activation_token
            )
        else:
            async def host(ctx: Context) -> None:
                ctx.require(SVC)

                async def child(_child_ctx: Context) -> None:
                    return None

                handle = await ctx.mount(child, name="owned-child")
                child_handles.append(handle)

            await root.mount(host, name="host", inject=(SVC,))
            target_handle = child_handles[0]
            call = target_handle.acquire_call(target_handle.activation_token)

        effect = registration[0]
        with pytest.raises(CompositionError) as caught:
            await effect.aclose()
        assert caught.value.code == "REENTRANT_CALL_WAIT"
        assert not root._providers[SVC].revoking  # pyright: ignore[reportPrivateUsage]
        call.release()
        released = True
        await effect.aclose()
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
    finally:
        if call is not None and not released:
            call.release()
        await root.dispose()


@pytest.mark.asyncio
async def test_registration_close_joiner_guard_rejects_existing_drain() -> None:
    """已有 registration close 的 joiner 仍须按实际 permit 立即拒绝。"""

    root = CompositionRoot("registration-joiner-guard")
    registration: list[Effect] = []
    consumer_context: list[Context] = []
    close_task: asyncio.Task[None] | None = None

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))

    async def consumer(ctx: Context) -> None:
        consumer_context.append(ctx)
        ctx.require(SVC)

    try:
        provider_fiber = await root.mount(provider, name="provider")
        consumer_fiber = await root.mount(consumer, name="consumer", inject=(SVC,))
        effect = registration[0]
        record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        call = consumer_fiber.context.fiber.acquire_call(
            consumer_fiber.context.fiber.activation_token
        )
        async with RuntimeScope(call) as scope:
            close_task = asyncio.create_task(effect.aclose())
            await scope.wait_admission_closed()
            with pytest.raises(CompositionError) as caught:
                await effect.aclose()
            assert caught.value.code == "REENTRANT_CALL_WAIT"
            assert not close_task.done()
            assert root._providers[SVC] is record  # pyright: ignore[reportPrivateUsage]
            assert record.revoking
            assert effect in provider_fiber.effects
            assert consumer_context[0].require(SVC) == "v"
        await close_task
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
    finally:
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_revoking_provider_raw_lookup_requires_target_access() -> None:
    """revoking 期间只保留 target-own permit，不把 ContextVar 传递当授权。"""

    root = CompositionRoot("registration-read-boundary")
    registration: list[Effect] = []
    consumer_context: list[Context] = []
    consumer_started = asyncio.Event()
    admission_closed = asyncio.Event()
    release_consumer = asyncio.Event()
    frozen_values: list[str] = []
    consumer_task: asyncio.Task[None] | None = None
    close_task: asyncio.Task[None] | None = None

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))

    async def consumer(ctx: Context) -> None:
        consumer_context.append(ctx)
        ctx.require(SVC)

    async def hold_consumer() -> None:
        fiber = consumer_context[0].fiber
        call = fiber.acquire_call(fiber.activation_token)
        async with RuntimeScope(call) as scope:
            consumer_started.set()
            await scope.wait_admission_closed()
            admission_closed.set()
            frozen_values.append(consumer_context[0].require(SVC))
            await release_consumer.wait()

    try:
        provider_fiber = await root.mount(provider, name="provider")
        await root.mount(consumer, name="consumer", inject=(SVC,))
        consumer_task = asyncio.create_task(hold_consumer())
        await consumer_started.wait()
        effect = registration[0]
        close_task = asyncio.create_task(effect.aclose())
        await admission_closed.wait()

        async with provider_fiber.context.runtime_scope():
            owner_context, value = root._service_provider(SVC)  # pyright: ignore[reportPrivateUsage]
            assert owner_context is provider_fiber.context and value == "v"
            async with provider_fiber.context.runtime_scope():
                nested_context, nested_value = root._service_provider(SVC)  # pyright: ignore[reportPrivateUsage]
                assert nested_context is owner_context and nested_value == value

            async def raw_child_lookup() -> None:
                with pytest.raises(RuntimeError, match="当前 runtime scope 不提供服务"):
                    root._service_provider(SVC)  # pyright: ignore[reportPrivateUsage]

            await asyncio.create_task(raw_child_lookup())

        assert frozen_values == ["v"]
        release_consumer.set()
        await asyncio.gather(consumer_task, close_task)
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
    finally:
        release_consumer.set()
        if consumer_task is not None and not consumer_task.done():
            consumer_task.cancel()
        tasks = [task for task in (consumer_task, close_task) if task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_unloading_nonrevoking_provider_defers_admission_to_runtime_scope() -> None:
    """非 revoking 的 UNLOADING raw lookup 返回 Context，实际 admission 再拒绝。"""

    root = CompositionRoot("unloading-provider-read-boundary")
    registration: list[Effect] = []
    dispose_task: asyncio.Task[None] | None = None

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))

    try:
        provider_fiber = await root.mount(provider, name="provider")
        token = provider_fiber.context.fiber.activation_token
        call = provider_fiber.context.fiber.acquire_call(token)
        async with RuntimeScope(call) as scope:
            dispose_task = asyncio.create_task(provider_fiber.dispose())
            await scope.wait_admission_closed()
            record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
            assert provider_fiber.state == FiberState.UNLOADING
            assert not record.revoking
            assert root._providers[SVC] is record  # pyright: ignore[reportPrivateUsage]
            owner_context, value = root._service_provider(SVC)  # pyright: ignore[reportPrivateUsage]
            assert owner_context is provider_fiber.context and value == "v"
            async with owner_context.runtime_scope():
                nested_context, nested_value = root._service_provider(SVC)  # pyright: ignore[reportPrivateUsage]
                assert nested_context is owner_context and nested_value == value

            async def new_task_admission() -> None:
                new_owner_context, new_value = root._service_provider(SVC)  # pyright: ignore[reportPrivateUsage]
                assert new_owner_context is owner_context
                assert new_value == value
                with pytest.raises(CompositionError) as caught:
                    async with new_owner_context.runtime_scope():
                        pass
                assert caught.value.code == "OWNER_UNAVAILABLE"

            await asyncio.create_task(new_task_admission())
        await dispose_task
        assert provider_fiber.state == FiberState.DISPOSED
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
    finally:
        if dispose_task is not None:
            await asyncio.gather(dispose_task, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["dispose", "effect"])
async def test_registration_close_rejects_exact_lifecycle_borrow(entrypoint: str) -> None:
    """consumer cleanup Task 的 exact lifecycle borrow 不能自等 registration。"""

    root = CompositionRoot(f"registration-lifecycle-guard-{entrypoint}")
    registration: list[Effect] = []
    consumer_effects: list[Effect] = []
    cleanup_tasks: list[asyncio.Task[object] | None] = []
    transition_tasks: list[asyncio.Task[object] | None] = []
    trigger_registration_close = True

    async def provider(ctx: Context) -> None:
        registration.append(await ctx.provide(SVC, "v"))
        await ctx.provide(SVC_B, "same-owner")

    async def consumer(ctx: Context) -> None:
        nonlocal trigger_registration_close
        ctx.require(SVC)

        async def cleanup() -> None:
            cleanup_tasks.append(asyncio.current_task())
            transition_tasks.append(ctx._fiber._transition_owner)  # pyright: ignore[reportPrivateUsage]
            if trigger_registration_close:
                await registration[0].aclose()

        consumer_effects.append(await ctx.effect(lambda: cleanup))

    try:
        await root.mount(provider, name="provider")
        consumer_fiber = await root.mount(consumer, name="consumer", inject=(SVC,))
        registration_effect = registration[0]
        record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        frozen_provider = consumer_fiber.dependency_store[SVC]
        before_composition = root._composition_revision  # pyright: ignore[reportPrivateUsage]

        if entrypoint == "dispose":
            with pytest.raises(CompositionError) as caught:
                await consumer_fiber.dispose()
        else:
            with pytest.raises(CompositionError) as caught:
                await consumer_effects[0].aclose()
        assert caught.value.code == "REENTRANT_LIFECYCLE_WAIT"
        assert cleanup_tasks[0] is not None
        assert cleanup_tasks[0] is not transition_tasks[0]
        if entrypoint == "dispose":
            assert transition_tasks[0] is not None
        assert root._composition_revision == before_composition  # pyright: ignore[reportPrivateUsage]
        assert not record.revoking
        assert root._providers[SVC] is record  # pyright: ignore[reportPrivateUsage]
        assert consumer_fiber.dependency_store[SVC] is frozen_provider
        assert registration_effect in record.owner.effects
        assert root.service_value(SVC_B) == "same-owner"

        trigger_registration_close = False
        if entrypoint == "dispose":
            await consumer_fiber.dispose()
        else:
            await consumer_effects[0].aclose()
        await registration_effect.aclose()
        assert SVC not in root._providers  # pyright: ignore[reportPrivateUsage]
        assert root.service_value(SVC_B) == "same-owner"
    finally:
        trigger_registration_close = False
        await root.dispose()


@pytest.mark.asyncio
async def test_cancelled_dispose_waiter_does_not_cancel_cleanup() -> None:
    root = CompositionRoot("cancel-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    consumer_unloading = asyncio.Event()
    call_started = asyncio.Event()
    release_call = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark_unloading() -> None:
            consumer_unloading.set()

        await ctx.effect(lambda: mark_unloading)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))

    async def actual_call() -> None:
        token = provider.context.fiber.activation_token
        async with provider.context.fiber.acquire_call(token):
            call_started.set()
            await release_call.wait()

    call_task = asyncio.create_task(actual_call())
    await call_started.wait()

    waiter = asyncio.create_task(provider.dispose())
    await consumer_unloading.wait()
    waiter.cancel()
    # _await_critical 取消后仍等物理清理结束才向等待者抛 CancelledError；
    # 必须先让在途调用完成，否则 await waiter 死锁。
    release_call.set()
    await call_task
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1, "取消等待者不得丢弃清理所有权"
    await root.dispose()


@pytest.mark.asyncio
async def test_missing_conflict_and_cycle_are_observable() -> None:
    root = CompositionRoot("dep-root")

    missing = await root.mount(
        _consumer_apply({"consumer_apply": 0}, []),
        name="missing",
        inject=(SVC,),
    )
    assert missing.state == FiberState.PENDING
    assert missing.missing_services == (SVC.name,)

    async def first(ctx: Context) -> None:
        await ctx.provide(SVC, "one")

    async def second(ctx: Context) -> None:
        await ctx.provide(SVC, "two")

    one = await root.mount(first, name="one")
    conflict = await root.mount(second, name="two")
    assert conflict.state == FiberState.FAILED
    assert _error_code(conflict.error) == "DUPLICATE_SERVICE"

    await conflict.dispose()
    await missing.dispose()
    await one.dispose()

    # A 需要 B 的服务、B 需要 A 的服务：互等确定停在 PENDING，
    # missing_services 可观察，不死循环也不混过 ACTIVE。
    async def needs_b(ctx: Context) -> None:
        ctx.require(SVC_B)
        await ctx.provide(SVC_A, "a")

    async def needs_a(ctx: Context) -> None:
        ctx.require(SVC_A)
        await ctx.provide(SVC_B, "b")

    a = await root.mount(needs_b, name="a", inject=(SVC_B,))
    b = await root.mount(needs_a, name="b", inject=(SVC_A,))
    assert a.state == FiberState.PENDING and b.state == FiberState.PENDING
    assert a.missing_services == (SVC_B.name,)
    assert b.missing_services == (SVC_A.name,)
    await root.dispose()


@pytest.mark.asyncio
async def test_stale_context_loses_capabilities_after_reload() -> None:
    """同一 Fiber 重载后，旧 Context 的一切能力入口统一拒绝且无副作用。

    bound method 路径与 plugins/react/plugin.py 的
    partial(react, capture_scope=ctx.capture_runtime_scope) 同形：
    捕获于 A activation 的方法在 B 之后调用必须在捕获处拒绝。
    """
    root = CompositionRoot("stale-ctx-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    seen: list[str] = []
    ctxs: list[Context] = []
    bound_captures: list[Callable[[], RuntimeScope]] = []

    async def consumer_apply(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        seen.append(ctx.require(SVC))
        ctxs.append(ctx)
        bound_captures.append(ctx.capture_runtime_scope)

    provider = await root.mount(_provider_apply(stats), name="provider")
    consumer = await root.mount(
        consumer_apply, name="consumer", inject=(SVC,)
    )
    root_ctx = root.context
    old_ctx = ctxs[0]
    assert consumer.context is old_ctx

    await provider.dispose()
    await root.mount(_provider_apply(stats, marker="w"), name="provider")

    assert stats["consumer_apply"] == 2 and seen == ["v1", "w2"]
    new_ctx = ctxs[1]
    assert consumer.context is new_ctx and new_ctx is not old_ctx
    assert root.context is root_ctx, "Root Context 身份不因局部替换改变"

    for entry in (
        lambda: old_ctx.require(SVC),
        lambda: old_ctx.get(SVC),
        lambda: old_ctx.fiber,
        bound_captures[0],
    ):
        with pytest.raises(CompositionError) as excinfo:
            entry()
        assert excinfo.value.code == "STALE_ACTIVATION"

    async def ghost(ctx: Context) -> None:
        pass

    for entry in (
        lambda: old_ctx.provide(SVC, "ghost"),
        lambda: old_ctx.mount(ghost, name="ghost"),
        lambda: old_ctx.runtime_scope().__aenter__(),
    ):
        with pytest.raises(CompositionError) as excinfo:
            await entry()
        assert excinfo.value.code == "STALE_ACTIVATION"

    assert {v.name for v in root.receipt().fibers} >= {"provider", "consumer"}
    assert "ghost" not in {v.name for v in root.receipt().fibers}
    assert root.service_value(SVC) == "w2"
    assert new_ctx.require(SVC) == "w2"
    await root.dispose()


@pytest.mark.asyncio
async def test_inflight_call_reads_own_activation_deps_during_drain() -> None:
    """排空期间已接纳调用仍可读本 activation 的 dependency_store。"""
    root = CompositionRoot("drain-read-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    svc2 = ServiceKey[str]("test.local.drain-svc2")
    dependent_unloading = asyncio.Event()
    call_started = asyncio.Event()
    release_call = asyncio.Event()
    ctxs: list[Context] = []
    read_during_drain: list[str] = []
    cleanup_dependency: list[str] = []

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)
        ctxs.append(ctx)
        await ctx.provide(svc2, "c")
        # Effect 清理在 dependency_store 重置前执行：旧 Context 在排空后
        # 仍是 current，可读本 activation 固定的依赖。
        await ctx.effect(
            lambda: lambda: cleanup_dependency.append(ctx.require(SVC))
        )

    async def dependent(ctx: Context) -> None:
        ctx.require(svc2)

        def mark() -> None:
            dependent_unloading.set()

        await ctx.effect(lambda: mark)

    provider = await root.mount(_provider_apply(stats), name="provider")
    consumer_fiber = await root.mount(consumer, name="consumer", inject=(SVC,))
    await root.mount(dependent, name="dependent", inject=(svc2,))

    async def actual_call() -> None:
        token = consumer_fiber.context.fiber.activation_token
        async with consumer_fiber.context.fiber.acquire_call(token):
            call_started.set()
            await release_call.wait()
            # 本 activation Context 在排空期间仍 current；
            # dependency_store 到 unload 末尾才清空。
            read_during_drain.append(ctxs[0].require(SVC))

    call_task = asyncio.create_task(actual_call())
    await call_started.wait()

    dispose_task = asyncio.create_task(provider.dispose())
    # dependent 的清理在 consumer._unload 的依赖方通知阶段置位，
    # 此后 consumer 确定性进入对本任务 call 的排空等待。
    await dependent_unloading.wait()
    assert consumer_fiber.state == FiberState.UNLOADING

    release_call.set()
    await asyncio.gather(call_task, dispose_task)
    assert read_during_drain == ["v1"], "在途调用须读到本 activation 固定的旧依赖"
    assert cleanup_dependency == ["v1"], "Effect 清理须读到本 activation 固定的旧依赖"
    assert stats["provider_close"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_stale_context_cannot_dispatch_into_new_listeners() -> None:
    """旧 Context 的五个事件分发入口不得触达新 activation 的 listener。

    同一 consumer Fiber 经历 A→B 重载；B 的 apply 经 ctx.on 注册真实
    listener。旧 A Context 的 emit/serial/parallel/transform/observe
    必须在到达 Root 事件注册表前 STALE_ACTIVATION：不调用 B 的
    listener、不创建 parallel listener task、不产生结果。新 B Context
    的五个入口仍按真实签名分发。
    """
    root = CompositionRoot("stale-event-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    ctxs: list[Context] = []
    calls: list[tuple[str, object]] = []

    emit_key = EmitEventKey[str]("test.local.emit")
    serial_key = SerialEventKey[str, str]("test.local.serial")
    parallel_key = ParallelEventKey[str]("test.local.parallel")
    transform_key = TransformEventKey[str](
        "test.local.transform", str, "payload 必须是 str"
    )
    observe_key = ObserveEventKey[str]("test.local.observe")

    def on_emit(payload: str) -> None:
        calls.append(("emit", payload))

    def on_serial(payload: str) -> Bail[str]:
        calls.append(("serial", payload))
        return Bail("serial-ok")

    async def on_parallel(payload: str) -> None:
        calls.append(("parallel", payload))

    def on_transform(payload: str) -> str:
        calls.append(("transform", payload))
        return payload + "!"

    def on_observe(payload: str) -> None:
        calls.append(("observe", payload))

    async def consumer_apply(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        ctx.require(SVC)
        ctxs.append(ctx)
        await ctx.on(emit_key, on_emit)
        await ctx.on(serial_key, on_serial)
        await ctx.on(parallel_key, on_parallel)
        await ctx.on(transform_key, on_transform)
        await ctx.on(observe_key, on_observe)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer_apply, name="consumer", inject=(SVC,))
    old_ctx = ctxs[0]

    await provider.dispose()
    await root.mount(_provider_apply(stats, marker="w"), name="provider")
    assert stats["consumer_apply"] == 2
    new_ctx = ctxs[1]

    with pytest.raises(CompositionError) as excinfo:
        old_ctx.emit(emit_key, "stale")
    assert excinfo.value.code == "STALE_ACTIVATION"

    for entry in (
        lambda: old_ctx.serial(serial_key, "stale"),
        lambda: old_ctx.parallel(parallel_key, "stale"),
        lambda: old_ctx.transform(transform_key, "stale"),
        lambda: old_ctx.observe(observe_key, "stale"),
    ):
        with pytest.raises(CompositionError) as excinfo:
            await entry()
        assert excinfo.value.code == "STALE_ACTIVATION"

    assert calls == [], "旧 Context 分发不得触达新 activation 的 listener 或建 task"

    new_ctx.emit(emit_key, "live")
    assert await new_ctx.serial(serial_key, "live") == Bail("serial-ok")
    await new_ctx.parallel(parallel_key, "live")
    assert await new_ctx.transform(transform_key, "live") == "live!"
    await new_ctx.observe(observe_key, "live")
    assert calls == [
        ("emit", "live"),
        ("serial", "live"),
        ("parallel", "live"),
        ("transform", "live"),
        ("observe", "live"),
    ]
    await root.dispose()


@pytest.mark.asyncio
async def test_stale_context_cannot_report_incident_or_require_owner() -> None:
    """旧 Context 的 report_incident/require_runtime_owner 必须先拒，
    不得产生 Incident 记录或读取 lease/service 状态。"""
    root = CompositionRoot("stale-write-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    ctxs: list[Context] = []

    async def consumer_apply(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        ctx.require(SVC)
        ctxs.append(ctx)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer_apply, name="consumer", inject=(SVC,))
    old_ctx = ctxs[0]

    await provider.dispose()
    await root.mount(_provider_apply(stats, marker="w"), name="provider")
    assert stats["consumer_apply"] == 2

    before = len(root.receipt().incidents)
    with pytest.raises(CompositionError) as excinfo:
        old_ctx.report_incident("stale.kind", "stale message")
    assert excinfo.value.code == "STALE_ACTIVATION"
    assert len(root.receipt().incidents) == before

    # 即使没有任何 runtime scope，拒绝也必须是 STALE_ACTIVATION 而非
    # lease/service 校验错误——证明检查在读取外部状态之前。
    with pytest.raises(CompositionError) as excinfo:
        old_ctx.require_runtime_owner(
            ServiceKey[object]("test.local.owner-key"), object()
        )
    assert excinfo.value.code == "STALE_ACTIVATION"
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_owner_uses_local_owner_call_and_rejects_inherited_or_foreign_context(
    tmp_path,
) -> None:
    """授权只接受本 Context 的 permit 或精确生命周期借用。"""

    owner_key = ServiceKey[object]("test.local.owner-service")
    other_key = ServiceKey[object]("test.local.other-service")
    root = CompositionRoot("runtime-owner-local")
    root_two = CompositionRoot("runtime-owner-foreign")
    owner_value = object()
    foreign_value = object()
    other_value = object()
    contexts: dict[str, Context] = {}

    async def owner_apply(ctx: Context) -> None:
        contexts["owner"] = ctx
        await ctx.provide(owner_key, owner_value)

    async def other_apply(ctx: Context) -> None:
        contexts["other"] = ctx
        await ctx.provide(other_key, other_value)

    async def foreign_apply(ctx: Context) -> None:
        contexts["foreign"] = ctx
        await ctx.provide(owner_key, foreign_value)

    owner = await root.mount(
        owner_apply,
        name="owner",
        runtime=PluginRuntime("owner", "runtime-owner-local", tmp_path, tmp_path, tmp_path, {}),
    )
    other = await root.mount(
        other_apply,
        name="other",
        runtime=PluginRuntime("other", "runtime-owner-local", tmp_path, tmp_path, tmp_path, {}),
    )
    foreign = await root_two.mount(
        foreign_apply,
        name="foreign-owner",
        runtime=PluginRuntime("foreign", "runtime-owner-foreign", tmp_path, tmp_path, tmp_path, {}),
    )
    owner_ctx = contexts["owner"]
    other_ctx = contexts["other"]
    try:
        with pytest.raises(CompositionError) as excinfo:
            owner_ctx.require_runtime_owner(owner_key, owner_value)
        assert excinfo.value.code == "OWNER_CALL_CONTEXT"

        async with other_ctx.runtime_scope():
            with pytest.raises(CompositionError) as excinfo:
                owner_ctx.require_runtime_owner(owner_key, owner_value)
            assert excinfo.value.code == "OWNER_CALL_CONTEXT"

        async with owner_ctx.runtime_scope():
            assert owner_ctx.require_runtime_owner(owner_key, owner_value) == "owner"
            with pytest.raises(CompositionError) as excinfo:
                owner_ctx.require_runtime_owner(owner_key, object())
            assert excinfo.value.code == "SERVICE_SCOPE_MISMATCH"

            child_errors: list[BaseException] = []

            async def raw_child() -> None:
                try:
                    owner_ctx.require_runtime_owner(owner_key, owner_value)
                except BaseException as error:
                    child_errors.append(error)

            await asyncio.create_task(raw_child())
            assert child_errors and _error_code(child_errors[0]) == "OWNER_CALL_CONTEXT"

        with pytest.raises(CompositionError) as excinfo:
            async with owner_ctx.runtime_scope():
                owner_ctx.require_runtime_owner(owner_key, foreign_value)
        assert excinfo.value.code == "SERVICE_SCOPE_MISMATCH"
    finally:
        await foreign.dispose()
        await root_two.dispose()
        await other.dispose()
        await owner.dispose()
        await root.dispose()


@pytest.mark.asyncio
async def test_runtime_owner_lifecycle_borrow_and_admitted_consumer_survive_provider_drain(
    tmp_path,
) -> None:
    """生命周期借用与排空中的已接纳消费者都读取本地 dependency store。"""

    key = ServiceKey[object]("test.local.lifecycle-owner")
    root = CompositionRoot("runtime-owner-lifecycle")
    value = object()
    lifecycle_auth: list[str] = []
    effect_auth: list[str] = []
    consumer_context: list[Context] = []
    work_started = asyncio.Event()
    release_work = asyncio.Event()
    admitted_auth: list[str] = []
    work: list[Callable[[], Coroutine[object, object, None]]] = []

    async def provider_apply(ctx: Context) -> None:
        await ctx.provide(key, value)

        async def cleanup() -> None:
            effect_auth.append(ctx.require_runtime_owner(key, value))

        await ctx.effect(lambda: cleanup, label="owner-guard-cleanup")

        async def on_started(_event: object) -> None:
            lifecycle_auth.append(ctx.require_runtime_owner(key, value))

        async def on_stopping(_event: object) -> None:
            lifecycle_auth.append(ctx.require_runtime_owner(key, value))

        await ctx.on(RUNTIME_STARTED, on_started)
        await ctx.on(RUNTIME_STOPPING, on_stopping)

    async def consumer_apply(ctx: Context) -> None:
        consumer_context.append(ctx)
        service = ctx.require(key)

        async def run() -> None:
            async with ctx.runtime_scope():
                work_started.set()
                await release_work.wait()
                admitted_auth.append(ctx.require_runtime_owner(key, service))

        work.append(run)

    provider = await root.mount(
        provider_apply,
        name="provider",
        runtime=PluginRuntime("provider", "runtime-owner-lifecycle", tmp_path, tmp_path, tmp_path, {}),
    )
    consumer = await root.mount(
        consumer_apply,
        name="consumer",
        inject=(key,),
        runtime=PluginRuntime("consumer", "runtime-owner-lifecycle", tmp_path, tmp_path, tmp_path, {}),
    )
    work_task = asyncio.create_task(work[0]())
    await work_started.wait()
    dispose_task = asyncio.create_task(provider.dispose())
    drain_started = asyncio.Event()
    asyncio.get_running_loop().call_soon(drain_started.set)
    await drain_started.wait()
    try:
        assert provider.state is FiberState.UNLOADING
        assert lifecycle_auth == ["provider"]
        assert not work_task.done()
        release_work.set()
        await work_task
        await dispose_task
        assert admitted_auth == ["consumer"]
        assert lifecycle_auth == ["provider", "provider"]
        assert effect_auth == ["provider"]
        assert consumer_context[0].fiber._fiber._in_flight_calls == {}
    finally:
        release_work.set()
        if not work_task.done():
            await work_task
        if not dispose_task.done():
            await dispose_task
        await root.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("release_source_first", [True, False])
async def test_retain_extends_protection_either_release_order(
    release_source_first: bool,
) -> None:
    """ACTIVE 下 _retain 派生同 activation 的独立许可；任意释放顺序
    下，最后一份 release 前真实 cleanup 不得发生。"""
    root = CompositionRoot("retain-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    dependent_unloading = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark() -> None:
            dependent_unloading.set()

        await ctx.effect(lambda: mark)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))

    token = provider.context.fiber.activation_token
    source = provider.context.fiber.acquire_call(token)
    retained = source._retain()
    assert retained.activation is source.activation
    assert retained is not source

    dispose_task = asyncio.create_task(provider.dispose())
    await dependent_unloading.wait()
    assert provider.state == FiberState.UNLOADING
    assert stats["provider_close"] == 0

    first, second = (
        (source, retained) if release_source_first else (retained, source)
    )
    first.release()
    assert stats["provider_close"] == 0, "仍有一份许可存活，不得提前清理"
    assert not dispose_task.done()

    second.release()
    await dispose_task
    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_retain_during_drain_and_retain_after_release_rejected() -> None:
    """UNLOADING 中源许可仍受保护故仍可 retain；源 release 后再
    retain 拒绝且不增加保护；全部释放后 dispose 正常完成。"""
    root = CompositionRoot("retain-drain-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    dependent_unloading = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark() -> None:
            dependent_unloading.set()

        await ctx.effect(lambda: mark)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))

    token = provider.context.fiber.activation_token
    source = provider.context.fiber.acquire_call(token)

    dispose_task = asyncio.create_task(provider.dispose())
    await dependent_unloading.wait()
    assert provider.state == FiberState.UNLOADING

    # 排空期源许可仍持有：retain 成功且不依赖 ACTIVE/当前 token。
    retained = source._retain()
    assert retained.activation is source.activation

    source.release()
    assert not dispose_task.done(), "派生许可仍保护 owner 资源"
    assert stats["provider_close"] == 0

    with pytest.raises(CompositionError) as excinfo:
        source._retain()
    assert excinfo.value.code == "OWNER_CALL_RELEASED"
    assert stats["provider_close"] == 0, "被拒的 retain 不得增加保护"

    retained.release()
    await dispose_task
    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_retain_rejects_task_not_holding_source() -> None:
    """别的 Task 仅拿到源许可引用而未获转交，retain 明确拒绝且
    不改变源许可；源释放后 dispose 正常完成（无泄漏）。"""
    root = CompositionRoot("retain-task-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    provider = await root.mount(_provider_apply(stats), name="provider")

    token = provider.context.fiber.activation_token
    source = provider.context.fiber.acquire_call(token)
    failure: list[BaseException] = []

    async def foreign_retain() -> None:
        try:
            source._retain()
        except BaseException as error:
            failure.append(error)

    other = asyncio.create_task(foreign_retain())
    await other
    assert len(failure) == 1
    assert _error_code(failure[0]) == "OWNER_CALL_CONTEXT"

    # 源许可不受失败的 retain 影响；释放后正常卸载。
    source.release()
    await provider.dispose()
    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_call_scope_capture_survives_unload_until_child_settles() -> None:
    """capture 返回前即持独立许可：parent close 与 child enter 之间
    发生局部 unload 也不提前清理；child 显式接管后记账归属本 Task。"""
    root = CompositionRoot("call-scope-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    dependent_unloading = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark() -> None:
            dependent_unloading.set()

        await ctx.effect(lambda: mark)

    async def unrelated(ctx: Context) -> None:
        pass

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))
    unrelated_fiber = await root.mount(unrelated, name="unrelated")

    token = provider.context.fiber.activation_token
    call = provider.context.fiber.acquire_call(token)
    scope = RuntimeScope(call)
    async with scope:
        assert _current_runtime_scope() is scope
        captured = scope.capture()
        assert captured._call.activation is call.activation
    # parent scope 已关闭释放其许可；captured 的独立许可仍在途。
    dispose_task = asyncio.create_task(provider.dispose())
    await dependent_unloading.wait()
    assert provider.state == FiberState.UNLOADING
    assert stats["provider_close"] == 0, "captured 许可未结算，不得提前清理"
    # 无关 owner 在同一段排空窗口仍可正常接纳。
    other = unrelated_fiber.context.fiber.acquire_call(
        unrelated_fiber.context.fiber.activation_token
    )
    other.release()

    async def child() -> None:
        async with captured:
            # 记账归属确已移交本 Task：仅持有者可 _retain；
            # 持有在途许可的任务等待 owner 卸载被自等待拒绝。
            extra = captured._call._retain()
            extra.release()
            with pytest.raises(CompositionError) as excinfo:
                await provider.dispose()
            assert excinfo.value.code == "REENTRANT_CALL_WAIT"
            assert _current_runtime_scope() is captured

    await asyncio.create_task(child())
    await dispose_task
    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1
    await root.dispose()


@pytest.mark.asyncio
async def test_admission_closed_wait_is_readonly_during_consumer_drain() -> None:
    """Admission wait is read-only while a hard consumer drains a frozen edge."""

    root = CompositionRoot("admission-closed-root")
    upstream_key = ServiceKey[str]("test.local.upstream")
    stats = {"target_close": 0, "unrelated_apply": 0, "unrelated_close": 0}
    consumer_unloading = asyncio.Event()
    release_consumer = asyncio.Event()
    child_entered = asyncio.Event()
    derived_entered = asyncio.Event()
    release_derived = asyncio.Event()
    target_context: list[Context] = []
    consumer_context: list[Context] = []

    async def upstream(ctx: Context) -> None:
        await ctx.provide(upstream_key, "upstream-frozen")

    async def target(ctx: Context) -> None:
        target_context.append(ctx)
        frozen = ctx.require(upstream_key)
        await ctx.provide(SVC, f"target:{frozen}")

        def close() -> None:
            stats["target_close"] += 1

        await ctx.effect(lambda: close)

    async def consumer(ctx: Context) -> None:
        consumer_context.append(ctx)
        assert ctx.require(SVC) == "target:upstream-frozen"

        async def cleanup() -> None:
            consumer_unloading.set()
            await release_consumer.wait()

        await ctx.effect(lambda: cleanup)

    async def unrelated(ctx: Context) -> None:
        stats["unrelated_apply"] += 1

        def cleanup() -> None:
            stats["unrelated_close"] += 1

        await ctx.effect(lambda: cleanup)

    await root.mount(upstream, name="upstream")
    target_fiber = await root.mount(target, name="target", inject=(upstream_key,))
    consumer_fiber = await root.mount(consumer, name="consumer", inject=(SVC,))
    unrelated_fiber = await root.mount(unrelated, name="unrelated")
    captured: RuntimeScope | None = None
    child_task: asyncio.Task[None] | None = None
    monitor_task: asyncio.Task[None] | None = None
    dispose_task: asyncio.Task[None] | None = None
    new_target_task: asyncio.Task[None] | None = None
    derived_scope: RuntimeScope | None = None
    try:
        async with target_context[0].runtime_scope():
            captured = target_context[0].capture_runtime_scope()

        monitor_started = asyncio.Event()
        monitor_scope: list[object | None] = []
        monitor_errors: list[BaseException] = []

        async def child() -> None:
            nonlocal derived_scope
            assert captured is not None
            async with captured:
                child_entered.set()
                await consumer_unloading.wait()
                derived_scope = captured.capture()
                async with derived_scope:
                    derived_entered.set()
                    assert target_fiber.state is FiberState.UNLOADING
                    assert target_context[0].require(SVC) == "target:upstream-frozen"
                    assert target_context[0].require(upstream_key) == "upstream-frozen"
                    await release_derived.wait()

        async def monitor() -> None:
            assert captured is not None
            monitor_started.set()
            await captured.wait_admission_closed()
            monitor_scope.append(_current_runtime_scope())
            try:
                captured.capture()
            except BaseException as error:
                monitor_errors.append(error)

        child_task = asyncio.create_task(child(), name="target-captured-child")
        await child_entered.wait()
        monitor_task = asyncio.create_task(monitor(), name="raw-admission-monitor")
        await monitor_started.wait()
        dispose_task = asyncio.create_task(target_fiber.dispose(), name="target-dispose")
        await consumer_unloading.wait()
        await monitor_task
        await derived_entered.wait()

        assert target_fiber.state is FiberState.UNLOADING
        assert consumer_fiber.state is FiberState.UNLOADING
        assert stats["target_close"] == 0
        assert monitor_scope == [None]
        assert len(monitor_errors) == 1
        assert _error_code(monitor_errors[0]) == "OWNER_CALL_CONTEXT"

        async def new_target_scope() -> None:
            assert target_context[0] is not None
            async with target_context[0].runtime_scope():
                raise AssertionError("UNLOADING target must not admit a new scope")

        new_target_task = asyncio.create_task(new_target_scope(), name="target-new-scope")
        with pytest.raises(CompositionError) as rejected:
            await new_target_task
        assert rejected.value.code == "OWNER_UNAVAILABLE"

        peer_context = unrelated_fiber.context
        peer_before = (
            peer_context,
            unrelated_fiber._activation_token,  # pyright: ignore[reportPrivateUsage]
            unrelated_fiber.state,
            tuple(unrelated_fiber.effects),
            unrelated_fiber._lifecycle_started,  # pyright: ignore[reportPrivateUsage]
            unrelated_fiber._stopping_completed,  # pyright: ignore[reportPrivateUsage]
        )
        async with unrelated_fiber.context.runtime_scope():
            assert unrelated_fiber.state is FiberState.ACTIVE
        peer_after = (
            unrelated_fiber.context,
            unrelated_fiber._activation_token,  # pyright: ignore[reportPrivateUsage]
            unrelated_fiber.state,
            tuple(unrelated_fiber.effects),
            unrelated_fiber._lifecycle_started,  # pyright: ignore[reportPrivateUsage]
            unrelated_fiber._stopping_completed,  # pyright: ignore[reportPrivateUsage]
        )
        assert peer_after == peer_before
        assert stats["unrelated_apply"] == 1
        assert stats["unrelated_close"] == 0

        release_derived.set()
        await child_task
        assert not dispose_task.done(), "consumer Effect gate must still own the frozen dependency"
        assert stats["target_close"] == 0
        release_consumer.set()
        await dispose_task
        assert target_fiber.state is FiberState.DISPOSED
        assert stats["target_close"] == 1
    finally:
        release_derived.set()
        release_consumer.set()
        if child_task is not None and not child_task.done():
            child_task.cancel()
        if child_task is not None:
            try:
                await child_task
            except BaseException:
                pass
        if derived_scope is not None and not derived_scope._closed:
            await derived_scope.close()
        if captured is not None and not captured._closed:
            await captured.close()
        if new_target_task is not None and not new_target_task.done():
            new_target_task.cancel()
        if new_target_task is not None:
            try:
                await new_target_task
            except BaseException:
                pass
        if monitor_task is not None and not monitor_task.done():
            monitor_task.cancel()
        if monitor_task is not None:
            try:
                await monitor_task
            except BaseException:
                pass
        if dispose_task is not None:
            try:
                await dispose_task
            except BaseException:
                pass
        await root.dispose()


@pytest.mark.asyncio
async def test_admission_waiter_cancel_is_local_and_self_wait_does_not_notify() -> None:
    """取消一个 waiter 不影响同 activation 的 waiter 或 permit。"""

    root = CompositionRoot("admission-waiter-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    consumer_unloading = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark() -> None:
            consumer_unloading.set()

        await ctx.effect(lambda: mark)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))
    token = provider.context.fiber.activation_token
    source = provider.context.fiber.acquire_call(token)
    source_scope = RuntimeScope(source)
    async with source_scope:
        captured_cancelled = source_scope.capture()
        captured_waiting = source_scope.capture()
        with pytest.raises(CompositionError) as excinfo:
            await provider.dispose()
        assert excinfo.value.code == "REENTRANT_CALL_WAIT"
        assert provider.state is FiberState.ACTIVE
        assert not captured_waiting._call._admission_closed.is_set()

    waiter_cancelled_started = asyncio.Event()
    waiter_waiting_started = asyncio.Event()
    waiter_observed_scope: list[object | None] = []

    async def waiter(scope: RuntimeScope, started: asyncio.Event) -> None:
        started.set()
        await scope.wait_admission_closed()
        waiter_observed_scope.append(_current_runtime_scope())

    waiter_cancelled = None
    waiter_waiting = None
    dispose_task = None
    try:
        waiter_cancelled = asyncio.create_task(
            waiter(captured_cancelled, waiter_cancelled_started)
        )
        waiter_waiting = asyncio.create_task(
            waiter(captured_waiting, waiter_waiting_started)
        )
        await waiter_cancelled_started.wait()
        await waiter_waiting_started.wait()
        waiter_cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter_cancelled
        assert not waiter_waiting.done()
        assert len(provider.context.fiber._fiber._in_flight_calls) == 2

        dispose_task = asyncio.create_task(provider.dispose())
        await consumer_unloading.wait()
        await waiter_waiting
        assert waiter_observed_scope == [None]
        assert len(provider.context.fiber._fiber._in_flight_calls) == 2

        await captured_cancelled.close()
        await captured_waiting.close()
        await dispose_task
        assert provider.state is FiberState.DISPOSED
        assert stats["provider_close"] == 1
    finally:
        await captured_cancelled.close()
        await captured_waiting.close()
        for task in (waiter_cancelled, waiter_waiting):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if dispose_task is not None and not dispose_task.done():
            await dispose_task
        await root.dispose()


@pytest.mark.asyncio
async def test_admission_closed_is_bound_to_each_real_activation() -> None:
    """真实 A→PENDING→B 替换不让旧 scope 观察或执行新 activation。"""

    root = CompositionRoot("admission-activation-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    seen: list[str] = []

    async def consumer(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        seen.append(ctx.require(SVC))

    provider = await root.mount(_provider_apply(stats), name="provider")
    consumer_fiber = await root.mount(
        consumer,
        name="consumer",
        inject=(SVC,),
    )
    old_context = consumer_fiber.context
    old_token = old_context.fiber.activation_token
    old_call = old_context.fiber.acquire_call(old_token)
    old_scope = RuntimeScope(old_call)
    async with old_scope:
        old_captured = old_scope.capture()
    old_wait = asyncio.create_task(old_captured.wait_admission_closed())
    provider_dispose = None
    new_captured = None
    new_wait = None
    consumer_dispose = None
    try:
        provider_dispose = asyncio.create_task(provider.dispose())
        await old_wait
        assert consumer_fiber.state is FiberState.UNLOADING
        assert provider.state is FiberState.UNLOADING
        assert not provider_dispose.done()

        old_event = old_captured._call._admission_closed
        await old_captured.close()
        await provider_dispose
        assert consumer_fiber.state is FiberState.PENDING

        replacement = await root.mount(
            _provider_apply(stats, marker="w"),
            name="provider",
        )
        assert replacement.state is FiberState.ACTIVE
        assert consumer_fiber.state is FiberState.ACTIVE
        new_context = consumer_fiber.context
        new_token = new_context.fiber.activation_token
        assert new_context is not old_context
        assert new_token is not old_token
        assert seen == ["v1", "w2"]
        assert old_captured._call._admission_closed is old_event
        assert old_event.is_set()
        await old_captured.wait_admission_closed()
        with pytest.raises(CompositionError) as excinfo:
            old_captured.capture()
        assert excinfo.value.code == "OWNER_CALL_CONTEXT"

        new_call = new_context.fiber.acquire_call(new_token)
        new_scope = RuntimeScope(new_call)
        async with new_scope:
            new_captured = new_scope.capture()
        new_wait = asyncio.create_task(new_captured.wait_admission_closed())
        assert not new_wait.done()
        consumer_dispose = asyncio.create_task(consumer_fiber.dispose())
        await new_wait
        assert new_captured._call._admission_closed is not old_event
        await new_captured.close()
        await consumer_dispose
    finally:
        await old_captured.close()
        if new_captured is not None:
            await new_captured.close()
        for task in (old_wait, new_wait):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if consumer_dispose is not None and not consumer_dispose.done():
            await consumer_dispose
        if provider_dispose is not None and not provider_dispose.done():
            await provider_dispose
        await root.dispose()


@pytest.mark.asyncio
async def test_call_scope_enter_and_close_ownership() -> None:
    """嵌套恢复、错误 Task close、重复 enter 与已释放许可的拒绝。"""
    root = CompositionRoot("scope-rules-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    provider = await root.mount(_provider_apply(stats), name="provider")
    token = provider.context.fiber.activation_token

    call = provider.context.fiber.acquire_call(token)
    scope = RuntimeScope(call)

    async with scope:
        with pytest.raises(RuntimeError):
            await scope.__aenter__()

        foreign: list[BaseException] = []

        async def foreign_close() -> None:
            try:
                await scope.close()
            except BaseException as error:
                foreign.append(error)

        await asyncio.create_task(foreign_close())
        assert len(foreign) == 1
        assert _error_code(foreign[0]) == "OWNER_CALL_CONTEXT"
        assert _current_runtime_scope() is scope, "被拒的 close 不得动绑定/释放"

        nested = scope.capture()
        async with nested:
            assert _current_runtime_scope() is nested
        assert _current_runtime_scope() is scope, "内层 close 须恢复外层绑定"

    assert _current_runtime_scope() is None
    with pytest.raises(RuntimeError):
        await scope.__aenter__()

    # 已释放许可不得绑 scope（不增加保护）。
    released = provider.context.fiber.acquire_call(token)
    released.release()
    with pytest.raises(CompositionError) as excinfo:
        RuntimeScope(released)
    assert excinfo.value.code == "OWNER_CALL_RELEASED"
    await root.dispose()


@pytest.mark.asyncio
async def test_call_scope_unentered_close_and_cancel_paths_do_not_leak() -> None:
    """未 enter close、重复 close、create_task 失败显式 close、
    运行前取消的 finally close 均释放各自独立许可；ContextVar
    隐式继承不给子 Task 授权。dispose 完成证明无泄漏。"""
    root = CompositionRoot("scope-leak-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    dependent_unloading = asyncio.Event()

    async def consumer(ctx: Context) -> None:
        ctx.require(SVC)

        def mark() -> None:
            dependent_unloading.set()

        await ctx.effect(lambda: mark)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer, name="consumer", inject=(SVC,))
    token = provider.context.fiber.activation_token

    scope = RuntimeScope(provider.context.fiber.acquire_call(token))
    async with scope:
        # 子 Task 隐式继承 ContextVar 不获得 scope 查询与 capture 权限。
        inherited: list[object] = []
        foreign_capture: list[BaseException] = []

        async def inheriting_child() -> None:
            inherited.append(_current_runtime_scope())
            try:
                scope.capture()
            except BaseException as error:
                foreign_capture.append(error)

        await asyncio.create_task(inheriting_child())
        assert inherited == [None]
        assert len(foreign_capture) == 1
        assert _error_code(foreign_capture[0]) == "OWNER_CALL_CONTEXT"

        # 未 enter close（重复调用幂等）。
        unentered = scope.capture()
        await unentered.close()
        await unentered.close()

        # create_task 真实失败路径（shell.py:359-365 结构）：先建
        # coroutine，创建失败后 coroutine.close + scope.close 再
        # re-raise；body 未执行，释放的只是捕获的独立许可。
        spawn_scope = scope.capture()
        spawn_ran: list[bool] = []

        async def spawn_body() -> None:
            async with spawn_scope:
                spawn_ran.append(True)

        def fail_create_task(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("task factory failed")

        operation = spawn_body()
        with pytest.raises(RuntimeError, match="task factory failed"):
            try:
                fail_create_task(operation)
            except BaseException:
                operation.close()
                await spawn_scope.close()
                raise
        assert spawn_ran == []

        # tasks.py _cancel_requested 分支形态：Task 进入 _run 后到达
        # finally 关闭 scope（与下面的 raw task-before-start 是两个事实）。
        captured = scope.capture()
        cancel_cleanup_done = asyncio.Event()

        async def cancelled_before_enter() -> None:
            try:
                raise asyncio.CancelledError
            finally:
                await captured.close()
                cancel_cleanup_done.set()

        task = asyncio.create_task(cancelled_before_enter())
        with pytest.raises(asyncio.CancelledError):
            await task
        await cancel_cleanup_done.wait()

        # 原生 Task 在首指令前被取消：body/finally 均不执行，由创建方
        # 显式 close 已捕获但未 enter 的 scope（默认非 eager task
        # factory 下 create 到 await 之间无调度点）。
        raw_scope = scope.capture()
        raw_ran: list[bool] = []

        async def raw_child() -> None:
            try:
                async with raw_scope:
                    raw_ran.append(True)
            finally:
                raw_ran.append(False)

        raw_task = asyncio.create_task(raw_child())
        raw_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await raw_task
        assert raw_ran == [], "首指令前取消：body 与 finally 都不得执行"
        await raw_scope.close()

    dispose_task = asyncio.create_task(provider.dispose())
    await dependent_unloading.wait()
    await dispose_task
    assert provider.state == FiberState.DISPOSED
    assert stats["provider_close"] == 1, "所有 scope 许可结算后清理才发生"
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_scope_admission_retain_and_rejection() -> None:
    """runtime_scope facade：ACTIVE 新接纳、嵌套 retain、stale/非 ACTIVE 拒绝。"""

    root = CompositionRoot("scope-facade-root")
    stats = {"provider_apply": 0, "provider_close": 0}
    provider = await root.mount(_provider_apply(stats), name="provider")
    ctx = provider.context

    # ACTIVE 首次进入：本 Task 无既有许可，走正常新接纳。
    async with ctx.runtime_scope():
        owned = ctx.fiber._fiber._call_owned_by_current_task()
        assert owned is not None, "ACTIVE 进入应新接纳一份许可"
        # 嵌套进入：经 retain 派生，不再要求 ACTIVE 重新接纳。
        async with ctx.runtime_scope():
            assert len(ctx.fiber._fiber._in_flight_calls) == 2
        assert len(ctx.fiber._fiber._in_flight_calls) == 1
    assert not ctx.fiber._fiber._in_flight_calls

    # capture 只能从已受保护工作派生；无许可 Task 拒绝。
    with pytest.raises(CompositionError) as excinfo:
        ctx.capture_runtime_scope()
    assert excinfo.value.code == "OWNER_CALL_CONTEXT"
    async with ctx.runtime_scope():
        captured = ctx.capture_runtime_scope()
    entered: list[object] = []

    async def child() -> None:
        async with captured:
            entered.append(_current_runtime_scope())

    await asyncio.create_task(child())
    assert entered == [captured]
    assert captured._closed, "child 退出时 __aexit__ 已在 entered Task 内结算"

    # 旧 Context 换代后拒绝，不隐式改绑新 activation。
    await provider.dispose()
    provider2 = await root.mount(_provider_apply(stats, marker="w"), name="provider")
    assert provider2.state == FiberState.ACTIVE
    with pytest.raises(CompositionError) as excinfo:
        async with ctx.runtime_scope():
            pass
    assert excinfo.value.code == "OWNER_UNAVAILABLE"
    await root.dispose()


@pytest.mark.asyncio
async def test_stale_context_spawn_closes_rejected_coroutine() -> None:
    """同一 Fiber 换代后的旧 Context 拒绝 spawn 并关闭用户 coroutine。"""

    root = CompositionRoot("stale-spawn-root")
    stats = {"provider_apply": 0, "provider_close": 0, "consumer_apply": 0}
    contexts: list[Context] = []

    async def consumer_apply(ctx: Context) -> None:
        stats["consumer_apply"] += 1
        ctx.require(SVC)
        contexts.append(ctx)

    provider = await root.mount(_provider_apply(stats), name="provider")
    await root.mount(consumer_apply, name="consumer", inject=(SVC,))
    old_context = contexts[0]

    await provider.dispose()
    await root.mount(_provider_apply(stats, marker="w"), name="provider")
    assert contexts[1] is not old_context

    ran: list[str] = []

    async def user_work() -> None:
        ran.append("ran")

    coroutine = user_work()
    with pytest.raises(CompositionError) as excinfo:
        await old_context.spawn(coroutine, name="stale-spawn")
    assert excinfo.value.code == "STALE_ACTIVATION"
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED
    assert ran == []
    await root.dispose()


@pytest.mark.asyncio
async def test_starting_failure_stops_resources_opened_by_prior_listener() -> None:
    """STARTING 中途失败仍为先前 listener 打开的资源派发 STOPPING。"""

    root = CompositionRoot("starting-failure-root")
    opened: list[str] = []
    calls = {"stop": 0}

    async def apply(ctx: Context) -> None:
        async def close_opened(_event: object) -> None:
            calls["stop"] += 1
            opened.clear()

        async def open_resource(_event: object) -> None:
            opened.append("resource")

        async def fail_after_open(_event: object) -> None:
            raise RuntimeError("STARTING listener failed")

        await ctx.on(RUNTIME_STOPPING, close_opened)
        await ctx.on(RUNTIME_STARTING, open_resource)
        await ctx.on(RUNTIME_STARTING, fail_after_open)

    fiber = await root.mount(apply, name="starting-failure")
    assert fiber.state == FiberState.FAILED
    assert isinstance(fiber.error, RuntimeError)
    assert calls["stop"] == 1
    assert opened == []
    await root.dispose()


@pytest.mark.asyncio
async def test_lifecycle_dispatch_health_and_spawn_gate() -> None:
    """STARTING→STARTED→required health→ACTIVE；生命周期借用不占许可；

    spawn 用户 coroutine 在 ready 前不运行；原生子 Task 不继承借用。
    """

    root = CompositionRoot("lifecycle-root")
    order: list[str] = []
    observations: list[str] = []
    started_entered = asyncio.Event()
    release_started = asyncio.Event()
    body_started = asyncio.Event()
    spawned_tasks: list[asyncio.Task[None]] = []

    async def apply(ctx: Context) -> None:
        order.append("apply")

        async def started(_event: object) -> None:
            order.append("started")
            # 生命周期借用：回调内 scope 不产生在途许可。
            async with ctx.runtime_scope():
                assert not ctx.fiber._fiber._in_flight_calls
            # 原生子 Task 继承 ContextVar 值但 Task 不匹配，不获权。
            errors: list[BaseException] = []

            async def native_child() -> None:
                try:
                    async with ctx.runtime_scope():
                        pass
                except BaseException as error:
                    errors.append(error)

            await asyncio.create_task(native_child())
            assert errors and _error_code(errors[0]) == "OWNER_UNAVAILABLE"

            async def user_work() -> None:
                order.append("spawn-ran")
                body_started.set()

            spawned = await ctx.spawn(user_work(), name="gated-work")
            spawned_tasks.append(spawned)
            observations.append("spawned")
            started_entered.set()
            await release_started.wait()

        await ctx.on(RUNTIME_STARTING, lambda _e: order.append("starting"))
        await ctx.on(RUNTIME_STARTED, started)
        await ctx.health("backend")
        await ctx.provide(SVC, "v1")

    mount_task = asyncio.create_task(root.mount(apply, name="lifecycle-plugin"))
    await started_entered.wait()
    views = {view.name: view for view in root.receipt().fibers}
    assert views["lifecycle-plugin"].state == FiberState.LOADING
    assert root.service_value(SVC) is None
    assert observations == ["spawned"]
    assert not body_started.is_set()
    assert not spawned_tasks[0].done()

    release_started.set()
    fiber = await mount_task
    assert order[:3] == ["apply", "starting", "started"]
    assert fiber.state == FiberState.ACTIVE
    # ready 闸放行后用户工作才运行；等 spawned Task 落定。
    assert observations == ["spawned"]
    await spawned_tasks[0]
    assert body_started.is_set()
    assert order[-1] == "spawn-ran"
    await root.dispose()


@pytest.mark.asyncio
async def test_spawn_cancel_during_start_closes_unstarted_user_coroutine() -> None:
    """启动取消在 ready 前清理 wrapper 与尚未启动的用户 coroutine。"""

    root = CompositionRoot("spawn-cancel-root")
    starting = asyncio.Event()
    ran: list[str] = []
    user_coroutines: list[Coroutine[object, object, None]] = []

    async def apply(ctx: Context) -> None:
        async def hold_start(_event: object) -> None:
            starting.set()
            await asyncio.Event().wait()

        await ctx.on(RUNTIME_STARTING, hold_start)

        async def user_work() -> None:
            ran.append("started")

        user_coroutine = user_work()
        user_coroutines.append(user_coroutine)
        _ = await ctx.spawn(user_coroutine, name="cancel-before-ready")

    mount_task = asyncio.create_task(root.mount(apply, name="spawn-cancel"))
    await starting.wait()
    mount_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await mount_task

    assert ran == []
    assert inspect.getcoroutinestate(user_coroutines[0]) == inspect.CORO_CLOSED
    await root.dispose()


@pytest.mark.asyncio
async def test_spawn_task_cancel_before_first_instruction_closes_user_coroutine() -> None:
    """原生 Task 首指令前取消时，Root 清理仍关闭用户 coroutine。"""

    root = CompositionRoot("spawn-raw-cancel-root")
    tasks: list[asyncio.Task[None]] = []
    coroutines: list[Coroutine[object, object, None]] = []
    ran: list[str] = []

    async def apply(ctx: Context) -> None:
        async def started(_event: object) -> None:
            async def user_work() -> None:
                ran.append("ran")

            coroutine = user_work()
            coroutines.append(coroutine)
            tasks.append(await ctx.spawn(coroutine, name="raw-cancel"))

        await ctx.on(RUNTIME_STARTED, started)

    await root.mount(apply, name="raw-cancel")
    task = tasks[0]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ran == []
    await root.dispose()
    assert inspect.getcoroutinestate(coroutines[0]) == inspect.CORO_CLOSED


@pytest.mark.asyncio
async def test_required_health_gates_activation() -> None:
    """required health 未就绪的 owner 不进入 ACTIVE；失败后清理→FAILED。"""

    root = CompositionRoot("health-root")
    stats = {"apply": 0, "close": 0}

    async def degraded(ctx: Context) -> None:
        stats["apply"] += 1
        await ctx.provide(SVC, "unhealthy")
        health = await ctx.health("backend")
        health.degrade("backend 未就绪")

        def close() -> None:
            stats["close"] += 1

        await ctx.effect(lambda: close)

    fiber = await root.mount(degraded, name="unhealthy")
    assert fiber.state == FiberState.FAILED
    assert isinstance(fiber.error, CompositionError)
    assert fiber.error.code == "UNHEALTHY_OWNER"
    assert stats["close"] == 1, "启动失败须先清理已获资源"
    assert root.service_value(SVC) is None, "FAILED owner 的服务不得对外可见"
    await root.dispose()


@pytest.mark.asyncio
async def test_stopping_failure_retains_and_retry_does_not_redispatch() -> None:
    """STOPPING 失败保留资源；retry 只重试失败阶段；已完成的 STOPPING 不重发。"""

    root = CompositionRoot("stopping-root")
    calls = {"stop": 0, "close": 0}
    fail_stop = {"stop": True, "close": True}
    stopped = asyncio.Event()

    async def apply(ctx: Context) -> None:
        async def stop(_event: object) -> None:
            calls["stop"] += 1
            if fail_stop["stop"]:
                raise RuntimeError("stop 失败")

        async def close() -> None:
            calls["close"] += 1
            if fail_stop["close"]:
                raise RuntimeError("effect 清理失败")

        await ctx.on(RUNTIME_STOPPING, stop)
        await ctx.effect(lambda: close)

    fiber = await root.mount(apply, name="stopping-plugin")
    assert fiber.state == FiberState.ACTIVE

    # 第一次 unload：STOPPING 失败 → 在 Effect 释放前传播，资源保留。
    with pytest.raises(RuntimeError):
        await fiber.dispose()
    assert calls == {"stop": 1, "close": 0}
    assert fiber.state != FiberState.DISPOSED

    # retry：STOPPING 成功置位，但 Effect 清理失败 → 仍保留。
    fail_stop["stop"] = False
    with pytest.raises(RuntimeError):
        await fiber.dispose()
    assert calls == {"stop": 2, "close": 1}

    # 再 retry：STOPPING 已完成不重发，仅重试失败的清理阶段。
    fail_stop["close"] = False
    await fiber.dispose()
    assert calls == {"stop": 2, "close": 2}
    assert fiber.state == FiberState.DISPOSED
    await root.dispose()


@pytest.mark.parametrize("close_entry", ("active", "fiber", "root"))
@pytest.mark.asyncio
async def test_effect_cleanup_borrow_applies_to_all_close_entries(
    close_entry: str,
) -> None:
    """Effect 清理在任何关闭入口（Fiber 卸载/业务主动 aclose）都经窄

    binder 获得生命周期借用；借用不占在途许可。"""

    root = CompositionRoot("effect-binder-root")
    seen: list[str] = []
    stored: dict[str, object] = {}

    async def apply(ctx: Context) -> None:
        async def cleanup() -> None:
            # _close_task 内经 binder 建立借用，不登记在途 call。
            async with ctx.runtime_scope():
                seen.append("cleanup-scope")
                assert not ctx.fiber._fiber._in_flight_calls

        stored["effect"] = await ctx.effect(lambda: cleanup)

    fiber = await root.mount(apply, name="binder-plugin")
    from agent.plugin_composition.effect import Effect

    effect = stored["effect"]
    assert isinstance(effect, Effect)
    root_cleanup: list[str] = []
    if close_entry == "active":
        # 业务主动 aclose：同一份借用语义，与发起 Task 无关。
        await effect.aclose()
    elif close_entry == "fiber":
        await fiber.dispose()
    else:
        async def close_root_effect() -> None:
            async with root.context.runtime_scope():
                assert not root.root_fiber._in_flight_calls
                root_cleanup.append("root")

        await root.context.effect(lambda: close_root_effect)
        await root.dispose()

    assert seen == ["cleanup-scope"]
    assert root_cleanup == (["root"] if close_entry == "root" else [])
    if close_entry != "root":
        await root.dispose()
    assert fiber.state == FiberState.DISPOSED


@pytest.mark.asyncio
async def test_owner_scoped_dispatch_isolation() -> None:
    """生命周期 dispatch 按准确 owner Fiber，不按 plugin 名广播。"""

    root = CompositionRoot("dispatch-root")
    hits: list[str] = []

    def make(
        callback_label: str, service_name: str,
    ) -> Callable[[Context], Coroutine[object, object, None]]:
        async def apply(ctx: Context) -> None:
            await ctx.on(RUNTIME_STARTED, lambda _e: hits.append(callback_label))
            await ctx.provide(ServiceKey[str](f"test.local.{service_name}"), service_name)

        return apply

    first = await root.mount(make("first", "first"), name="first")
    second = await root.mount(make("second", "second"), name="second")
    assert hits == ["first", "second"]
    # 单个 owner 重载只重发自己的生命周期事件。
    await first.dispose()
    hits.clear()
    await root.mount(make("first-v2", "first-v2"), name="first-v2")
    assert hits == ["first-v2"]
    assert second.state == FiberState.ACTIVE
    await root.dispose()


@pytest.mark.asyncio
async def test_active_late_provide_notifies_exact_key_and_reprovide() -> None:
    """ACTIVE provide wakes only its exact pending consumer and can reprovide."""

    root = CompositionRoot("active-provide-root")
    stats = {
        "late_apply": 0,
        "late_close": 0,
        "peer_apply": 0,
        "peer_close": 0,
    }
    seen: list[str] = []

    async def provider(_ctx: Context) -> None:
        pass

    async def late(ctx: Context) -> None:
        stats["late_apply"] += 1
        seen.append(ctx.require(SVC))

        async def close() -> None:
            stats["late_close"] += 1

        await ctx.effect(lambda: close)

    async def peer_provider(ctx: Context) -> None:
        await ctx.provide(SVC_A, "peer")

    async def peer(ctx: Context) -> None:
        stats["peer_apply"] += 1
        assert ctx.require(SVC_A) == "peer"

        async def close() -> None:
            stats["peer_close"] += 1

        await ctx.effect(lambda: close)

    try:
        provider_fiber = await root.mount(provider, name="provider")
        late_fiber = await root.mount(late, name="late", inject=(SVC,))
        await root.mount(peer_provider, name="peer-provider")
        peer_fiber = await root.mount(peer, name="peer", inject=(SVC_A,))
        provider_context = provider_fiber.context
        provider_activation = provider_context.fiber.activation_token
        assert late_fiber.state == FiberState.PENDING
        assert peer_fiber.state == FiberState.ACTIVE

        first_effect = await provider_context.provide(SVC, "v1")
        first_record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert late_fiber.state == FiberState.ACTIVE
        assert seen == ["v1"]
        assert stats["late_apply"] == 1
        assert stats["peer_apply"] == 1

        async with late_fiber.context.runtime_scope():
            assert late_fiber.context.require(SVC) == "v1"

        await first_effect.aclose()
        assert late_fiber.state == FiberState.PENDING
        assert stats["late_close"] == 1
        assert peer_fiber.state == FiberState.ACTIVE
        assert stats["peer_apply"] == 1 and stats["peer_close"] == 0

        second_effect = await provider_context.provide(SVC, "v2")
        second_record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert provider_fiber.context is provider_context
        assert provider_context.fiber.activation_token is provider_activation
        assert second_record is not first_record
        assert second_record.revision > first_record.revision
        assert late_fiber.state == FiberState.ACTIVE
        assert seen == ["v1", "v2"]
        assert stats["late_apply"] == 2
        assert stats["peer_apply"] == 1
        async with late_fiber.context.runtime_scope():
            assert late_fiber.context.require(SVC) == "v2"
        await second_effect.aclose()
    finally:
        await root.dispose()

    assert stats["late_close"] == 2
    assert stats["peer_close"] == 1


@pytest.mark.asyncio
async def test_loading_provide_waits_for_active_stage_and_preserves_priority() -> None:
    """LOADING provides register only; ACTIVE owns the single notification stage."""

    root = CompositionRoot("loading-provide-root")
    started = asyncio.Event()
    release = asyncio.Event()
    counts = {"svc": 0, "svc_a": 0}

    async def consumer(ctx: Context) -> None:
        counts["svc"] += 1
        ctx.require(SVC)

    async def consumer_a(ctx: Context) -> None:
        counts["svc_a"] += 1
        ctx.require(SVC_A)

    async def provider(ctx: Context) -> None:
        await ctx.provide(SVC, "v")
        await ctx.provide(SVC_A, "a")

        async def on_started(_event: object) -> None:
            started.set()
            await release.wait()

        await ctx.on(RUNTIME_STARTED, on_started)

    mount_task: asyncio.Task[object] | None = None
    try:
        svc_consumer = await root.mount(consumer, name="svc-consumer", inject=(SVC,))
        a_consumer = await root.mount(consumer_a, name="a-consumer", inject=(SVC_A,))
        mount_task = asyncio.create_task(root.mount(provider, name="provider"))
        await started.wait()
        assert not mount_task.done()
        assert root.service_value(SVC) is None
        assert root.service_value(SVC_A) is None
        assert svc_consumer.state == FiberState.PENDING
        assert a_consumer.state == FiberState.PENDING
        assert counts == {"svc": 0, "svc_a": 0}

        release.set()
        provider_fiber = await mount_task
        assert provider_fiber.state == FiberState.ACTIVE
        assert root.service_value(SVC) == "v"
        assert root.service_value(SVC_A) == "a"
        assert svc_consumer.state == FiberState.ACTIVE
        assert a_consumer.state == FiberState.ACTIVE
        assert counts == {"svc": 1, "svc_a": 1}
        root.freeze()
    finally:
        release.set()
        if mount_task is not None:
            await asyncio.gather(mount_task, return_exceptions=True)
        await root.dispose()


@pytest.mark.asyncio
async def test_active_provide_keeps_duplicate_and_frozen_errors_before_effect() -> None:
    """ACTIVE preflight preserves duplicate/frozen priority and leaves no Effect."""

    root = CompositionRoot("active-provide-priority-root")

    async def provider(ctx: Context) -> None:
        await ctx.provide(SVC, "v")

    try:
        provider_fiber = await root.mount(provider, name="provider")
        before_effects = tuple(provider_fiber.effects)
        with pytest.raises(CompositionError) as duplicate:
            await provider_fiber.context.provide(SVC, "duplicate")
        assert duplicate.value.code == "DUPLICATE_SERVICE"
        assert tuple(provider_fiber.effects) == before_effects

        root.freeze()
        with pytest.raises(CompositionError) as frozen:
            await provider_fiber.context.provide(SVC_A, "frozen")
        assert frozen.value.code == "COMPOSITION_FROZEN"
        assert tuple(provider_fiber.effects) == before_effects
        assert root.service_value(SVC) == "v"
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_late_provide_failure_is_local_to_required_downstream() -> None:
    """A consumer failure does not fail its provider, sibling, or unrelated branch."""

    root = CompositionRoot("late-provide-local-failure-root")
    svc_c = ServiceKey[str]("test.local.svc-c")
    stats = {
        "provider_apply": 0,
        "provider_close": 0,
        "sibling": 0,
        "unrelated": 0,
        "unrelated_close": 0,
    }
    contexts: dict[str, Context] = {}

    async def provider(ctx: Context) -> None:
        stats["provider_apply"] += 1
        contexts["provider"] = ctx

        def close() -> None:
            stats["provider_close"] += 1

        await ctx.effect(lambda: close)

    async def bad_consumer(ctx: Context) -> None:
        ctx.require(SVC)
        await ctx.provide(svc_c, "bad")
        raise RuntimeError("consumer startup failed")

    async def downstream(ctx: Context) -> None:
        ctx.require(svc_c)

    async def sibling(ctx: Context) -> None:
        stats["sibling"] += 1
        ctx.require(SVC)

    async def unrelated(ctx: Context) -> None:
        stats["unrelated"] += 1
        contexts["unrelated"] = ctx

        def close() -> None:
            stats["unrelated_close"] += 1

        await ctx.effect(lambda: close)
        await ctx.provide(SVC_A, "unrelated")

    try:
        provider_fiber = await root.mount(provider, name="provider")
        bad_fiber = await root.mount(bad_consumer, name="bad", inject=(SVC,))
        downstream_fiber = await root.mount(
            downstream,
            name="downstream",
            inject=(svc_c,),
        )
        sibling_fiber = await root.mount(sibling, name="sibling", inject=(SVC,))
        unrelated_fiber = await root.mount(unrelated, name="unrelated")

        root_token = root.instance_token
        provider_context = contexts["provider"]
        unrelated_context = contexts["unrelated"]
        provider_activation = provider_context.fiber.activation_token
        unrelated_activation = unrelated_context.fiber.activation_token
        provider_effects = tuple(provider_fiber.effects)
        unrelated_effects = tuple(unrelated_fiber.effects)
        late_effect = await provider_context.provide(SVC, "provider")

        assert provider_fiber.state == FiberState.ACTIVE
        assert bad_fiber.state == FiberState.FAILED
        assert isinstance(bad_fiber.error, RuntimeError)
        assert downstream_fiber.state == FiberState.PENDING
        assert downstream_fiber.error is None
        assert downstream_fiber.missing_services == (svc_c.name,)
        assert sibling_fiber.state == FiberState.ACTIVE
        assert unrelated_fiber.state == FiberState.ACTIVE
        assert stats == {
            "provider_apply": 1,
            "provider_close": 0,
            "sibling": 1,
            "unrelated": 1,
            "unrelated_close": 0,
        }
        assert root.instance_token is root_token
        assert provider_fiber.context is provider_context
        assert provider_context.fiber.activation_token is provider_activation
        assert tuple(provider_fiber.effects[: len(provider_effects)]) == provider_effects
        assert late_effect in provider_fiber.effects
        assert unrelated_fiber.context is unrelated_context
        assert unrelated_context.fiber.activation_token is unrelated_activation
        assert tuple(unrelated_fiber.effects) == unrelated_effects
        assert root.service_value(SVC) == "provider"
        assert root.service_value(SVC_A) == "unrelated"

        async with provider_context.runtime_scope():
            assert provider_context.require(SVC) == "provider"
        async with sibling_fiber.context.runtime_scope():
            assert sibling_fiber.context.require(SVC) == "provider"
        async with unrelated_context.runtime_scope():
            assert unrelated_context.require(SVC_A) == "unrelated"

        for unavailable_fiber in (bad_fiber, downstream_fiber):
            with pytest.raises(CompositionError) as unavailable:
                async with unavailable_fiber.context.runtime_scope():
                    pass
            assert unavailable.value.code == "OWNER_UNAVAILABLE"

        assert bad_fiber.error is not downstream_fiber.error
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_late_provide_cleanup_failure_retains_registration_until_retry() -> None:
    """Notification cleanup failure retains both the consumer and provider effect."""

    root = CompositionRoot("late-provide-cleanup-failure-root")
    cleanup_calls = 0

    async def provider(_ctx: Context) -> None:
        pass

    async def bad_consumer(ctx: Context) -> None:
        nonlocal cleanup_calls
        ctx.require(SVC)

        def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            if cleanup_calls == 1:
                raise OSError("first cleanup failed")

        await ctx.effect(lambda: cleanup)
        raise RuntimeError("consumer apply failed")

    async def peer_provider(ctx: Context) -> None:
        await ctx.provide(SVC_A, "peer")

    async def peer_consumer(ctx: Context) -> None:
        assert ctx.require(SVC_A) == "peer"

    try:
        provider_fiber = await root.mount(provider, name="provider")
        bad_fiber = await root.mount(bad_consumer, name="bad", inject=(SVC,))
        peer_provider_fiber = await root.mount(peer_provider, name="peer-provider")
        peer_consumer_fiber = await root.mount(
            peer_consumer,
            name="peer-consumer",
            inject=(SVC_A,),
        )
        provider_context = provider_fiber.context

        with pytest.raises(BaseExceptionGroup) as caught:
            await provider_context.provide(SVC, "provider")

        assert len(caught.value.exceptions) == 1
        fiber_failure = caught.value.exceptions[0]
        assert isinstance(fiber_failure, BaseExceptionGroup)
        assert len(fiber_failure.exceptions) == 2
        apply_error, cleanup_error = fiber_failure.exceptions
        assert isinstance(apply_error, RuntimeError)
        assert str(apply_error) == "consumer apply failed"
        assert isinstance(cleanup_error, OSError)
        assert str(cleanup_error) == "first cleanup failed"

        record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert provider_fiber.state == FiberState.ACTIVE
        assert record.owner is provider_fiber
        assert bad_fiber.dependency_store[SVC] is record
        assert not record.revoking
        assert len(provider_fiber.effects) == 1
        assert bad_fiber.state == FiberState.UNLOADING
        assert len(bad_fiber.effects) == 1
        assert cleanup_calls == 1
        assert peer_provider_fiber.state == FiberState.ACTIVE
        assert peer_consumer_fiber.state == FiberState.ACTIVE
        assert root.service_value(SVC) == "provider"
        assert root.service_value(SVC_A) == "peer"
        async with peer_consumer_fiber.context.runtime_scope():
            assert peer_consumer_fiber.context.require(SVC_A) == "peer"

        with pytest.raises(CompositionError) as duplicate:
            await provider_context.provide(SVC, "duplicate")
        assert duplicate.value.code == "DUPLICATE_SERVICE"
        assert cleanup_calls == 1
        assert len(provider_fiber.effects) == 1
        assert len(bad_fiber.effects) == 1

        extra_effect = await provider_context.provide(SVC_B, "extra")
        extra_record = root._providers[SVC_B]  # pyright: ignore[reportPrivateUsage]
        assert extra_effect in provider_fiber.effects
        assert extra_record.owner is provider_fiber
        assert bad_fiber.state == FiberState.UNLOADING
        assert bad_fiber.dependency_store[SVC] is record
        assert cleanup_calls == 1
        async with provider_context.runtime_scope():
            assert provider_context.require(SVC) == "provider"
            assert provider_context.require(SVC_B) == "extra"

        await bad_fiber.dispose()
        assert cleanup_calls == 2
        assert bad_fiber.state == FiberState.DISPOSED
        assert root._providers[SVC] is record  # pyright: ignore[reportPrivateUsage]
        assert root._providers[SVC_B] is extra_record  # pyright: ignore[reportPrivateUsage]
        assert provider_fiber.state == FiberState.ACTIVE
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_cancelled_late_provide_waits_for_physical_notification() -> None:
    """Caller cancellation cannot abandon a running consumer notification."""

    root = CompositionRoot("cancelled-late-provide-root")
    entered = asyncio.Event()
    release = asyncio.Event()
    cleanup_calls = 0

    async def provider(_ctx: Context) -> None:
        pass

    async def consumer(ctx: Context) -> None:
        nonlocal cleanup_calls
        ctx.require(SVC)
        entered.set()
        await release.wait()

        def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        await ctx.effect(lambda: cleanup)

    provide_task: asyncio.Task[Effect] | None = None
    try:
        provider_fiber = await root.mount(provider, name="provider")
        consumer_fiber = await root.mount(consumer, name="consumer", inject=(SVC,))
        provide_task = asyncio.create_task(provider_fiber.context.provide(SVC, "v"))
        await entered.wait()
        provide_task.cancel()
        cancellation_boundary = asyncio.Event()
        asyncio.get_running_loop().call_soon(cancellation_boundary.set)
        await cancellation_boundary.wait()
        assert not provide_task.done()
        record = root._providers[SVC]  # pyright: ignore[reportPrivateUsage]
        assert provider_fiber.state == FiberState.ACTIVE
        assert not record.revoking
        assert len(provider_fiber.effects) == 1

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await provide_task
        assert consumer_fiber.state == FiberState.ACTIVE
        assert cleanup_calls == 0
        assert root._providers[SVC] is record  # pyright: ignore[reportPrivateUsage]
    finally:
        release.set()
        if provide_task is not None:
            await asyncio.gather(provide_task, return_exceptions=True)
        await root.dispose()
    assert cleanup_calls == 1
