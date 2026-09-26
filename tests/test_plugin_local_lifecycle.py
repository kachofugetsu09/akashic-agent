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
from typing import Callable, Coroutine
import pytest
from agent.plugin_composition import CompositionError, CompositionRoot, Context, RuntimeScope, ServiceKey
from agent.plugin_composition.model import FiberState, PluginRuntime

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
