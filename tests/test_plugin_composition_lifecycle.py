from __future__ import annotations
import pytest
from agent.plugin_composition import CompositionRoot, EmitEventKey
from bus.event_bus import EventBus
from core.memory.events import MemoryWritten

_MEMORY_WRITTEN_EVENT = EmitEventKey[MemoryWritten]("test.memory.written")

@pytest.mark.asyncio
async def test_failed_consumer_cleanup_keeps_provider_until_explicit_retry():
    """消费者关闭失败后，原资源及依赖仍由同一 Root 持有。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("cleanup-retry")
    service = ServiceKey[list[str]]("test.cleanup-resource")
    events: list[str] = []
    attempts = 0

    async def provider(ctx):
        await ctx.effect(lambda: lambda: events.append("provider-close"))
        await ctx.provide(service, events)

    async def consumer(ctx):
        resource = ctx.require(service)

        def close():
            nonlocal attempts
            attempts += 1
            resource.append("consumer-close")
            if attempts == 1:
                raise OSError("resource still open")

        await ctx.effect(lambda: close)

    await root.mount(provider, name="provider")
    await root.mount(consumer, name="consumer", inject=(service,))
    with pytest.raises(BaseExceptionGroup):
        await root.dispose()
    assert events == ["consumer-close"]

    await root.dispose()
    assert events == ["consumer-close", "consumer-close", "provider-close"]
    await root.dispose()
    assert events == ["consumer-close", "consumer-close", "provider-close"]

@pytest.mark.asyncio
async def test_emit_event_listener_failure_is_fail_loud() -> None:
    root = CompositionRoot("emit-event-failure")

    def fail(_: object) -> None:
        raise RuntimeError("observe failed")

    async def plugin(ctx) -> None:
        await ctx.on(EmitEventKey[object]("test.emit.failure"), fail)

    await root.mount(plugin, name="failing-emit-plugin")
    try:
        with pytest.raises(RuntimeError, match="observe failed"):
            root.context.emit(EmitEventKey[object]("test.emit.failure"), object())
    finally:
        await root.dispose()

@pytest.mark.asyncio
async def test_event_bus_does_not_bridge_into_plugin_composition() -> None:
    observed: list[MemoryWritten] = []
    root = CompositionRoot("event-bus-is-core-only")

    async def plugin(ctx) -> None:
        await ctx.on(_MEMORY_WRITTEN_EVENT, observed.append)

    await root.mount(plugin, name="composition-observer")
    bus = EventBus()
    try:
        await bus.fanout(_memory_written_event())
    finally:
        try:
            await bus.aclose()
        finally:
            await root.dispose()

    assert observed == []

def _memory_written_event() -> MemoryWritten:
    return MemoryWritten(
        session_key="session",
        channel="test",
        chat_id="chat",
        action="supersede",
        source_ref="session@post_response",
        superseded_ids=["memory-1"],
    )

@pytest.mark.asyncio
async def test_pending_initial_dependency_resolves_then_frozen_teardown_closes_consumers_first():
    """初始待定依赖可激活；退出顺序按实际依赖而非挂载顺序。"""
    from agent.plugin_composition import ServiceKey

    root = CompositionRoot("initial-resolution")
    service = ServiceKey[list[str]]("test.initial.service")
    events = []

    async def consumer(ctx):
        shared = ctx.require(service)
        events.append("consumer-start")
        await ctx.effect(lambda: lambda: shared.append("consumer-close"))

    async def provider(ctx):
        await ctx.effect(lambda: lambda: events.append("provider-close"))
        await ctx.provide(service, events)

    await root.mount(consumer, name="consumer", inject=(service,))
    assert events == []
    await root.mount(provider, name="provider")
    assert events == ["consumer-start"]
    await root.dispose()
    assert events == ["consumer-start", "consumer-close", "provider-close"]
