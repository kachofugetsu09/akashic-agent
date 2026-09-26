import asyncio
import pytest
from agent.plugin_composition import CompositionError, CompositionRoot, Context
from agent.plugin_composition.tasks import Tasks

@pytest.mark.asyncio
async def test_raw_child_does_not_inherit_parent_plugin_scope():
    """A raw child Task inherits values, but cannot capture the parent permit."""

    root = CompositionRoot("task-raw-child-root")
    contexts: list[Context] = []

    async def apply(ctx: Context) -> None:
        contexts.append(ctx)

    fiber = await root.mount(apply, name="task-owner")
    ctx = contexts[0]
    tasks = Tasks()
    errors: list[str | None] = []

    async def operation(_task):
        with pytest.raises(CompositionError) as excinfo:
            ctx.capture_runtime_scope()
        errors.append(excinfo.value.code)

    async def raw_child() -> None:
        task = await tasks.admit("key", lambda slot: slot.start(operation))
        await task.join()

    async with ctx.runtime_scope():
        await asyncio.create_task(raw_child())

    assert errors == ["OWNER_CALL_CONTEXT"]
    await tasks.close()
    await fiber.dispose()
    await root.dispose()
