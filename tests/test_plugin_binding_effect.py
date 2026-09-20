import pytest

from agent.plugin_composition import CompositionRoot, CompositionError, ServiceKey


@pytest.mark.asyncio
async def test_binding_contributors_end_with_the_service_effect() -> None:
    """动态归档贡献与原服务同生共死，其他 consumer 不能覆盖它。"""
    root = CompositionRoot("binding-lifetime")
    key = ServiceKey("binding-lifetime.value")
    try:
        effect = await root.context.provide(key, "owned", binding_contributors=lambda: (root.context,))
        assert root.binding_contributors(key) == (root.context,)
        with pytest.raises(CompositionError, match="已由"):
            await root.context.provide(key, "other", binding_contributors=lambda: ())
        await effect.aclose()
        with pytest.raises(RuntimeError, match="已失效"):
            root.binding_contributors(key)
    finally:
        await root.dispose()
