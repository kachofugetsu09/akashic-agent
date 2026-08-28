from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from agent.plugin_composition.context import CompositionRoot, PluginRuntime
from agent.plugin_composition.model import CompositionError, ServiceKey
from agent.plugin_composition.tool_catalog import (
    TOOL_CATALOG,
    PluginToolDefinition,
    PluginTools,
    _freeze_plugin_tools,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from agent.looping.core import _disable_candidate_side_effect_tools
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.events import InboundMessage
from bus.event_bus import EventBus


def _definition(*, name: str = "inspect_repository") -> PluginToolDefinition:
    return PluginToolDefinition(
        name=name,
        description="Inspect one repository without changing it.",
        parameters={
            "type": "object",
            "properties": {
                "repository": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["repository"],
            "additionalProperties": False,
        },
        handler_export="runtime.inspect_repository",
        risk="read-only",
    )


def _runtime(tmp_path: Path, plugin_id: str) -> PluginRuntime:
    plugin_dir = tmp_path / plugin_id
    plugin_dir.mkdir(exist_ok=True)
    return PluginRuntime(
        plugin_id=plugin_id,
        generation_id="test-generation",
        plugin_dir=plugin_dir,
        data_dir=plugin_dir / "data",
        workspace=plugin_dir / "workspace",
        config=None,
    )


class _MarketplaceWriteTool(Tool):
    name = "marketplace_write"
    description = "Write through one marketplace plugin."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return "ok"


@pytest.mark.parametrize(
    "parameters",
    [
        {"type": "array", "items": {"type": "string"}},
        {
            "type": "object",
            "properties": {"repository": {"type": "string"}},
            "required": ["missing"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": True,
        },
    ],
)
def test_tool_definition_rejects_malformed_schema_before_registration(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        PluginToolDefinition(
            name="inspect_repository",
            description="Inspect one repository.",
            parameters=parameters,
            handler_export="runtime.inspect_repository",
        )


def test_tool_definition_rejects_non_string_schema_key() -> None:
    with pytest.raises(TypeError, match="object key 必须是字符串"):
        PluginToolDefinition(
            name="inspect_repository",
            description="Inspect one repository.",
            parameters={
                "type": "object",
                "properties": {1: {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            handler_export="runtime.inspect_repository",
        )


def test_candidate_side_effect_fence_uses_full_marketplace_plugin_id() -> None:
    registry = ToolRegistry(validate_semantic_schema=False)
    registry.register(
        _MarketplaceWriteTool(),
        risk="external-side-effect",
        source_type="plugin",
        source_name="watcher@github",
    )
    generation = SimpleNamespace(
        instance=SimpleNamespace(name="watcher"),
        contributions=SimpleNamespace(mcp_servers={}),
    )
    snapshot = SimpleNamespace(
        generations=MappingProxyType({"watcher@github": generation}),
        mcp_server_registry=None,
    )
    message = InboundMessage("web", "hua", "1", "inspect")

    _disable_candidate_side_effect_tools(
        message,
        frozenset({"watcher@github"}),
        registry,
        snapshot,
    )

    assert message.metadata["disabled_tools"] == ["marketplace_write"]


@pytest.mark.asyncio
async def test_provided_service_resolves_its_exact_bound_tool(tmp_path: Path) -> None:
    capability = ServiceKey[object]("fixture.lookup.v1")
    root = CompositionRoot("provided-tool")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def apply(ctx) -> None:
        _ = await ctx.provide(capability, object())
        await ctx.require(TOOL_CATALOG).register(
            ctx,
            _definition(name="fixture_lookup"),
            provided_for=capability,
        )

    _ = await root.mount(
        apply,
        name="fixture",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "fixture"),
    )
    _ = _freeze_plugin_tools(
        tools,
        root.instance_token,
        {"fixture": "fixture:generation"},
        root.plugin_service_owners(),
    )

    assert tools.from_provide(capability) == "fixture_lookup"
    await root.dispose()


@pytest.mark.asyncio
async def test_empty_provided_marker_fails_loud_when_tool_is_requested(
    tmp_path: Path,
) -> None:
    marker = ServiceKey[object]("fixture.marker.v1")
    root = CompositionRoot("empty-provided-tool")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def apply(ctx) -> None:
        _ = await ctx.provide(marker, object())

    _ = await root.mount(
        apply,
        name="fixture",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "fixture"),
    )
    _ = _freeze_plugin_tools(
        tools,
        root.instance_token,
        {"fixture": "fixture:generation"},
        root.plugin_service_owners(),
    )

    with pytest.raises(CompositionError) as raised:
        tools.from_provide(marker)
    assert raised.value.code == "PROVIDED_TOOL_NOT_BOUND"
    await root.dispose()


@pytest.mark.asyncio
async def test_bound_tool_requires_an_existing_provided_service(tmp_path: Path) -> None:
    capability = ServiceKey[object]("fixture.missing.v1")
    root = CompositionRoot("missing-provide")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def apply(ctx) -> None:
        await ctx.require(TOOL_CATALOG).register(
            ctx,
            _definition(name="missing_lookup"),
            provided_for=capability,
        )

    _ = await root.mount(
        apply,
        name="consumer",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "missing-consumer"),
    )
    with pytest.raises(CompositionError) as raised:
        _ = _freeze_plugin_tools(
            tools,
            root.instance_token,
            {"missing-consumer": "consumer:generation"},
            root.plugin_service_owners(),
        )
    assert raised.value.code == "PROVIDED_SERVICE_MISSING"
    await root.dispose()


@pytest.mark.asyncio
async def test_bound_tool_must_share_its_service_owner(tmp_path: Path) -> None:
    capability = ServiceKey[object]("fixture.owned.v1")
    root = CompositionRoot("owner-mismatch")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def provide(ctx) -> None:
        _ = await ctx.provide(capability, object())

    async def bind(ctx) -> None:
        await ctx.require(TOOL_CATALOG).register(
            ctx,
            _definition(name="foreign_lookup"),
            provided_for=capability,
        )

    _ = await root.mount(provide, name="provider", runtime=_runtime(tmp_path, "owner"))
    _ = await root.mount(
        bind,
        name="consumer",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "foreign-consumer"),
    )
    with pytest.raises(CompositionError) as raised:
        _ = _freeze_plugin_tools(
            tools,
            root.instance_token,
            {
                "owner": "provider:generation",
                "foreign-consumer": "consumer:generation",
            },
            root.plugin_service_owners(),
        )
    assert raised.value.code == "PROVIDED_TOOL_OWNER_MISMATCH"
    await root.dispose()


@pytest.mark.asyncio
async def test_one_provided_service_binds_at_most_one_tool(tmp_path: Path) -> None:
    capability = ServiceKey[object]("fixture.one-tool.v1")
    root = CompositionRoot("duplicate-provided-tool")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def apply(ctx) -> None:
        _ = await ctx.provide(capability, object())
        for name in ("lookup_one", "lookup_two"):
            await ctx.require(TOOL_CATALOG).register(
                ctx,
                _definition(name=name),
                provided_for=capability,
            )

    _ = await root.mount(
        apply,
        name="provider",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "duplicate-provider"),
    )
    with pytest.raises(CompositionError) as raised:
        _ = _freeze_plugin_tools(
            tools,
            root.instance_token,
            {"duplicate-provider": "provider:generation"},
            root.plugin_service_owners(),
        )
    assert raised.value.code == "DUPLICATE_PROVIDED_TOOL"
    await root.dispose()


@pytest.mark.asyncio
async def test_provided_tool_resolution_uses_complete_active_snapshot(
    tmp_path: Path,
) -> None:
    capability = ServiceKey[object]("fixture.lookup.v1")
    provider_root = CompositionRoot("stable-provider")
    provider_tools = PluginTools(provider_root.instance_token)
    _ = await provider_root.context.provide(TOOL_CATALOG, provider_tools)

    async def provide(ctx) -> None:
        _ = await ctx.provide(capability, object())
        await ctx.require(TOOL_CATALOG).register(
            ctx,
            _definition(name="stable_lookup"),
            provided_for=capability,
        )

    _ = await provider_root.mount(
        provide,
        name="provider",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "provider"),
    )
    provider_catalog = _freeze_plugin_tools(
        provider_tools,
        provider_root.instance_token,
        {"provider": "provider:generation"},
        provider_root.plugin_service_owners(),
    )

    candidate_root = CompositionRoot("candidate-consumer")
    candidate_tools = PluginTools(candidate_root.instance_token)
    _ = await candidate_root.context.provide(TOOL_CATALOG, candidate_tools)
    _ = _freeze_plugin_tools(
        candidate_tools,
        candidate_root.instance_token,
        {},
        candidate_root.plugin_service_owners(),
    )
    lease = SimpleNamespace(
        active=True,
        snapshot=SimpleNamespace(plugin_tool_catalog=provider_catalog),
    )
    token = bind_runtime_snapshot(lease)  # type: ignore[arg-type]
    try:
        assert candidate_tools.from_provide(capability) == "stable_lookup"
    finally:
        reset_runtime_snapshot(token)
    await candidate_root.dispose()
    await provider_root.dispose()


@pytest.mark.asyncio
async def test_provided_tool_resolution_uses_compiled_catalog_in_background(
    tmp_path: Path,
) -> None:
    capability = ServiceKey[object]("fixture.background_lookup.v1")
    provider_root = CompositionRoot("background-provider")
    provider_tools = PluginTools(provider_root.instance_token)
    _ = await provider_root.context.provide(TOOL_CATALOG, provider_tools)

    async def provide(ctx) -> None:
        _ = await ctx.provide(capability, object())
        await ctx.require(TOOL_CATALOG).register(
            ctx,
            _definition(name="background_lookup"),
            provided_for=capability,
        )

    _ = await provider_root.mount(
        provide,
        name="provider",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "background-provider"),
    )
    provider_catalog = _freeze_plugin_tools(
        provider_tools,
        provider_root.instance_token,
        {"background-provider": "provider:generation"},
        provider_root.plugin_service_owners(),
    )
    consumer_tools = PluginTools(object())
    consumer_tools._bind_runtime_catalog(provider_catalog)

    assert consumer_tools.from_provide(capability) == "background_lookup"
    await provider_root.dispose()


@pytest.mark.asyncio
async def test_tool_catalog_identity_is_content_based_and_root_local(
    tmp_path: Path,
) -> None:
    async def build(root_name: str):
        root = CompositionRoot(root_name)
        tools = PluginTools(root.instance_token)
        _ = await root.context.provide(TOOL_CATALOG, tools)

        async def handler(context, arguments):
            _ = context, arguments
            return root_name

        async def apply(ctx) -> None:
            await ctx.require(TOOL_CATALOG).register(ctx, _definition(), handler)

        fiber = await root.mount(
            apply,
            name="github-watch",
            inject=(TOOL_CATALOG,),
            runtime=_runtime(tmp_path, "github-watch"),
        )
        catalog = _freeze_plugin_tools(
            tools,
            root.instance_token,
            {"github-watch": f"{root_name}:generation"},
        )
        return root, fiber, catalog

    candidate_root, candidate_fiber, candidate = await build("candidate")
    formal_root, formal_fiber, formal = await build("formal")

    assert candidate.identity == formal.identity
    assert candidate.root_instance_token is candidate_root.instance_token
    assert formal.root_instance_token is formal_root.instance_token
    assert candidate.root_instance_token is not formal.root_instance_token
    assert candidate["inspect_repository"].is_live()
    assert formal["inspect_repository"].is_live()
    candidate_handler = candidate["inspect_repository"].handler
    formal_handler = formal["inspect_repository"].handler
    assert candidate_handler is not None and formal_handler is not None
    assert candidate_handler is not formal_handler
    assert await candidate_handler(object(), {}) == "candidate"
    assert await formal_handler(object(), {}) == "formal"

    await candidate_fiber.dispose()
    assert not candidate["inspect_repository"].is_live()
    assert formal["inspect_repository"].is_live()
    await formal_fiber.dispose()
    await candidate_root.dispose()
    await formal_root.dispose()


@pytest.mark.asyncio
async def test_frozen_tool_catalog_rejects_generation_map_drift(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("tools:generation")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def apply(ctx) -> None:
        await ctx.require(TOOL_CATALOG).register(ctx, _definition())

    _ = await root.mount(
        apply,
        name="github-watch",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "github-watch"),
    )
    _ = _freeze_plugin_tools(
        tools,
        root.instance_token,
        {"github-watch": "generation:one"},
    )
    with pytest.raises(RuntimeError, match="generation identity 已冻结"):
        _freeze_plugin_tools(
            tools,
            root.instance_token,
            {"github-watch": "generation:two"},
        )
    await root.dispose()


@pytest.mark.asyncio
async def test_tool_catalog_rejects_duplicate_name_without_partial_registration(
    tmp_path: Path,
) -> None:
    root = CompositionRoot("tools:test")
    tools = PluginTools(root.instance_token)
    _ = await root.context.provide(TOOL_CATALOG, tools)

    async def register(ctx) -> None:
        await ctx.require(TOOL_CATALOG).register(ctx, _definition())

    first = await root.mount(
        register,
        name="first",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "first"),
    )
    second = await root.mount(
        register,
        name="second",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "second"),
    )
    assert second.state.value == "failed"
    assert isinstance(second.error, CompositionError)
    assert second.error.code == "DUPLICATE_PLUGIN_TOOL"

    catalog = _freeze_plugin_tools(
        tools,
        root.instance_token,
        {"first": "first:generation"},
    )
    assert tuple(catalog) == ("inspect_repository",)
    assert catalog["inspect_repository"].plugin_id == "first"
    await first.dispose()
    await root.dispose()


@pytest.mark.asyncio
async def test_tool_facade_rejects_cross_root_registration(
    tmp_path: Path,
) -> None:
    owner_root = CompositionRoot("owner:test")
    foreign_root = CompositionRoot("foreign:test")
    tools = PluginTools(owner_root.instance_token)
    _ = await foreign_root.context.provide(TOOL_CATALOG, tools)

    async def apply(ctx) -> None:
        await ctx.require(TOOL_CATALOG).register(ctx, _definition())

    fiber = await foreign_root.mount(
        apply,
        name="foreign",
        inject=(TOOL_CATALOG,),
        runtime=_runtime(tmp_path, "foreign"),
    )
    assert fiber.state.value == "failed"
    assert isinstance(fiber.error, CompositionError)
    assert fiber.error.code == "PLUGIN_TOOLS_SERVICE_ROOT_MISMATCH"

    await owner_root.dispose()
    await foreign_root.dispose()


@pytest.mark.asyncio
async def test_manager_compiles_and_executes_exact_v3_tool_binding(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "github-watch"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import TOOL_CATALOG, PluginToolDefinition\n"
        "api_version = 3\n"
        "name = 'github-watch'\n"
        "version = '1.0.0'\n"
        "inject = (TOOL_CATALOG,)\n"
        "bound_data_dir = None\n"
        "async def inspect_repository(context, arguments):\n"
        "    return context.turn_id + ':' + str(arguments['repository']) + ':' + str(bound_data_dir)\n"
        "async def mount_tools(ctx):\n"
        "    await ctx.require(TOOL_CATALOG).register(ctx, PluginToolDefinition(\n"
        "        name='inspect_repository',\n"
        "        description='Inspect one repository.',\n"
        "        parameters={\n"
        "            'type': 'object',\n"
        "            'properties': {'repository': {'type': 'string'}},\n"
        "            'required': ['repository'],\n"
        "            'additionalProperties': False,\n"
        "        },\n"
        "        handler_export='inspect_repository',\n"
        "        risk='read-only',\n"
        "    ))\n"
        "async def apply(ctx, config):\n"
        "    global bound_data_dir\n"
        "    bound_data_dir = ctx.data_root\n"
        "    await ctx.mount(mount_tools, inject=(TOOL_CATALOG,))\n",
        encoding="utf-8",
    )
    registry = ToolRegistry(validate_semantic_schema=False)
    manager = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        tool_registry=registry,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()
    snapshot = manager.current_snapshot
    assert snapshot is not None
    assert snapshot.plugin_tool_catalog is not None
    assert snapshot.composition_root is not None
    assert (
        snapshot.plugin_tool_catalog.root_instance_token
        is snapshot.composition_root.instance_token
    )
    assert snapshot.tool_registry is not None
    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    snapshot.tool_registry.set_context(turn_id="turn:test")
    try:
        result = await snapshot.tool_registry.execute(
            "inspect_repository",
            {"repository": "akashic-agent"},
            raise_errors=True,
        )
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
    assert isinstance(result, str)
    assert result.startswith("turn:test:akashic-agent:")
    assert "plugin-validation" not in result

    stable_binding = snapshot.plugin_tool_catalog["inspect_repository"]
    source = (plugin_dir / "plugin.py").read_text(encoding="utf-8")
    (plugin_dir / "plugin.py").write_text(
        source.replace("version = '1.0.0'", "version = '2.0.0'"),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("github-watch")
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_catalog = candidate.runtime_snapshot.plugin_tool_catalog
    assert candidate_catalog is not None
    candidate_binding = candidate_catalog["inspect_repository"]
    assert candidate_catalog is not snapshot.plugin_tool_catalog
    assert candidate_catalog.identity == snapshot.plugin_tool_catalog.identity

    transaction = manager.snapshot_store.begin_publish(candidate.runtime_snapshot)
    await manager.snapshot_store.commit_latest(transaction)
    candidate_lease = manager.snapshot_store.lease(selector="latest")
    candidate_token = bind_runtime_snapshot(candidate_lease)
    candidate.runtime_snapshot.tool_registry.set_context(turn_id="turn:candidate")
    try:
        candidate_result = await candidate.runtime_snapshot.tool_registry.execute(
            "inspect_repository",
            {"repository": "akashic-agent"},
            raise_errors=True,
        )
    finally:
        reset_runtime_snapshot(candidate_token)
        await candidate_lease.release()
    assert isinstance(candidate_result, str)
    assert "turn:candidate:akashic-agent:" in candidate_result
    assert "plugin-validation" in candidate_result

    await manager.snapshot_store.discard_latest(candidate.runtime_snapshot)
    assert not candidate_binding.is_live()
    await manager.discard_prepared("github-watch")
    await manager.terminate_all()
    assert not stable_binding.is_live()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("candidate_tool", "expected_tool"),
    (("recall_v2", "recall_v2"), (None, None)),
)
async def test_overlay_publish_switches_stable_background_consumer_catalog(
    tmp_path: Path,
    candidate_tool: str | None,
    expected_tool: str | None,
) -> None:
    plugin_dir = tmp_path / "plugins"
    provider_dir = plugin_dir / "memory-provider"
    consumer_dir = plugin_dir / "wake-consumer"
    provider_dir.mkdir(parents=True)
    consumer_dir.mkdir(parents=True)

    def provider_source(version: str, tool_name: str | None) -> str:
        source = "".join(
            (
                "from agent.plugin_composition import TOOL_CATALOG, "
                "PluginToolDefinition, ServiceKey\n"
                "MEMORY_RECALL = ServiceKey[object]('memory.recall.v1')\n"
                "api_version = 3\n"
                "name = 'memory-provider'\n"
                f"version = '{version}'\n",
                "inject = (TOOL_CATALOG,)\n" if tool_name else "inject = ()\n",
                "async def recall(context, arguments): return 'ok'\n"
                "async def apply(ctx, config):\n"
                "    await ctx.provide(MEMORY_RECALL, object())\n",
            )
        )
        if tool_name is None:
            return source
        return source + (
            "    await ctx.require(TOOL_CATALOG).register(\n"
            "        ctx, PluginToolDefinition(\n"
            f"            name='{tool_name}', description='Recall memory.',\n"
            "            parameters={'type':'object','properties':{},'required':[],"
            "'additionalProperties':False},\n"
            "            handler_export='recall', risk='read-only'),\n"
            "        recall, provided_for=MEMORY_RECALL)\n"
        )

    (provider_dir / "plugin.py").write_text(
        provider_source("1.0.0", "recall_v1"),
        encoding="utf-8",
    )
    (consumer_dir / "plugin.py").write_text(
        "from agent.plugin_composition import TOOL_CATALOG, ServiceKey\n"
        "MEMORY_RECALL = ServiceKey[object]('memory.recall.v1')\n"
        "api_version = 3\n"
        "name = 'wake-consumer'\n"
        "version = '1.0.0'\n"
        "inject = (TOOL_CATALOG, MEMORY_RECALL)\n"
        "tools = None\n"
        "async def apply(ctx, config):\n"
        "    global tools\n"
        "    ctx.require(MEMORY_RECALL)\n"
        "    tools = ctx.require(TOOL_CATALOG)\n"
        "def resolved(): return tools.from_provide(MEMORY_RECALL)\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_dir],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None
    consumer = stable.generations["wake-consumer"].instance.module
    assert consumer.resolved() == "recall_v1"

    (provider_dir / "plugin.py").write_text(
        provider_source("2.0.0", candidate_tool),
        encoding="utf-8",
    )
    candidate = await manager.prepare_candidate("memory-provider")
    assert candidate is not None and candidate.runtime_snapshot is not None
    candidate_catalog = candidate.runtime_snapshot.plugin_tool_catalog
    assert candidate_catalog is not None
    assert set(candidate_catalog) == ({candidate_tool} if candidate_tool else set())
    assert consumer.tools in candidate.runtime_snapshot.plugin_tool_facades
    transaction = manager.snapshot_store.begin_publish(candidate.runtime_snapshot)
    await manager.snapshot_store.commit_latest(transaction)
    assert consumer.resolved() == "recall_v1"

    candidate.runtime_snapshot.accepting_leases = False
    manager.snapshot_store.seal_candidate_validation(candidate.runtime_snapshot)
    with pytest.raises(RuntimeError, match="rollback fixture"):
        await manager.snapshot_store.promote_latest(
            after_open=lambda: (_ for _ in ()).throw(RuntimeError("rollback fixture"))
        )
    assert consumer.resolved() == "recall_v1"

    candidate.runtime_snapshot.accepting_leases = False
    _ = await manager.snapshot_store.promote_latest()
    if expected_tool is None:
        with pytest.raises(CompositionError) as raised:
            _ = consumer.resolved()
        assert raised.value.code == "PROVIDED_TOOL_NOT_BOUND"
    else:
        assert consumer.resolved() == expected_tool

    await manager.terminate_all()


@pytest.mark.asyncio
async def test_manager_rejects_malformed_tool_handler_before_publication(
    tmp_path: Path,
) -> None:
    plugin_dir = tmp_path / "plugins" / "broken-tool"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(
        "from agent.plugin_composition import TOOL_CATALOG, PluginToolDefinition\n"
        "api_version = 3\n"
        "name = 'broken-tool'\n"
        "version = '1.0.0'\n"
        "inject = (TOOL_CATALOG,)\n"
        "async def broken(arguments):\n"
        "    return 'bad'\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(TOOL_CATALOG).register(ctx, PluginToolDefinition(\n"
        "        name='broken_tool', description='Broken Tool.',\n"
        "        parameters={'type': 'object', 'properties': {}, 'required': [], "
        "'additionalProperties': False},\n"
        "        handler_export='broken',\n"
        "    ))\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugin_dir.parent],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )

    await manager.load_all()
    assert manager.current_snapshot is None
    assert manager.snapshot_store.current is None
    assert manager.generation("broken-tool") is None
    assert not (tmp_path / "workspace" / "plugin-data" / "broken-tool-builtin").exists()
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_incremental_load_rolls_back_new_tool_data_root_on_admission_failure(
    tmp_path: Path,
) -> None:
    plugins = tmp_path / "plugins"
    stable_dir = plugins / "stable-v3"
    stable_dir.mkdir(parents=True)
    (stable_dir / "plugin.py").write_text(
        "api_version = 3\n"
        "name = 'stable-v3'\n"
        "version = '1.0.0'\n"
        "async def apply(ctx, config):\n"
        "    return None\n",
        encoding="utf-8",
    )
    manager = PluginManager(
        plugin_dirs=[plugins],
        event_bus=EventBus(),
        tool_registry=ToolRegistry(validate_semantic_schema=False),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "cache",
    )
    await manager.load_all()
    stable = manager.current_snapshot
    assert stable is not None

    broken_dir = plugins / "broken-tool"
    broken_dir.mkdir()
    (broken_dir / "plugin.py").write_text(
        "from agent.plugin_composition import TOOL_CATALOG, PluginToolDefinition\n"
        "api_version = 3\n"
        "name = 'broken-tool'\n"
        "version = '1.0.0'\n"
        "inject = (TOOL_CATALOG,)\n"
        "async def broken(arguments):\n"
        "    return 'bad'\n"
        "async def apply(ctx, config):\n"
        "    await ctx.require(TOOL_CATALOG).register(ctx, PluginToolDefinition(\n"
        "        name='broken_tool', description='Broken Tool.',\n"
        "        parameters={'type': 'object', 'properties': {}, 'required': [], "
        "'additionalProperties': False}, handler_export='broken'))\n",
        encoding="utf-8",
    )
    await manager.load_all()

    assert manager.current_snapshot is stable
    assert manager.generation("broken-tool") is None
    assert not (tmp_path / "workspace" / "plugin-data" / "broken-tool-builtin").exists()
    await manager.terminate_all()
