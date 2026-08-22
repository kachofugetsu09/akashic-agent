from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest

from agent.plugin_composition.context import CompositionRoot, PluginRuntime
from agent.plugin_composition.model import CompositionError
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
    assert not (
        tmp_path / "workspace" / "plugin-data" / "broken-tool-builtin"
    ).exists()
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
