from __future__ import annotations

# pyright: reportPrivateUsage=false

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest

from agent.plugin_composition import (
    PLUGIN_TOOLS,
    CompositionRoot,
    PluginRuntime,
    PluginToolContribution,
    PluginTools,
    ServiceKey,
)
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.plugins.snapshot import (
    RuntimeSnapshot,
    RuntimeSnapshotCompiler,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from agent.tool_hooks import HookOutcome, ToolHook
from agent.tool_hooks.executor import ToolExecutor
from agent.tool_hooks.types import HookContext, ToolExecutionRequest
from agent.tools.base import Tool
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


class _ExistingTool(Tool):
    name = "composition_echo"
    description = "existing tool"
    parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return "existing"


class _CatalogTool(Tool):
    name = "catalog-placeholder"
    description = "Catalog placeholder"
    parameters = {"type": "object"}

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        marker: str,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.marker = marker

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return self.marker


class _RewriteHook(ToolHook):
    name = "rewrite"
    event = "pre_tool_use"

    def matches(self, ctx: HookContext) -> bool:
        return ctx.request.tool_name == "composition_echo"

    async def run(self, ctx: HookContext) -> HookOutcome:
        return HookOutcome(updated_input={"text": "rewritten"})


class _LeakedPluginTools(PluginTools):
    def _register(
        self,
        plugin_id: str,
        contribution: PluginToolContribution,
    ) -> Callable[[], None]:
        _ = super()._register(plugin_id, contribution)
        return lambda: None


def _write_plugin(root: Path, source: str) -> None:
    plugin_dir = root / "tool_probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")


def _manager(
    tmp_path: Path,
    registry: ToolRegistry | None,
) -> PluginManager:
    return PluginManager(
        plugin_dirs=[tmp_path / "plugins"],
        event_bus=EventBus(),
        tool_registry=registry,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


def _tool_snapshot(
    *,
    marker: str,
    name: str = "catalog_probe",
    description: str = "Catalog probe",
    parameters: dict[str, Any] | None = None,
    risk: str = "read-only",
    always_on: bool = False,
    preloadable: bool = True,
    requires_turn_search: bool = False,
    search_hint: str | None = "probe",
    source_type: str = "plugin",
    source_name: str = "catalog-source",
    owner: str = "catalog-owner",
) -> RuntimeSnapshot:
    registry = ToolRegistry(follow_runtime_snapshot=False)
    registry.register(
        _CatalogTool(
            name=name,
            description=description,
            parameters=parameters
            or {
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["text"],
            },
            marker=marker,
        ),
        risk=risk,
        always_on=always_on,
        preloadable=preloadable,
        requires_turn_search=requires_turn_search,
        search_hint=search_hint,
        source_type=source_type,
        source_name=source_name,
        owner=owner,
    )
    return RuntimeSnapshotCompiler().compile({}, tool_registry=registry)


def test_effective_tool_catalog_fields_change_snapshot_identity() -> None:
    variants = (
        {},
        {"name": "catalog_other"},
        {"description": "Other description"},
        {
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "integer"}},
                "required": ["text"],
            }
        },
        {"risk": "write"},
        {"always_on": True},
        {"preloadable": False},
        {"requires_turn_search": True},
        {"search_hint": "other"},
        {"source_type": "builtin"},
        {"source_name": "other-source"},
        {"owner": "other-owner"},
    )

    snapshots = tuple(
        _tool_snapshot(marker=f"variant-{index}", **fields)
        for index, fields in enumerate(variants)
    )

    assert len({snapshot.snapshot_id for snapshot in snapshots}) == len(variants)
    assert all(snapshot.tool_registry is not None for snapshot in snapshots)


def test_tool_catalog_identity_normalizes_schema_and_excludes_handler() -> None:
    implicit = _tool_snapshot(marker="first")
    explicit = _tool_snapshot(
        marker="second",
        parameters={
            "required": ["text"],
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
            },
            "additionalProperties": False,
            "type": "object",
        },
    )

    assert implicit.snapshot_id == explicit.snapshot_id
    assert implicit.tool_registry is not None
    assert explicit.tool_registry is not None
    assert implicit.tool_registry.catalog_identity() == (
        explicit.tool_registry.catalog_identity()
    )


def test_tool_catalog_identity_ignores_registration_order() -> None:
    first = ToolRegistry(follow_runtime_snapshot=False)
    second = ToolRegistry(follow_runtime_snapshot=False)
    tools = (
        _CatalogTool(
            name="catalog_alpha",
            description="Alpha",
            parameters={"type": "object"},
            marker="alpha",
        ),
        _CatalogTool(
            name="catalog_beta",
            description="Beta",
            parameters={"type": "object"},
            marker="beta",
        ),
    )
    for tool in tools:
        first.register(tool, owner="fixture")
    for tool in reversed(tools):
        second.register(tool, owner="fixture")

    assert first.catalog_identity() == second.catalog_identity()


def test_tool_catalog_default_owners_are_explicitly_reproducible() -> None:
    cases = (
        ("builtin", "", "core"),
        ("plugin", "meme", "meme"),
        ("mcp", "github", "github"),
    )
    for index, (source_type, source_name, owner) in enumerate(cases):
        implicit = ToolRegistry(follow_runtime_snapshot=False)
        explicit = ToolRegistry(follow_runtime_snapshot=False)
        tool = _CatalogTool(
            name=f"owner_{index}",
            description="Owner probe",
            parameters={"type": "object"},
            marker="owner",
        )
        implicit.register(
            tool,
            source_type=source_type,
            source_name=source_name,
        )
        explicit.register(
            tool,
            source_type=source_type,
            source_name=source_name,
            owner=owner,
        )

        assert implicit.catalog_identity() == explicit.catalog_identity()


@pytest.mark.asyncio
async def test_pending_root_does_not_freeze_tool_collector(tmp_path: Path) -> None:
    dependency = ServiceKey[object]("fixture.pending")
    root = CompositionRoot("pending-tools")
    tools = PluginTools()
    _ = await root.context.provide(PLUGIN_TOOLS, tools)

    async def plugin(ctx) -> None:
        await tools.register(ctx, _ExistingTool(), risk="read-only")

    plugin_dir = tmp_path / "pending-plugin"
    plugin_dir.mkdir()
    _ = await root.mount(
        plugin,
        name="tool_probe",
        inject=(PLUGIN_TOOLS, dependency),
        runtime=PluginRuntime(
            plugin_id="tool_probe",
            plugin_dir=plugin_dir,
            data_dir=plugin_dir / "data",
            workspace=plugin_dir / "workspace",
            config=object(),
        ),
    )
    manager = _manager(tmp_path, ToolRegistry())

    with pytest.raises(RuntimeError, match="不能冻结 Tool catalog"):
        manager._compile_snapshot_tools({}, composition_root=root)

    async def provider(ctx) -> None:
        await ctx.provide(dependency, object())

    _ = await root.mount(provider, name="provider")
    assert root.receipt().errors == ()
    assert set(tools.freeze()) == {"tool_probe"}

    await root.dispose()


@pytest.mark.asyncio
async def test_v3_tool_compiles_into_snapshot_and_runs_owned_events(
    tmp_path: Path,
) -> None:
    _write_plugin(tmp_path / "plugins", _tool_plugin_source())
    manager = _manager(tmp_path, ToolRegistry())

    await manager.load_all()

    generation = manager.generation("tool_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    assert snapshot.tool_registry is not None
    registry = snapshot.tool_registry
    module = generation.instance.module
    document = registry.get_document("composition_echo")
    assert document is not None
    assert (
        document.risk,
        document.always_on,
        document.search_hint,
        document.source_type,
        document.source_name,
    ) == ("read-only", True, "echo probe", "plugin", "tool_probe")
    assert registry.get_registered_order() == ["composition_echo"]
    assert registry.get_non_preloadable_names() == {"composition_echo"}
    assert snapshot.composition_topology is not None
    assert snapshot.composition_topology.services == (
        "core.plugin_assets",
        "core.timer",
        "core.tools",
    )

    lease = manager.snapshot_store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        executor = ToolExecutor([_RewriteHook()])
        result = await executor.execute(
            ToolExecutionRequest(
                call_id="call-success",
                tool_name="composition_echo",
                arguments={"text": "original"},
                source="passive",
                session_key="session-1",
                channel="mobile",
                chat_id="chat-1",
            ),
            lambda name, arguments: registry.execute(
                name,
                arguments,
                raise_errors=True,
            ),
        )
        assert result.status == "success"
        assert result.output == "echo:rewritten"
        assert module.events == [
            ("before", "rewritten", "passive"),
            ("invoke", "rewritten"),
            ("after", "success", "echo:rewritten"),
        ]
        with pytest.raises(TypeError):
            cast(Any, module.before_arguments)["text"] = "mutated"

        module.mode = "deny"
        denied = await executor.execute(
            ToolExecutionRequest(
                call_id="call-denied",
                tool_name="composition_echo",
                arguments={"text": "blocked"},
                source="subagent",
            ),
            lambda name, arguments: registry.execute(
                name,
                arguments,
                raise_errors=True,
            ),
        )
        assert denied.status == "denied"
        assert denied.output == "blocked by probe"
        assert module.calls == 1
        assert module.events[-2:] == [
            ("before", "rewritten", "subagent"),
            ("after", "denied", "blocked by probe"),
        ]

        module.mode = "fail"
        failed = await executor.execute(
            ToolExecutionRequest(
                call_id="call-failed",
                tool_name="composition_echo",
                arguments={"text": "failed"},
                source="proactive",
            ),
            lambda name, arguments: registry.execute(
                name,
                arguments,
                raise_errors=True,
            ),
        )
        assert failed.status == "error"
        assert "before probe failed" in cast(str, failed.output)
        assert module.calls == 1
        assert module.events[-2:] == [
            ("before", "rewritten", "proactive"),
            ("after", "error", cast(str, failed.output)),
        ]
    finally:
        reset_runtime_snapshot(token)
        await lease.release()

    root = snapshot.composition_root
    assert root is not None
    assert "tool_probe:tool:composition_echo" in root.receipt().effects
    await manager.terminate_all()
    assert root.receipt().effects == ()
    assert root.receipt().services == ()


@pytest.mark.asyncio
async def test_v3_tool_conflict_rejects_candidate_without_mutating_base(
    tmp_path: Path,
) -> None:
    _write_plugin(tmp_path / "plugins", _tool_plugin_source())
    registry = ToolRegistry()
    existing = _ExistingTool()
    registry.register(existing)
    manager = _manager(tmp_path, registry)

    await manager.load_all()

    assert manager.generation("tool_probe") is None
    assert manager.current_snapshot is None
    assert registry.get_tool("composition_echo") is existing
    assert registry.get_registered_order() == ["composition_echo"]
    gate = manager.latest_gate("tool_probe")
    assert gate is not None
    assert gate.status == "failed"
    snapshot_check = next(
        item for item in gate.checks if item.check_id == "runtime_snapshot"
    )
    assert snapshot_check.status == "failed"
    assert "插件工具名称重复" in cast(str, snapshot_check.evidence)
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_v3_tool_declaration_fails_when_core_registry_is_unavailable(
    tmp_path: Path,
) -> None:
    _write_plugin(tmp_path / "plugins", _tool_plugin_source())
    manager = _manager(tmp_path, None)

    await manager.load_all()

    assert manager.generation("tool_probe") is None
    assert manager.current_snapshot is None
    gate = manager.latest_gate("tool_probe")
    assert gate is not None and gate.status == "failed"
    snapshot_check = next(
        item for item in gate.checks if item.check_id == "runtime_snapshot"
    )
    assert "Core 没有配置 ToolRegistry" in cast(str, snapshot_check.evidence)
    await manager.terminate_all()


@pytest.mark.asyncio
async def test_tool_registration_effect_rolls_back_duplicate_before_freeze(
    tmp_path: Path,
) -> None:
    registrations, errors = await _duplicate_registration_fixture(
        tmp_path,
        PluginTools,
    )

    assert any("插件 Tool 名称重复" in error for error in errors)
    assert registrations == {}


@pytest.mark.asyncio
async def test_tool_registration_oracle_kills_leaked_disposer_mutant(
    tmp_path: Path,
) -> None:
    correct, _ = await _duplicate_registration_fixture(
        tmp_path / "correct",
        PluginTools,
    )
    mutant, _ = await _duplicate_registration_fixture(
        tmp_path / "mutant",
        _LeakedPluginTools,
    )

    assert correct == {}
    assert set(mutant) == {"tool_probe"}


async def _duplicate_registration_fixture(
    root_dir: Path,
    tools_type: type[PluginTools],
) -> tuple[dict[str, tuple[PluginToolContribution, ...]], tuple[str, ...]]:
    """Run one duplicate declaration through real Fiber rollback."""

    # 1. Mount the same plugin behavior against production and mutant collectors.
    root = CompositionRoot(f"tool-duplicate:{tools_type.__name__}")
    tools = tools_type()
    _ = await root.context.provide(PLUGIN_TOOLS, tools)

    async def plugin(ctx) -> None:
        await tools.register(ctx, _ExistingTool(), risk="read-only")
        await tools.register(ctx, _ExistingTool(), risk="read-only")

    plugin_dir = root_dir / "plugin"
    plugin_dir.mkdir(parents=True)
    _ = await root.mount(
        plugin,
        name="tool_probe",
        inject=(PLUGIN_TOOLS,),
        runtime=PluginRuntime(
            plugin_id="tool_probe",
            plugin_dir=plugin_dir,
            data_dir=plugin_dir / "data",
            workspace=plugin_dir / "workspace",
            config=object(),
        ),
    )

    # 2. Observe the collector before Root disposal hides it behind service cleanup.
    registrations = dict(tools.freeze())
    errors = root.receipt().errors
    await root.dispose()
    return registrations, errors


def _tool_plugin_source() -> str:
    return (
        "from agent.plugin_composition import Bail, PLUGIN_TOOLS\n"
        "from agent.tool_hooks.executor import (\n"
        "    TOOL_EXECUTION_AFTER, TOOL_EXECUTION_BEFORE,\n"
        ")\n"
        "from agent.tools.base import Tool\n"
        "api_version = 3\n"
        "name = 'tool_probe'\n"
        "version = '1.0.0'\n"
        "inject = (PLUGIN_TOOLS,)\n"
        "events = []\n"
        "before_arguments = None\n"
        "calls = 0\n"
        "mode = 'pass'\n"
        "class Echo(Tool):\n"
        "    name = 'composition_echo'\n"
        "    description = 'Echo one value'\n"
        "    parameters = {\n"
        "        'type': 'object',\n"
        "        'properties': {'text': {'type': 'string'}},\n"
        "        'required': ['text'],\n"
        "    }\n"
        "    async def execute(self, text):\n"
        "        global calls\n"
        "        calls += 1\n"
        "        events.append(('invoke', text))\n"
        "        return f'echo:{text}'\n"
        "    def validate_params(self, params, *, schema=None):\n"
        "        del params, schema\n"
        "        return ['plugin override must not run']\n"
        "    def to_schema(self):\n"
        "        raise AssertionError('plugin formatter must not run')\n"
        "tool = Echo()\n"
        "async def before(event):\n"
        "    global before_arguments\n"
        "    before_arguments = event.arguments\n"
        "    events.append(('before', event.arguments['text'], event.source))\n"
        "    if mode == 'deny':\n"
        "        return Bail('blocked by probe')\n"
        "    if mode == 'fail':\n"
        "        raise RuntimeError('before probe failed')\n"
        "async def after(event):\n"
        "    events.append(('after', event.status, event.output))\n"
        "async def apply(ctx, config):\n"
        "    del config\n"
        "    await ctx.require(PLUGIN_TOOLS).register(\n"
        "        ctx, tool, risk='read-only', always_on=True,\n"
        "        preloadable=False, search_hint='echo probe',\n"
        "    )\n"
        "    await ctx.on(TOOL_EXECUTION_BEFORE, before)\n"
        "    await ctx.on(TOOL_EXECUTION_AFTER, after)\n"
    )


@pytest.mark.asyncio
async def test_v3_snapshot_freezes_catalog_but_keeps_handler(tmp_path: Path) -> None:
    _write_plugin(tmp_path / "plugins", _tool_plugin_source())
    manager = _manager(tmp_path, ToolRegistry())
    await manager.load_all()

    generation = manager.generation("tool_probe")
    snapshot = manager.current_snapshot
    assert generation is not None and snapshot is not None
    assert isinstance(generation.instance, ComposablePlugin)
    assert snapshot.tool_registry is not None
    registry = snapshot.tool_registry
    module = generation.instance.module
    snapshot_id = snapshot.snapshot_id
    catalog_identity = registry.catalog_identity()
    schemas = registry.get_schemas()
    search_results = registry.search("Echo one value")
    document = registry.get_document("composition_echo")
    documents = registry.get_documents()
    assert document is not None and len(documents) == 1

    module.tool.name = "mutated_name"
    module.tool.description = "mutated description"
    module.tool.parameters["properties"]["text"]["type"] = "integer"
    module.tool.parameters["required"] = []
    document.description = "mutated document"
    documents[0].search_hint = "mutated hint"

    assert snapshot.snapshot_id == snapshot_id
    assert registry.catalog_identity() == catalog_identity
    assert registry.get_schemas() == schemas
    assert registry.search("Echo one value") == search_results
    current_document = registry.get_document("composition_echo")
    assert current_document is not None
    assert current_document.description == "Echo one value"
    assert current_document.search_hint == "echo probe"
    assert (
        await registry.execute(
            "composition_echo",
            {"text": "still-valid"},
            raise_errors=True,
        )
        == "echo:still-valid"
    )
    with pytest.raises(ValueError, match="text 应为 string 类型"):
        await registry.execute(
            "composition_echo",
            {"text": 1},
            raise_errors=True,
        )
    await manager.terminate_all()
