from pathlib import Path
import json
import shutil
from collections.abc import Mapping
from typing import cast

import pytest

from agent.plugin_composition.bindings import BINDINGS, Bindings
from agent.plugin_composition.models import ToolCall as ModelToolCall
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.tool_search.plugin import TOOL_SEARCH_PRESENTATION, TOOL_SEARCH_TOOLS
from plugins.tools.api import MessageReply
from plugins.tools.menu import ToolMenu
from plugins.tools.plugin import ALL_TOOLS, TOOLS, ToolView, open_tool
from session.log import MessageLog
from session.message import CallRef
from tests.test_tool_bindings import write_plugins


def _manager(tmp_path, sources, log):
    return PluginManager(
        plugin_dirs=sources,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
        message_log=log,
    )


def _sources(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    target = sources / "target/plugin.py"
    target.write_text(
        target.read_text()
        .replace(
            "from plugins.tools.execution import Result",
            "from plugins.tools.execution import Result",
        )
        .replace(
            'raise ValueError("value must be text")',
            'return "value must be text"',
        )
    )
    prepare = sources / "prepare/plugin.py"
    prepare.write_text(
        prepare.read_text().replace(
            'return {"value": "restore:" + arguments["value"]}',
            'return {"value": ("restore:" + arguments["value"]) if isinstance(arguments["value"], str) else arguments["value"]}',
        )
    )
    target.write_text(
        target.read_text()
        .replace(
            '    await ctx.require(inject[0]).register(\n        ctx, name="example"',
            "    await ctx.require(inject[0]).declare_group(ctx)\n"
            '    await ctx.require(inject[0]).register(\n        ctx, name="example"',
        )
        .replace(
            "        open=open_target,\n    )",
            "        open=open_target,\n    )\n"
            "    await ctx.require(inject[0]).register(\n"
            '        ctx, name="example_status", description="Read example status",\n'
            '        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"], "additionalProperties": False},\n'
            "        open=open_target,\n"
            "    )",
        )
    )
    shutil.copytree(
        Path(__file__).parents[1] / "plugins/tool_search",
        sources / "tool_search",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return sources


def _unexpected_reply(call_ref: CallRef) -> MessageReply:
    raise AssertionError(f"unexpected message tool execution: {call_ref}")


@pytest.mark.asyncio
async def test_search_presentation_keeps_fixed_schemas_and_executes_awarded_ref(tmp_path):
    sources = _sources(tmp_path)
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            bindings = ctx.require(BINDINGS)
            view = ToolView.combine(
                ctx.require(ALL_TOOLS)(), cast(ToolView, ctx.require(TOOL_SEARCH_TOOLS))
            )
            presentation = ctx.require(TOOL_SEARCH_PRESENTATION)(view)
            menu = ToolMenu(
                catalog,
                bindings,
                catalog.execution(lambda binding, arguments: _allow()),
                _unexpected_reply,
                view=view,
                presentation=presentation,
            )
            assert [row["function"]["name"] for row in menu.schemas] == [
                "tool_search",
                "tool_call",
            ]

            decoded = menu.decode(ModelToolCall('search', 'tool_search', {'query': 'example'}))
            (search_binding, search_arguments) = decoded.binding_id, decoded.arguments
            assert search_binding is not None
            async with open_tool(bindings, search_binding) as search:
                prepared = await search.prepare(search_arguments)
                assert isinstance(prepared, Mapping)
                result = await search.invoke(
                    "search",
                    prepared,
                )
            value = result.parts[0].value
            assert isinstance(value, str)
            payload = json.loads(value)
            assert payload["matched_groups"][0]["owner"] == "target"
            schema = payload["matched_groups"][0]["tools"][0]["function"]
            assert schema["name"] == "example"
            assert schema["parameters"]["required"] == ["value"]
            assert [
                tool["function"]["name"]
                for tool in payload["matched_groups"][0]["tools"]
            ] == [
                "example",
                "example_status",
            ]
            assert [row["function"]["name"] for row in menu.schemas] == [
                "tool_search",
                "tool_call",
            ]

            decoded = menu.decode(ModelToolCall('call', 'tool_call', {'name': 'example', 'arguments': {'value': ' ok '}}))
            (binding, arguments) = decoded.binding_id, decoded.arguments
            assert binding is not None
            executed = await catalog.execution(
                lambda identity, final: _allow()
            ).execute("valid", binding, arguments)
            assert executed.parts[0].value == "A:restore: ok"
            invalid = await catalog.execution(
                lambda identity, final: _allow()
            ).execute("invalid", binding, {"value": 7})
            assert invalid.outcome == "error"
            assert invalid.parts[0].value == "value must be text"
            rejected = menu.decode(
                ModelToolCall(
                    "unknown",
                    "tool_call",
                    {"name": "missing", "arguments": {}},
                )
            )
            assert not rejected.accepted
            assert rejected.rejection is not None
            assert rejected.rejection["name"] == "tool_call"
            rejection_error = rejected.rejection["error"]
            assert isinstance(rejection_error, str)
            assert "获授 view" in rejection_error
            narrow = ToolView.combine(
                catalog.view(view.select("example")),
                cast(ToolView, ctx.require(TOOL_SEARCH_TOOLS)),
            )
            narrow_menu = ToolMenu(
                catalog,
                bindings,
                catalog.execution(lambda binding, arguments: _allow()),
                _unexpected_reply,
                view=narrow,
                presentation=ctx.require(TOOL_SEARCH_PRESENTATION)(narrow),
            )
            rejected = narrow_menu.decode(
                ModelToolCall(
                    "not-awarded",
                    "tool_call",
                    {"name": "example_status", "arguments": {"value": "ok"}},
                )
            )
            assert not rejected.accepted
            assert rejected.rejection is not None
            assert rejected.rejection["name"] == "tool_call"
            rejection_error = rejected.rejection["error"]
            assert isinstance(rejection_error, str)
            assert "获授 view" in rejection_error
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_standard_web_is_directly_callable_without_search(tmp_path):
    """conversation 的 Web 基础工具在首次模型请求中即可直接调用。"""
    sources = _sources(tmp_path)
    shutil.copytree(Path(__file__).parents[1] / "plugins/standard_web", sources / "standard_web",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            catalog = ctx.require(TOOLS)
            view = ToolView.combine(
                ctx.require(ALL_TOOLS)(), cast(ToolView, ctx.require(TOOL_SEARCH_TOOLS)),
            )
            menu = ToolMenu(catalog, ctx.require(BINDINGS),
                           catalog.execution(lambda binding, arguments: _allow()), _unexpected_reply,
                           view=view, presentation=ctx.require(TOOL_SEARCH_PRESENTATION)(view))
            names = {row["function"]["name"] for row in menu.schemas}
            assert {"web_fetch", "web_search"} <= names
            assert "example" not in names
            for name, arguments in (("web_fetch", {"url": "https://example.com"}),
                                    ("web_search", {"query": "weather"})):
                decoded = menu.decode(ModelToolCall('direct', name, arguments))
                (binding, _) = decoded.binding_id, decoded.arguments
                assert binding is not None
                async with open_tool(ctx.require(BINDINGS), binding) as tool:
                    assert await tool.prepare(arguments) == arguments
    finally:
        await host.terminate_all()
        log.close()


async def _allow() -> Mapping[str, object]:
    return {"allowed": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["removed", "changed"])
async def test_fixed_bindings_use_archived_schema_without_rebinding_current_provider(
    tmp_path, replacement
):
    sources = _sources(tmp_path)
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            bindings = Bindings(log, host._archive, host.open_binding)
            catalog = ctx.require(TOOLS)
            old = catalog.bind(ctx.require(ALL_TOOLS)().select("example"), bindings)
        await host.terminate_all()
        target = sources / "target"
        if replacement == "removed":
            shutil.rmtree(target)
            shutil.rmtree(sources / "prepare")
        else:
            module = target / "plugin.py"
            module.write_text(
                module.read_text()
                .replace("Example target A", "Example target B")
                .replace('"required": ["value"]', '"required": []')
            )
        restored = _manager(tmp_path, [sources], log)
        await restored.load_all()
        async with lease_runtime_snapshot(restored.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            bindings = Bindings(log, restored._archive, restored.open_binding)
            menu = ToolMenu(
                ctx.require(TOOLS),
                bindings,
                ctx.require(TOOLS).execution(lambda binding, arguments: _allow()),
                _unexpected_reply,
                fixed_bindings={"example": old},
            )
            schema = menu.schemas[0]["function"]
            assert schema["description"] == "Example target A"
            assert schema["parameters"]["required"] == ("value",)
            if replacement == "removed":
                with pytest.raises(PermissionError, match="获授 view"):
                    ctx.require(ALL_TOOLS)().select("example")
            else:
                current = ctx.require(ALL_TOOLS)().select("example")
                assert current.description["description"] == "Example target B"
                parameters = current.description["parameters"]
                assert isinstance(parameters, Mapping)
                assert parameters["required"] == ()
            decoded = menu.decode(ModelToolCall('old', 'example', {'value': 'old'}))
            (identity, arguments) = decoded.binding_id, decoded.arguments
            assert identity is not None
            assert identity == old
            result = await ctx.require(TOOLS).execution(
                lambda binding, final: _allow()
            ).execute("archived", identity, arguments)
            assert result.parts[0].value == "A:restore:old"
        await restored.terminate_all()
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_group_search_prioritizes_explicit_names_and_explains_risk_filter():
    """大型模糊命中组不能挤走明确工具名；风险过滤不静默隐藏原因。"""
    from plugins.tool_search.plugin import Query, SearchTool, _search

    def entry(name, description, risk="read-only"):
        return {"schema": {"type": "function", "function": {
            "name": name, "description": description, "parameters": {"type": "object"},
        }}, "risk": risk, "search_hint": None}
    groups: tuple[dict[str, object], ...] = tuple(dict[str, object](owner=f"decoy_{number}", tools=tuple(
        entry(f"unrelated_{number}_{index}", "computer browser tab playwright 打开网页 读取")
        for index in range(12)
    )) for number in range(12)) + ({"owner": "computer", "tools": (
        entry("computer", "Read or operate browser and desktop UI", "external-side-effect"),
        entry("status", "Read current status"),
    )},)
    for text in ("computer", "computer browser tab playwright 打开网页 读取"):
        matched = _search(groups, Query(query=text))
        assert matched[0]["owner"] == "computer"
        assert len(cast(tuple[object, ...], matched[0]["tools"])) == 2
    filtered = await SearchTool(groups).invoke("filtered", {
        "query": "computer browser tab playwright 打开网页 读取", "allowed_risk": ["read-only"],
    })
    value = json.loads(cast(str, filtered.parts[0].value))
    assert value["excluded_by_risk"] == [{"owner": "computer", "name": "computer", "risk": "external-side-effect"}]
    assert value["risk_tip"]
