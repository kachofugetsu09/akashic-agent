from pathlib import Path
import json
import shutil
from collections.abc import Mapping

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
            "from plugins.tools.execution import Result\nfrom plugins.tools.api import InvalidArguments",
        )
        .replace(
            'raise ValueError("value must be text")',
            'raise InvalidArguments("value must be text")',
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
                ctx.require(ALL_TOOLS)(), ctx.require(TOOL_SEARCH_TOOLS)
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

            search_binding, search_arguments = menu.decode(
                ModelToolCall("search", "tool_search", {"query": "example"})
            )
            async with open_tool(bindings, search_binding) as search:
                result = await search.invoke(
                    "search",
                    await search.prepare(search_arguments),
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

            binding, arguments = menu.decode(
                ModelToolCall(
                    "call",
                    "tool_call",
                    {"name": "example", "arguments": {"value": " ok "}},
                )
            )
            executed = await catalog.execution(
                lambda identity, final: _allow()
            ).execute("valid", binding, arguments)
            assert executed.parts[0].value == "A:restore: ok"
            invalid = await catalog.execution(
                lambda identity, final: _allow()
            ).execute("invalid", binding, {"value": 7})
            assert invalid.outcome == "error"
            assert invalid.parts[0].value == "value must be text"
            with pytest.raises(PermissionError, match="获授 view"):
                menu.decode(
                    ModelToolCall(
                        "unknown",
                        "tool_call",
                        {"name": "missing", "arguments": {}},
                    )
                )
            narrow = ToolView.combine(
                catalog.view(view.select("example")),
                ctx.require(TOOL_SEARCH_TOOLS),
            )
            narrow_menu = ToolMenu(
                catalog,
                bindings,
                catalog.execution(lambda binding, arguments: _allow()),
                _unexpected_reply,
                view=narrow,
                presentation=ctx.require(TOOL_SEARCH_PRESENTATION)(narrow),
            )
            with pytest.raises(PermissionError, match="获授 view"):
                narrow_menu.decode(
                    ModelToolCall(
                        "not-awarded",
                        "tool_call",
                        {"name": "example_status", "arguments": {"value": "ok"}},
                    )
                )
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
            identity, arguments = menu.decode(
                ModelToolCall("old", "example", {"value": "old"})
            )
            assert identity == old
            result = await ctx.require(TOOLS).execution(
                lambda binding, final: _allow()
            ).execute("archived", identity, arguments)
            assert result.parts[0].value == "A:restore:old"
        await restored.terminate_all()
    finally:
        await host.terminate_all()
        log.close()
