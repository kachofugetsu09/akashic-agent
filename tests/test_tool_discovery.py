from pathlib import Path
import shutil
from collections.abc import Mapping
from typing import cast

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import lease_runtime_snapshot
from bus.event_bus import EventBus
from plugins.tools.menu import ToolMenu
from plugins.tools.api import MessageReply, result_message_id
from plugins.tools.plugin import TOOLS, open_tool
from session.log import MessageLog
from session.message import CallRef, ContentPart, ContentReferences, Input, Output, ToolCall, ToolResult
from tests.test_tool_bindings import write_plugins


def _candidate(metadata: Mapping[str, object], name: str) -> str:
    candidates = cast(Mapping[str, Mapping[str, object]], metadata["candidates"])
    binding_id = candidates[name]["binding_id"]
    assert isinstance(binding_id, str)
    return binding_id


async def _reject_authorize(binding_id: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
    raise AssertionError(f"discovery test unexpectedly authorized {binding_id}: {arguments}")


def _unexpected_reply(call_ref: CallRef) -> MessageReply:
    raise AssertionError(f"discovery test unexpectedly created a reply for {call_ref}")


def _manager(tmp_path, sources, log):
    return PluginManager(
        plugin_dirs=sources,
        event_bus=EventBus(),
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
        message_log=log,
    )


_ALWAYS_ON = tuple(f"always_on_{index}" for index in range(9))


def _add_always_on_tools(sources, names=_ALWAYS_ON):
    """在真实 target 插件中增加固定工具，保持 discovery 走生产注册路径。"""
    target = sources / "target/plugin.py"
    source = target.read_text()
    marker = "        open=open_target,\n    )\n"
    registrations = "".join(
        "    await ctx.require(inject[0]).register(\n"
        f"        ctx, name={name!r}, description={name!r},\n"
        '        parameters={"type": "object"}, open=open_target, always_on=True,\n'
        "    )\n"
        for name in names
    )
    assert source.count(marker) == 1
    target.write_text(source.replace(marker, marker + registrations))


def _discovery_sources(tmp_path, *, always_on=()):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    _add_always_on_tools(sources, always_on)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    return sources


@pytest.mark.asyncio
async def test_discovery_archive_and_log_restore_keep_exact_candidates_without_full_fleet(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            menu = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("example", "tool_search"),
                            reader=log.reader("s"), source="chat")
            assert [item["function"]["name"] for item in menu.schemas] == ["tool_search"]
            search_id = menu.bind("tool_search")
            original_id = _candidate(bindings.describe(search_id, TOOLS), "example")
            restricted = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("tool_search",),
                                  reader=log.reader("s"), source="chat")
            _ = restricted.schemas
            assert bindings.describe(restricted.bind("tool_search"), TOOLS)["candidates"] == {}
        # 归档搜索本身没有注册 example，仍能从自己的固定目录选中它。
        async with bindings.open(search_id, TOOLS) as (archived, metadata):
            assert [item["name"] for item in archived.descriptions()] == ["tool_search"]
            async with archived.open(metadata) as tool:
                arguments = await tool.prepare({"query": "select:example", "allowed_risk": ["read-write"]})
                result = await tool.invoke("search", arguments)
                assert result.parts[-1].value == (original_id,)
        log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append("u", Input(()))
        log.writer("s", author="assistant", source="chat", body_types=(Output,), content={},
                   check_call=menu.check_call).append("search", Output((ToolCall(search_id, {}),), "continue"))
        ref = CallRef("search", 0)
        log.writer("s", author="tool", source="chat", body_types=(ToolResult,), call_ref=ref,
                   content={"tool.selection": lambda part: menu.check_selection(ref, part)}).append(
                       "selected", ToolResult(ref, "success", (result.parts[-1],)))
        await host.terminate_all()
        # 新安装版本的同名工具改变，尚未结束的日志选择仍固定旧版本。
        target = sources / "target/plugin.py"
        target.write_text(target.read_text().replace("target A", "target B").replace('"A:"', '"B:"'))
        host = _manager(tmp_path, [sources], log)
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            current = catalog.bind("example", bindings)
            assert current != original_id
            restored = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("example", "tool_search"),
                                reader=log.reader("s"), source="chat")
            assert {item["function"]["name"] for item in restored.schemas} == {"example", "tool_search"}
            assert restored.bind("example") == original_id
            closed = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("tool_search",),
                              reader=log.reader("s"), source="chat")
            assert [item["function"]["name"] for item in closed.schemas] == ["tool_search"]
        async with open_tool(bindings, original_id) as tool:
            result = await tool.invoke("old", await tool.prepare({"value": "value"}))
            assert result.parts[0].value == "A:restore:value"
        with pytest.raises(ValueError, match="原调用"):
            menu.check_selection(ref, ContentPart("tool.selection", (search_id,)))
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_discovery_menu_keeps_fixed_tools_above_legacy_default_and_selects_candidate(tmp_path):
    sources = _discovery_sources(tmp_path, always_on=_ALWAYS_ON)
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    names = (*_ALWAYS_ON, "example", "tool_search")
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            menu = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=names,
                            reader=log.reader("s"), source="chat")
            assert [item["function"]["name"] for item in menu.schemas] == [*_ALWAYS_ON, "tool_search"]
            search_id = menu.bind("tool_search")
            candidate = _candidate(bindings.describe(search_id, TOOLS), "example")
            async with open_tool(bindings, search_id) as search:
                result = await search.invoke(
                    "search", await search.prepare({"query": "select:example", "allowed_risk": ["read-write"]})
                )
            assert result.parts[-1].value == (candidate,)
        log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append(
            "u", Input(())
        )
        log.writer("s", author="assistant", source="chat", body_types=(Output,), content={},
                   check_call=menu.check_call).append(
                       "search", Output((ToolCall(search_id, {}),), "continue")
                   )
        ref = CallRef("search", 0)
        log.writer("s", author="tool", source="chat", body_types=(ToolResult,), call_ref=ref,
                   content={"tool.selection": lambda part: menu.check_selection(ref, part)}).append(
                       "selected", ToolResult(ref, "success", (result.parts[-1],))
                   )
        assert [item["function"]["name"] for item in menu.schemas] == [*_ALWAYS_ON, "tool_search", "example"]
        assert menu.bind("example") == candidate
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_tool_search_invalid_call_returns_error_then_corrected_call_succeeds_in_same_output(tmp_path):
    sources = _discovery_sources(tmp_path)
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)

            async def authorize(binding_id, arguments):
                return {"policy": "test"}

            execution = catalog.execution(authorize)
            menu = ToolMenu(catalog, bindings, execution, _unexpected_reply,
                            names=("example", "tool_search"), reader=log.reader("s"), source="chat")
            _ = menu.schemas
            search_id = menu.bind("tool_search")
            output = log.writer("s", author="assistant", source="chat", body_types=(Output,),
                                content={}, check_call=menu.check_call)
            output.append("searches", Output((
                ToolCall(search_id, {}),
                ToolCall(search_id, {"query": "select:example"}),
            ), "continue"))

            def reply(index):
                ref = CallRef("searches", index)
                writer = log.writer("s", author="tool", source="chat", body_types=(ToolResult,),
                                    content={"text": lambda part: ContentReferences(),
                                             "tool.selection": lambda part: menu.check_selection(ref, part)},
                                    call_ref=ref)
                return MessageReply(result_message_id(ref), ref, log.reader("s"), writer, lambda: None)

            invalid = await execution.execute_call(reply(0))
            corrected = await execution.execute_call(reply(1))
            assert invalid.outcome == "error"
            assert isinstance(invalid.parts[0].value, str)
            assert "query" in invalid.parts[0].value
            assert corrected.outcome == "success"
            assert corrected.parts[-1].value == (_candidate(bindings.describe(search_id, TOOLS), "example"),)
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_discovery_menu_capacity_counts_fixed_tools_and_rejects_evicted_candidate(tmp_path):
    sources = _discovery_sources(tmp_path, always_on=_ALWAYS_ON)
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    names = (*_ALWAYS_ON, "example", "tool_search")
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            source_menu = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=names,
                                   reader=log.reader("s"), source="chat")
            _ = source_menu.schemas
            search_id = source_menu.bind("tool_search")
            candidate = _candidate(bindings.describe(search_id, TOOLS), "example")
            async with open_tool(bindings, search_id) as search:
                result = await search.invoke(
                    "search", await search.prepare({"query": "select:example", "allowed_risk": ["read-write"]})
                )
        log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append(
            "u", Input(())
        )
        log.writer("s", author="assistant", source="chat", body_types=(Output,), content={},
                   check_call=source_menu.check_call).append(
                       "search", Output((ToolCall(search_id, {}),), "continue")
                   )
        ref = CallRef("search", 0)
        log.writer("s", author="tool", source="chat", body_types=(ToolResult,), call_ref=ref,
                   content={"tool.selection": lambda part: source_menu.check_selection(ref, part)}).append(
                       "selected", ToolResult(ref, "success", (result.parts[-1],))
                   )
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            fits = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=names,
                            reader=log.reader("s"), source="chat", limit=len(_ALWAYS_ON) + 2)
            assert [item["function"]["name"] for item in fits.schemas] == [*_ALWAYS_ON, "tool_search", "example"]
            assert fits.bind("example") == candidate
            full = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=names,
                            reader=log.reader("s"), source="chat", limit=len(_ALWAYS_ON) + 1)
            assert [item["function"]["name"] for item in full.schemas] == [*_ALWAYS_ON, "tool_search"]
            with pytest.raises(PermissionError, match="未获授工具"):
                full.bind("example")
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_discovery_menu_rejects_fixed_tools_over_model_capacity(tmp_path):
    sources = _discovery_sources(tmp_path, always_on=_ALWAYS_ON)
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    names = (*_ALWAYS_ON, "example", "tool_search")
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            menu = ToolMenu(
                catalog, bindings, catalog.execution(_reject_authorize), _unexpected_reply,
                names=names, reader=log.reader("s"), source="chat", limit=len(_ALWAYS_ON)
            )
            with pytest.raises(ValueError, match=r"required=10 limit=9"):
                _ = menu.schemas
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["abandon", "complete"])
async def test_late_discovery_result_is_saved_but_does_not_unlock_next_segment(tmp_path, boundary):
    from session.message import Control

    sources = tmp_path / "plugins"
    write_plugins(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            menu = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("example", "tool_search"),
                            reader=log.reader("s"), source="chat")
            _ = menu.schemas
            search_id = menu.bind("tool_search")
            candidate = _candidate(bindings.describe(search_id, TOOLS), "example")
            def writer(kind, **kwargs):
                return log.writer("s", author="test", source="chat", body_types=(kind,), **kwargs)
            writer(Input, content={}).append("u1", Input(()))
            writer(Output, content={}, check_call=menu.check_call).append(
                "call", Output((ToolCall(search_id, {}),), "continue"))
            if boundary == "abandon":
                writer(Control, content={}).append("closed", Control("abandon", 1))
            else:
                writer(Output, content={}).append("closed", Output((), "complete"))
            writer(Input, content={}).append("u2", Input(()))
            ref = CallRef("call", 0)
            writer(ToolResult, call_ref=ref,
                   content={"tool.selection": lambda part: menu.check_selection(ref, part)}).append(
                       "late", ToolResult(ref, "success", (ContentPart("tool.selection", (candidate,)),)))
            assert [item["function"]["name"] for item in menu.schemas] == ["tool_search"]
            assert log.reader("s").get("late") is not None
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_search_requirement_cannot_be_bypassed_by_configuration(tmp_path):
    from contextlib import asynccontextmanager
    from plugins.tools.menu import check_menu

    sources = tmp_path / "plugins"
    write_plugins(sources)
    target = sources / "target/plugin.py"
    target.write_text(target.read_text().replace("open=open_target,", "open=open_target, requires_search=True,"))
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            with pytest.raises(ValueError, match="缺少发现能力"):
                check_menu(catalog, ("example",))
            @asynccontextmanager
            async def unused(candidates):
                raise AssertionError("invalid registration must not open a tool")
                yield
            with pytest.raises(ValueError, match="同时常驻"):
                await catalog.register(catalog._ctx, name="contradiction",
                                       description="invalid", parameters={"type": "object"},
                                       open=unused, always_on=True, requires_search=True)
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_fixed_menu_uses_original_directory_after_targets_are_uninstalled(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = _manager(tmp_path, [sources], log)
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            original = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("example", "tool_search"),
                                reader=log.reader("s"), source="chat")
            _ = original.schemas
            search = original.bind("tool_search")
            candidate = _candidate(bindings.describe(search, TOOLS), "example")
            identities = {"example": candidate, "tool_search": search}
            with pytest.raises(ValueError, match="配置入口"):
                catalog.bind("example", bindings, configuration={})
            assert catalog.bind("example", bindings, configuration=None) == candidate
            restricted = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=("tool_search",),
                                  reader=log.reader("s"), source="chat")
            _ = restricted.schemas
            empty_search = restricted.bind("tool_search")
        await host.terminate_all()
        for name in ("target", "prepare", "tool_search"):
            shutil.rmtree(sources / name)
        host = _manager(tmp_path, [sources], log)
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            execution = catalog.execution(_reject_authorize)
            assert catalog.descriptions() == ()
            with pytest.raises(ValueError, match="完整对应"):
                ToolMenu(catalog, bindings, execution, _unexpected_reply, names=tuple(identities), reader=log.reader("s"),
                         source="chat", fixed_bindings={"example": candidate})
            with pytest.raises(ValueError, match="候选与允许目录"):
                ToolMenu(catalog, bindings, execution, _unexpected_reply, names=tuple(identities), reader=log.reader("s"),
                         source="chat", fixed_bindings={**identities, "tool_search": empty_search})
            log.writer("conflict", author="assistant", source="chat", body_types=(Output,), content={},
                       check_call=lambda call: None).append("foreign-search", Output((ToolCall(empty_search, {}),), "continue"))
            conflict = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=tuple(identities),
                                reader=log.reader("conflict"), source="chat", fixed_bindings=identities)
            with pytest.raises(PermissionError, match="未闭合调用"):
                _ = conflict.schemas
            fixed = ToolMenu(catalog, bindings, execution, _unexpected_reply, names=tuple(identities),
                             reader=log.reader("s"), source="chat", fixed_bindings=identities)
            identities.clear()  # 调用者后改目录不改变已建立的程序。
            assert [row["function"]["name"] for row in fixed.schemas] == ["tool_search"]
            assert fixed.bind("tool_search") == search
            async with open_tool(bindings, search) as target:
                selected = await target.invoke("search", await target.prepare({
                    "query": "select:example", "allowed_risk": ["read-write"]}))
            log.writer("s", author="user", source="chat", body_types=(Input,), content={}).append("u", Input(()))
            log.writer("s", author="assistant", source="chat", body_types=(Output,), content={},
                       check_call=fixed.check_call).append("search", Output((ToolCall(search, {}),), "continue"))
            ref = CallRef("search", 0)
            log.writer("s", author="tool", source="chat", body_types=(ToolResult,), call_ref=ref,
                       content={"tool.selection": lambda part: fixed.check_selection(ref, part)}).append(
                           "selected", ToolResult(ref, "success", (selected.parts[-1],)))
            assert {row["function"]["name"] for row in fixed.schemas} == {"example", "tool_search"}
            assert fixed.bind("example") == candidate
            log.writer("s", author="assistant", source="chat", body_types=(Output,), content={},
                       check_call=fixed.check_call).append("call", Output((ToolCall(candidate, {"value": "x"}),), "continue"))
            log.writer("s", author="assistant", source="chat", body_types=(Output,), content={}).append(
                "complete", Output((), "complete"))
            # 闭段预载也只能使用固定目录，不能重新查当前插件。
            assert {row["function"]["name"] for row in fixed.schemas} == {"example", "tool_search"}
            assert fixed.bind("example") == candidate
            async with open_tool(bindings, candidate) as target:
                result = await target.invoke("fixed", await target.prepare({"value": "input"}))
                assert result.parts[0].value == "A:restore:input"
    finally:
        await host.terminate_all()
        log.close()


@pytest.mark.asyncio
async def test_risk_filtered_computer_reports_why_and_exact_name_can_be_selected():
    import json
    from plugins.tool_search.plugin import SearchTool
    candidates = {f"unrelated_{number}": {"binding_id": f"other-{number}", "tool": {
        "name": f"unrelated_{number}", "risk": "read-only", "search_hint": "打开网页读取 computer browser tab playwright",
        "description": "打开网页读取 computer browser tab playwright",
    }} for number in range(12)}
    candidates["computer"] = {"binding_id": "computer-binding", "tool": {
        "name": "computer", "risk": "external-side-effect", "search_hint": None,
        "description": "Read or operate browser and desktop UI",
    }}
    tool = SearchTool(candidates)
    mixed = {"query": "computer browser tab playwright 打开网页 读取", "top_k": 10}
    ranked = await tool.invoke("mixed", await tool.prepare(mixed))
    assert json.loads(ranked.parts[0].value)["selected"][0] == "computer"
    filtered = await tool.invoke("filtered", await tool.prepare({
        **mixed, "allowed_risk": ["read-only", "read-write"],
    }))
    payload = json.loads(filtered.parts[0].value)
    assert "computer" not in payload["selected"]
    assert payload["excluded_by_risk"] == [{"name": "computer", "risk": "external-side-effect"}]
    assert "computer-binding" not in filtered.parts[-1].value
    selected = await tool.invoke("exact", await tool.prepare({"query": "select:computer"}))
    assert selected.parts[-1].value == ("computer-binding",)
