from pathlib import Path
import shutil

import pytest

from agent.plugin_composition.bindings import Bindings
from agent.plugins.snapshot import lease_runtime_snapshot
from plugins.tools.menu import ToolMenu
from plugins.tools.plugin import TOOLS, open_tool
from session.log import MessageLog
from session.message import CallRef, ContentPart, Input, Output, ToolCall, ToolResult
from tests.test_tool_bindings import manager, write_plugins


@pytest.mark.asyncio
async def test_discovery_archive_and_log_restore_keep_exact_candidates_without_full_fleet(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [sources])
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            menu = ToolMenu(catalog, bindings, None, None, names=("example", "tool_search"),
                            reader=log.reader("s"), source="chat")
            assert [item["function"]["name"] for item in menu.schemas] == ["tool_search"]
            search_id = menu.bind("tool_search")
            original_id = bindings.describe(search_id, TOOLS)["candidates"]["example"]["binding_id"]
            restricted = ToolMenu(catalog, bindings, None, None, names=("tool_search",),
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
        host = manager(tmp_path, [sources])
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            current = catalog.bind("example", bindings)
            assert current != original_id
            restored = ToolMenu(catalog, bindings, None, None, names=("example", "tool_search"),
                                reader=log.reader("s"), source="chat")
            assert {item["function"]["name"] for item in restored.schemas} == {"example", "tool_search"}
            assert restored.bind("example") == original_id
            closed = ToolMenu(catalog, bindings, None, None, names=("tool_search",),
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
@pytest.mark.parametrize("boundary", ["abandon", "complete"])
async def test_late_discovery_result_is_saved_but_does_not_unlock_next_segment(tmp_path, boundary):
    from session.message import Control

    sources = tmp_path / "plugins"
    write_plugins(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [sources])
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            menu = ToolMenu(catalog, bindings, None, None, names=("example", "tool_search"),
                            reader=log.reader("s"), source="chat")
            _ = menu.schemas
            search_id = menu.bind("tool_search")
            candidate = bindings.describe(search_id, TOOLS)["candidates"]["example"]["binding_id"]
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
    host = manager(tmp_path, [sources])
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
                await catalog.register(snapshot.composition_root.context, name="contradiction",
                                       description="invalid", parameters={"type": "object"},
                                       open=unused, always_on=True, requires_search=True)
    finally:
        await host.terminate_all()


@pytest.mark.asyncio
async def test_fixed_menu_uses_original_directory_after_targets_are_uninstalled(tmp_path):
    sources = tmp_path / "plugins"
    write_plugins(sources)
    shutil.copytree(Path(__file__).parents[1] / "plugins/tool_search", sources / "tool_search",
                    ignore=shutil.ignore_patterns("__pycache__"))
    log = MessageLog(tmp_path / "sessions.db")
    host = manager(tmp_path, [sources])
    try:
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            original = ToolMenu(catalog, bindings, None, None, names=("example", "tool_search"),
                                reader=log.reader("s"), source="chat")
            _ = original.schemas
            search = original.bind("tool_search")
            candidate = bindings.describe(search, TOOLS)["candidates"]["example"]["binding_id"]
            identities = {"example": candidate, "tool_search": search}
            with pytest.raises(ValueError, match="配置入口"):
                catalog.bind("example", bindings, configuration={})
            assert catalog.bind("example", bindings, configuration=None) == candidate
            restricted = ToolMenu(catalog, bindings, None, None, names=("tool_search",),
                                  reader=log.reader("s"), source="chat")
            _ = restricted.schemas
            empty_search = restricted.bind("tool_search")
        await host.terminate_all()
        for name in ("target", "prepare", "tool_search"):
            shutil.rmtree(sources / name)
        host = manager(tmp_path, [sources])
        await host.load_all()
        bindings = Bindings(log, host._archive, host.open_binding)
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            catalog = snapshot.composition_root.context.require(TOOLS)
            assert catalog.descriptions() == ()
            with pytest.raises(ValueError, match="完整对应"):
                ToolMenu(catalog, bindings, None, None, names=tuple(identities), reader=log.reader("s"),
                         source="chat", fixed_bindings={"example": candidate})
            with pytest.raises(ValueError, match="候选与允许目录"):
                ToolMenu(catalog, bindings, None, None, names=tuple(identities), reader=log.reader("s"),
                         source="chat", fixed_bindings={**identities, "tool_search": empty_search})
            log.writer("conflict", author="assistant", source="chat", body_types=(Output,), content={},
                       check_call=lambda call: None).append("foreign-search", Output((ToolCall(empty_search, {}),), "continue"))
            conflict = ToolMenu(catalog, bindings, None, None, names=tuple(identities),
                                reader=log.reader("conflict"), source="chat", fixed_bindings=identities)
            with pytest.raises(PermissionError, match="未闭合调用"):
                _ = conflict.schemas
            fixed = ToolMenu(catalog, bindings, None, None, names=tuple(identities),
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
