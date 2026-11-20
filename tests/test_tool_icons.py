from __future__ import annotations

from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugin_composition import CompositionRoot, PluginRuntime
from agent.plugin_composition.tasks import TASKS, PluginTasks
from plugins.content.contract import CONTENT, Content
from plugins.ledger.contract import (
    BINDINGS,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    Bindings,
    Message,
    MessageCatalog,
    MessagePage,
    MessageWriters,
    Output,
    OwnerState,
    ToolCall,
)
from plugins.subagent.contract import AGENT_RUNNERS, AgentResult, AgentRunner
from plugins.tools.contract import (
    ALL_TOOLS,
    TOOL_DISPLAY,
    TOOLS,
    ToolDisplay,
    ToolRef,
    normalize_tool_icon,
)
from plugins.tools.plugin import apply as tools_apply
from plugins.ui.contract import (
    PLUGIN_UI,
    UI_SLOTS,
    MessageDisplayProviders,
    message_rows,
)
from plugins.ui.plugin_ui import PluginUiSlots
from plugins.ui.queries import LivePluginUiProvider
from plugins.userinformation.plugin import apply as userinfo_apply


class _MemoryBindings:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, object]] = {}

    def bind(self, _service: object, metadata: Mapping[str, object], *, contributors: object = ()) -> str:
        _ = contributors
        key = f"b-{len(self._records) + 1}"
        self._records[key] = dict(metadata)
        return key

    def describe(self, identity: str, _service: object) -> Mapping[str, object]:
        return self._records[identity]


def _runtime(tmp_path: Path, plugin_id: str) -> PluginRuntime:
    return PluginRuntime(
        plugin_id,
        plugin_id + ":1",
        tmp_path,
        tmp_path / plugin_id / "data",
        tmp_path / "workspace",
        {},
    )


async def _empty_follow(**_kw: object):
    if False:
        yield {}


def _catalog_stub() -> MessageCatalog:
    return cast(MessageCatalog, cast(object, SimpleNamespace(follow=_empty_follow)))


def _content_stub() -> Content:
    return cast(Content, cast(object, SimpleNamespace()))


def _writers_stub() -> MessageWriters:
    return cast(MessageWriters, cast(object, SimpleNamespace()))


def _owner_state_stub() -> OwnerState:
    store = SimpleNamespace(scan=lambda **_kw: [], read=lambda _k: None)
    return cast(
        OwnerState,
        cast(object, SimpleNamespace(open=lambda _ctx: store, open_scoped=lambda _ctx, _name: store)),
    )


def test_normalize_tool_icon_accepts_lucide_and_safe_svg() -> None:
    assert normalize_tool_icon(None) is None
    assert dict(normalize_tool_icon("terminal") or {}) == {"kind": "lucide", "name": "terminal"}

    svg = '<svg viewBox="0 0 24 24"><path d="M12 3v18"/><circle cx="12" cy="12" r="3" fill="currentColor"/></svg>'
    normalized = normalize_tool_icon(svg)
    assert normalized is not None
    assert normalized["kind"] == "svg"
    assert normalized["viewBox"] == "0 0 24 24"
    nodes = cast(tuple[Mapping[str, Any], ...], normalized["nodes"])
    assert len(nodes) == 2
    assert nodes[0]["tag"] == "path"
    assert nodes[1]["attrs"]["fill"] == "currentColor"


@pytest.mark.parametrize(
    "bad_icon",
    [
        "",
        "   ",
        "bad icon name!",
        "<svg><script>alert(1)</script></svg>",
        '<svg viewBox="0 0 24 24"><path d="M0 0" onclick="alert(1)"/></svg>',
        '<svg viewBox="0 0 24 24"><path d="M0 0" stroke="#ff0000"/></svg>',
        '<svg viewBox="0 0 24 24"><path d="M0 0" fill="red"/></svg>',
        '<!DOCTYPE svg><svg viewBox="0 0 24 24"><path d="M0 0"/></svg>',
        "<svg viewBox='0 0 24 24'></svg>",
        {"kind": "unknown"},
    ],
)
def test_normalize_tool_icon_rejects_unsafe_or_invalid(bad_icon: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        _ = normalize_tool_icon(bad_icon)


@pytest.mark.asyncio
async def test_tool_catalog_icon_resolution_and_binding_compatibility(tmp_path: Path) -> None:
    """组级图标、单工具覆写、不污染 ToolRef.description，且归档 ToolCall 可投影到前端。"""
    root = CompositionRoot("tool-icons")
    plugin_tasks = PluginTasks()
    _ = await root.context.provide(TASKS, plugin_tasks)
    bindings = _MemoryBindings()
    _ = await root.context.provide(BINDINGS, cast(Bindings, bindings))
    _ = await root.context.provide(CONTENT, _content_stub())
    _ = await root.context.provide(MESSAGE_CATALOG, _catalog_stub())
    _ = await root.context.provide(MESSAGE_WRITERS, _writers_stub())
    _ = await root.context.provide(OWNER_STATE, _owner_state_stub())

    _ = await root.mount(
        tools_apply,
        name="tools",
        inject=(CONTENT, BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, TASKS),
        runtime=_runtime(tmp_path, "tools"),
    )
    catalog = root.context.require(TOOLS)
    read_display = root.context.require(TOOL_DISPLAY)

    @asynccontextmanager
    async def open_target(_state: Mapping[str, object]):
        yield cast(Any, SimpleNamespace(idempotent=True))

    refs: dict[str, ToolRef] = {}

    async def sample_plugin(ctx: Any) -> None:
        tools = ctx.require(TOOLS)
        _ = await tools.declare_group(ctx, description="示例插件", icon="pen-tool")
        refs["group_default"] = await tools.register(
            ctx,
            name="draw_rect",
            description="画矩形",
            parameters={"type": "object", "properties": {}},
            open=open_target,
        )
        refs["override"] = await tools.register(
            ctx,
            name="preview_html",
            description="预览网页",
            parameters={"type": "object", "properties": {}},
            open=open_target,
            icon="layout-template",
        )

    plugin_fiber = await root.mount(
        sample_plugin,
        name="sample_plugin",
        inject=(TOOLS,),
        runtime=_runtime(tmp_path, "sample_plugin"),
    )

    try:
        # 1. icon 不写入 ToolRef.description，不污染模型 schema 与历史 binding 描述核对
        assert "icon" not in refs["group_default"].description
        assert "icon" not in refs["override"].description

        b_default = await catalog.bind(refs["group_default"], cast(Bindings, bindings))
        b_override = await catalog.bind(refs["override"], cast(Bindings, bindings))

        # 2. TOOL_DISPLAY 正确解析组级默认与单工具覆写
        d1 = read_display(b_default)
        d2 = read_display(b_override)
        assert d1 == ToolDisplay("draw_rect", "sample_plugin", {"kind": "lucide", "name": "pen-tool"})
        assert d2 == ToolDisplay("preview_html", "sample_plugin", {"kind": "lucide", "name": "layout-template"})

        # 3. _tool_call 投影包含 icon 字段
        msg = Message(
            message_id="m1",
            session_id="s1",
            seq=1,
            recorded_at=datetime.now(timezone.utc),
            author="assistant",
            source="akashic",
            body=Output(parts=(ToolCall(binding_id=b_override, arguments={}),), finish="continue"),
        )
        page = MessagePage(
            messages=(msg,),
            attachments={"m1": ()},
            bindings={b_override: bindings.describe(b_override, TOOLS)},
            through_seq=1,
            has_more=False,
        )
        rows = message_rows(page, providers=MessageDisplayProviders(tool_display=read_display))
        body = cast(Mapping[str, Any], rows[0]["body"])
        part = cast(Mapping[str, Any], cast(list[Any], body["parts"])[0])
        assert part["kind"] == "tool_call"
        assert part["owner"] == "sample_plugin"
        assert part["icon"] == {"kind": "lucide", "name": "layout-template"}
    finally:
        await plugin_fiber.dispose()

    # 4. 插件卸载后历史 binding 读取平滑回退 icon=None（不报错）
    d_unloaded = read_display(b_override)
    assert d_unloaded == ToolDisplay("preview_html", "sample_plugin", None)
    await plugin_tasks.close()
    await root.dispose()


@pytest.mark.asyncio
async def test_subagent_runner_forwards_icon_to_delegate_tools(tmp_path: Path) -> None:
    """AgentRunner 携带的 SVG icon 经 subagent 注册链透传给 delegate_* 工具。"""
    from plugins.subagent.runners import Runners

    root = CompositionRoot("subagent-runner-icon")
    plugin_tasks = PluginTasks()
    _ = await root.context.provide(TASKS, plugin_tasks)
    bindings = _MemoryBindings()
    _ = await root.context.provide(BINDINGS, cast(Bindings, bindings))
    _ = await root.context.provide(CONTENT, _content_stub())
    _ = await root.context.provide(MESSAGE_CATALOG, _catalog_stub())
    _ = await root.context.provide(MESSAGE_WRITERS, _writers_stub())
    _ = await root.context.provide(OWNER_STATE, _owner_state_stub())

    _ = await root.mount(
        tools_apply,
        name="tools",
        inject=(CONTENT, BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, TASKS),
        runtime=_runtime(tmp_path, "tools"),
    )
    catalog = root.context.require(TOOLS)
    read_display = root.context.require(TOOL_DISPLAY)

    runners = Runners(root.context, max_active=2, result_chars=4000)
    _ = await root.context.provide(AGENT_RUNNERS, runners)

    codex_svg = '<svg viewBox="0 0 24 24"><path d="M12 2.5 20 7v10l-8 4.5L4 17V7Z"/></svg>'

    async def runner_plugin(ctx: Any) -> None:
        async def run(_task: object, _emit: object) -> AgentResult:
            return AgentResult("completed", "ok")

        _ = await ctx.require(AGENT_RUNNERS).register(
            ctx,
            AgentRunner("codex", "Codex 代理", True, "read-only", run, icon=codex_svg),
        )

    _ = await root.mount(
        runner_plugin,
        name="subagent_codex",
        inject=(AGENT_RUNNERS, TOOLS),
        runtime=_runtime(tmp_path, "subagent_codex"),
    )
    try:
        all_tools = root.context.require(ALL_TOOLS)()
        ref = all_tools.select("delegate_codex")
        b_id = await catalog.bind(ref, cast(Bindings, bindings))
        display = read_display(b_id)
        assert display.owner == "subagent_codex"
        assert display.icon is not None
        assert display.icon["kind"] == "svg"
        nodes = cast(tuple[Mapping[str, Any], ...], display.icon["nodes"])
        assert nodes[0]["tag"] == "path"
    finally:
        await plugin_tasks.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_userinformation_registers_static_group_icon_and_tool_icon_slot(tmp_path: Path) -> None:
    """userinformation 同时声明后端组级图标与前端 tool.icon 动态槽位。"""
    root = CompositionRoot("userinfo-icon")
    plugin_tasks = PluginTasks()
    _ = await root.context.provide(TASKS, plugin_tasks)
    bindings = _MemoryBindings()
    _ = await root.context.provide(BINDINGS, cast(Bindings, bindings))
    _ = await root.context.provide(CONTENT, _content_stub())
    _ = await root.context.provide(MESSAGE_CATALOG, _catalog_stub())
    _ = await root.context.provide(MESSAGE_WRITERS, _writers_stub())
    _ = await root.context.provide(OWNER_STATE, _owner_state_stub())

    _ = await root.mount(
        tools_apply,
        name="tools",
        inject=(CONTENT, BINDINGS, MESSAGE_CATALOG, MESSAGE_WRITERS, OWNER_STATE, TASKS),
        runtime=_runtime(tmp_path, "tools"),
    )

    async def ui_apply(ctx: Any) -> None:
        slots = PluginUiSlots(ctx)
        _ = await ctx.provide(UI_SLOTS, slots)
        _ = await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, slots))

    _ = await root.mount(ui_apply, name="ui", runtime=_runtime(tmp_path, "ui"))

    plugin_dir = Path(__file__).resolve().parents[1] / "plugins" / "userinformation"
    ui_dir = tmp_path / "userinformation"
    ui_dir.mkdir(parents=True, exist_ok=True)
    _ = shutil.copy2(plugin_dir / "plugin_ui.js", ui_dir / "plugin_ui.js")
    rt = PluginRuntime("userinformation", "userinformation:1", ui_dir, tmp_path / "data", tmp_path / "ws", {})

    _ = await root.mount(userinfo_apply, name="userinformation", inject=(TOOLS,), runtime=rt)
    try:
        catalog = root.context.require(TOOLS)
        all_tools = root.context.require(ALL_TOOLS)()
        ref = all_tools.select("user_information_query")
        b_id = await catalog.bind(ref, cast(Bindings, bindings))
        display = root.context.require(TOOL_DISPLAY)(b_id)
        assert display.icon == {"kind": "lucide", "name": "user-round-search"}

        ui_catalog = await cast(LivePluginUiProvider, root.context.require(PLUGIN_UI)).catalog()
        raw_items = cast(list[Mapping[str, Any]], ui_catalog["items"])
        items = {str(item["id"]): item for item in raw_items}
        assert items["userinformation"]["slots"] == ["tool.icon"]
    finally:
        await cast(LivePluginUiProvider, root.context.require(PLUGIN_UI)).aclose()
        await plugin_tasks.close()
        await root.dispose()
