"""O/C1: bounded UI work is isolated by owner and drains its real Root scope."""
import asyncio
from pathlib import Path
import threading
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, Context, PluginRuntime
from plugins.ui.contract import PluginUiDefinition, UI_SLOTS
from plugins.ui.contract import PluginUiQueryTimeout
from plugins.ui.contract import PLUGIN_UI
from plugins.ui.queries import LivePluginUiProvider
from plugins.ui.plugin_ui import PluginUiSlots


@pytest.mark.asyncio
async def test_slow_ui_owner_leaves_capacity_and_queued_timeout_never_runs(tmp_path, monkeypatch):
    root = CompositionRoot("ui-isolation")
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    lock = threading.Lock()
    calls = []

    async def slots(ctx):
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    await root.mount(slots, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))
    (tmp_path / "panel.js").write_text("export default {};\n")

    async def slow(ctx):
        def query(method, payload, *, session_id, turn_id):
            with lock:
                calls.append(payload["id"])
                if len(calls) >= 4:
                    loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "test did not release the physical query"
            return {"owner": "slow"}
        await ctx.require(UI_SLOTS).register_plugin_ui(ctx, PluginUiDefinition("panel.js"), query=query)

    async def fast(ctx):
        def query(method, payload, *, session_id, turn_id):
            return {"owner": "fast"}
        await ctx.require(UI_SLOTS).register_plugin_ui(ctx, PluginUiDefinition("panel.js"), query=query)

    for name, apply in (("slow", slow), ("fast", fast)):
        await root.mount(apply, name=name, inject=(UI_SLOTS,),
                         runtime=PluginRuntime(name, "ui", tmp_path, tmp_path, tmp_path, {}))
    catalog = await provider.catalog()
    items = cast(list[dict[str, object]], catalog["items"])
    revisions = {str(item["id"]): str(item["revision"]) for item in items}

    async def query(owner, identity):
        return await provider.query(owner, revisions[owner], "read", {"id": identity},
                                    session_id=None, turn_id=None)

    jobs = [asyncio.create_task(query("slow", i)) for i in range(8)]
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert await asyncio.wait_for(query("fast", "fast"), 1) == {"owner": "fast"}
        jobs[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await jobs[0]
        monkeypatch.setattr("plugins.ui.queries.PLUGIN_UI_QUERY_TIMEOUT_SECONDS", 0.05)
        with pytest.raises(PluginUiQueryTimeout):
            await query("slow", "withdrawn")
        closing = asyncio.create_task(provider.aclose())
        tick = asyncio.Event()
        loop.call_soon(tick.set)
        await tick.wait()
        assert not closing.done()
        release.set()
        await asyncio.gather(*jobs[1:])
        await closing
        assert "withdrawn" not in calls
        assert sorted(calls) == list(range(8))
    finally:
        release.set()
        await asyncio.gather(*jobs, return_exceptions=True)
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_plugin_ui_slot_invalidation_follows_and_validates_owner(tmp_path):
    root = CompositionRoot("ui-slot-invalidation")
    emitted: list[dict[str, object]] = []
    got_first = asyncio.Event()
    got_second = asyncio.Event()

    async def slots(ctx):
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    await root.mount(slots, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))
    ui_slots = root.context.require(UI_SLOTS)
    (tmp_path / "panel.js").write_text("export default {};\n")

    captured: dict[str, object] = {}

    async def groups_owner(ctx):
        captured["groups"] = ctx
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx,
            PluginUiDefinition("panel.js", slots=("navigation.groups",)),
            query=lambda method, payload, *, session_id, turn_id: {},
        )

    async def other_owner(ctx):
        captured["other"] = ctx
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx,
            PluginUiDefinition("panel.js", slots=("header.end",)),
            query=lambda method, payload, *, session_id, turn_id: {},
        )

    for plugin_name, apply in (("projects", groups_owner), ("other", other_owner)):
        await root.mount(
            apply,
            name=plugin_name,
            inject=(UI_SLOTS,),
            runtime=PluginRuntime(plugin_name, "ui", tmp_path, tmp_path, tmp_path, {}),
        )

    async def emit(frame: dict[str, object]) -> None:
        emitted.append(frame)
        if len(emitted) == 1:
            got_first.set()
        elif len(emitted) == 2:
            got_second.set()

    follower = asyncio.create_task(provider.follow_slots(emit))
    try:
        # 让 follow_slots 先建立基线并挂起等待。
        await asyncio.sleep(0)
        with pytest.raises(ValueError, match="未登记 slot"):
            ui_slots.invalidate_slot(captured["other"], "navigation.groups")  # type: ignore[arg-type]

        ui_slots.invalidate_slot(captured["groups"], "navigation.groups")  # type: ignore[arg-type]
        await asyncio.wait_for(got_first.wait(), 2)
        assert emitted[0] == {"slot": "navigation.groups", "epoch": 1}

        ui_slots.invalidate_slot(captured["groups"], "navigation.groups")  # type: ignore[arg-type]
        await asyncio.wait_for(got_second.wait(), 2)
        assert emitted[1] == {"slot": "navigation.groups", "epoch": 2}
    finally:
        await provider.aclose()
        await asyncio.wait_for(follower, 2)
        await root.dispose()


@pytest.mark.asyncio
async def test_plugin_ui_v1_v2_catalog_and_file_asset_isolation(tmp_path: Path) -> None:
    from plugins.ui.contract import PluginUiPluginUnavailable, PluginUiStaleRevision
    from plugins.ui.plugin_ui import resolve_plugin_ui_asset

    root = CompositionRoot("ui-v2-files")
    kubai_dir = tmp_path / "kubai_icons"
    chunks_dir = kubai_dir / "static" / "chunks"
    chunks_dir.mkdir(parents=True)
    (kubai_dir / "plugin_ui.js").write_text("export default { slots: {} };\n", encoding="utf-8")
    (kubai_dir / "plugin_ui.css").write_text(".kubai-icons-root {}\n", encoding="utf-8")
    (chunks_dir / "a.js").write_text("export const a = 1;\n", encoding="utf-8")
    (chunks_dir / "b.js").write_text("export const b = 2;\n", encoding="utf-8")

    akasha_dir = tmp_path / "akasha"
    akasha_dir.mkdir(parents=True)
    (akasha_dir / "message_ui.js").write_text("export default { slots: {} };\n", encoding="utf-8")

    async def slots(ctx: Context) -> None:
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    await root.mount(slots, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))

    async def mount_kubai(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx,
            PluginUiDefinition(
                "plugin_ui.js",
                stylesheet="plugin_ui.css",
                slots=(),
                assets_dir="static/chunks",
                client_kernel=True,
            ),
            query=lambda method, payload, *, session_id, turn_id: {},
        )

    async def mount_akasha(ctx: Context) -> None:
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx,
            PluginUiDefinition(
                "message_ui.js",
                slots=("scope.badge", "scope.meta"),
            ),
            query=lambda method, payload, *, session_id, turn_id: {},
        )

    try:
        await root.mount(
            mount_kubai,
            name="kubai_icons",
            inject=(UI_SLOTS,),
            runtime=PluginRuntime("kubai_icons", "ui", kubai_dir, tmp_path, tmp_path, {}),
        )
        await root.mount(
            mount_akasha,
            name="akasha",
            inject=(UI_SLOTS,),
            runtime=PluginRuntime("akasha", "ui", akasha_dir, tmp_path, tmp_path, {}),
        )

        # 1. protocol=1 隐藏纯 v2 插件并过滤混合插件的 v2 slot，保护已打开的旧页面。
        v1 = await provider.catalog(protocol=1)
        v1_items = cast(list[dict[str, object]], v1["items"])
        assert [item["id"] for item in v1_items] == ["akasha"]
        assert v1_items[0]["slots"] == ["scope.badge"]

        # 2. protocol=2 隐藏纯微内核插件，protocol=3 暴露完整插件、bundle_sha256 与实际入口文件名。
        v2 = await provider.catalog(protocol=2)
        assert v2["catalog_revision"] != v1["catalog_revision"]
        v2_by_id = {str(item["id"]): item for item in cast(list[dict[str, object]], v2["items"])}
        assert set(v2_by_id) == {"akasha"}
        assert v2_by_id["akasha"]["module_path"] == "message_ui.js"
        assert set(cast(list[str], v2_by_id["akasha"]["slots"])) == {"scope.badge", "scope.meta"}

        v3 = await provider.catalog(protocol=3)
        v3_by_id = {str(item["id"]): item for item in cast(list[dict[str, object]], v3["items"])}
        assert set(v3_by_id) == {"akasha", "kubai_icons"}
        kubai_item = v3_by_id["kubai_icons"]
        rev = str(kubai_item["revision"])
        bundle_sha = str(kubai_item["bundle_sha256"])
        assert kubai_item["module_path"] == "plugin_ui.js"
        assert kubai_item["stylesheet_path"] == "plugin_ui.css"
        assert kubai_item["client_kernel"] is True

        # 3. 路径式文件读取返回单文件独立 sha256 与原始 bytes，错摘要或缺失路径均被拒。
        chunk_a = await provider.file_asset("kubai_icons", rev, bundle_sha, "static/chunks/a.js")
        chunk_b = await provider.file_asset("kubai_icons", rev, bundle_sha, "static/chunks/b.js")
        assert chunk_a["content"] == b"export const a = 1;\n"
        assert chunk_b["content"] == b"export const b = 2;\n"
        assert chunk_a["sha256"] != chunk_b["sha256"]
        assert chunk_a["media_type"] == "text/javascript"

        with pytest.raises(PluginUiStaleRevision):
            await provider.file_asset("kubai_icons", rev, "0" * 64, "static/chunks/a.js")
        with pytest.raises(PluginUiPluginUnavailable):
            await provider.file_asset("kubai_icons", rev, bundle_sha, "static/chunks/missing.js")

        # 4. 修改任意 chunk 内容会改变 bundle_sha256；非法文件类型与越界链接在登记阶段即被拒。
        before_asset = resolve_plugin_ui_asset(
            kubai_dir, module="plugin_ui.js", stylesheet="plugin_ui.css",
            slots=(), assets_dir="static/chunks", client_kernel=True,
        )
        (chunks_dir / "b.js").write_text("export const b = 3;\n", encoding="utf-8")
        after_asset = resolve_plugin_ui_asset(
            kubai_dir, module="plugin_ui.js", stylesheet="plugin_ui.css",
            slots=(), assets_dir="static/chunks", client_kernel=True,
        )
        assert before_asset.bundle_sha256 != after_asset.bundle_sha256

        (chunks_dir / "evil.svg").write_text("<svg/>", encoding="utf-8")
        with pytest.raises(RuntimeError, match="不支持的文件类型"):
            resolve_plugin_ui_asset(
                kubai_dir, module="plugin_ui.js", stylesheet=None,
                slots=(), assets_dir="static/chunks", client_kernel=True,
            )
        (chunks_dir / "evil.svg").unlink()

        outside = tmp_path / "secret.js"
        outside.write_text("export const s = 1;\n", encoding="utf-8")
        (chunks_dir / "escape.js").symlink_to(outside)
        with pytest.raises(RuntimeError, match="非法文件"):
            resolve_plugin_ui_asset(
                kubai_dir, module="plugin_ui.js", stylesheet=None,
                slots=(), assets_dir="static/chunks", client_kernel=True,
            )
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_chat_api_catalog_protocol_http_query_coercion(tmp_path: Path) -> None:
    """验证 HTTP 层 /api/chat/plugin-ui/catalog 对 ?protocol=1|2 的真实查询参数解析（防 Literal 整数 422 回归）。"""
    import httpx
    from plugins.akashic_clients.chat_api import create_chat_app
    from plugins.akashic_clients.web_chat import WebChatChannel

    root = CompositionRoot("chat-api-protocol")
    plugin_dir = tmp_path / "demo"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin_ui.js").write_text("export default { slots: {} };\n", encoding="utf-8")

    async def providers(ctx: Context) -> None:
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    async def demo_plugin(ctx: Context) -> None:
        async def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
            del method, payload, session_id, turn_id
            return {}
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx, PluginUiDefinition(module="plugin_ui.js", slots=("scope.badge", "scope.meta")), query=query,
        )

    await root.mount(providers, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))

    try:
        await root.mount(
            demo_plugin, name="demo", inject=(UI_SLOTS,),
            runtime=PluginRuntime("demo", "release", plugin_dir, tmp_path, tmp_path, {}),
        )
        app = create_chat_app(workspace=tmp_path, channel=cast(WebChatChannel, object()), plugin_ui_provider=provider)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_default = await client.get("/api/chat/plugin-ui/catalog")
            assert r_default.status_code == 200
            assert r_default.json()["items"][0]["slots"] == ["scope.badge"]

            r_v1 = await client.get("/api/chat/plugin-ui/catalog?protocol=1")
            assert r_v1.status_code == 200
            assert r_v1.json()["items"][0]["slots"] == ["scope.badge"]

            r_v2 = await client.get("/api/chat/plugin-ui/catalog?protocol=2")
            assert r_v2.status_code == 200
            assert r_v2.json()["items"][0]["slots"] == ["scope.badge", "scope.meta"]

            r_v3 = await client.get("/api/chat/plugin-ui/catalog?protocol=3")
            assert r_v3.status_code == 200
            assert r_v3.json()["items"][0]["client_kernel"] is False

            r_bad = await client.get("/api/chat/plugin-ui/catalog?protocol=99")
            assert r_bad.status_code == 422
    finally:
        await provider.aclose()
        await root.dispose()


@pytest.mark.asyncio
async def test_plugin_ui_protocol_v3_client_kernel_projection(tmp_path: Path) -> None:
    root = CompositionRoot("plugin-ui-v3-kernel")
    kernel_dir = tmp_path / "pure_kernel"
    legacy_empty_dir = tmp_path / "legacy_empty"
    hybrid_dir = tmp_path / "hybrid"
    for d in (kernel_dir, legacy_empty_dir, hybrid_dir):
        d.mkdir(parents=True)
        (d / "plugin_ui.js").write_text("export function activate() {}\nexport default { slots: {} };\n", encoding="utf-8")

    async def providers(ctx: Context) -> None:
        directory = PluginUiSlots(ctx)
        await ctx.provide(UI_SLOTS, directory)
        await ctx.provide(PLUGIN_UI, LivePluginUiProvider(ctx, directory))

    async def pure_kernel_plugin(ctx: Context) -> None:
        async def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
            del method, payload, session_id, turn_id
            return {"ok": True}
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx, PluginUiDefinition(module="plugin_ui.js", slots=(), client_kernel=True), query=query,
        )

    async def legacy_empty_plugin(ctx: Context) -> None:
        async def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
            del method, payload, session_id, turn_id
            return {}
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx, PluginUiDefinition(module="plugin_ui.js", slots=()), query=query,
        )

    async def hybrid_plugin(ctx: Context) -> None:
        async def query(method: str, payload: dict[str, object], *, session_id: str | None, turn_id: str | None) -> dict[str, object]:
            del method, payload, session_id, turn_id
            return {}
        await ctx.require(UI_SLOTS).register_plugin_ui(
            ctx, PluginUiDefinition(module="plugin_ui.js", slots=("scope.badge", "scope.meta"), client_kernel=True), query=query,
        )

    await root.mount(providers, name="ui")
    provider = cast(LivePluginUiProvider, root.context.require(PLUGIN_UI))

    try:
        await root.mount(pure_kernel_plugin, name="pure_kernel", inject=(UI_SLOTS,), runtime=PluginRuntime("pure_kernel", "release", kernel_dir, tmp_path, tmp_path, {}))
        await root.mount(legacy_empty_plugin, name="legacy_empty", inject=(UI_SLOTS,), runtime=PluginRuntime("legacy_empty", "release", legacy_empty_dir, tmp_path, tmp_path, {}))
        await root.mount(hybrid_plugin, name="hybrid", inject=(UI_SLOTS,), runtime=PluginRuntime("hybrid", "release", hybrid_dir, tmp_path, tmp_path, {}))

        v1 = await provider.catalog(protocol=1)
        v1_by_id = {item["id"]: item for item in cast(list[dict[str, object]], v1["items"])}
        assert "pure_kernel" not in v1_by_id
        assert "legacy_empty" in v1_by_id
        assert v1_by_id["hybrid"]["slots"] == ["scope.badge"]

        v2 = await provider.catalog(protocol=2)
        v2_by_id = {item["id"]: item for item in cast(list[dict[str, object]], v2["items"])}
        assert "pure_kernel" not in v2_by_id
        assert "legacy_empty" in v2_by_id
        assert v2_by_id["hybrid"]["slots"] == ["scope.badge", "scope.meta"]

        v3 = await provider.catalog(protocol=3)
        v3_by_id = {item["id"]: item for item in cast(list[dict[str, object]], v3["items"])}
        assert v3_by_id["pure_kernel"]["client_kernel"] is True
        assert v3_by_id["pure_kernel"]["slots"] == []
        assert v3_by_id["legacy_empty"]["client_kernel"] is False
        assert v3_by_id["hybrid"]["client_kernel"] is True
    finally:
        await provider.aclose()
        await root.dispose()
