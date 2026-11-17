"""验证跨插件目录/注册表在贡献方 LOADING 或 UNLOADING 排空期间不会暴露未就绪项或把全量读操作拖垮。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.plugin_composition import CompositionRoot, Context, FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.assets.contract import INSTALLED_ASSETS
from plugins.assets.plugin import apply as apply_assets
from plugins.channels.contract import ChannelInboundMessage
from plugins.content.plugin import check_text
from plugins.delivery.contract import DELIVERY_SENDERS
from plugins.delivery.senders import Senders
from plugins.ledger.artifact_store import ArtifactStore
from plugins.ledger.contract import (
    BINDINGS,
    MESSAGE_CATALOG,
    MESSAGE_WRITERS,
    OWNER_STATE,
    SESSION_ADMISSION,
    ContentPart,
    Input,
)
from plugins.ledger.bindings import Bindings
from plugins.ledger.log import MessageLog
from plugins.ledger.services import MessageWriters, OwnerState, SessionAdmission
from plugins.onboarding.contract import ONBOARDING, Ability, Step
from plugins.onboarding.plugin import apply as apply_onboarding
from plugins.plugin_settings.contract import SETTINGS_FORMS, SettingsField, SettingsForm
from plugins.plugin_settings.plugin import apply as apply_plugin_settings
from plugins.sources.contract import SOURCES_V5 as SOURCES, AsyncSource
from plugins.sources.plugin import apply as apply_sources
from plugins.standard_web.contract import WEB_PROVIDERS
from plugins.standard_web.web import Providers as WebProviders, SearchRequest, WebConfig
from plugins.ui.contract import SETTINGS_SLOT, UI, Configuration


@pytest.mark.asyncio
async def test_assets_open_skips_loading_and_unloading_contributors(tmp_path: Path) -> None:
    root = CompositionRoot("test")
    consumer_ctx: list[Context] = []
    ready_dir = tmp_path / "ready-plugin"
    (ready_dir / "skills").mkdir(parents=True)
    flaky_dir = tmp_path / "flaky-plugin"
    (flaky_dir / "skills").mkdir(parents=True)

    unload_gate = asyncio.Event()
    in_flight_entered = asyncio.Event()

    async def mount_assets(ctx: Context) -> None:
        await apply_assets(ctx)

    async def mount_ready(ctx: Context) -> None:
        _ = await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")

    async def mount_flaky(ctx: Context) -> None:
        _ = await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
        # 登记一个常驻作用域让卸载停在 UNLOADING（且 _activation_token=None）以便命中竞态
        await ctx.provide(ServiceKey("test.flaky.ping"), lambda: ctx)

    async def mount_consumer(ctx: Context) -> None:
        consumer_ctx.append(ctx)

    await root.mount(mount_assets, name="assets")
    await root.mount(
        mount_ready,
        name="ready",
        inject=(INSTALLED_ASSETS,),
        runtime=PluginRuntime("ready@local", "local", ready_dir, tmp_path / "d1", tmp_path, {}),
    )
    await root.mount(
        mount_flaky,
        name="flaky",
        inject=(INSTALLED_ASSETS,),
        runtime=PluginRuntime("flaky@local", "local", flaky_dir, tmp_path / "d2", tmp_path, {}),
    )
    await root.mount(
        mount_consumer,
        name="consumer",
        inject=(INSTALLED_ASSETS,),
        runtime=PluginRuntime("consumer@local", "local", tmp_path, tmp_path / "dc", tmp_path, {}),
    )

    try:
        consumer = consumer_ctx[0]
        assets_svc = root.service_value(INSTALLED_ASSETS)
        assert assets_svc is not None

        # 1. 在途持有 flaky 调用，随后卸载 flaky，使其卡在 UNLOADING（effect 尚未 cleanup，_entries 仍有 flaky）
        flaky_fiber = next(f for f in root.fibers() if f.name == "flaky")
        flaky_ctx = root.service_value(ServiceKey("test.flaky.ping"))()  # type: ignore[misc]

        async def hold_flaky() -> None:
            async with flaky_ctx.runtime_scope():
                in_flight_entered.set()
                await unload_gate.wait()

        try:
            async with asyncio.timeout(3):
                holder = asyncio.create_task(hold_flaky())
                await in_flight_entered.wait()
                unloader = asyncio.create_task(flaky_fiber.dispose())
                while flaky_fiber.state is not FiberState.UNLOADING:
                    await asyncio.sleep(0.01)

                # 2. 此时 flaky 仍在 Assets._entries 中，但已是 UNLOADING；assets(consumer) 与 open() 不应报错也不应包含 flaky
                async with consumer.runtime_scope():
                    listed = assets_svc(consumer)
                    assert [a.owner_id for a in listed] == ["ready@local"]
                    async with assets_svc.open(consumer, category="skills") as opened:
                        assert [a.owner_id for a in opened] == ["ready@local"]

                unload_gate.set()
                await holder
                await unloader
        finally:
            unload_gate.set()
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_registries_filter_non_active_contributors(tmp_path: Path) -> None:
    log = MessageLog(tmp_path / "sessions.db")
    store = ArtifactStore(tmp_path / "sessions.db")
    root = CompositionRoot("test")
    unloading_gate = asyncio.Event()
    unloading_entered = asyncio.Event()
    try:
        async def core(ctx: Context) -> None:
            for key, val in [
                (MESSAGE_CATALOG, log.catalog()),
                (MESSAGE_WRITERS, MessageWriters(log)),
                (OWNER_STATE, OwnerState(log)),
                (SESSION_ADMISSION, SessionAdmission(log)),
                (BINDINGS, Bindings(log, ctx)),
                (RUNTIME_CATALOG, lambda _ctx: {"plugins": []}),
                (UI, SimpleNamespace(
                    register=lambda *_a, **_kw: asyncio.sleep(0),
                    register_configuration=lambda *_a, **_kw: None,
                )),
            ]:
                await ctx.provide(key, val)

        async def mount_delivery(ctx: Context) -> None:
            await ctx.provide(DELIVERY_SENDERS, Senders(ctx))

        async def mount_onboarding(ctx: Context) -> None:
            await apply_onboarding(ctx)

        async def mount_settings(ctx: Context) -> None:
            await apply_plugin_settings(ctx)

        async def mount_sources(ctx: Context) -> None:
            await apply_sources(ctx)

        await root.mount(core, name="core")
        await root.mount(mount_delivery, name="delivery", inject=(BINDINGS,))
        await root.mount(mount_onboarding, name="onboarding", inject=(RUNTIME_CATALOG,))
        await root.mount(mount_settings, name="plugin_settings", inject=(UI,))
        await root.mount(mount_sources, name="sources", inject=(MESSAGE_WRITERS,))

        async def mount_web(ctx: Context) -> None:
            await ctx.provide(WEB_PROVIDERS, WebProviders(ctx, WebConfig()))

        await root.mount(mount_web, name="standard_web")

        # 挂载一个即将处于 UNLOADING 状态的插件，向上述全部注册表登记条目
        unloading_ctx_holder: list[Context] = []

        async def mount_unloading_contrib(ctx: Context) -> None:
            unloading_ctx_holder.append(ctx)
            senders = cast(Senders, ctx.require(DELIVERY_SENDERS))
            await senders.candidate(ctx, name="flaky_chan", title="Flaky", route="/flaky", status=lambda: {"enabled": True})

            @asynccontextmanager
            async def open_sender():
                yield SimpleNamespace(idempotent=True, send=lambda *_a, **_kw: None)

            await senders.register(ctx, name="flaky_chan", open=open_sender, idempotent=True)

            await ctx.require(ONBOARDING).register(
                ctx,
                Step(
                    key="flaky_step",
                    title="Flaky Step",
                    group="other",
                    route="/flaky",
                    status=lambda: asyncio.sleep(0, result={"configured": True}),
                ),
            )
            await ctx.require(SETTINGS_FORMS).register(
                ctx,
                SettingsForm("Flaky", "desc", (SettingsField("token", "Token"),)),
                SimpleNamespace(
                    read=lambda: asyncio.sleep(0, result={}),
                    save=lambda *_a: asyncio.sleep(0, result={}),
                    receipt=lambda *_a: {},
                ),
            )
            await ctx.require(WEB_PROVIDERS).register_search(
                ctx,
                SimpleNamespace(search=lambda _req: asyncio.sleep(0, result={"ok": True})),
                label="FlakyWeb",
            )
            await ctx.require(SOURCES).register(
                ctx,
                name="flaky_src",
                channels=("flaky",),
                open=lambda _sid: None,  # type: ignore[arg-type]
                needs_reply=lambda _r: asyncio.sleep(0, result=False),
                accept=lambda *_a: asyncio.sleep(0, result=None),  # type: ignore[arg-type]
            )

        await root.mount(
            mount_unloading_contrib,
            name="flaky_contrib",
            inject=(DELIVERY_SENDERS, ONBOARDING, SETTINGS_FORMS, WEB_PROVIDERS, SOURCES),
            runtime=PluginRuntime("flaky@local", "local", tmp_path, tmp_path / "fd", tmp_path, {}),
        )

        # 卡住 flaky_contrib 在 UNLOADING 排空状态
        assert unloading_ctx_holder, [(f.name, f.state, f.error) for f in root.fibers()]
        flaky_ctx = unloading_ctx_holder[0]
        flaky_fiber = next(f for f in root.fibers() if f.name == "flaky_contrib")

        async def hold() -> None:
            async with flaky_ctx.runtime_scope():
                unloading_entered.set()
                await unloading_gate.wait()

        try:
            async with asyncio.timeout(3):
                holder = asyncio.create_task(hold())
                await unloading_entered.wait()
                unloader = asyncio.create_task(flaky_fiber.dispose())
                while flaky_fiber.state is not FiberState.UNLOADING:
                    await asyncio.sleep(0.01)

            # 逐项核对：全部跳过 UNLOADING 贡献者，不抛 CompositionError
            senders = root.service_value(DELIVERY_SENDERS)
            assert senders is not None
            assert senders.candidates() == ()
            assert senders.bind_all(root.service_value(BINDINGS)) == {}  # type: ignore[arg-type]

            onboarding = root.service_value(ONBOARDING)
            assert onboarding is not None
            catalog = await onboarding.catalog()
            assert catalog["steps"] == []
            with pytest.raises(KeyError):
                await onboarding.status("flaky@local/flaky_step")

            forms = root.service_value(SETTINGS_FORMS)
            assert forms is not None
            assert forms.entries() == ()
            with pytest.raises(KeyError):
                forms.entry("flaky@local")

            web_providers = root.service_value(WEB_PROVIDERS)
            assert web_providers is not None
            assert web_providers.choices("search") == []
            res = await web_providers.search(SearchRequest(query="hello", num_results=3))
            assert "error" in res

            sources_svc = root.service_value(SOURCES)
            assert sources_svc is not None
            assert sources_svc.entries() == ()
            with pytest.raises(ValueError, match="输入渠道没有来源"):
                await sources_svc.accept(
                    "s1",
                    "m1",
                    ChannelInboundMessage("flaky", "u", "c", "hi", datetime.now(UTC), {}),
                )

            unloading_gate.set()
            await holder
            await unloader
        finally:
            unloading_gate.set()
    finally:
        await root.dispose()
        store.close()
        log.close()
