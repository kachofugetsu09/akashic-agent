"""验证 runtime_inspection 是可选的普通 provider，而非 Core 业务实现。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import CompositionError, CompositionRoot, PluginRuntime, ServiceKey
from agent.plugin_composition.rpc import rpc_method_key
from agent.plugin_composition.runtime_catalog import (
    RUNTIME_CATALOG,
    build_runtime_catalog,
)
from agent.plugins.generation import PluginGeneration
from agent.plugins.scope import PluginScope
from agent.plugins.static_manifest import StaticPluginManifest
from plugins.runtime_inspection import plugin
from plugins.runtime_inspection.inspection import (
    SCHEDULER_INSPECTION,
    SKILL_INSPECTION,
)


def _payload(value: object) -> Mapping[str, object]:
    """窄化 inspection RPC 的结构化返回值。"""
    if not isinstance(value, Mapping):
        raise AssertionError("inspection RPC 必须返回对象")
    return cast(Mapping[str, object], value)


def _rows(value: object) -> tuple[Mapping[str, object], ...]:
    """窄化 inspection RPC 中的列表行。"""
    if not isinstance(value, (list, tuple)):
        raise AssertionError("inspection RPC rows 必须是列表")
    rows: list[Mapping[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise AssertionError("inspection RPC row 必须是对象")
        rows.append(cast(Mapping[str, object], row))
    return tuple(rows)


def test_runtime_catalog_projects_selected_pre_fiber_failure(tmp_path: Path) -> None:
    root = CompositionRoot("selected-failed")
    generation = PluginGeneration(
        plugin_id="broken",
        generation_id="broken:g1",
        module_path="_broken",
        source_revision="source",
        config_revision="config",
        plugin_dir=tmp_path,
        data_dir=tmp_path / "data",
        instance=None,
        scope=PluginScope("broken", generation_id="broken:g1"),
        static_manifest=StaticPluginManifest(
            name="broken",
            version="1.0.0",
            api_version=3,
            python=(),
            identity_digest="identity",
        ),
        state="failed",
        archive_ref="a" * 64,
        load_error=ImportError("import blocked"),
    )
    catalog = build_runtime_catalog(
        root,
        {"broken": generation},
        {"broken": [generation]},
    )
    item = next(
        item for item in _rows(catalog["plugins"])
        if item["id"] == "broken"
    )
    composition = _payload(item["composition"])
    assert item["api_version"] == 3
    assert item["archive_ref"] == "a" * 64
    assert item["state"] == "failed"
    assert item["load_error"] == "import blocked"
    assert item["cleanup_pending"] is True
    assert composition["ready"] is False
    assert composition["fibers"] == []


class _Scheduler:
    def list_jobs(self) -> tuple[Mapping[str, object], ...]:
        return ({"id": "external-job", "name": "外部任务"},)

    def get_job(self, job_id: str) -> Mapping[str, object] | None:
        return {"id": job_id, "name": "外部任务"}


class _Skills:
    async def list_skills(self) -> tuple[Mapping[str, object], ...]:
        return ({"name": "external-skill", "available": True},)


@pytest.mark.asyncio
async def test_runtime_inspection_plugin_owns_documents_and_optional_providers(
    tmp_path: Path,
) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
    (memory / "SELF.md").write_text("# Self\n", encoding="utf-8")
    (memory / "VEDA.md").write_text("# VEDA\n", encoding="utf-8")
    runtime = PluginRuntime(
        plugin_id="runtime_inspection",
        generation_id="runtime-inspection:g1",
        plugin_dir=tmp_path,
        data_dir=tmp_path / "plugin-data",
        workspace=tmp_path,
        config={},
        workspace_files=plugin.workspace_files,
    )
    root = CompositionRoot("runtime-inspection")

    async def mount_runtime(ctx):
        await plugin.apply(ctx)

    await root.mount(mount_runtime, name=plugin.name, runtime=runtime)
    async def call(method, params=None):
        operation = root.context.require(rpc_method_key("inspection/" + method))
        return await operation.invoke(operation.params.model_validate(params or {}), None)
    documents = _payload(await call("documents.list"))
    assert [item["id"] for item in _rows(documents["items"])] == [
        "memory",
        "self",
        "veda",
    ]
    memory = _payload(await call("documents.get", {"document_id": "memory"}))
    assert memory["markdown"] == "# Memory\n"
    skills = _payload(await call("skills.list"))
    assert _payload(skills["unavailable"])["code"] == "skills_unavailable"
    jobs = _payload(await call("jobs.list"))
    assert _payload(jobs["unavailable"])["code"] == "scheduler_unavailable"

    scheduler = _Scheduler()
    skills = _Skills()

    async def mount_business(ctx):
        await ctx.provide(SCHEDULER_INSPECTION, scheduler)
        await ctx.provide(SKILL_INSPECTION, skills)

    business = await root.mount(mount_business, name="business-providers")
    jobs = _payload(await call("jobs.list"))
    assert tuple(_rows(jobs["items"])) == ({"id": "external-job", "name": "外部任务"},)
    skills = _payload(await call("skills.list"))
    assert tuple(_rows(skills["items"])) == ({"name": "external-skill", "available": True},)
    await business.dispose()
    jobs = _payload(await call("jobs.list"))
    assert _payload(jobs["unavailable"])["code"] == "scheduler_unavailable"
    skills = _payload(await call("skills.list"))
    assert _payload(skills["unavailable"])["code"] == "skills_unavailable"
    await root.dispose()


@pytest.mark.asyncio
async def test_runtime_inspection_provider_propagates_business_failure(
    tmp_path: Path,
) -> None:
    runtime = PluginRuntime(
        plugin_id="runtime_inspection",
        generation_id="runtime-inspection:g1",
        plugin_dir=tmp_path,
        data_dir=tmp_path / "plugin-data",
        workspace=tmp_path,
        config={},
        workspace_files=plugin.workspace_files,
    )
    root = CompositionRoot("runtime-inspection-failure")

    async def mount_runtime(ctx):
        await plugin.apply(ctx)

    await root.mount(mount_runtime, name=plugin.name, runtime=runtime)

    class BrokenScheduler(_Scheduler):
        def list_jobs(self) -> tuple[Mapping[str, object], ...]:
            raise RuntimeError("scheduler read failed")

    async def mount_business(ctx):
        await ctx.provide(SCHEDULER_INSPECTION, BrokenScheduler())

    await root.mount(mount_business, name="business-providers")
    operation = root.context.require(rpc_method_key("inspection/jobs.list"))
    with pytest.raises(RuntimeError, match="scheduler read failed"):
        await operation.invoke(operation.params.model_validate({}), None)
    await root.dispose()


@pytest.mark.asyncio
async def test_client_inspection_binds_live_probe_scope(tmp_path: Path) -> None:
    """客户端经真实 probe Context 读取 live catalog 与技能 RPC。"""
    from contextlib import asynccontextmanager

    from plugins.akashic_clients.capabilities import INSPECTION_SKILLS_LIST
    from plugins.akashic_clients.runtime_inspection import (
        RuntimeInspectionError,
        ScopedRpcRuntimeInspection,
    )

    for relative in ("MEMORY.md", "SELF.md", "VEDA.md"):
        path = tmp_path / "memory" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    runtime = PluginRuntime(
        plugin_id="runtime_inspection",
        generation_id="runtime-inspection:g1",
        plugin_dir=tmp_path,
        data_dir=tmp_path / "plugin-data",
        workspace=tmp_path,
        config={},
        workspace_files=plugin.workspace_files,
    )
    root = CompositionRoot("live-probe")
    await root.context.provide(
        RUNTIME_CATALOG,
        lambda _ctx: build_runtime_catalog(root, {}),
    )
    await root.mount(plugin.apply, name=plugin.name, runtime=runtime)

    async def probe_apply(ctx):
        ctx.require(RUNTIME_CATALOG)
        ctx.require(INSPECTION_SKILLS_LIST)

    probe = await root.mount(
        probe_apply,
        name="inspection-probe",
        inject=(RUNTIME_CATALOG, INSPECTION_SKILLS_LIST),
        runtime=PluginRuntime(
            plugin_id="inspection-probe",
            generation_id="inspection-probe:g1",
            plugin_dir=tmp_path,
            data_dir=tmp_path / "probe-data",
            workspace=tmp_path,
            config={},
        ),
    )

    @asynccontextmanager
    async def open_scope():
        async with probe.context.runtime_scope():
            yield probe.context

    service = ScopedRpcRuntimeInspection(open_scope)
    try:
        with pytest.raises(RuntimeInspectionError) as captured:
            await service.list_capabilities()
        assert captured.value.code == "mcp_provider_unavailable"

        from agent.plugin_composition.mcp_slots import MCP_SERVERS

        class RegisteredTargets:
            root_instance_token = root.instance_token

            def catalog(self):
                return [{"owner_id": "probe", "name": "server", "status": "declared"}]

        await root.context.provide(MCP_SERVERS, RegisteredTargets())

        async def mount_skills(ctx):
            await ctx.provide(SKILL_INSPECTION, _Skills())

        await root.mount(mount_skills, name="skills", runtime=runtime)
        catalog = await service.list_capabilities()
        assert catalog["mcp_servers"] == [
            {"owner_id": "probe", "name": "server", "status": "declared"},
        ]
        assert catalog["skills"] == [{"name": "external-skill", "available": True}]
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_live_catalog_tracks_manager_replacement_without_lifecycle_side_effects(
    tmp_path: Path,
) -> None:
    """The catalog follows current Fibers while rejecting raw or foreign scopes."""
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from tests.fixtures.plugin_workspace import initialize_plugin_workspace

    runtime_key = (
        "from agent.plugin_composition import RUNTIME_STARTED, ServiceKey\n"
        "RUNTIME_CATALOG = ServiceKey('core.runtime_catalog.v1')\n"
        "C_SERVICE = ServiceKey('probe.c-service')\n"
        "MOUNT_B = ServiceKey('probe.mount-b')\n"
    )

    def source(version: str) -> str:
        return (
            "api_version = 3\n"
            "name = 'probe'\n"
            f"version = {version!r}\n"
            f"{runtime_key}"
            "inject = ()\n"
            "async def apply(ctx):\n"
            "    await ctx.health('probe-health', required=True)\n"
            "    async def mount_b():\n"
            "        async def child_c(cctx):\n"
            "            reader = cctx.require(RUNTIME_CATALOG)\n"
            "            async def read_from_c():\n"
            "                async with cctx.runtime_scope():\n"
            "                    result = reader(cctx)\n"
            "                    if not result['plugins']:\n"
            "                        raise RuntimeError('C runtime catalog is empty')\n"
            "                    return result\n"
            "            await cctx.provide(C_SERVICE, read_from_c)\n"
            "        await ctx.mount(child_c, name='probe-c', inject=(RUNTIME_CATALOG,))\n"
            "        async def child_b(bctx):\n"
            "            reader = bctx.require(RUNTIME_CATALOG)\n"
            "            read_from_c = bctx.require(C_SERVICE)\n"
            "            async def on_started(_event):\n"
            "                result = reader(bctx)\n"
            "                if not result['plugins']:\n"
            "                    raise RuntimeError('B runtime catalog is empty')\n"
            "                c_result = await read_from_c()\n"
            "                if not c_result['plugins']:\n"
            "                    raise RuntimeError('C runtime catalog is empty')\n"
            f"                bctx.report_incident('probe_loaded', 'version {version}')\n"
            "            await bctx.on(RUNTIME_STARTED, on_started)\n"
            "        await ctx.mount(child_b, name='probe-child', inject=(RUNTIME_CATALOG, C_SERVICE))\n"
            "    await ctx.provide(MOUNT_B, mount_b)\n"
        )

    plugins = tmp_path / "plugins"
    probe_dir = plugins / "probe"
    unrelated_dir = plugins / "unrelated"
    probe_dir.mkdir(parents=True)
    unrelated_dir.mkdir(parents=True)
    probe_file = probe_dir / "plugin.py"
    probe_file.write_text(source("1.0.0"), encoding="utf-8")
    unrelated_dir.joinpath("plugin.py").write_text(
        "api_version = 3\nname = 'unrelated'\nversion = '1.0.0'\n"
        "async def apply(ctx):\n    return None\n",
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    manager = PluginManager(
        [plugins],
        event_bus=EventBus(),
        workspace=workspace,
        installed_cache_root=tmp_path / "cache",
    )
    foreign = CompositionRoot("foreign")
    try:
        await manager.load_all()
        root = manager._live_root
        generation = manager.generation("probe")
        unrelated = manager.generation("unrelated")
        assert root is not None and generation is not None and generation.fiber is not None
        assert unrelated is not None and unrelated.fiber is not None
        unrelated_fiber = unrelated.fiber
        unrelated_context = unrelated_fiber.context
        old_fiber = generation.fiber
        mount_b = old_fiber.context.require(ServiceKey("probe.mount-b"))
        async with old_fiber.context.runtime_scope():
            await mount_b()
        old_child = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "probe-child"
        )
        old_c = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "probe-c"
        )
        old_child_context = old_child.context
        reader = old_child_context.require(RUNTIME_CATALOG)
        with pytest.raises(CompositionError) as missing_permit:
            reader(old_child_context)
        assert missing_permit.value.code == "OWNER_CALL_CONTEXT"
        with pytest.raises(CompositionError) as child_missing_permit:
            reader(old_child.context)
        assert child_missing_permit.value.code == "OWNER_CALL_CONTEXT"
        with pytest.raises(CompositionError) as undeclared:
            async with old_fiber.context.runtime_scope():
                reader(old_fiber.context)
        assert undeclared.value.code == "UNDECLARED_SERVICE"

        before = (
            root.frozen,
            root._composition_revision,  # pyright: ignore[reportPrivateUsage]
            tuple(fiber.fiber_id for fiber in root._fibers.values()),  # pyright: ignore[reportPrivateUsage]
        )
        async with old_child_context.runtime_scope():
            first = reader(old_child_context)

            inherited_errors: list[str] = []

            async def raw_child() -> None:
                with pytest.raises(CompositionError) as inherited:
                    reader(old_child_context)
                inherited_errors.append(inherited.value.code)

            await asyncio.create_task(raw_child())
            parent_after_child = reader(old_child_context)
        assert inherited_errors == ["OWNER_CALL_CONTEXT"]
        assert parent_after_child["snapshot_id"] == first["snapshot_id"]
        assert not old_child._in_flight_calls  # pyright: ignore[reportPrivateUsage]
        after = (
            root.frozen,
            root._composition_revision,  # pyright: ignore[reportPrivateUsage]
            tuple(fiber.fiber_id for fiber in root._fibers.values()),  # pyright: ignore[reportPrivateUsage]
        )
        assert before == after
        first_item = next(item for item in _rows(first["plugins"]) if item["id"] == "probe")
        first_item_composition = _payload(first_item["composition"])
        assert first_item_composition["ready"] is True
        assert first_item_composition["health"] == [
            {
                "owner": "probe",
                "name": "probe-health",
                "required": True,
                "healthy": True,
                "reason": None,
            }
        ]
        assert first_item_composition["incident_count"] == 1
        assert _rows(first_item_composition["recent_incidents"])[0]["fiber_id"] == old_child.fiber_id
        assert _payload(first["mcp_unavailable"])["code"] == "mcp_provider_unavailable"

        await foreign.context.provide(RUNTIME_CATALOG, reader)
        foreign_fiber = await foreign.mount(
            lambda ctx: None,
            name="foreign-probe",
            inject=(RUNTIME_CATALOG,),
            runtime=PluginRuntime(
                plugin_id="foreign-probe",
                generation_id="foreign-probe:g1",
                plugin_dir=tmp_path,
                data_dir=tmp_path / "foreign-data",
                workspace=tmp_path,
                config={},
            ),
        )
        async with foreign_fiber.context.runtime_scope():
            with pytest.raises(RuntimeError, match="不属于当前 live Root"):
                reader(foreign_fiber.context)

        same_fiber_context = old_child.context
        await old_c.effects[0].aclose()

        async def replacement_c_reader():
            async with old_c.context.runtime_scope():
                return old_c.context.require(RUNTIME_CATALOG)(old_c.context)

        await old_c.context.provide(ServiceKey("probe.c-service"), replacement_c_reader)
        await old_child.reconcile()
        assert old_child.context is not same_fiber_context
        with pytest.raises(CompositionError) as stale:
            reader(same_fiber_context)
        assert stale.value.code == "STALE_ACTIVATION"
        same_reader = old_child.context.require(RUNTIME_CATALOG)
        async with old_child.context.runtime_scope():
            same = same_reader(old_child.context)
        same_item = next(item for item in _rows(same["plugins"]) if item["id"] == "probe")
        same_item_composition = _payload(same_item["composition"])
        assert same_item_composition["incident_count"] == 2
        assert {
            item["fiber_id"]
            for item in _rows(same_item_composition["recent_incidents"])
        } == {old_child.fiber_id}

        probe_file.write_text(source("2.0.0"), encoding="utf-8")
        await manager.reconcile_changed()
        fresh = manager.generation("probe")
        assert fresh is not None and fresh is not generation and fresh.fiber is not None
        assert manager._live_root is root
        assert manager.generation("unrelated") is unrelated
        assert unrelated.fiber is unrelated_fiber
        assert unrelated.fiber.context is unrelated_context
        assert unrelated.fiber.state.value == "active"
        assert old_fiber not in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(CompositionError) as replaced:
            async with old_fiber.context.runtime_scope():
                pass
        assert replaced.value.code == "OWNER_UNAVAILABLE"
        with pytest.raises(CompositionError) as old_child_unloaded:
            async with old_child.context.runtime_scope():
                pass
        assert old_child_unloaded.value.code == "OWNER_UNAVAILABLE"
        new_mount_b = fresh.fiber.context.require(ServiceKey("probe.mount-b"))
        async with fresh.fiber.context.runtime_scope():
            await new_mount_b()
        new_child = next(
            fiber for fiber in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
            if fiber.name == "probe-child"
        )
        assert old_child not in root._fibers.values()  # pyright: ignore[reportPrivateUsage]
        assert new_child is not old_child
        new_child_context = new_child.context
        new_reader = new_child_context.require(RUNTIME_CATALOG)
        async with new_child_context.runtime_scope():
            second = new_reader(new_child_context)
        second_item = next(item for item in _rows(second["plugins"]) if item["id"] == "probe")
        second_item_composition = _payload(second_item["composition"])
        assert second_item["generation_id"] == fresh.generation_id
        assert second_item["generation_id"] != first_item["generation_id"]
        assert second_item_composition["incident_count"] == 1
        assert _rows(second_item_composition["recent_incidents"])[0]["fiber_id"] == new_child.fiber_id
        assert _rows(second_item_composition["recent_incidents"])[0]["message"] == "version 2.0.0"
        assert {item.fiber_id for item in root.receipt().incidents} >= {
            old_child.fiber_id,
            new_child.fiber_id,
        }
    finally:
        await foreign.dispose()
        await manager.terminate_all()


@pytest.mark.asyncio
async def test_client_translates_runtime_catalog_unavailable() -> None:
    """Core 的中立 unavailable 结果在插件边界转换为既有错误合同。"""
    from contextlib import asynccontextmanager

    from agent.plugin_composition import CompositionRoot
    from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
    from plugins.akashic_clients.runtime_inspection import (
        RuntimeInspectionError,
        ScopedRpcRuntimeInspection,
    )

    root = CompositionRoot("catalog-unavailable")
    await root.context.provide(
        RUNTIME_CATALOG,
        lambda _ctx: {
            "snapshot_id": "live-root:1",
            "plugins": [],
            "mcp_unavailable": {
                "code": "mcp_catalog_unavailable",
                "message": "MCP 工具目录暂不可用，声明的服务按需启动",
            }
        },
    )

    @asynccontextmanager
    async def open_scope():
        yield root.context

    try:
        service = ScopedRpcRuntimeInspection(open_scope)
        with pytest.raises(RuntimeInspectionError) as captured:
            await service.list_capabilities()
        assert captured.value.code == "mcp_catalog_unavailable"
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_manager_catalog_accepts_only_current_channel_request(tmp_path: Path) -> None:
    """真实 Channel 请求沿原 Context 读目录，离开请求或换 task 后拒绝。"""
    from agent.plugin_composition.channels import CHANNELS, CHANNEL_INPUT, ChannelCapability, ChannelDefinition, ChannelReady, InboundIdentity, StopReceipt
    from agent.plugins.manager import PluginManager
    from bus.event_bus import EventBus
    from plugins.channels import plugin as channels_plugin
    from plugins.akashic_clients.runtime_inspection import RuntimeInspectionError, ScopedRpcRuntimeInspection
    from agent.plugin_composition.runtime_catalog import RUNTIME_MCP_DETAIL
    from tests.fixtures.plugin_workspace import initialize_plugin_workspace

    source = tmp_path / "plugins" / "probe"
    source.mkdir(parents=True)
    source.joinpath("plugin.py").write_text(
        "api_version = 3\nname = 'probe'\nversion = '1.0.0'\n"
        "async def apply(ctx):\n    pass\n", encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    initialize_plugin_workspace(workspace)
    manager = PluginManager([source.parent], event_bus=EventBus(), workspace=workspace,
                            installed_cache_root=tmp_path / "cache")
    contexts = []
    opened = asyncio.Event()

    class Adapter:
        def __init__(self, context):
            self.context = context
            contexts.append(context)

        async def start(self):
            return ChannelReady(self.context.binding_token)

        async def deliver(self, request):
            raise AssertionError("目录测试不发送外部消息")

        def attach_runtime(self, ports):
            self.ports = ports

        def open_admission(self):
            opened.set()

        def close_admission(self):
            pass

        async def stop(self):
            return StopReceipt(self.context.binding_token, True)

    try:
        await manager.load_all()
        root = manager.live_root
        assert root is not None
        runtime = PluginRuntime("client", "client:g1", source, tmp_path / "data", workspace, {})
        async def reject_input(*args, **kwargs):
            raise AssertionError("目录测试不接受输入消息")

        await root.context.provide(CHANNEL_INPUT, reject_input)
        await root.mount(channels_plugin.apply, name="channels", inject=channels_plugin.inject, runtime=runtime)

        async def contribute(ctx):
            await ctx.require(CHANNELS).register(
                ctx, ChannelDefinition("catalog", frozenset({ChannelCapability.INBOUND}), Adapter,
                                       InboundIdentity.PROVIDER_MESSAGE_ID),
            )

        fiber = await root.mount(contribute, name="client", inject=(CHANNELS, CHANNEL_INPUT, RUNTIME_CATALOG, RUNTIME_MCP_DETAIL), runtime=runtime)
        assert contexts, root.receipt().incidents
        await asyncio.wait_for(opened.wait(), 5)
        context = contexts[0]
        async with context.open_scope() as request:
            reader = request.require(RUNTIME_CATALOG)
            catalog = reader(request)
            assert next(item for item in _rows(catalog["plugins"]) if item["id"] == "probe")
            assert not hasattr(request, "root_instance_token")

            async def unowned_child():
                with pytest.raises(CompositionError, match="请求作用域已关闭"):
                    reader(request)

            await asyncio.create_task(unowned_child())
        with pytest.raises(CompositionError, match="请求作用域已关闭"):
            reader(request)

        # 没有 MCP provider 应按既有 unavailable 合同返回，而非 Context AttributeError。
        inspection = ScopedRpcRuntimeInspection(context.open_scope)
        for call in (inspection.list_capabilities(), inspection.get_mcp("probe", "server")):
            with pytest.raises(RuntimeInspectionError) as missing:
                await call
            assert missing.value.code == "mcp_provider_unavailable"
        await fiber.dispose()
        with pytest.raises(KeyError, match="catalog"):
            async with context.open_scope():
                raise AssertionError("旧 channel 不应再接纳请求")
    finally:
        await manager.terminate_all()
