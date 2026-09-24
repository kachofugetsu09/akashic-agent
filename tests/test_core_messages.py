"""正式 Core 构造只取得消息与资源 owner，不重开旧回复执行权。"""
import asyncio

from agent.plugin_composition.ui import UI
from contextlib import closing
import sqlite3
from collections.abc import Mapping

import pytest

from agent.config_models import Config
from bootstrap import tools as bootstrap
from bootstrap.app_server import build_control_service
from core.net.http import SharedHttpResources
from plugins.sources.plugin import SOURCES
from agent.plugin_composition.bindings import BINDINGS
from session.log import MessageCatalog, MessageLog
from session.message import ContentPart, Control, Input
from session.store import SessionStore
from tests.fixtures.formal_plugins import (
    FULL_RUNTIME_PLUGINS,
    MINIMAL_MESSAGE_PLUGINS,
    install_formal_plugins,
)


async def _model_command(core, payload: dict[str, object]) -> dict[str, object]:
    """Configure the installed Models owner through its public RPC boundary."""

    service = build_control_service(core)
    resolve = service.resolve_method
    if resolve is None:
        raise AssertionError("ControlService 未提供动态 RPC resolver")
    async with resolve("models/command") as operation:
        if operation is None:
            raise AssertionError("live Root 未提供 models/command")
        params = operation.params.model_validate(payload)
        result = await operation.invoke(params, None)
    assert isinstance(result, dict)
    assert result.get("status") == 200, result
    body = result.get("body")
    assert isinstance(body, dict)
    return body


async def _registered_source(manager, name):
    generation = manager.generation("sources@fixture")
    if generation is None or generation.fiber is None:
        raise AssertionError("sources generation 未建立")
    context = generation.fiber.context
    async with context.runtime_scope():
        matches = tuple(item for item in context.require(SOURCES).entries()
                        if item.name == name)
    assert len(matches) == 1
    return matches[0]


@pytest.mark.asyncio
async def test_core_opens_message_schema_and_real_source_without_legacy_execution(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_home, _ = install_formal_plugins(tmp_path, MINIMAL_MESSAGE_PLUGINS)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http,
                                        clear_stale_session_admissions=True,
                                        plugin_dirs=[])
    try:
        await core.start()
        source = await _registered_source(core.plugin_manager, "conversation")
        async with source.context.runtime_scope():
            message = await source.open("local:one").accept(
                "input-one", Input((ContentPart("text", "保存原始输入"),)),
            )
        assert isinstance(message.body, Input)
        assert MessageCatalog(core.message_log).reader("local:one").snapshot() == (message,)
        assert "conversation" in await core.inspect_modules()
        with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert not tables & {"turns", "turn_items", "turn_events", "turn_requests"}
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()
    reopened = MessageLog(workspace / "sessions.db")
    try:
        assert MessageCatalog(reopened).reader("local:one").snapshot() == (message,)
    finally:
        reopened.close()


def test_core_rejects_legacy_schema_before_admission_cleanup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    legacy = SessionStore(workspace / "sessions.db")
    legacy.close()
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        before = tuple(connection.iterdump())
    with pytest.raises(RuntimeError, match="schema|迁移"):
        bootstrap.build_core_runtime(Config(), workspace, SharedHttpResources(),
                                     clear_stale_session_admissions=True)
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        assert tuple(connection.iterdump()) == before


@pytest.mark.asyncio
async def test_core_loads_complete_builtin_message_composition(tmp_path, monkeypatch):
    """完整内置候选必须同时装配，防止分项夹具遗漏依赖冲突。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_home, _ = install_formal_plugins(
        tmp_path, FULL_RUNTIME_PLUGINS, configure_materials=True,
        initialize_persona=True,
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http, plugin_dirs=[])
    try:
        await core.start()
        root = core.plugin_manager.live_root
        assert root is not None
        source = await _registered_source(core.plugin_manager, "conversation")
        async with source.context.runtime_scope():
            message = await source.open("local:one").accept(
                "input-one", Input((ContentPart("text", "尚未启动回复服务"),)),
            )
        assert MessageCatalog(core.message_log).reader("local:one").snapshot() == (
            message,
        )

        async def reply_stopped():
            async for recorded in MessageCatalog(core.message_log).reader("local:one").follow(
                after_seq=message.seq,
            ):
                if isinstance(recorded.body, Control) and recorded.body.action == "failure":
                    return recorded

        failure = await asyncio.wait_for(reply_stopped(), 10)
        assert failure is not None
        assert failure.source == "conversation"
        from plugins.tools.plugin import ALL_TOOLS, TOOLS

        tools_context, tools = root._service_provider(TOOLS)
        view_context, all_tools = root._service_provider(ALL_TOOLS)
        async with view_context.runtime_scope():
            push = all_tools().select("message_push")
        binding = await tools.bind_scoped(push, tools_context.require(BINDINGS))
        async with tools_context.runtime_scope():
            async def authorize(binding, arguments):
                return {"approved": True}
            result = await tools.execution(authorize).execute("offline-push", binding,
                {"target_channel": "akashic", "target_chat_id": "room", "message": "离线时也保存"})
            assert result.outcome == "success"
        pushed = MessageCatalog(core.message_log).reader("akashic:room").snapshot()
        assert len(pushed) == 1 and pushed[0].body.parts[0].value == "离线时也保存"
        from plugins.context.materials import MATERIALS
        assert root.context.require(MATERIALS) is not None
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


@pytest.mark.asyncio
async def test_default_runtime_starts_settings_without_embedding(tmp_path, monkeypatch):
    """首次真实内置组合的后台生命周期完成，未配置记忆不会阻断模型设置。"""
    from agent.plugin_composition import MODEL_CATALOG

    workspace = tmp_path / "workspace"
    plugin_home, _ = install_formal_plugins(
        tmp_path, FULL_RUNTIME_PLUGINS, configure_materials=True,
        initialize_persona=True,
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http, plugin_dirs=[])
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        root = core.plugin_manager.live_root
        assert root is not None
        catalog = root.context.require(MODEL_CATALOG).snapshot()
        assert not catalog.role_bindings
        assert catalog.default_embedding_model_id is None
        health = [item for item in root.receipt().health if item.owner == "akasha@fixture"]
        assert len(health) == 1 and not health[0].required and not health[0].healthy
        assert health[0].reason is not None and "embedding" in health[0].reason
        assert not (workspace / "memory/akasha.db").exists()
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


@pytest.mark.asyncio
async def test_saved_embedding_enables_same_root_and_space_change_preserves_graph(tmp_path, monkeypatch):
    """真实设置服务保存后启用记忆；换空间时不发请求、不改原图。"""
    from aiohttp import web
    from agent.plugin_composition import MODEL_CATALOG, ModelUnavailableError
    from agent.plugins.archive import PluginArchive
    from plugins.akasha.infrastructure.persistence import logical_state_sha256
    from plugins.context.materials import MATERIALS
    from plugins.content.plugin import CONTENT
    from plugins.tools.plugin import ALL_TOOLS, TOOLS
    from session.message import Output

    learned = asyncio.Event()
    calls = []
    authorization = []

    async def embeddings(request):
        body = await request.json()
        calls.append(body)
        authorization.append(request.headers.get("Authorization"))
        if "saved answer" in body["input"]:
            learned.set()
        return web.json_response({
            "data": [{"index": i, "embedding": [0.6, 0.8]}
                     for i, _ in enumerate(body["input"])],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })

    provider = web.Application()

    async def models(_request):
        return web.json_response({"data": [{"id": "first"}, {"id": "second"}]})

    provider.router.add_get("/v1/models", models)
    provider.router.add_get("/changed/v1/models", models)
    provider.router.add_post("/v1/embeddings", embeddings)
    runner = web.AppRunner(provider)
    await runner.setup()
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()
    workspace = tmp_path / "workspace"
    plugin_home, _ = install_formal_plugins(
        tmp_path, FULL_RUNTIME_PLUGINS, configure_materials=True,
        initialize_persona=True,
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http, plugin_dirs=[])
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        root = core.plugin_manager.live_root
        assert root is not None
        await _model_command(core, {
            "type": "add_connection", "expected_revision": 0,
            "connection_id": "local", "name": "Local",
            "driver_id": "openai-compatible",
            "endpoint": f"http://127.0.0.1:{port}/v1",
            "auth_identity": "fixture", "credential": {"api_key": "fixture"},
        })
        capabilities = {"embedding_dimensions": 2, "embedding_normalization": "unit"}
        await _model_command(core, {
            "type": "add_model", "expected_revision": 1,
            "model_id": "first", "connection_id": "local", "kind": "embedding",
            "model": "first", "capabilities": capabilities, "capability_sources": {},
        })
        await _model_command(core, {
            "type": "set_default", "expected_revision": 2,
            "role": None, "model_id": "first",
        })
        assert core.plugin_manager.live_root is root
        materials_context, materials_owner = root._service_provider(MATERIALS)
        content_context, content = root._service_provider(CONTENT)
        tools_context, tools = root._service_provider(TOOLS)
        async with materials_owner.bind() as materials:
            await materials.prepare((), "conversation")
        async with content.bind() as content_owner:
            core.message_log.writer(
                "fixture", author="user", source="conversation", body_types=(Input,),
                content=content_owner.checks,
            ).append("u", Input((ContentPart("text", "saved memory"),)))
            core.message_log.writer(
                "fixture", author="assistant", source="conversation", body_types=(Output,),
                content=content_owner.checks, check_call=lambda call: None,
            ).append("a", Output((ContentPart("text", "saved answer"),), "complete"))
        await asyncio.wait_for(learned.wait(), 10)
        async with materials_owner.bind() as materials:
            await materials.prepare(
                core.message_log.reader("fixture").snapshot(), "conversation"
            )
        assert any("saved answer" in call["input"] for call in calls)
        akasha_generation = core.plugin_manager.generation("akasha@fixture")
        assert akasha_generation is not None and akasha_generation.fiber is not None
        akasha_fiber = akasha_generation.fiber
        akasha_health = tuple(
            item for item in akasha_fiber.root.receipt().health
            if item.owner == akasha_fiber.name
        )
        assert len(akasha_health) == 1 and akasha_health[0].healthy
        graph = workspace / "memory/akasha.db"
        before = logical_state_sha256(graph)

        async with tools_context.runtime_scope():
            bindings = tools_context.require(BINDINGS)
            all_tools = tools_context.require(ALL_TOOLS)
            binding = await tools.bind_scoped(
                all_tools().select("recall_memory"), bindings,
            )
            binding_description = bindings.describe(binding, TOOLS)
            assert isinstance(binding_description, Mapping)
            binding_state = binding_description.get("state")
            assert isinstance(binding_state, Mapping)
            saved = binding_state.get("embedding_binding")
            assert isinstance(saved, str)
            descriptor = core.message_log.read_binding(saved)
            archive = PluginArchive(workspace / "runtime/plugin-archives")
            assert isinstance(descriptor, Mapping)
            root_ref = descriptor.get("root_ref")
            assert isinstance(root_ref, str)
            root_descriptor = archive.read_descriptor(root_ref)
            components = root_descriptor.get("components")
            assert isinstance(components, (list, tuple))
            bound_plugins = {archive.read_descriptor(ref)["plugin_id"] for ref in components}
            assert {"models@fixture", "openai-compatible@fixture"} <= bound_plugins
            outer = core.message_log.read_binding(binding)
            assert isinstance(outer, Mapping)
            outer_root_ref = outer.get("root_ref")
            assert isinstance(outer_root_ref, str)
            outer_descriptor = archive.read_descriptor(outer_root_ref)
            outer_components = outer_descriptor.get("components")
            assert isinstance(outer_components, (list, tuple))
            outer_plugins = {archive.read_descriptor(ref)["plugin_id"] for ref in outer_components}
            assert bound_plugins <= outer_plugins

            async def authorize(binding, arguments):
                return {"approved": True}

            await _model_command(core, {
                "type": "add_model", "expected_revision": 3,
                "model_id": "second", "connection_id": "local", "kind": "embedding",
                "model": "second", "capabilities": capabilities, "capability_sources": {},
            })
            await _model_command(core, {
                "type": "set_default", "expected_revision": 4,
                "role": None, "model_id": "second",
            })
            sent = len(calls)
            async with materials_owner.bind() as materials:
                result = await materials.prepare(
                    core.message_log.reader("fixture").snapshot(), "conversation"
                )
            reminders = result["reminders"]
            assert isinstance(reminders, tuple)
            status = next(part["text"] for part in reminders if part["name"] == "status")
            assert "召回不可用" in status and "重建" in status
            assert logical_state_sha256(graph) == before and len(calls) == sent
            recalled = await tools.execution(authorize).execute(
                "old-model-after-default-switch", binding, {"query": "saved memory"}
            )
            assert recalled.outcome == "success" and calls[-1]["model"] == "first"
            assert any(part.kind == "akasha.recall" for part in recalled.parts)
            await _model_command(core, {
                "type": "set_default", "expected_revision": 5,
                "role": None, "model_id": "first",
            })
            async with materials_owner.bind() as materials:
                result = await materials.prepare(
                    core.message_log.reader("fixture").snapshot(), "conversation"
                )
            reminders = result["reminders"]
            assert isinstance(reminders, tuple)
            assert not any(part["name"] == "status" for part in reminders)
            akasha_health = tuple(
                item for item in akasha_fiber.root.receipt().health
                if item.owner == akasha_fiber.name
            )
            assert len(akasha_health) == 1 and akasha_health[0].healthy
            assert logical_state_sha256(graph) == before
            assert core.plugin_manager.live_root is root
            await _model_command(core, {
                "type": "update_connection", "expected_revision": 6,
                "connection_id": "local", "name": "Local", "auth_identity": "fixture",
                "endpoint": f"http://127.0.0.1:{port}/changed/v1",
            })
            sent = len(calls)
            with pytest.raises(ModelUnavailableError, match="配置已变化"):
                await tools.execution(authorize).execute(
                    "endpoint-drift", binding, {"query": "saved memory"}
                )
            assert len(calls) == sent and logical_state_sha256(graph) == before
            update = await _model_command(core, {
                "type": "update_connection", "expected_revision": 7,
                "connection_id": "local", "name": "Local", "auth_identity": "fixture",
                "endpoint": f"http://127.0.0.1:{port}/v1",
            })
            from plugins.models.store import ModelsStore
            registry = ModelsStore(
                workspace / "model-registry.sqlite3",
                backup_dir=workspace / "runtime/model-backups", writable=True,
            )
            revision = update["revision"]
            await registry.credential_handle("local", "fixture").refresh({"api_key": "rotated"})
            refreshed = await tools.execution(authorize).execute(
                "refreshed-token", binding, {"query": "saved memory"}
            )
            assert refreshed.outcome == "success" and authorization[-1] == "Bearer rotated"
            assert logical_state_sha256(graph) == before

        catalog_context, catalog = root._service_provider(MODEL_CATALOG)
        async with catalog_context.runtime_scope():
            assert catalog.snapshot().revision == revision

    finally:
        try:
            await core.bus.aclose()
        finally:
            try:
                await core.stop()
            finally:
                try:
                    await http.aclose()
                finally:
                    await runner.cleanup()


def test_telegram_channel_is_the_formal_owner_and_factory_is_closed(tmp_path):
    """Telegram ownership is a normal plugin binding, without Core construction."""

    from agent.plugin_composition import CredentialRef
    from plugins.telegram_channel.config import TelegramChannelConfig
    from plugins.telegram_channel.plugin import Config, name

    config = Config(
        enabled=True,
        token=CredentialRef(("token",)),
        allow_from=("alice",),
    )
    assert isinstance(config, TelegramChannelConfig)
    assert name == "telegram_channel"
    assert config.token == CredentialRef(("token",))


@pytest.mark.asyncio
async def test_app_real_socket_default_reply_and_shutdown(tmp_path, monkeypatch):
    """真实 App、SDK、内置回复和本地 provider 串通，并清理本次 readiness。"""
    import asyncio
    import json
    import socket
    from aiohttp import web
    from akashic_sdk import AsyncAkashic
    from bootstrap.app import AppRuntime
    from bootstrap.runtime_readiness import RuntimeReadiness

    calls = []
    async def models(request):
        return web.json_response({"data": [{"id": "fixture"}]})
    async def chat(request):
        body = await request.json()
        calls.append(body)
        assert body["stream"] is True
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "启动回复成功"}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}},
        ]
        payload = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return web.Response(text=payload, content_type="text/event-stream")
    provider = web.Application()
    provider.router.add_get("/v1/models", models)
    provider.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(provider)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()
    workspace = tmp_path / "workspace"
    plugin_home, _ = install_formal_plugins(
        tmp_path, FULL_RUNTIME_PLUGINS, configure_materials=True,
        initialize_persona=True,
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    ready = asyncio.Event()
    class Readiness(RuntimeReadiness):
        def mark_ready(self):
            super().mark_ready()
            ready.set()
    readiness = Readiness(workspace, "fixture-app")
    config = Config()
    app = AppRuntime(config, workspace, readiness=readiness)
    task = asyncio.create_task(app.run())
    try:
        ready_task = asyncio.create_task(ready.wait())
        done, _ = await asyncio.wait((task, ready_task), timeout=20,
                                     return_when=asyncio.FIRST_COMPLETED)
        ready_task.cancel()
        if task in done:
            await task
        assert ready.is_set() and readiness.path.exists()
        assert app.core is not None and app.app_server is not None
        assert app.restart_gate is not None
        assert app.core.restart_gate.boot_id == readiness.boot_id
        assert app.core.plugin_manager._host_boot_id == readiness.boot_id
        await _model_command(app.core, {
            "type": "add_connection",
            "expected_revision": 0,
            "connection_id": "local",
            "name": "Local",
            "driver_id": "openai-compatible",
            "endpoint": f"http://127.0.0.1:{port}/v1",
            "auth_identity": "fixture",
            "credential": {"api_key": "fixture"},
        })
        await _model_command(app.core, {
            "type": "add_model",
            "expected_revision": 1,
            "model_id": "chat",
            "connection_id": "local",
            "kind": "chat",
            "model": "fixture",
            "capabilities": {
                "context_window": 32000,
                "max_output_tokens": 1024,
                "supports_tool_calls": True,
            },
            "capability_sources": {},
        })
        await _model_command(app.core, {
            "type": "set_default",
            "expected_revision": 2,
            "role": "default",
            "model_id": "chat",
        })
        async with await AsyncAkashic.connect(str(app.app_server.endpoint)) as client:
            session = (await client.session_create())["session_id"]
            async with await client.session_follow(session) as following:
                ack = await client.message_send(session, "启动检查", message_id="app-input")
                assert ack["message_id"] == "app-input"
                async with asyncio.timeout(15):
                    async for event in following.events():
                        if event["type"] == "messages.appended" and any(
                            item["body"].get("finish") == "complete" for item in event["items"]
                        ):
                            break
            messages = (await client.message_read(session))["items"]
            assert any(part.get("value") == "启动回复成功" for item in messages for part in item["body"].get("parts", []))
            assert len(calls) == 1
            assert any("启动检查" in str(message["content"]) for message in calls[0]["messages"])
            import httpx
            with closing(sqlite3.connect(workspace / "sessions.db")) as database:
                before = tuple(database.iterdump())
            root = app.core.plugin_manager.live_root
            assert root is not None
            ui_context, ui = root._service_provider(UI)
            async with ui_context.runtime_scope():
                catalog = ui.catalog()
            module = next(
                item
                for item in catalog.modules
                if item.plugin_id == "workbench-ui@fixture"
            )
            headers = {"x-akashic-web-snapshot": root.generation_id,
                "x-akashic-web-catalog": catalog.identity,
                "x-akashic-web-module": module.plugin_id, "x-akashic-web-generation": module.generation_id}
            async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=app.dashboard_server.config.uds), base_url="http://fixture", headers=headers) as dashboard:
                directory = await dashboard.get("/api/dashboard/sessions")
                assert directory.status_code == 200
                assert any(item["key"] == session for item in directory.json()["items"])
                latest = await dashboard.get(f"/api/dashboard/sessions/{session}/messages", params={"limit": 1})
                assert latest.status_code == 200
                tail = latest.json()
                assert tail["has_more"] and len(tail["items"]) == 1
                assert tail["items"][0]["body"]["finish"] == "complete"
                earlier = await dashboard.get(f"/api/dashboard/sessions/{session}/messages", params={
                    "before_seq": tail["next_before_seq"], "through_seq": tail["through_seq"], "limit": 1})
                assert earlier.status_code == 200 and earlier.json()["items"][0]["id"] == "app-input"
                for method, path in [("DELETE", f"/api/dashboard/sessions/{session}"),
                                     ("PATCH", "/api/dashboard/messages/app-input"),
                                     ("POST", "/api/dashboard/messages/batch-delete")]:
                    rejected = await dashboard.request(method, path, json={})
                    assert rejected.status_code == 403
            with closing(sqlite3.connect(workspace / "sessions.db")) as database:
                assert tuple(database.iterdump()) == before
                tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                assert not tables & {"turns", "turn_items", "turn_events", "turn_requests"}
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await runner.cleanup()
    assert not readiness.path.exists()
    assert not readiness.ready
