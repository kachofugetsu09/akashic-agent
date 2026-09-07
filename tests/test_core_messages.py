"""正式 Core 构造只取得消息与资源 owner，不重开旧回复执行权。"""
import shutil
from contextlib import closing
import sqlite3
from pathlib import Path

import pytest

from agent.config_models import Config
from agent.plugins.snapshot import lease_runtime_snapshot
from bootstrap import tools as bootstrap
from core.net.http import SharedHttpResources
from plugins.conversation.plugin import CONVERSATION
from agent.plugin_composition.bindings import BINDINGS
from session.log import MessageCatalog, MessageLog
from session.message import ContentPart, Input
from session.store import SessionStore


@pytest.mark.asyncio
async def test_core_opens_message_schema_and_real_source_without_legacy_execution(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins/conversation", source / "conversation")
    shutil.copytree(Path(__file__).parents[1] / "plugins/sources", source / "sources")
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setattr(bootstrap, "_resolve_plugin_dirs", lambda _: [source])
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(system_prompt=""), workspace, http,
                                        clear_stale_session_admissions=True)
    try:
        await core.start()
        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
            async with snapshot.composition_root.context.runtime_scope():
                source = snapshot.composition_root.context.require(CONVERSATION)("local:one")
                message = await source.accept("input-one", Input((ContentPart("text", "保存原始输入"),)))
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
        bootstrap.build_core_runtime(Config(system_prompt=""), workspace, SharedHttpResources(),
                                     clear_stale_session_admissions=True)
    with closing(sqlite3.connect(workspace / "sessions.db")) as connection:
        assert tuple(connection.iterdump()) == before


@pytest.mark.asyncio
async def test_core_loads_complete_builtin_message_composition(tmp_path, monkeypatch):
    """完整内置候选必须同时装配，防止分项夹具遗漏依赖冲突。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    from bootstrap.init_workspace import init_workspace
    _ = init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    source = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins", source,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setattr(bootstrap, "_resolve_plugin_dirs", lambda _: [source])
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(system_prompt=""), workspace, http)
    try:
        await core.start()
        snapshot = core.plugin_manager.current_snapshot
        assert snapshot is not None
        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as lease:
            async with lease.composition_root.context.runtime_scope():
                conversation = lease.composition_root.context.require(CONVERSATION)("local:one")
                message = await conversation.accept("input-one", Input((ContentPart("text", "尚未启动回复服务"),)))
        assert MessageCatalog(core.message_log).reader("local:one").snapshot() == (message,)
        from plugins.tools.plugin import TOOLS
        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as lease:
            ctx = lease.composition_root.context
            async with ctx.runtime_scope():
                tools = ctx.require(TOOLS)
                binding = tools.bind("message_push", ctx.require(BINDINGS))
                async def authorize(binding, arguments):
                    return {"approved": True}
                result = await tools.execution(authorize).execute("offline-push", binding,
                    {"target_channel": "akashic", "target_chat_id": "room", "message": "离线时也保存"})
                assert result.outcome == "success"
        pushed = MessageCatalog(core.message_log).reader("akashic:room").snapshot()
        assert len(pushed) == 1 and pushed[0].body.parts[0].value == "离线时也保存"
        from plugins.context.materials import MATERIALS
        assert snapshot.composition_root.context.require(MATERIALS) is not None
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


@pytest.mark.asyncio
async def test_default_runtime_starts_settings_without_embedding(tmp_path, monkeypatch):
    """首次真实内置组合的后台生命周期完成，未配置记忆不会阻断模型设置。"""
    from agent.plugin_composition import MODEL_CATALOG
    from bootstrap.init_workspace import init_workspace

    workspace = tmp_path / "workspace"
    _ = init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(system_prompt=""), workspace, http)
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            catalog = ctx.require(MODEL_CATALOG).snapshot()
            assert not catalog.role_bindings
            assert catalog.default_embedding_model_id is None
            health = [item for item in snapshot.composition_root.receipt().health if item.owner == "akasha"]
            assert len(health) == 1 and not health[0].required and not health[0].healthy
            assert "embedding" in health[0].reason
        assert not (workspace / "memory/akasha.db").exists()
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()


@pytest.mark.asyncio
async def test_saved_embedding_enables_same_root_and_space_change_preserves_graph(tmp_path, monkeypatch):
    """真实设置服务保存后启用记忆；换空间时不发请求、不改原图。"""
    from aiohttp import web
    from agent.plugin_composition import AddConnection, AddModel, SetDefaultModel, UpdateConnection, ModelKind, ModelCapabilities, CapabilitySources, ModelUnavailableError
    from agent.plugins.model_control import RuntimeModelControl
    from bootstrap.init_workspace import init_workspace
    from plugins.context.materials import MATERIALS
    from plugins.akasha.infrastructure.persistence import logical_state_sha256
    from plugins.content.plugin import CONTENT
    from session.message import Output

    import asyncio
    learned = asyncio.Event()
    calls = []
    authorization = []
    async def embeddings(request):
        body = await request.json()
        calls.append(body)
        authorization.append(request.headers.get("Authorization"))
        if "saved answer" in body["input"]:
            learned.set()
        return web.json_response({"data": [{"index": i, "embedding": [0.6, 0.8]}
            for i, _ in enumerate(body["input"])], "usage": {"prompt_tokens": 2, "total_tokens": 2}})
    provider = web.Application()
    async def models(request):
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
    _ = init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    source = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins", source,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    monkeypatch.setattr(bootstrap, "_resolve_plugin_dirs", lambda _: [source])
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(system_prompt=""), workspace, http)
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        root = core.plugin_manager.current_snapshot
        control = RuntimeModelControl(core.plugin_manager.snapshot_store)
        await control.apply(AddConnection(0, "local", "Local", "openai-compatible",
            f"http://127.0.0.1:{port}/v1", "fixture", {"api_key": "fixture"}))
        capabilities = ModelCapabilities(embedding_dimensions=2, embedding_normalization="unit")
        await control.apply(AddModel(1, "first", "local", ModelKind.EMBEDDING, "first", capabilities, CapabilitySources()))
        await control.apply(SetDefaultModel(2, None, "first"))
        assert core.plugin_manager.current_snapshot is root
        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
            ctx = snapshot.composition_root.context
            async with ctx.runtime_scope():
                async with ctx.require(MATERIALS).bind() as materials:
                    await materials.prepare((), "conversation")
                # 输入直接写 fixture Session，避免未配置聊天模型干扰本次记忆验收。
                async with ctx.require(CONTENT).bind() as content:
                    core.message_log.writer("fixture", author="user", source="conversation", body_types=(Input,),
                        content=content.checks).append("u", Input((ContentPart("text", "saved memory"),)))
                    core.message_log.writer("fixture", author="assistant", source="conversation", body_types=(Output,),
                        content=content.checks, check_call=lambda call: None).append("a", Output((ContentPart("text", "saved answer"),), "complete"))
                await asyncio.wait_for(learned.wait(), 10)
                async with ctx.require(MATERIALS).bind() as materials:
                    await materials.prepare(core.message_log.reader("fixture").snapshot(), "conversation")
                assert any("saved answer" in call["input"] for call in calls)
                assert all(item.healthy for item in snapshot.composition_root.receipt().health if item.owner == "akasha")
                graph = workspace / "memory/akasha.db"
                before = logical_state_sha256(graph)
                from plugins.tools.plugin import TOOLS
                from agent.plugins.archive import PluginArchive
                tools = ctx.require(TOOLS)
                binding = tools.bind("recall_memory", ctx.require(BINDINGS))
                saved = ctx.require(BINDINGS).describe(binding, TOOLS)["state"]["embedding_binding"]
                descriptor = core.message_log.read_binding(saved)
                archive = PluginArchive(workspace / "runtime/plugin-archives")
                components = archive.read_descriptor(descriptor["root_ref"])["components"]
                assert {archive.read_descriptor(ref)["plugin_id"] for ref in components} == {"models", "openai-compatible"}
                outer = core.message_log.read_binding(binding)
                outer_components = archive.read_descriptor(outer["root_ref"])["components"]
                assert "openai-compatible" not in {archive.read_descriptor(ref)["plugin_id"] for ref in outer_components}
                shutil.rmtree(source / "openai_compatible")
                shutil.rmtree(source / "models")
                async def authorize(binding, arguments):
                    return {"approved": True}
                await control.apply(AddModel(3, "second", "local", ModelKind.EMBEDDING, "second", capabilities, CapabilitySources()))
                await control.apply(SetDefaultModel(4, None, "second"))
                sent = len(calls)
                async with ctx.require(MATERIALS).bind() as materials:
                    result = await materials.prepare(core.message_log.reader("fixture").snapshot(), "conversation")
                status = next(part.value for part in result.context if part.kind == "akasha.status")
                assert status["available"] is False and "重建" in status["reason"]
                assert logical_state_sha256(graph) == before and len(calls) == sent
                recalled = await tools.execution(authorize).execute("old-model-after-default-switch", binding, {"query": "saved memory"})
                assert recalled.outcome == "success" and calls[-1]["model"] == "first"
                assert any(part.kind == "akasha.recall" for part in recalled.parts)
                await control.apply(SetDefaultModel(5, None, "first"))
                async with ctx.require(MATERIALS).bind() as materials:
                    result = await materials.prepare(core.message_log.reader("fixture").snapshot(), "conversation")
                assert not any(part.kind == "akasha.status" for part in result.context)
                assert all(item.healthy for item in snapshot.composition_root.receipt().health if item.owner == "akasha")
                assert logical_state_sha256(graph) == before
                assert core.plugin_manager.current_snapshot is root
                # 当前同名连接配置漂移不能重定向已经准备的模型调用。
                await control.apply(UpdateConnection(6, "local", "Local", "fixture", endpoint=f"http://127.0.0.1:{port}/changed/v1"))
                sent = len(calls)
                with pytest.raises(ModelUnavailableError, match="配置已变化"):
                    await tools.execution(authorize).execute("endpoint-drift", binding, {"query": "saved memory"})
                assert len(calls) == sent and logical_state_sha256(graph) == before
                await control.apply(UpdateConnection(7, "local", "Local", "fixture", endpoint=f"http://127.0.0.1:{port}/v1"))
                # 同一 auth identity 的 token 刷新是既有凭据 owner 的正常路径。
                from plugins.models.store import ModelsStore
                registry = ModelsStore(workspace / "model-registry.sqlite3", backup_dir=workspace / "runtime/model-backups", writable=True)
                revision = (await control.catalog()).revision
                await registry.credential_handle("local", "fixture").refresh({"api_key": "rotated"})
                assert (await control.catalog()).revision == revision
                refreshed = await tools.execution(authorize).execute("refreshed-token", binding, {"query": "saved memory"})
                assert refreshed.outcome == "success" and authorization[-1] == "Bearer rotated"
                assert logical_state_sha256(graph) == before

    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("sender_enabled", [False, True])
async def test_app_checks_sender_before_starting_native_receiver(tmp_path, monkeypatch, sender_enabled):
    """实际 App 装配拒绝缺失 Sender，收件渠道没有提前联网。"""
    from agent.config_models import TelegramChannelConfig
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from infra.channels.telegram_channel import TelegramChannel

    workspace = tmp_path / "workspace"
    _ = init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    started = []
    async def start(self, context):
        started.append(self.name)
        if not sender_enabled:
            raise AssertionError("receiver must not start")
    monkeypatch.setattr(TelegramChannel, "start", start)
    config = Config(system_prompt="")
    config.channels.chat.enabled = False
    config.channels.telegram = TelegramChannelConfig(token="fixture:token", channel_name="private_bot")
    if sender_enabled:
        sender = workspace / "plugin-data/telegram_sender-builtin/config.local.toml"
        sender.parent.mkdir(parents=True, exist_ok=True)
        sender.write_text('enabled = true\nchannel = "private_bot"\ntoken = "fixture:token"\n')
    app = AppRuntime(config, workspace)
    try:
        if sender_enabled:
            await app.start()
            assert started == ["private_bot"]
        else:
            with pytest.raises(RuntimeError, match="private_bot"):
                await app.start()
            assert not started
        assert app.app_server is not None
    finally:
        await app.shutdown()


@pytest.mark.asyncio
async def test_app_real_socket_default_reply_and_shutdown(tmp_path, monkeypatch):
    """真实 App、SDK、内置回复和本地 provider 串通，并清理本次 readiness。"""
    import asyncio
    import json
    import socket
    from aiohttp import web
    from akashic_sdk import AsyncAkashic
    from agent.plugin_composition import AddConnection, AddModel, SetDefaultModel, ModelKind, ModelRole, ModelCapabilities, CapabilitySources
    from agent.plugins.model_control import RuntimeModelControl
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
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
    _ = init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    ready = asyncio.Event()
    class Readiness(RuntimeReadiness):
        def mark_ready(self):
            super().mark_ready()
            ready.set()
    readiness = Readiness(workspace, "fixture-app")
    config = Config(system_prompt="")
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
        control = RuntimeModelControl(app.core.plugin_manager.snapshot_store)
        await control.apply(AddConnection(0, "local", "Local", "openai-compatible",
            f"http://127.0.0.1:{port}/v1", "fixture", {"api_key": "fixture"}))
        await control.apply(AddModel(1, "chat", "local", ModelKind.CHAT, "fixture",
            ModelCapabilities(context_window=32000, max_output_tokens=1024, supports_tool_calls=True), CapabilitySources()))
        await control.apply(SetDefaultModel(2, ModelRole.DEFAULT, "chat"))
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
            snapshot = app.core.plugin_manager.current_snapshot
            module = next(item for item in snapshot.web_ui_catalog.modules if item.plugin_id == "workbench-ui")
            headers = {"x-akashic-web-snapshot": snapshot.snapshot_id,
                "x-akashic-web-catalog": snapshot.web_ui_catalog.identity,
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
