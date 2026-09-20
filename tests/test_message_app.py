"""临时 workspace 中运行真正 App 装配和控制 socket。"""
import httpx
import pytest

from akashic_sdk import AsyncAkashic
from agent.config_models import Config
from bootstrap.app import AppRuntime
from session.log import MessageLog
from tests.fixtures.formal_plugins import FULL_RUNTIME_PLUGINS, install_formal_plugins


@pytest.mark.asyncio
async def test_app_starts_web_and_control_with_message_owners(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plugin_home, _ = install_formal_plugins(
        tmp_path,
        FULL_RUNTIME_PLUGINS,
        configure_materials=True,
        initialize_persona=True,
    )
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(plugin_home))
    app = AppRuntime(Config(), workspace)
    try:
        await app.start()
        async with await AsyncAkashic.connect(str(app.app_server.endpoint)) as client:
            session = (await client.session_create())["session_id"]
            await client.message_send(session, "App 实际输入", message_id="app-input")
            page = await client.message_read(session)
            assert page["items"][0]["id"] == "app-input"
        chat_socket = workspace / "runtime" / "chat.sock"
        assert chat_socket.is_socket()
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=str(chat_socket)),
            base_url="http://testserver",
        ) as web:
            sessions = await web.get("/api/chat/sessions")
            assert sessions.status_code == 200
            assert any(item["key"] == session for item in sessions.json()["items"])
            messages = await web.get(f"/api/chat/sessions/{session}/messages")
            assert messages.status_code == 200
            assert messages.json()["items"][0]["id"] == "app-input"
    finally:
        await app.shutdown()
    log = MessageLog(workspace / "sessions.db")
    try:
        assert log.reader(session).snapshot()[0].message_id == "app-input"
    finally:
        log.close()
