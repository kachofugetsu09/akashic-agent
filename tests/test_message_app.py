"""临时 workspace 中运行真正 App 装配和控制 socket。"""
import shutil
from pathlib import Path

import httpx
import pytest

from akashic_sdk import AsyncAkashic
from agent.config_models import Config
from bootstrap.app import AppRuntime
from bootstrap import tools as bootstrap
from session.log import MessageLog


@pytest.mark.asyncio
async def test_app_starts_web_and_control_with_message_owners(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "plugins"
    shutil.copytree(Path(__file__).parents[1] / "plugins/conversation", source / "conversation")
    shutil.copytree(Path(__file__).parents[1] / "plugins/sources", source / "sources")
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.setattr(bootstrap, "_resolve_plugin_dirs", lambda _: [source])
    app = AppRuntime(Config(), workspace)
    try:
        await app.start()
        async with await AsyncAkashic.connect(str(app.app_server.endpoint)) as client:
            session = (await client.session_create())["session_id"]
            await client.message_send(session, "App 实际输入", message_id="app-input")
            page = await client.message_read(session)
            assert page["items"][0]["id"] == "app-input"
        assert app.web_chat_channel is not None
        assert app.channel_host.channels
        chat_app = app.chat_server.config.app
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=chat_app),
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
