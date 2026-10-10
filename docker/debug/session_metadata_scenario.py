"""真实 SQLite、Chat HTTP/WebSocket 与可选浏览器夹具；不连接正式 workspace。"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager, closing
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from agent.plugin_composition.messages import SessionAdmin
from plugins.akashic_clients.chat_api import create_chat_app
from plugins.akashic_clients.navigation import NavigationPreferences
from plugins.akashic_clients.web_chat import WebChatChannel
from session.log import MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Input, Output


def append(log, key, text, *, output=False):
    log.ensure_session(key, SessionAttributes())
    writer = log.writer(key, author="assistant" if output else "user", source="conversation",
        body_types=(Output,) if output else (Input,), content={"text": lambda _: ContentReferences()})
    body = Output((ContentPart("text", text),), "complete") if output else Input((ContentPart("text", text),))
    return writer.append(f"{key}:{log.reader(key).head() + 1}", body)


def fixture(log, workspace, static=None):
    """只补隔离场景的管理入口；目录读取与通知使用正式 Chat API。"""
    admin = SessionAdmin(log)

    @asynccontextmanager
    async def admins():
        yield admin

    app = create_chat_app(workspace=workspace, channel=WebChatChannel(),
        messages=log.catalog(), session_admin_scope=admins,
        navigation=NavigationPreferences(lambda: log.owner("scenario-navigation")))

    @app.post("/__scenario/auto-title")
    async def auto_title(title: str, session: str = "akashic:auto"):
        return {"written": await admin.set_title_if_unset(session, title)}

    @app.get("/__scenario/state")
    def state():
        with log._listener_lock:
            listeners = len(log._listeners)
        return {"listeners": listeners, "heads": dict(log.catalog().snapshot_heads())}

    @app.get("/api/shell/state")
    def shell_state():
        return {"status": "ready", "configured": True, "chatReady": True}

    if static is not None:
        app.router.routes[:] = [route for route in app.router.routes if route.path not in ("/", "/assets")]
        app.mount("/assets", StaticFiles(directory=static))

        @app.get("/")
        def index():
            return FileResponse(static / "index.html")

    return app


def verify(log, workspace):
    """双连接观察标题与删除提交，断线重连补读，并核对消息及目录排序保全。"""
    before = tuple(log.reader(key).read() for key in ("akashic:auto", "akashic:other"))
    times = tuple(log._connection.execute("SELECT key,updated_at FROM sessions ORDER BY key"))
    changed = {"type": "sessions.changed", "version": 2}
    app = fixture(log, workspace)
    with TestClient(app) as client:
        with client.websocket_connect("/ws?watch_sessions=true") as first, client.websocket_connect("/ws?watch_sessions=true") as second:
            assert first.receive_json() == second.receive_json() == changed
            # 1. 首条回复已完成；自动命名不依赖活跃会话或新的消息。
            assert client.post("/__scenario/auto-title", params={"title": "迟到的自动标题"}).json() == {"written": True}
            assert first.receive_json() == second.receive_json() == changed
            rows = client.get("/api/chat/sessions").json()["items"]
            assert next(row for row in rows if row["key"] == "akashic:auto")["title"] == "迟到的自动标题"
            # 2. HTTP 手动改名、软删与恢复共享同一提交通知。
            for action, body in (("rename", {"title": "另一窗口改名"}), ("delete", None), ("undelete", None)):
                response = client.post(f"/api/chat/sessions/akashic:other/{action}", json=body)
                assert response.status_code == 200, response.text
                assert first.receive_json() == second.receive_json() == changed
            # 3. 相同标题和竞争失败不发通知；ping 作为处理完成的连接屏障。
            assert client.post("/__scenario/auto-title", params={"title": "不可覆盖"}).json() == {"written": False}
            assert client.post("/api/chat/sessions/akashic:other/rename", json={"title": "另一窗口改名"}).status_code == 200
            first.send_json({"type": "ping", "request_id": "no-change"})
            assert first.receive_json() == {"type": "pong", "request_id": "no-change"}
        assert client.get("/__scenario/state").json()["listeners"] == 0
        # 4. 离线修改通过重新订阅的首次失效通知补读；旧客户端只收到原协议。
        assert client.post("/api/chat/sessions/akashic:auto/rename", json={"title": "离线期间改名"}).status_code == 200
        with client.websocket_connect("/ws?watch_sessions=true") as connected:
            assert connected.receive_json() == changed
            rows = client.get("/api/chat/sessions").json()["items"]
            assert next(row for row in rows if row["key"] == "akashic:auto")["title"] == "离线期间改名"
        with client.websocket_connect("/ws") as legacy:
            legacy.send_json({"type": "ping", "request_id": "legacy"})
            assert legacy.receive_json() == {"type": "pong", "request_id": "legacy"}
        assert client.get("/__scenario/state").json()["listeners"] == 0
    assert before == tuple(log.reader(key).read() for key in ("akashic:auto", "akashic:other"))
    assert times == tuple(log._connection.execute("SELECT key,updated_at FROM sessions ORDER BY key"))
    # 5. 消息追加沿原消息流，不额外广播元数据变化。
    with TestClient(app) as client, client.websocket_connect("/ws?watch_sessions=true") as connected:
        assert connected.receive_json() == changed
        append(log, "akashic:auto", "普通后续消息")
        connected.send_json({"type": "ping", "request_id": "message-only"})
        assert connected.receive_json() == {"type": "pong", "request_id": "message-only"}
    print(json.dumps({"status": "passed", "connections": 2, "messages_preserved": 4,
        "updated_at_preserved": True, "reconnect_refreshed": True, "listeners_after_close": 0,
        "metadata_notification_on_message_append": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=int, metavar="PORT")
    parser.add_argument("--static", type=Path, help="npm run build:chat 产物目录")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="akashic-session-metadata-") as directory:
        workspace = Path(directory)
        with closing(MessageLog(workspace / "sessions.db")) as log:
            for key, text in (("akashic:auto", "等待自动标题"), ("akashic:other", "另一个会话")):
                append(log, key, text)
                append(log, key, "正文回复已完成", output=True)
            if args.serve:
                import uvicorn
                uvicorn.run(fixture(log, workspace, args.static), host="127.0.0.1", port=args.serve, log_level="warning")
            else:
                verify(log, workspace)
