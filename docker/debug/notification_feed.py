"""用临时真实消息库验证通知补发；--serve 提供隔离的 Android 通知验收入口。"""
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager, closing
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.akashic_clients.notifications import NotificationFeed, notification_events
from plugins.akashic_clients.services import MessageCatalogPort
from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Output


def append(log: MessageLog, session: str, name: str, finish: str = "complete") -> None:
    writer = log.writer(session, author="assistant", source="conversation", body_types=(Output,),
                        content={"text": lambda _: ContentReferences()})
    writer.append(name, Output((ContentPart("text", name),), finish))


def feed_for(log: MessageLog) -> NotificationFeed:
    @asynccontextmanager
    async def catalog():
        yield cast(MessageCatalogPort, log.catalog())
    return NotificationFeed(catalog, prefix="akashic:")


async def read_event(stream):
    raw = await asyncio.wait_for(anext(stream), 5)
    name, payload = raw.decode().strip().split("\n")
    return name.removeprefix("event: "), json.loads(payload.removeprefix("data: "))


async def verify(log: MessageLog) -> None:
    """覆盖分页、旧时间戳、多会话和页面中断，读取不能改写数据库。"""
    feed = feed_for(log)
    append(log, "akashic:a", "old")
    cursor = await feed.baseline()
    for i in range(235):
        append(log, "akashic:a", f"offline-{i}")
    append(log, "akashic:a", "quiet", "quiet")
    for i in range(205):
        append(log, f"akashic:session-{i}", f"session-{i}")
    append(log, "telegram:other", "other-channel")
    # 测试数据故意使用同一旧时刻，证明续传不依赖客户端或服务器时钟。
    log._connection.execute("UPDATE messages SET ts='2020-01-01T00:00:00+00:00'")
    log._connection.commit()
    before = tuple(log._connection.iterdump())
    stream = notification_events(feed, cursor, poll_interval=0, heartbeat=0)
    seen = set()
    try:
        kind, data = await read_event(stream)
        assert kind == "ready" and data["cursor"] == cursor
        while True:
            kind, data = await read_event(stream)
            if kind == "message":
                seen.add(data["message_id"])
                cursor[data["session_id"]] = data["seq"]
                if len(seen) == 17:
                    break
            elif kind == "cursor":
                cursor[data["session_id"]] = data["seq"]
    finally:
        await stream.aclose()
    # 中断发生在一页内；使用客户端已经确认的进度重连。
    stream = notification_events(feed, cursor, poll_interval=0, heartbeat=0)
    try:
        await read_event(stream)
        while True:
            kind, data = await read_event(stream)
            if kind == "heartbeat":
                break
            if kind == "message":
                assert data["message_id"] not in seen
                seen.add(data["message_id"])
                cursor[data["session_id"]] = data["seq"]
            elif kind == "cursor":
                cursor[data["session_id"]] = data["seq"]
    finally:
        await stream.aclose()
    assert seen == {f"offline-{i}" for i in range(235)} | {f"session-{i}" for i in range(205)}
    assert tuple(log._connection.iterdump()) == before
    assert log._connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    print(json.dumps({"result": "passed", "notifications": len(seen), "sessions": 206,
                      "cases": ["initial_baseline", "offline_pages", "directory_pages", "old_equal_timestamps",
                                "mid_page_disconnect", "quiet_filter", "channel_filter", "read_only_integrity"]}))


def serve(log: MessageLog, workspace: Path, port: int) -> None:
    """真机夹具使用真实 chat API 和消息库；控制入口仅监听本机回环地址。"""
    import uvicorn
    from fastapi.responses import HTMLResponse
    from plugins.akashic_clients.chat_api import create_chat_app
    from plugins.akashic_clients.web_chat import WebChatChannel

    app = create_chat_app(workspace=workspace, channel=WebChatChannel(),
                          messages=cast(MessageCatalogPort, log.catalog()))

    @app.get("/api/shell/state")
    def state():
        return {"status": "ready", "configured": True, "chatReady": True}

    @app.post("/experiment/append")
    def add(name: str, session: str = "akashic:phone"):
        append(log, session, name)
        return {"message_id": name}

    # 夹具首页不伪装聊天功能，只验证壳的导航和真实通知消费。
    for route in list(app.router.routes):
        if getattr(route, "path", None) == "/":
            app.router.routes.remove(route)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return """<html><meta name="viewport" content="width=device-width"><body>
        <h1>Akashic 通知验收</h1><p>临时消息库，不连接正式 Agent。</p><p id="session"></p>
        <script>window.akashicOpenSession=id=>{document.getElementById('session').textContent=id;return true};
        const id=new URLSearchParams(location.search).get('session');if(id)akashicOpenSession(id);</script>
        </body></html>"""

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", type=int, metavar="PORT")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="akashic-notification-check-") as root:
        with closing(MessageLog(Path(root) / "sessions.db")) as log:
            if args.serve:
                serve(log, Path(root), args.serve)
            else:
                asyncio.run(verify(log))
