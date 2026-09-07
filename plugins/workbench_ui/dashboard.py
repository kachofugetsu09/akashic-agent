"""工作台只读取当前 Message 目录，不取得数据库或数据管理权限。"""
from __future__ import annotations

from typing import Literal
from dataclasses import asdict
import json

from fastapi import FastAPI, HTTPException, Query

from agent.plugin_composition import DashboardContext
from agent.plugin_composition.messages import MESSAGE_CATALOG
from agent.plugins.snapshot import get_current_runtime_snapshot
from infra.channels.message_view import session_row
from session.log import InvalidPage, MessageCatalog
from session.message_codec import encode_body


def _catalog() -> MessageCatalog:
    snapshot = get_current_runtime_snapshot()
    if snapshot is None or snapshot.composition_root is None:
        raise RuntimeError("工作台请求缺少实际插件 snapshot")
    return snapshot.composition_root.context.require(MESSAGE_CATALOG)


def register(app: FastAPI, context: DashboardContext) -> None:
    """注册只读页面；请求 middleware 持有实际 generation，注册不打开运行数据。"""
    @app.get("/api/dashboard/sessions")
    async def list_sessions(
        prefix: str = "", visibility: Literal["listed", "internal"] | None = None,
        cursor: list[str] | None = Query(default=None, min_length=2, max_length=2),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, object]:
        try:
            page = _catalog().sessions(prefix=prefix, visibility=visibility, limit=limit,
                after=None if cursor is None else (cursor[0], cursor[1]))
        except InvalidPage as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"version": 2, "items": [{**session_row(entry), "attributes": asdict(entry.attributes)} for entry in page.items],
                "total": page.total, "next_cursor": page.next_cursor}

    @app.get("/api/dashboard/sessions/{session_id:path}/messages")
    async def list_messages(
        session_id: str, before_seq: int | None = Query(default=None, ge=0),
        through_seq: int | None = Query(default=None, ge=-1),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict[str, object]:
        try:
            page = _catalog().reader(session_id).read_tail(
                before_seq=before_seq, through_seq=through_seq, limit=limit,
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="会话不存在") from error
        except InvalidPage as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {"version": 2, "session_id": session_id, "items": [{"id": message.message_id, "session_id": message.session_id,
                    "seq": message.seq, "timestamp": message.recorded_at.isoformat(),
                    "author": message.author, "source": message.source,
                    "body": json.loads(encode_body(message.body))} for message in page.messages],
                "before_seq": before_seq, "through_seq": page.through_seq,
                "next_before_seq": page.messages[0].seq if page.messages else None,
                "has_more": page.has_more}
