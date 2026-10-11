"""独立的只读 Unix Gateway 示例；只公开会话和消息查询，不接纳写入。"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import Context
from plugins.ledger.contract import MESSAGE_CATALOG
from plugins.ui.contract import message_rows, session_row

api_version = 3
name = "readonly-gateway"
version = "1.0.0"
inject = (MESSAGE_CATALOG,)


class Sessions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    cursor: list[str] | None = Field(default=None, min_length=2, max_length=2)
    limit: int = Field(default=50, ge=1, le=200)


class Messages(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    session_id: str = Field(min_length=1, max_length=512)
    after_seq: int = Field(default=-1, ge=-1)
    through_seq: int | None = Field(default=None, ge=-1)
    limit: int = Field(default=50, ge=1, le=200)


class Initialize(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    protocolVersion: Literal["2.0"]
    clientInfo: dict[str, str]
    workspaceToken: None = None


async def apply(ctx: Context) -> None:
    """每个请求借用本代许可，监听和已接纳连接由同一个 effect 关闭。"""
    catalog = ctx.require(MESSAGE_CATALOG)
    endpoint = ctx.data_root / "readonly.sock"
    tasks: set[asyncio.Task] = set()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        tasks.add(task)
        state = "new"
        try:
            while raw := await reader.readline():
                request = json.loads(raw)
                if request.get("method") == "initialized" and state == "waiting":
                    state = "ready"
                    continue
                identity = request["id"]
                method, params = request["method"], request.get("params", {})
                response = {"jsonrpc": "2.0", "id": identity}
                try:
                    async with ctx.runtime_scope():
                        if method == "initialize" and state == "new":
                            Initialize.model_validate(params)
                            state = "waiting"
                            result = {"protocolVersion": "2.0", "serverInfo": {"name": name, "version": version},
                                      "workspace": str(ctx.runtime.workspace), "capabilities": {"messageLog": True}}
                        elif state != "ready":
                            raise ValueError("complete initialize/initialized first")
                        elif method == "session/list":
                            query = Sessions.model_validate(params)
                            page = catalog.sessions(visibility="listed", limit=query.limit,
                                after=None if query.cursor is None else (query.cursor[0], query.cursor[1]))
                            result = {"version": 2, "items": [session_row(item) for item in page.items],
                                      "total": page.total, "next_cursor": page.next_cursor}
                        elif method == "message/read":
                            query = Messages.model_validate(params)
                            page = catalog.reader(query.session_id).read_page(after_seq=query.after_seq,
                                through_seq=query.through_seq, limit=query.limit)
                            result = {"version": 2, "session_id": query.session_id, "items": message_rows(page),
                                "after_seq": query.after_seq, "through_seq": page.through_seq,
                                "next_after_seq": page.messages[-1].seq if page.messages else query.after_seq,
                                "has_more": page.has_more}
                        else:
                            response["error"] = {"code": -32601, "message": "read-only Gateway: method unavailable"}
                        if "error" not in response:
                            response["result"] = result
                except (ValidationError, ValueError) as error:
                    response["error"] = {"code": -32602, "message": str(error)}
                writer.write(json.dumps(response, ensure_ascii=False).encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            finally:
                tasks.remove(task)

    async def start():
        server = await asyncio.start_unix_server(serve, path=str(endpoint))
        os.chmod(endpoint, 0o600)
        async def close():
            server.close()
            await server.wait_closed()
            for task in tuple(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Python 3.13 的 Unix Server 会自动删除 socket；3.12 由 owner 清理。
            endpoint.unlink(missing_ok=True)
        return close

    await ctx.effect(start, label="readonly-gateway")
    await ctx.endpoint("gateway", protocol="jsonrpc+unix", address=str(endpoint))
