"""终端的浏览器入口：列出、新开、关闭会话终端；WebSocket 承载输入、尺寸与实时输出。"""
from __future__ import annotations

import asyncio
from contextlib import suppress
import json

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from agent.plugin_composition import DashboardContext
from plugins.host_execution.contract import (
    TERMINAL_MAX_COLS, TERMINAL_MAX_ROWS, TERMINAL_MAX_WRITE_BYTES, TerminalInfo, UnknownTerminalError,
)

from .sessions import SESSION_TERMINALS, SessionTerminals

inject = (SESSION_TERMINALS,)
_SESSION = Query(min_length=1, max_length=512)


def register(app: FastAPI, context: DashboardContext) -> None:
    @app.get("/api/dashboard/terminal/terminals")
    async def terminals(session: str = _SESSION) -> dict[str, object]:
        return {"terminals": [_view(item) for item in await context.require(SESSION_TERMINALS).list(session)]}

    @app.post("/api/dashboard/terminal/terminals")
    async def open_terminal(
        session: str = _SESSION,
        cols: int = Body(ge=1, le=TERMINAL_MAX_COLS), rows: int = Body(ge=1, le=TERMINAL_MAX_ROWS),
    ) -> dict[str, object]:
        try:
            return _view(await context.require(SESSION_TERMINALS).open(session, cols=cols, rows=rows))
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.delete("/api/dashboard/terminal/terminals/{terminal_id}")
    async def close_terminal(terminal_id: int, session: str = _SESSION) -> dict[str, object]:
        return {"closed": await context.require(SESSION_TERMINALS).close(session, terminal_id)}

    @app.websocket("/api/dashboard/terminal/attach")
    async def attach(socket: WebSocket) -> None:
        target = _attach_target(socket)
        if target is None:
            return await socket.close(code=1008, reason="终端参数无效")
        await socket.accept()
        await _relay(socket, context.require(SESSION_TERMINALS), *target)


def _view(info: TerminalInfo) -> dict[str, object]:
    return {"id": info.terminal_id, "shell": info.shell, "cwd": info.cwd, "exited": info.exited, "exit_code": info.exit_code}


def _attach_target(socket: WebSocket) -> tuple[str, int, int] | None:
    params = socket.query_params
    session = params.get("session", "")
    try:
        terminal_id, offset = int(params.get("terminal", "")), int(params.get("offset", "0"))
    except ValueError:
        return None
    if not 0 < len(session) <= 512 or terminal_id < 1 or offset < 0:
        return None
    return session, terminal_id, offset


# 输出按偏移推给浏览器（二进制帧）；偏移跳过表示更早输出已滚出缓冲，先通知浏览器重置再回放。
async def _relay(socket: WebSocket, terminals: SessionTerminals, session: str, terminal_id: int, offset: int) -> None:
    async def output() -> None:
        position = offset
        while True:
            chunk = await terminals.read(session, terminal_id, offset=position)
            if chunk.offset != position:
                await socket.send_text(json.dumps({"type": "reset", "offset": chunk.offset}))
            if chunk.data:
                await socket.send_bytes(chunk.data)
            position = chunk.end
            if chunk.exited:
                await socket.send_text(json.dumps({"type": "exit", "code": chunk.exit_code}))
                return

    async def input() -> None:
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return
            await _apply_input(terminals, session, terminal_id, message.get("text"))

    tasks = {asyncio.create_task(output()), asyncio.create_task(input())}
    code, reason = 1000, ""
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            await task
    except UnknownTerminalError:
        code, reason = 4404, "终端已关闭"
    except (WebSocketDisconnect, ValueError):
        code = 1008
    finally:
        for task in tasks:
            _ = task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError, WebSocketDisconnect, UnknownTerminalError, ValueError):
                await task
    if socket.client_state is WebSocketState.CONNECTED:
        with suppress(RuntimeError):
            await socket.close(code=code, reason=reason)


# 浏览器只发两种文本帧：{"type":"input","data":...} 与 {"type":"resize","cols":..,"rows":..}。
async def _apply_input(terminals: SessionTerminals, session: str, terminal_id: int, text: object) -> None:
    if not isinstance(text, str):
        raise ValueError("终端输入帧必须是文本")
    message = json.loads(text)
    match message:
        case {"type": "input", "data": str(data)}:
            encoded = data.encode("utf-8")
            if len(encoded) > TERMINAL_MAX_WRITE_BYTES:
                raise ValueError("单次终端输入过长")
            await terminals.write(session, terminal_id, encoded)
        case {"type": "resize", "cols": int(cols), "rows": int(rows)}:
            await terminals.resize(session, terminal_id, cols=cols, rows=rows)
        case _:
            raise ValueError("未知的终端输入帧")
