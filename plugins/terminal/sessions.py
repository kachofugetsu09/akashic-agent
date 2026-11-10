"""会话终端：终端事实归 host.terminals.v1，本插件只按会话命名 owner，并选定会话的工作目录。"""
from __future__ import annotations

from pathlib import Path

from agent.plugin_composition import Context, ServiceKey
from plugins.host_execution.contract import TERMINALS, TerminalInfo, TerminalOutput
from plugins.working_directory.contract import WORKING_DIRECTORY

# 浏览器长轮询一次最多等这么久；断线时最多晚这么久释放 generation 借用。
READ_WAIT_MS = 15_000


class SessionTerminals:
    """每个会话一组终端；owner_key 即会话，插件换代后从执行主机列出即可接回。"""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    async def list(self, session: str) -> list[TerminalInfo]:
        async with self._ctx.runtime_scope():
            return await self._ctx.require(TERMINALS).list(self._ctx, session)

    async def open(self, session: str, *, cols: int, rows: int) -> TerminalInfo:
        async with self._ctx.runtime_scope():
            return await self._ctx.require(TERMINALS).open(
                self._ctx, session, cwd=self._directory(session), cols=cols, rows=rows,
            )

    async def write(self, session: str, terminal_id: int, data: bytes) -> None:
        async with self._ctx.runtime_scope():
            await self._ctx.require(TERMINALS).write(self._ctx, session, terminal_id, data)

    async def resize(self, session: str, terminal_id: int, *, cols: int, rows: int) -> None:
        async with self._ctx.runtime_scope():
            await self._ctx.require(TERMINALS).resize(self._ctx, session, terminal_id, cols=cols, rows=rows)

    async def read(self, session: str, terminal_id: int, *, offset: int) -> TerminalOutput:
        async with self._ctx.runtime_scope():
            return await self._ctx.require(TERMINALS).read(
                self._ctx, session, terminal_id, offset=offset, wait_ms=READ_WAIT_MS,
            )

    async def close(self, session: str, terminal_id: int) -> bool:
        async with self._ctx.runtime_scope():
            return await self._ctx.require(TERMINALS).close(self._ctx, session, terminal_id)

    # 会话设过工作目录就在那里打开，与 Agent 的命令同一处；没设或读不到时由执行主机退回主目录。
    def _directory(self, session: str) -> Path | None:
        with self._ctx.borrow(WORKING_DIRECTORY) as directories:
            if directories is None:
                return None
            try:
                path = directories.snapshot(session).path
            except ValueError:
                return None
        return None if path is None else Path(path)


SESSION_TERMINALS = ServiceKey[SessionTerminals]("terminal.sessions.v1")
