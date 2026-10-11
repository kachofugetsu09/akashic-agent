"""附件端口的独立 SQLite BLOB 实现；示例只接纳本地文件。"""
from __future__ import annotations

import hashlib
import mimetypes
import sqlite3
from contextlib import closing
from pathlib import Path

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.ledger.contract import (
    ARTIFACT_IMPORT, ARTIFACT_READ, AttachmentKind, AttachmentRef,
)

api_version = 3
name = "artifact_provider"
version = "1.0.0"
inject = ()


class Lease:
    """不可变字节快照拥有自己的寿命，不把数据库连接交给消费者。"""

    def __init__(self, ref: AttachmentRef, data: bytes):
        self.ref, self._data = ref, data
        self._closed = False

    async def read_bytes(self, *, max_bytes: int) -> bytes:
        if self.ref.size_bytes > max_bytes:
            raise ValueError("附件超过读取上限")
        return await self.read_chunk(offset=0, max_bytes=max_bytes)

    async def read_chunk(self, *, offset: int, max_bytes: int) -> bytes:
        if self._closed:
            raise RuntimeError("附件租约已关闭")
        if offset < 0 or max_bytes < 0:
            raise ValueError("读取范围不能为负")
        return self._data[offset:offset + max_bytes]

    async def aclose(self) -> None:
        self._closed = True
        self._data = b""


class Blobs:
    """每次读写使用独立连接；只按内容摘要发布不可变附件。"""

    def __init__(self, path: Path):
        self._path = path

    def initialize(self) -> None:
        with closing(sqlite3.connect(self._path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS blobs (sha256 TEXT PRIMARY KEY, data BLOB NOT NULL)")

    async def import_source(self, source: str, kind: AttachmentKind) -> AttachmentRef:
        """在真实事务中保存文件内容，不读取 Ledger 的实现或数据库。"""
        if "://" in source:
            raise ValueError("此示例 provider 仅支持本地文件")
        path = Path(source)
        def save() -> AttachmentRef:
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            with closing(sqlite3.connect(self._path)) as db, db:
                db.execute("INSERT OR IGNORE INTO blobs VALUES (?, ?)", (digest, data))
                if db.execute("SELECT data FROM blobs WHERE sha256=?", (digest,)).fetchone()[0] != data:
                    raise ValueError("已发布附件与摘要冲突")
            return AttachmentRef(digest, kind, path.name, mimetypes.guess_type(path.name)[0], len(data), digest)
        return await run_file_io(save)

    async def acquire(self, ref: AttachmentRef) -> Lease:
        """读取完成才创建租约，取消时不会遗留连接或文件描述符。"""
        def read() -> bytes:
            with closing(sqlite3.connect(self._path)) as db:
                row = db.execute("SELECT data FROM blobs WHERE sha256=?", (ref.sha256,)).fetchone()
            if row is None:
                raise LookupError(ref.artifact_id)
            data = row[0]
            if len(data) != ref.size_bytes or hashlib.sha256(data).hexdigest() != ref.sha256:
                raise ValueError("附件引用与内容不一致")
            return data
        return Lease(ref, await run_file_io(read))


async def apply(ctx: Context) -> None:
    blobs = Blobs(ctx.data_root / "blobs.db")
    await run_file_io(blobs.initialize)
    await ctx.provide(ARTIFACT_IMPORT, blobs)
    await ctx.provide(ARTIFACT_READ, blobs)
