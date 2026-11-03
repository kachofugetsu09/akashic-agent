"""设计文档的持久存放：每个会话一个目录，文档是 .fig 字节，名字与修订号记在同目录的 index.json。"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from agent.plugin_composition import ServiceKey

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_DOCUMENTS = 50
MAX_NAME = 80


class DesignStoreError(RuntimeError):
    pass


class RevisionConflict(DesignStoreError):
    def __init__(self, current: int) -> None:
        super().__init__(f"文档已更新到修订 {current}")
        self.current = current


@dataclass(frozen=True, slots=True)
class DesignDocument:
    id: str
    name: str
    revision: int
    updated_at: str
    size: int


# 修订号单调递增：浏览器保存带上自己基于的修订，不一致就拒绝，避免覆盖 Agent 或另一个标签页刚写的内容。
class DesignStore:
    def __init__(self, root: Path) -> None:
        self._root = root
        # 索引的读改写串行：两次新建或保存同时到达时不会互相覆盖 index.json。
        self._index = asyncio.Lock()

    async def list(self, session: str) -> list[DesignDocument]:
        index = await asyncio.to_thread(self._load, session)
        return sorted(index.values(), key=lambda item: item.updated_at, reverse=True)

    async def find(self, session: str, document: str) -> DesignDocument | None:
        return (await asyncio.to_thread(self._load, session)).get(document)

    async def create(self, session: str, name: str) -> DesignDocument:
        async with self._index:
            return await asyncio.to_thread(self._create, session, name)

    async def rename(self, session: str, document: str, name: str) -> DesignDocument:
        async with self._index:
            return await asyncio.to_thread(self._rename, session, document, name)

    async def read(self, session: str, document: str) -> tuple[DesignDocument, bytes]:
        async with self._index:
            return await asyncio.to_thread(self._read, session, document)

    async def write(self, session: str, document: str, data: bytes, *, base: int) -> DesignDocument:
        if len(data) > MAX_DOCUMENT_BYTES:
            raise DesignStoreError(f"设计文档超过 {MAX_DOCUMENT_BYTES // 1024 // 1024} MB")
        async with self._index:
            return await asyncio.to_thread(self._write, session, document, data, base)

    def _create(self, session: str, name: str) -> DesignDocument:
        index = self._load(session)
        if len(index) >= MAX_DOCUMENTS:
            raise DesignStoreError(f"每个会话最多 {MAX_DOCUMENTS} 份设计文档")
        created = DesignDocument(secrets.token_hex(8), _name(name), 0, _now(), 0)
        index[created.id] = created
        self._save(session, index)
        return created

    def _rename(self, session: str, document: str, name: str) -> DesignDocument:
        index = self._load(session)
        index[document] = replace(_known(index, document), name=_name(name))
        self._save(session, index)
        return index[document]

    def _read(self, session: str, document: str) -> tuple[DesignDocument, bytes]:
        current = _known(self._load(session), document)
        data = (self._folder(session) / f"{document}.fig").read_bytes() if current.revision else b""
        return current, data

    def _write(self, session: str, document: str, data: bytes, base: int) -> DesignDocument:
        index = self._load(session)
        current = _known(index, document)
        if current.revision != base:
            raise RevisionConflict(current.revision)
        saved = replace(current, revision=current.revision + 1, updated_at=_now(), size=len(data))
        _atomic_write(self._folder(session) / f"{document}.fig", data)
        index[document] = saved
        self._save(session, index)
        return saved

    # 会话 ID 不直接当目录名：取摘要，任何字符都落在同一层目录里。
    def _folder(self, session: str) -> Path:
        return self._root / hashlib.sha256(session.encode("utf-8")).hexdigest()[:32]

    def _load(self, session: str) -> dict[str, DesignDocument]:
        path = self._folder(session) / "index.json"
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        return {row["id"]: DesignDocument(**row) for row in rows}

    def _save(self, session: str, index: dict[str, DesignDocument]) -> None:
        rows = [asdict(item) for item in index.values()]
        _atomic_write(self._folder(session) / "index.json", json.dumps(rows, ensure_ascii=False).encode("utf-8"))


def _known(index: dict[str, DesignDocument], document: str) -> DesignDocument:
    current = index.get(document)
    if current is None:
        raise DesignStoreError(f"没有这份设计文档：{document}")
    return current


def _name(raw: str) -> str:
    name = " ".join(raw.split())[:MAX_NAME]
    return name or "未命名设计"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    _ = temporary.write_bytes(data)
    os.replace(temporary, path)


DESIGN_STORE = ServiceKey[DesignStore]("design.store.v1")
