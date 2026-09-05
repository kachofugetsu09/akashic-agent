"""归档调用读取已发布学习图，不借用或争抢正式 writer。"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from agent.plugin_composition.bindings import Bindings
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog

from ..domain.model import MemoryConfig
from ..infrastructure.consumption import Consumption
from .consumer import MessageConsumer, run_memory_job
from .cycle import MemoryCycle


def _copy_published(source: Path, target: Path) -> None:
    """SQLite 原生备份固定同一版本，复制期间发布新图不会混合消费出处。"""
    with closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as incoming:
        with closing(sqlite3.connect(target)) as outgoing:
            incoming.backup(outgoing)
            if outgoing.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("学习图只读快照完整性检查失败")


@asynccontextmanager
async def read_memory(
    path: Path, *, legacy_index: Path | None, catalog: MessageCatalog,
    embeddings: MessageEmbeddings, bindings: Bindings, config: MemoryConfig,
    embedding_space: tuple[str, int] | None = None,
) -> AsyncGenerator[tuple[MemoryCycle, Consumption]]:
    """只从一致副本恢复图，重用原 binding 校验；不给调用者提交或正式文件权限。"""
    # 1. 来源必须已经发布；缺文件由只读 SQLite 打开明确失败，不能创建空图。
    with TemporaryDirectory(prefix="akasha-read-") as temporary:
        snapshot = Path(temporary) / "memory.db"
        await run_memory_job(lambda: _copy_published(path, snapshot))
        # 2. 复用完整恢复校验；恢复器的本地 lease 只保护临时副本。
        restored = await MessageConsumer.load(
            snapshot, legacy_index=legacy_index, catalog=catalog,
            embeddings=embeddings, bindings=bindings, config=config,
        )
        try:
            if embedding_space is not None:
                restored.check_embedding_space(*embedding_space, bindings)
            yield restored.cycle, restored.state
        finally:
            restored.close()
