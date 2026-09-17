"""用与在线学习同一条路径从 canonical 来源全量重建 Akasha 图。"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agent.plugin_composition.bindings import Bindings
from agent.plugin_composition.messages import MessageCatalog, MessageEmbeddings

from ..domain.model import MemoryConfig
from ..infrastructure.consumption import Consumption
from ..infrastructure.persistence import (
    canonical_json,
    load_consumption,
    logical_state_sha256,
    memory_turn_count,
    sha256_file,
)
from .consumer import MessageConsumer


@dataclass(frozen=True, slots=True)
class RebuildReport:
    """报告一次完整重建的确定性身份与非确定性运行成本。"""

    turns: int
    sessions: int
    embedded_messages: int
    elapsed_seconds: float
    database_sha256: str
    logical_state_sha256: str
    memory_path: str
    backup_path: str
    completed_at: str


async def rebuild_from_catalog(
    *,
    catalog: MessageCatalog,
    embeddings: MessageEmbeddings,
    bindings: Bindings,
    config: MemoryConfig,
    learning_binding: str,
    embed_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
    memory_path: Path,
    backup_root: Path,
) -> RebuildReport:
    """唯一重建实现：空图 + 无切换上界，跑与在线相同的 MessageConsumer。"""

    # 1. 先固定恢复点，再生成候选；失败不触碰已发布的学习图。
    started = time.perf_counter()
    backup_dir = backup_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8]
    backup_path = _backup_existing(memory_path, backup_dir)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = memory_path.with_name(f".{memory_path.name}.rebuild-{uuid4().hex}.candidate")
    embedded = 0

    async def counting_embed(texts: list[str]) -> list[list[float]]:
        nonlocal embedded
        vectors = await embed_batch(texts)
        embedded += len(texts)
        return vectors

    try:
        # 2. 空进度且没有切换上界，等价于把全部历史按因果顺序重放一遍。
        consumer = MessageConsumer(
            candidate, turns=[], state=Consumption(cutover_heads=()), config=config,
        )
        try:
            _ = await consumer.consume(
                catalog=catalog, learning_binding=learning_binding, embeddings=embeddings,
                bindings=bindings, embed_batch=counting_embed,
            )
            turns = tuple(consumer.cycle.turns)
        finally:
            consumer.close()
        count = len(turns)
        sessions = len({turn.session_key for turn in turns})
        _verify_candidate(candidate, count)

        # 3. 先发布索引身份，再原子替换学习图；崩溃窗口只会留下可重建的候选文件。
        if count == 0:
            candidate.unlink(missing_ok=True)
            memory_path.unlink(missing_ok=True)
            database_sha256 = ""
            state_sha256 = ""
        else:
            os.replace(candidate, memory_path)
            _fsync_directory(memory_path.parent)
            database_sha256 = sha256_file(memory_path)
            state_sha256 = logical_state_sha256(memory_path)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise

    completed_at = datetime.now(UTC).isoformat()
    report = RebuildReport(
        turns=count,
        sessions=sessions,
        embedded_messages=embedded,
        elapsed_seconds=round(time.perf_counter() - started, 3),
        database_sha256=database_sha256,
        logical_state_sha256=state_sha256,
        memory_path=str(memory_path),
        backup_path=str(backup_path) if backup_path is not None else "",
        completed_at=completed_at,
    )
    _write_manifest(backup_dir, report, config)
    return report


def _backup_existing(memory_path: Path, backup_dir: Path) -> Path | None:
    """存在已发布学习图时先做 SQLite 原生备份，再允许替换。"""

    if not memory_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / "memory-before.db"
    with closing(sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)) as incoming:
        with closing(sqlite3.connect(target)) as outgoing:
            incoming.backup(outgoing)
            if outgoing.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Akasha 重建前的学习图备份完整性检查失败")
    return target


def _verify_candidate(candidate: Path, count: int) -> None:
    """候选必须是自描述、可恢复且与本次学习节点数一致的学习图。"""

    with closing(sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise ValueError(f"Akasha 重建候选完整性检查失败: {integrity}")
    if memory_turn_count(candidate) != count:
        raise ValueError("Akasha 重建候选的节点数与本次学习结果不一致")
    state = load_consumption(candidate)
    if state is None:
        raise ValueError("Akasha 重建候选缺少消费出处")
    state.check_count(count)
    if state.cutover_heads:
        raise ValueError("Akasha 完整重建不能保留切换上界")


def _write_manifest(
    backup_dir: Path, report: RebuildReport, config: MemoryConfig,
) -> None:
    """在同一恢复点目录留下可审阅的重建回执。"""

    backup_dir.mkdir(parents=True, exist_ok=True)
    payload: Mapping[str, object] = {
        "schema_version": 1,
        "report": asdict(report),
        "config": asdict(config),
    }
    temporary = Path(tempfile.mkstemp(prefix="manifest.", suffix=".tmp", dir=backup_dir)[1])
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, backup_dir / "manifest.json")
    _fsync_directory(backup_dir)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def manifest_json(report: RebuildReport) -> str:
    """给命令回执使用的单行 JSON。"""

    return json.dumps(asdict(report), ensure_ascii=False, separators=(",", ":"))
