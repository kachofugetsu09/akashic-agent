"""为旧学习图固定切换出处；不重放、不改索引或消息。"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from agent.migrations.session_db_backup import backup_sqlite_database
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.infrastructure.consumption import Consumption, LegacyPrefix, turns_digest
from plugins.akasha.infrastructure.loader import load_turns
from plugins.akasha.infrastructure.persistence import (
    canonical_json, check_memory_schema, load_consumption, load_memory_state,
    logical_state_sha256, memory_turn_count, sha256_file,
)
from plugins.akasha.infrastructure.sparse_index import sparse_index_state_sha256


def cutover_akasha(
    *, memory: Path, index: Path, heads: Mapping[str, int],
    config: MemoryConfig, backup_root: Path,
) -> bool:
    """在停止接纳的安装锁内发布消费起点；已发布但未落 Yoyo 账时可重试。"""
    # 1. 验证原图与旧索引；只将图中已学习部分封为旧前缀。
    check_memory_schema(memory)
    existing = load_consumption(memory)
    index_digest = sparse_index_state_sha256(index)
    count = memory_turn_count(memory) if existing is None else existing.legacy_prefix.count
    prefix = load_turns(index)[:count]
    if len(prefix) != count:
        raise ValueError("旧索引不足以恢复已学习前缀")
    for turn in prefix:
        if ((turn.user_text.strip() and turn.user_dense is None)
            or (turn.assistant_text.strip() and turn.assistant_dense is None)):
            raise ValueError(f"旧学习材料缺少固定向量: {turn.turn_id}")
    legacy = LegacyPrefix(count=count, index_state_sha256=index_digest,
                          turns_digest=turns_digest(prefix))
    if existing is not None:
        if existing.legacy_prefix != legacy:
            raise ValueError("重复切换时旧学习前缀身份已改变")
        if any(seq > heads.get(session, -1) for session, seq in existing.cutover_heads):
            raise ValueError("切换后日志前缀已减少，需要显式数据恢复")
        return False
    _ = load_memory_state(memory, turns=prefix, config=config,
                          source_index_sha256=None, source_index_state_sha256=index_digest)
    state = Consumption(legacy_prefix=legacy, cutover_heads=tuple(sorted(heads.items())))
    graph_digest = logical_state_sha256(memory)
    index_file_digest = sha256_file(index)

    # 2. 备份之后在单库副本只增加元数据；学习图与旧索引一字不重算。
    backup = backup_sqlite_database(memory, backup_root, migration="akasha-message-consumption-v1")
    candidate = memory.with_name(f".{memory.name}.cutover-{uuid4().hex}.candidate")
    with (
        closing(sqlite3.connect(f"file:{backup}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(candidate)) as target,
    ):
        source.backup(target)
        _ = target.execute("INSERT INTO metadata VALUES ('consumer_state_json', ?)",
                           (canonical_json(state.model_dump(mode="json")),))
        target.commit()
    candidate.chmod(0o600)
    check_memory_schema(candidate)
    _ = load_memory_state(candidate, turns=prefix, config=config,
                          source_index_sha256=None, source_index_state_sha256=index_digest)
    if load_consumption(candidate) != state:
        raise ValueError("切换候选的消费出处不一致")
    if logical_state_sha256(candidate, include_consumption=False) != graph_digest:
        raise ValueError("切换候选改变了原学习图")
    if sha256_file(index) != index_file_digest:
        raise ValueError("切换期间旧索引发生改变")

    # 3. 候选已经完整重开验证；文件与目录耐久后 Yoyo 才可提交完成记录。
    with candidate.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(candidate, memory)
    directory = os.open(memory.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return True
