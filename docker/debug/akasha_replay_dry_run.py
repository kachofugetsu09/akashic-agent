#!/usr/bin/env python3
"""在副本上离线重放 Akasha 学习图。

这个线束用 core 的 MessageCatalog 读副本 sessions.db，按 turn projection 完成
完整重放，并把结果写到独立的候选文件。它不接触正式 workspace，也不调用
embedding provider：缺少固定向量的闭段按消费状态明确跳过。

注意：线束无法复现真实 runtime 的 binding 身份（binding 摘要包含插件归档
generation），所以产物里的 learning_binding 只对本线束有效，禁止用于正式发布。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, closing
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.plugin_composition import ServiceKey
from agent.plugin_contracts import Message
from agent.plugin_contracts.turn_effects import post_commit_effect
from plugins.akasha.application.rebuild import rebuild_from_catalog
from plugins.akasha.domain.model import MemoryConfig
from plugins.akasha.learning import AKASHA_LEARNING, Learning, LearningConfig
from plugins.turn_projection.plugin import TurnProjection
from session.embedding_store import MessageEmbeddings
from session.log import MessageCatalog, MessageLog

DEFAULT_SOURCES = ("conversation", "programmatic", "legacy-unattributed")


class DryRunBindings:
    """离线重放的最小 Bindings：只提供本次固定的学习规则。"""

    def __init__(self, *, learning: Learning, rule: LearningConfig) -> None:
        self._learning = learning
        self._rule = rule

    def describe(self, identity: str, service: ServiceKey[object]) -> Mapping[str, object]:
        raise RuntimeError(f"dry run 不读取 binding descriptor: {service.name}")

    @asynccontextmanager
    async def open(
        self, identity: str, service: ServiceKey[object],
    ) -> AsyncIterator[tuple[object, Mapping[str, object]]]:
        if service.name != AKASHA_LEARNING.name:
            raise RuntimeError(f"dry run 不提供 binding 服务: {service.name}")
        yield self._learning, self._rule.model_dump()


def _embedding_space(sessions_db: Path) -> tuple[str, int]:
    """从副本的固定向量表读出唯一 embedding 空间身份与维度。"""

    with closing(sqlite3.connect(f"file:{sessions_db}?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT model, MAX(dim) FROM message_embeddings GROUP BY model"
        ).fetchall()
    if len(rows) != 1:
        raise SystemExit(f"副本必须只有一个 embedding 空间，实际为: {rows}")
    model, dim = str(rows[0][0]), int(rows[0][1])
    if not model or dim <= 0:
        raise SystemExit("副本的 embedding 空间身份或维度无效")
    return model, dim


def _dry_run_identity(rule: LearningConfig) -> str:
    """线束本地绑定身份；它不代表任何真实 runtime binding。"""

    digest = hashlib.sha256(rule.model_dump_json().encode()).hexdigest()
    return "dry-run:" + digest[:40]


async def run(arguments: argparse.Namespace) -> dict[str, object]:
    sessions_db = arguments.workspace / "sessions.db"
    if not sessions_db.is_file():
        raise SystemExit(f"副本缺少 sessions.db: {sessions_db}")
    memory_model, dimension = _embedding_space(sessions_db)
    sources = tuple(arguments.sources) if arguments.sources else DEFAULT_SOURCES
    rule = LearningConfig(embedding_model=memory_model, dimension=dimension, sources=sources)
    log = MessageLog(sessions_db)
    try:
        learning = Learning(
            TurnProjection(),
            owner="akasha",
            post_commit_effect=lambda message: post_commit_effect(message.metadata).value,
        )
        bindings = DryRunBindings(learning=learning, rule=rule)

        async def forbidden_embed(texts: list[str]) -> list[list[float]]:
            raise RuntimeError(f"离线重放不得调用 embedding provider: {len(texts)} 条")

        report = await rebuild_from_catalog(
            catalog=MessageCatalog(log),
            embeddings=MessageEmbeddings(log),
            bindings=bindings,
            config=MemoryConfig(),
            learning_binding=_dry_run_identity(rule),
            embed_batch=forbidden_embed,
            memory_path=arguments.out,
            backup_root=arguments.backup_root,
            skip_missing_embeddings=True,
        )
    finally:
        log.close()
    payload = asdict(report)
    payload["embedding_model"] = memory_model
    payload["dimension"] = dimension
    payload["sources"] = list(sources)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="在副本上离线重放 Akasha 学习图")
    parser.add_argument("--workspace", type=Path, required=True, help="包含 sessions.db 的副本目录")
    parser.add_argument("--out", type=Path, required=True, help="候选学习图输出路径")
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--sources", nargs="*", default=None)
    arguments = parser.parse_args()
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    report = asyncio.run(run(arguments))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
