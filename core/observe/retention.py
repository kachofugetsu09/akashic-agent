"""淘汰策略：定期清理过期的 observe 数据。

规则：
  turns:        保留 180 天（error IS NOT NULL 永久保留）
  rag_queries:  保留  90 天（error IS NOT NULL 永久保留）
  proactive_decisions: 保留 90 天（error IS NOT NULL 永久保留）

触发：启动时后台跑一次，距上次清理超过 24h 才执行。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.observe.db import open_db

logger = logging.getLogger("observe.retention")

_RETENTION_DAYS = {
    "turns": 180,
    "rag_queries": 90,
    "proactive_decisions": 90,
}
_STAMP_FILE = ".last_cleanup"


def _stamp_path(db_path: Path) -> Path:
    return db_path.parent / _STAMP_FILE


def _should_run(db_path: Path) -> bool:
    stamp = _stamp_path(db_path)
    if not stamp.exists():
        return True
    import time

    age_hours = (time.time() - stamp.stat().st_mtime) / 3600
    return age_hours >= 24


def _run_cleanup(db_path: Path) -> None:
    conn = open_db(db_path)
    try:
        deleted: dict[str, int] = {}
        with conn:
            for table, days in _RETENTION_DAYS.items():
                cutoff = f"datetime('now', '-{days} days')"
                cur = conn.execute(
                    f"DELETE FROM {table} WHERE ts < {cutoff} AND error IS NULL"
                )
                deleted[table] = cur.rowcount

        logger.info("observe retention done: %s", deleted)
        _ = _stamp_path(db_path).write_text("ok")
    except Exception:
        logger.exception("observe retention failed")
    finally:
        conn.close()


async def run_retention_if_needed(db_path: Path) -> None:
    """在 asyncio 后台跑清理（用 run_in_executor 避免阻塞事件循环）。"""
    if not db_path.exists():
        return
    if not _should_run(db_path):
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _run_cleanup, db_path)
