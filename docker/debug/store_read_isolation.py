"""在真实 SQLite 写事务未结束时验证插件只读快照；只使用临时数据。"""

from datetime import UTC, datetime
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from plugins.drift.store import DriftStore
from plugins.eventmail.store import EventMailStore


def check_store(path: Path, store: EventMailStore | DriftStore) -> None:
    """读取已提交快照，不等待另一个连接的写锁或吸收未提交状态。"""
    store.initialize()
    now = datetime.now(UTC)
    if isinstance(store, EventMailStore):
        store.submit("feed", "batch", [{"item_id": "one", "revision": "r1", "payload": {}}])
        update = "UPDATE content_state SET state_version = state_version + 1"
    else:
        store.propose("one", "r1", {}, now)
        update = "UPDATE proposals SET state_version = state_version + 1"
    before = store.snapshot(now)
    writer = sqlite3.connect(path)
    try:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(update)
        assert store.snapshot(now) == before
        writer.commit()
        assert store.snapshot(now) != before
        assert writer.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        writer.close()


def main() -> None:
    """逐一检查两个 owner，任一失败都返回非零。"""
    with TemporaryDirectory(prefix="akashic-store-reads-") as directory:
        root = Path(directory)
        for kind in (EventMailStore, DriftStore):
            path = root / f"{kind.__name__}.db"
            check_store(path, kind(path))
            print(f"PASS {kind.__name__}: committed snapshot during active writer")


if __name__ == "__main__":
    main()
