import logging
import json
import sqlite3
import threading
from pathlib import Path

from infra.persistence.json_store import atomic_write_text
from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

_CONSOLIDATION_MARKER_PREFIX = "<!-- consolidation:"
_CONSOLIDATION_MARKER_SUFFIX = " -->"
_CONSOLIDATION_TAIL_BYTES = 1024 * 1024
DEFAULT_SELF_MD = """# Akashic 的自我认知

## 人格与形象
- 我是 Akashic，一个直接、温暖、主动参与思考的长期协作伙伴。
- 我优先给出结论，再补充必要细节；不把自己伪装成没有立场的工具。

## 我对当前用户的理解
- 我会从长期记忆中逐步形成对当前用户的理解，不在缺少证据时编造画像。

## 我们关系的定义
- 我与当前用户的关系以透明、尊重边界和持续协作为基础。
"""


class MemoryStore:
    """Markdown 记忆文件：
    - MEMORY.md：稳定用户档案
    - SELF.md：Akashic 自我认知
    - PENDING.md：对话中提取的长期记忆候选
    """

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.pending_file = self.memory_dir / "PENDING.md"
        self.self_file = self.memory_dir / "SELF.md"
        self._consolidation_db = self.memory_dir / "consolidation_writes.db"
        self._consolidation_lock = threading.Lock()
        # 确保 PENDING.md 始终存在，避免首次运行时找不到文件
        if not self.pending_file.exists():
            self.pending_file.touch()
        self._init_consolidation_db()
        # 崩溃恢复：启动时若遗留 snapshot，回滚合并
        self._recover_pending_snapshot()

    # ── 长期记忆（MEMORY.md）─────────────────────────────

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        atomic_write_text(self.memory_file, content, domain="memory")

    # ── SELF.md（Akashic 自我模型）─────────────────────────────

    def read_self(self) -> str:
        if self.self_file.exists():
            return self.self_file.read_text(encoding="utf-8")
        return ""

    def write_self(self, content: str) -> None:
        atomic_write_text(self.self_file, content, domain="memory")

    # ── 待处理事实（对话 → optimizer 缓冲区）───────────

    def read_pending(self) -> str:
        if self.pending_file.exists():
            return self._strip_consolidation_markers(
                self.pending_file.read_text(encoding="utf-8")
            )
        return ""

    def append_pending(self, facts: str) -> None:
        """追加对话中提取的增量事实片段，不触碰 MEMORY.md。"""
        if not facts or not facts.strip():
            return
        with self._consolidation_lock:
            with open(self.pending_file, "a", encoding="utf-8") as f:
                _ = f.write(facts.rstrip() + "\n")

    def append_pending_once(
        self,
        facts: str,
        *,
        source_ref: str,
        kind: str = "pending",
    ) -> bool:
        """按 source_ref 幂等追加 PENDING，避免重启后重复 consolidation。"""
        text = facts.strip()
        if not text:
            return False
        return self._append_once_with_index(
            target_file=self.pending_file,
            text=text,
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=False,
        )

    def read_consolidation_receipt(
        self,
        source_ref: str,
        *,
        kind: str,
    ) -> dict[str, object] | None:
        """Read one immutable JSON receipt from the consolidation index."""

        src = source_ref.strip()
        kd = kind.strip()
        if not src or not kd:
            raise ValueError("receipt source_ref/kind 不能为空")
        with self._consolidation_lock:
            conn = sqlite3.connect(str(self._consolidation_db), timeout=30.0)
            try:
                row = conn.execute(
                    "SELECT payload FROM consolidation_writes "
                    "WHERE source_ref=? AND kind=?",
                    (src, kd),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        raw = row[0]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"receipt payload 缺失: {src}:{kd}")
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise ValueError(f"receipt payload 必须是 JSON object: {src}:{kd}")
        return {str(key): value for key, value in decoded.items()}

    def write_consolidation_receipt(
        self,
        source_ref: str,
        payload: dict[str, object],
        *,
        kind: str,
    ) -> dict[str, object]:
        """Persist one immutable JSON receipt and reject same-key drift."""

        src = source_ref.strip()
        kd = kind.strip()
        if not src or not kd:
            raise ValueError("receipt source_ref/kind 不能为空")
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._consolidation_lock:
            conn = sqlite3.connect(str(self._consolidation_db), timeout=30.0)
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload FROM consolidation_writes "
                    "WHERE source_ref=? AND kind=?",
                    (src, kd),
                ).fetchone()
                if row is not None:
                    existing = row[0]
                    if existing != encoded:
                        raise ValueError(f"receipt 内容冲突: {src}:{kd}")
                    conn.execute("COMMIT")
                    return dict(payload)
                conn.execute(
                    "INSERT INTO consolidation_writes "
                    "(source_ref, kind, payload, trailing_blank_line, done_at) "
                    "VALUES (?, ?, ?, 0, datetime('now'))",
                    (src, kd, encoded),
                )
                conn.execute("COMMIT")
                return dict(payload)
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.close()

    def clear_pending(self) -> None:
        """optimizer 归档后清空 PENDING.md。"""
        with self._consolidation_lock:
            atomic_write_text(self.pending_file, "", domain="memory")

    # ── 两阶段提交（供 MemoryOptimizer 使用）──────────────────────

    @property
    def _snapshot_path(self) -> Path:
        return self.pending_file.with_name("PENDING.snapshot.md")

    def snapshot_pending(self) -> str:
        """Phase-1：原子移走 PENDING.md，返回其内容。

        rename 之后 append_pending 会写入新建的 PENDING.md，
        与本次快照完全隔离，不会丢失后续增量。
        调用前会自动处理上次崩溃遗留的 snapshot。
        """
        self._recover_pending_snapshot()
        with self._consolidation_lock:
            if not self.pending_file.exists() or self.pending_file.stat().st_size == 0:
                return ""
            # POSIX rename 是原子操作：rename 完成后新追加写入全新的 PENDING.md
            _ = self.pending_file.rename(self._snapshot_path)
            return self._strip_consolidation_markers(
                self._snapshot_path.read_text(encoding="utf-8")
            )

    def commit_pending_snapshot(self) -> None:
        """Phase-2 成功：merge 已完成，删除快照。"""
        with self._consolidation_lock:
            if self._snapshot_path.exists():
                self._snapshot_path.unlink()
            # 保持 PENDING.md 常驻，避免“已归档后文件消失”带来的状态歧义
            if not self.pending_file.exists():
                self.pending_file.touch()

    def rollback_pending_snapshot(self) -> None:
        """Phase-2 失败：将快照内容合并回 PENDING.md，不丢失任何数据。

        快照（较旧）在前，运行期新追加（较新）在后。
        """
        with self._consolidation_lock:
            if not self._snapshot_path.exists():
                return
            snap_text = self._snapshot_path.read_text(encoding="utf-8")
            new_text = (
                self.pending_file.read_text(encoding="utf-8")
                if self.pending_file.exists()
                else ""
            )
            merged = (
                snap_text.rstrip() + "\n" + new_text if new_text.strip() else snap_text
            )
            atomic_write_text(self.pending_file, merged, domain="memory")
            self._snapshot_path.unlink()
        logger.info("[memory] PENDING snapshot 已回滚合并")

    def _recover_pending_snapshot(self) -> None:
        """启动时或 snapshot_pending 前调用，处理上次崩溃遗留的快照。"""
        if self._snapshot_path.exists():
            logger.warning("[memory] 检测到遗留 PENDING.snapshot.md，执行崩溃回滚")
            self.rollback_pending_snapshot()

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    @staticmethod
    def _consolidation_marker(source_ref: str, kind: str) -> str:
        src = source_ref.replace("\n", " ").strip()
        kd = kind.replace("\n", " ").strip()
        return f"{_CONSOLIDATION_MARKER_PREFIX}{src}:{kd}{_CONSOLIDATION_MARKER_SUFFIX}"

    @staticmethod
    def _strip_consolidation_markers(text: str) -> str:
        lines = text.splitlines()
        kept = [
            line
            for line in lines
            if not (
                line.startswith(_CONSOLIDATION_MARKER_PREFIX)
                and line.endswith(_CONSOLIDATION_MARKER_SUFFIX)
            )
        ]
        return "\n".join(kept).strip()

    def _init_consolidation_db(self) -> None:
        conn = sqlite3.connect(str(self._consolidation_db))
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_writes (
                    source_ref TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT,
                    trailing_blank_line INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT NOT NULL,
                    PRIMARY KEY (source_ref, kind)
                )""")
            cols = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(consolidation_writes)"
                ).fetchall()
            }
            if "payload" not in cols:
                conn.execute("ALTER TABLE consolidation_writes ADD COLUMN payload TEXT")
            if "trailing_blank_line" not in cols:
                conn.execute(
                    "ALTER TABLE consolidation_writes ADD COLUMN trailing_blank_line INTEGER NOT NULL DEFAULT 0"
                )
            conn.commit()
        finally:
            conn.close()

    def _append_once_with_index(
        self,
        *,
        target_file: Path,
        text: str,
        source_ref: str,
        kind: str,
        trailing_blank_line: bool,
    ) -> bool:
        """在文件和 SQLite 索引之间执行一次幂等追加。"""
        # 1. 校验调用方已建立的字符串契约，并生成稳定 marker。
        src = source_ref.strip()
        kd = kind.strip()
        if not src or not kd or not text:
            return False
        marker = self._consolidation_marker(src, kd)

        # 2. 锁定索引事务，恢复已记录但文件缺失的写入。
        with self._consolidation_lock:
            conn = sqlite3.connect(str(self._consolidation_db), timeout=30.0)
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload, trailing_blank_line FROM consolidation_writes WHERE source_ref=? AND kind=?",
                    (src, kd),
                ).fetchone()
                if row is not None:
                    existing_payload = row[0]
                    existing_trailing_raw = row[1]
                    if existing_payload is not None and not isinstance(
                        existing_payload, str
                    ):
                        raise TypeError("consolidation payload must be text")
                    if not isinstance(existing_trailing_raw, int):
                        raise TypeError("consolidation trailing flag must be an integer")
                    if existing_trailing_raw not in (0, 1):
                        raise ValueError("consolidation trailing flag must be 0 or 1")
                    existing_trailing = bool(existing_trailing_raw)
                    if not self._file_contains_marker(target_file, marker):
                        if not existing_payload:
                            raise ValueError(
                                "consolidation index payload is missing for file recovery"
                            )
                        with open(target_file, "a", encoding="utf-8") as f:
                            f.write(marker + "\n")
                            f.write(existing_payload.rstrip() + "\n")
                            if existing_trailing:
                                f.write("\n")
                    conn.execute("COMMIT")
                    return False

                # 恢复路径：若历史崩溃发生在“文件已写，索引未写”，用尾部扫描补索引并跳过重复写。
                if self._tail_contains_marker(target_file, marker):
                    conn.execute(
                        "INSERT OR REPLACE INTO consolidation_writes(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, ?, datetime('now'))",
                        (src, kd, text, 1 if trailing_blank_line else 0),
                    )
                    conn.execute("COMMIT")
                    return False

                # 3. 先追加 marker 和内容，再提交索引事务。
                with open(target_file, "a", encoding="utf-8") as f:
                    f.write(marker + "\n")
                    f.write(text.rstrip() + "\n")
                    if trailing_blank_line:
                        f.write("\n")

                conn.execute(
                    "INSERT OR REPLACE INTO consolidation_writes(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, ?, datetime('now'))",
                    (src, kd, text, 1 if trailing_blank_line else 0),
                )
                conn.execute("COMMIT")
                return True
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    @staticmethod
    def _tail_contains_marker(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            take = min(size, _CONSOLIDATION_TAIL_BYTES)
            if take <= 0:
                return False
            f.seek(size - take)
            tail = f.read(take).decode("utf-8")
            return marker in tail

    @staticmethod
    def _file_contains_marker(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        needle = marker.encode("utf-8")
        if not needle:
            return False
        carry = b""
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                data = carry + chunk
                if needle in data:
                    return True
                if len(needle) > 1:
                    carry = data[-(len(needle) - 1) :]
                else:
                    carry = b""
        return False
