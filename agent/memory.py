import logging
import re
import sqlite3
import threading
from pathlib import Path

from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

_CONSOLIDATION_MARKER_PREFIX = "<!-- consolidation:"
_CONSOLIDATION_MARKER_SUFFIX = " -->"
_CONSOLIDATION_TAIL_BYTES = 1024 * 1024
_JOURNAL_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class MemoryStore:
    """Five-layer memory:
    - MEMORY.md   : stable user profile, sole writer = MemoryOptimizer
    - SELF.md     : Akashic self-model & relationship understanding, updated by Optimizer
    - PENDING.md  : incremental facts extracted during conversations
    - HISTORY.md  : grep-searchable event log, permanent append
    - RECENT_CONTEXT.md : compacted recent context snapshot for proactive/drift
    - journal/    : per-day event timeline, append-only YYYY-MM-DD.md
    """

    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.journal_dir = ensure_dir(self.memory_dir / "journal")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.recent_context_file = self.memory_dir / "RECENT_CONTEXT.md"
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

    # ── long-term memory (MEMORY.md) ─────────────────────────────

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def append_history_once(
        self,
        entry: str,
        *,
        source_ref: str,
        kind: str = "history_entry",
    ) -> bool:
        """按 source_ref 幂等追加 HISTORY，避免重启后重复 consolidation。"""
        text = (entry or "").strip()
        if not text:
            return False
        return self._append_once_with_index(
            target_file=self.history_file,
            text=text,
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=True,
        )

    def read_history(self, max_chars: int = 0) -> str:
        """读取 HISTORY.md，并过滤 consolidation 标记行。"""
        if not self.history_file.exists():
            return ""
        text = self.history_file.read_text(encoding="utf-8")
        text = self._strip_consolidation_markers(text)
        if max_chars > 0 and len(text) > max_chars:
            return text[-max_chars:]
        return text

    # ── journal/ (per-day event timeline) ───────────────────────────

    def append_journal(
        self,
        date_str: str,
        entry: str,
        *,
        source_ref: str = "",
        kind: str = "journal",
    ) -> bool:
        date_str = date_str.strip()
        text = (entry or "").strip()
        if not _JOURNAL_DATE_RE.fullmatch(date_str) or not text:
            return False
        journal_file = self.journal_dir / f"{date_str}.md"
        if not journal_file.exists():
            journal_file.write_text(f"# {date_str}\n\n", encoding="utf-8")
        if source_ref:
            return self._append_once_with_index(
                target_file=journal_file,
                text=text,
                source_ref=source_ref,
                kind=kind,
                trailing_blank_line=True,
            )
        with open(journal_file, "a", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n\n")
        return True

    # ── RECENT_CONTEXT.md (compacted recent context) ──────────────

    def read_recent_context(self) -> str:
        if self.recent_context_file.exists():
            return self.recent_context_file.read_text(encoding="utf-8")
        return ""

    def write_recent_context(self, content: str) -> None:
        self.recent_context_file.write_text(content, encoding="utf-8")

    # ── SELF.md (Akashic self-model) ──────────────────────────────

    def read_self(self) -> str:
        if self.self_file.exists():
            return self.self_file.read_text(encoding="utf-8")
        return ""

    def write_self(self, content: str) -> None:
        self.self_file.write_text(content, encoding="utf-8")

    # ── pending facts (conversation → optimizer buffer) ───────────

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
        with open(self.pending_file, "a", encoding="utf-8") as f:
            f.write(facts.rstrip() + "\n")

    def append_pending_once(
        self,
        facts: str,
        *,
        source_ref: str,
        kind: str = "pending",
    ) -> bool:
        """按 source_ref 幂等追加 PENDING，避免重启后重复 consolidation。"""
        text = (facts or "").strip()
        if not text:
            return False
        return self._append_once_with_index(
            target_file=self.pending_file,
            text=text,
            source_ref=source_ref,
            kind=kind,
            trailing_blank_line=False,
        )

    def clear_pending(self) -> None:
        """optimizer 归档后清空 PENDING.md。"""
        self.pending_file.write_text("", encoding="utf-8")

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
        if not self.pending_file.exists() or self.pending_file.stat().st_size == 0:
            return ""
        # POSIX rename 是原子操作：rename 完成后新追加写入全新的 PENDING.md
        self.pending_file.rename(self._snapshot_path)
        return self._strip_consolidation_markers(
            self._snapshot_path.read_text(encoding="utf-8")
        )

    def commit_pending_snapshot(self) -> None:
        """Phase-2 成功：merge 已完成，删除快照。"""
        if self._snapshot_path.exists():
            self._snapshot_path.unlink()
        # 保持 PENDING.md 常驻，避免“已归档后文件消失”带来的状态歧义
        if not self.pending_file.exists():
            self.pending_file.touch()

    def rollback_pending_snapshot(self) -> None:
        """Phase-2 失败：将快照内容合并回 PENDING.md，不丢失任何数据。

        快照（较旧）在前，运行期新追加（较新）在后。
        """
        if not self._snapshot_path.exists():
            return
        snap_text = self._snapshot_path.read_text(encoding="utf-8")
        new_text = (
            self.pending_file.read_text(encoding="utf-8")
            if self.pending_file.exists()
            else ""
        )
        merged = snap_text.rstrip() + "\n" + new_text if new_text.strip() else snap_text
        self.pending_file.write_text(merged, encoding="utf-8")
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
        src = (source_ref or "").replace("\n", " ").strip()
        kd = (kind or "").replace("\n", " ").strip()
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
        marker = self._consolidation_marker(source_ref, kind)
        src = (source_ref or "").strip()
        kd = (kind or "").strip()
        if not src or not kd or not text:
            return False

        with self._consolidation_lock:
            conn = sqlite3.connect(str(self._consolidation_db), timeout=30.0)
            try:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT payload, trailing_blank_line FROM consolidation_writes WHERE source_ref=? AND kind=?",
                    (src, kd),
                ).fetchone()
                if row is not None:
                    existing_payload = row[0] or ""
                    existing_trailing = bool(int(row[1] or 0))
                    if not self._file_contains_marker(target_file, marker):
                        if existing_payload:
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
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                take = min(size, _CONSOLIDATION_TAIL_BYTES)
                if take <= 0:
                    return False
                f.seek(size - take)
                tail = f.read(take).decode("utf-8", errors="ignore")
                return marker in tail
        except Exception:
            return False

    @staticmethod
    def _file_contains_marker(path: Path, marker: str) -> bool:
        if not path.exists():
            return False
        needle = marker.encode("utf-8")
        if not needle:
            return False
        carry = b""
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    data = carry + chunk
                    if needle in data:
                        return True
                    if len(needle) > 1:
                        carry = data[-(len(needle) - 1) :]
                    else:
                        carry = b""
        except Exception:
            return False
        return False
