import logging
import re
from pathlib import Path

from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)

_QUESTIONS_SECTION = "## 想了解的问题"
_NOW_SECTIONS_ORDER = ["## 近期进行中", "## 待确认事项", _QUESTIONS_SECTION]


class MemoryStore:
    """Five-layer memory:
    - MEMORY.md   : stable user profile, sole writer = MemoryOptimizer
    - SELF.md     : Akashic self-model & relationship understanding, updated by Optimizer
    - PENDING.md  : incremental facts extracted during conversations
    - NOW.md      : short-term state (ongoing tasks, schedule, open questions)
    - HISTORY.md  : grep-searchable event log, permanent append
    """
    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.pending_file = self.memory_dir / "PENDING.md"
        self.self_file = self.memory_dir / "SELF.md"
        self.now_file = self.memory_dir / "NOW.md"
        # 确保 PENDING.md 始终存在，避免首次运行时找不到文件
        if not self.pending_file.exists():
            self.pending_file.touch()
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

    # ── SELF.md (Akashic self-model) ──────────────────────────────

    def read_self(self) -> str:
        if self.self_file.exists():
            return self.self_file.read_text(encoding="utf-8")
        return ""

    def write_self(self, content: str) -> None:
        self.self_file.write_text(content, encoding="utf-8")

    # ── NOW.md (short-term state) ─────────────────────────────────

    def read_now(self) -> str:
        if self.now_file.exists():
            return self.now_file.read_text(encoding="utf-8")
        return ""

    def write_now(self, content: str) -> None:
        self.now_file.write_text(content, encoding="utf-8")

    def read_now_ongoing(self) -> str:
        """从 NOW.md 提取 '## 近期进行中' section 正文（不含标题行）。"""
        return self._extract_now_section(self.read_now(), "## 近期进行中")

    # ── pending facts (conversation → optimizer buffer) ───────────

    def read_pending(self) -> str:
        if self.pending_file.exists():
            return self.pending_file.read_text(encoding="utf-8")
        return ""

    def append_pending(self, facts: str) -> None:
        """追加对话中提取的增量事实片段，不触碰 MEMORY.md。"""
        if not facts or not facts.strip():
            return
        with open(self.pending_file, "a", encoding="utf-8") as f:
            f.write(facts.rstrip() + "\n")

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
        return self._snapshot_path.read_text(encoding="utf-8")

    def commit_pending_snapshot(self) -> None:
        """Phase-2 成功：merge 已完成，删除快照。"""
        if self._snapshot_path.exists():
            self._snapshot_path.unlink()

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

    # ── questions（操作 NOW.md 中的 '## 想了解的问题' section）────

    def read_questions(self) -> str:
        """从 NOW.md 读取 '## 想了解的问题' section 内容。"""
        _, questions, _ = self._split_now_questions(self.read_now())
        if not questions:
            return ""
        lines = [_QUESTIONS_SECTION, ""]
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        return "\n".join(lines) + "\n"

    def write_questions(self, content: str) -> None:
        """覆写 NOW.md 中的 '## 想了解的问题' section，保留其他 section。"""
        text = self.read_now()
        prefix, _, suffix = self._split_now_questions(text)
        parts = []
        if prefix.strip():
            parts.append(prefix.rstrip())
        if content.strip():
            parts.append(content.strip())
        if suffix.strip():
            parts.append(suffix.strip())
        self.write_now("\n\n".join(parts) + "\n")

    def remove_questions_by_indices(self, indices: list[int]) -> None:
        """从 NOW.md 的问题 section 移除 1-based 指定序号，剩余问题重新编号。"""
        text = self.read_now()
        prefix, questions, suffix = self._split_now_questions(text)
        if not questions or not indices:
            return
        remove_set = {int(i) for i in indices if int(i) > 0}
        remaining = [q for i, q in enumerate(questions, 1) if i not in remove_set]

        section_lines = [_QUESTIONS_SECTION, ""]
        section_lines += [f"{i}. {q}" for i, q in enumerate(remaining, 1)]
        section = "\n".join(section_lines)

        parts = []
        if prefix.strip():
            parts.append(prefix.rstrip())
        parts.append(section)
        if suffix.strip():
            parts.append(suffix.strip())
        self.write_now("\n\n".join(parts) + "\n")

    def _extract_now_section(self, text: str, header: str) -> str:
        """提取 NOW.md 中指定 ## 标题 section 的正文（不含标题行本身）。"""
        pattern = re.compile(
            r"^" + re.escape(header) + r"\s*\n(.*?)(?=\n^## |\Z)",
            re.DOTALL | re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            return ""
        return m.group(1).strip()

    def _split_now_questions(self, text: str) -> tuple[str, list[str], str]:
        """把 NOW.md 文本拆成 (questions_section 之前的内容, 问题列表, section 之后的内容)。"""
        # 找到 ## 想了解的问题 section 的起始位置
        pattern = re.compile(
            r"^(## 想了解的问题)\s*$",
            re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            # 没有该 section，prefix = 全文，suffix = 空
            return text, [], ""

        prefix = text[: m.start()]

        # 找下一个同级 ## 标题（section 结束）
        rest = text[m.end():]
        next_section = re.search(r"^## ", rest, re.MULTILINE)
        if next_section:
            body = rest[: next_section.start()]
            suffix = rest[next_section.start():]
        else:
            body = rest
            suffix = ""

        questions: list[str] = []
        for line in body.splitlines():
            mo = re.match(r"^\d+\.\s+(.+)", line)
            if mo:
                questions.append(mo.group(1).strip())

        return prefix, questions, suffix
