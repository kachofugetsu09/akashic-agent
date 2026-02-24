import re
from pathlib import Path

from utils.helpers import ensure_dir


class MemoryStore:
    """Three-layer memory:
    - MEMORY.md   : stable user profile, append-only, sole writer = MemoryOptimizer
    - PENDING.md  : incremental facts extracted during conversations, append-only
    - HISTORY.md  : grep-searchable event log, permanent append
    """
    def __init__(self,workspace: Path):
        self.memory_dir = ensure_dir(workspace/ "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.pending_file = self.memory_dir / "PENDING.md"
        # 确保 PENDING.md 始终存在，避免首次运行时找不到文件
        if not self.pending_file.exists():
            self.pending_file.touch()

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

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

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""

    # ── questions file ────────────────────────────────────────────

    @property
    def _questions_file(self) -> Path:
        return self.memory_dir / "QUESTIONS.md"

    def read_questions(self) -> str:
        if self._questions_file.exists():
            return self._questions_file.read_text(encoding="utf-8")
        return ""

    def write_questions(self, content: str) -> None:
        self._questions_file.write_text(content, encoding="utf-8")

    def remove_questions_by_indices(self, indices: list[int]) -> None:
        """从 QUESTIONS.md 移除 1-based 指定序号的问题，剩余问题重新编号。"""
        text = self.read_questions()
        if not text.strip() or not indices:
            return
        remove_set = {int(i) for i in indices if int(i) > 0}

        header: list[str] = []
        questions: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^\d+\.\s+(.+)", line)
            if m:
                questions.append(m.group(1))
            else:
                header.append(line)

        remaining = [q for i, q in enumerate(questions, 1) if i not in remove_set]
        numbered = [f"{i}. {q}" for i, q in enumerate(remaining, 1)]

        parts = [l for l in header if l.strip()]
        if numbered:
            parts += [""] + numbered
        self.write_questions("\n".join(parts) + "\n")