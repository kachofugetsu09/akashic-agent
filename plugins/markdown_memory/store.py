from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, cast

from infra.persistence.json_store import atomic_write_text

from .documents import MEMORY, MEMORY_TEMPLATE, SCOPE_FILE, SELF, TOPICS, Workspace, parse_ops, split_frontmatter

logger = logging.getLogger(__name__)
_APPLY_ATTEMPTS = 3


class ConcurrentChange(RuntimeError):
    """叠加期间文件被其他 writer 持续修改；draft 保留待处理，由调用者稍后重试。"""


DEFAULT_SELF_MD = """# Akashic 的自我认知

## 人格与形象
- 我是 Akashic，一个直接、温暖、主动参与思考的长期协作伙伴。
- 我优先给出结论，再补充必要细节；不把自己伪装成没有立场的工具。

## 我对当前用户的理解
- 我会从长期记忆中逐步形成对当前用户的理解，不在缺少证据时编造画像。

## 我们关系的定义
- 我与当前用户的关系以透明、尊重边界和持续协作为基础。
"""


def content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class MarkdownProfileStore:
    """拥有 memory/ 下的档案文件与按 source_ref 幂等的提炼 receipt。"""

    def __init__(self, root: Path, receipts_path: Path) -> None:
        self.root = root
        self.receipts_path = receipts_path
        (root / TOPICS).mkdir(parents=True, exist_ok=True)
        # 缺失或只有空白的画像写入合法标题结构；有内容的手写文件保持原样。
        if not (root / MEMORY).exists() or not (root / MEMORY).read_text(encoding="utf-8").strip():
            atomic_write_text(root / MEMORY, MEMORY_TEMPLATE, domain="markdown_memory")
        if not (root / SELF).exists():
            atomic_write_text(root / SELF, DEFAULT_SELF_MD, domain="markdown_memory")
        self._init_schema()

    # 旧 PENDING 队列与未完成的旧版 draft 须先人工处理，不在新版本里兼容。
    def check_layout(self) -> None:
        for name in ("PENDING.md", "PENDING.snapshot.md"):
            path = self.root / name
            if path.exists() and path.read_text(encoding="utf-8").strip():
                raise RuntimeError(f"memory/{name} 仍有旧队列内容：请手动并入 MEMORY.md 或 topics/")
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            refs = [row[0] for row in conn.execute(
                "SELECT DISTINCT source_ref FROM consolidation_writes WHERE kind IN (?, ?)",
                ("markdown_memory_draft_v1", "markdown_self_draft_v1"))]
        if any(not self._legacy_applied(ref) for ref in refs):
            raise RuntimeError("存在未完成的旧版 Markdown draft：请先用旧版本完成恢复再升级")

    def read(self, doc: str) -> str:
        path = self._path(doc)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def folders(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for marker in sorted((self.root / TOPICS).glob(f"*/{SCOPE_FILE}")):
            scope = split_frontmatter(marker.read_text(encoding="utf-8"))[0].get("scope")
            if scope:
                result[marker.parent.name] = scope
        return result

    # 一次提炼可见的文件：两份档案、全局主题与当前 scope 文件夹。
    def workspace(self, scope: str | None) -> Workspace:
        folders = self.folders()
        names = [MEMORY, SELF, *(f"{TOPICS}/{path.name}" for path in sorted((self.root / TOPICS).glob("*.md")))]
        folder = next((name for name, key in folders.items() if key == scope), None)
        if folder is not None:
            names += [f"{TOPICS}/{folder}/{path.name}" for path in sorted((self.root / TOPICS / folder).glob("*.md"))]
        return Workspace({name: self.read(name) for name in names}, folders, scope)

    # 注入视图：可写工作区之外，再给出其他文件夹的说明。
    def view(self, scope: str | None) -> tuple[Workspace, dict[str, str]]:
        workspace = self.workspace(scope)
        current = workspace.folder()
        others = {name: split_frontmatter(self.read(f"{TOPICS}/{name}/{SCOPE_FILE}"))[0].get("description", "")
                  for name in workspace.folders if name != current}
        return workspace, others

    def ages_days(self, docs: list[str]) -> dict[str, int]:
        now = time.time()
        return {doc: int((now - self._path(doc).stat().st_mtime) // 86400)
                for doc in docs if self._path(doc).exists()}

    # 整理到期时返回本轮要整理的 scope（None 为全局），按上次整理后新应用的提炼 draft 计数。
    def consolidation_scopes(self, hours: int, min_drafts: int) -> tuple[str | None, ...]:
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            last = conn.execute("SELECT max(done_at) FROM consolidation_writes WHERE kind=?",
                                ("markdown_consolidated_v1",)).fetchone()[0]
            if last is not None and conn.execute("SELECT datetime(?, ?) > datetime('now')",
                                                 (last, f"+{hours} hours")).fetchone()[0]:
                return ()
            rows = conn.execute(
                "SELECT draft.payload FROM consolidation_writes AS applied JOIN consolidation_writes AS draft "
                "ON draft.source_ref=applied.source_ref AND draft.kind='markdown_ops_v1' "
                "WHERE applied.kind='markdown_ops_applied_v1' AND applied.source_ref NOT LIKE 'consolidation:%' "
                "AND applied.done_at > ?", (last or "",)).fetchall()
        if len(rows) < min_drafts:
            return ()
        scopes = {cast(dict[str, object], json.loads(row[0])).get("scope") for row in rows}
        return (None, *sorted(scope for scope in scopes if isinstance(scope, str)))

    def mark_consolidated(self, stamp: str, scopes: tuple[str | None, ...]) -> None:
        _ = self._write_once(f"consolidation:{stamp}", "markdown_consolidated_v1", {"scopes": list(scopes)})

    def read_draft(self, source_ref: str) -> dict[str, object] | None:
        return self._read_receipt(source_ref, "markdown_ops_v1")

    def write_draft(self, source_ref: str, payload: dict[str, object], *,
                    session_key: str, generation: int) -> dict[str, object]:
        # 操作与会话进度同一事务提交，恢复时不会出现只有操作的提炼 draft。
        self._write_once_many(source_ref, (
            ("markdown_ops_v1", payload),
            ("markdown_projection_order_v1", {"session_key": session_key, "generation": generation}),
        ))
        return dict(payload)

    # 不属于任何会话进度的操作（如整理），只落 draft，由 pending 恢复路径兜底。
    def write_ops(self, source_ref: str, payload: dict[str, object]) -> dict[str, object]:
        return self._write_once(source_ref, "markdown_ops_v1", payload)

    def is_applied(self, source_ref: str) -> bool:
        return (self._read_receipt(source_ref, "markdown_ops_applied_v1") is not None
                or self._legacy_applied(source_ref))

    def _legacy_applied(self, source_ref: str) -> bool:
        return all(self._read_receipt(source_ref, f"markdown_{document}_applied_v1") is not None
                   for document in ("memory", "self"))

    def latest_applied(self, session_key: str) -> tuple[str, int] | None:
        """从已有顺序与双文件 receipt 读取进度，不建立第二份消费 ledger。"""
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            rows = conn.execute("SELECT source_ref, payload FROM consolidation_writes WHERE kind=?",
                                ("markdown_projection_order_v1",)).fetchall()
        latest: tuple[str, int] | None = None
        seen: set[int] = set()
        for source_ref, payload in rows:
            raw: object = json.loads(payload)
            if not isinstance(raw, dict):
                raise ValueError("Markdown 顺序 receipt schema 无效")
            order = cast(dict[str, object], raw)
            if set(order) != {"session_key", "generation"}:
                raise ValueError("Markdown 顺序 receipt schema 无效")
            key, generation = order["session_key"], order["generation"]
            if not isinstance(key, str) or type(generation) is not int or generation < 0:
                raise ValueError("Markdown 顺序 receipt 身份无效")
            if key != session_key or not self.is_applied(source_ref):
                continue
            if generation in seen:
                raise ValueError("同一 Session generation 出现多条已应用档案")
            seen.add(generation)
            if latest is None or generation > latest[1]:
                latest = (source_ref, generation)
        return latest

    def pending_source_refs(self) -> tuple[str, ...]:
        """已落盘操作但尚未确认应用的 draft，按写入顺序恢复。"""
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            rows = conn.execute("SELECT source_ref FROM consolidation_writes WHERE kind=? ORDER BY done_at, source_ref",
                                ("markdown_ops_v1",)).fetchall()
        return tuple(row[0] for row in rows if not self.is_applied(row[0]))

    # 在调用者持有文件锁时把操作叠加到当前文件；幂等叠加使崩溃后重放收敛。
    def apply(self, source_ref: str) -> None:
        if self.is_applied(source_ref):
            return
        draft = self.read_draft(source_ref)
        if draft is None:
            raise RuntimeError(f"Markdown 操作 draft 缺失: {source_ref}")
        # 提炼 draft 必须带会话进度；整理 draft 不推进任何会话进度。
        if (self._read_receipt(source_ref, "markdown_projection_order_v1") is None
                and not source_ref.startswith("consolidation:")):
            raise RuntimeError(f"Markdown 提炼 draft 缺少会话进度，拒绝应用: {source_ref}")
        scope = draft.get("scope")
        if scope is not None and not isinstance(scope, str):
            raise ValueError(f"Markdown draft scope 无效: {source_ref}")
        ops = parse_ops({"ops": draft.get("ops")})
        for _attempt in range(_APPLY_ATTEMPTS):
            # 1. 读取当前文件并叠加；不持锁的原生工具可能在其间写入，替换前再比对一次。
            workspace = self.workspace(scope)
            before = dict(workspace.texts)
            workspace.apply(ops, strict=False)
            changed = {doc: text for doc, text in workspace.texts.items() if before.get(doc) != text}
            if any(self.read(doc) != before.get(doc, "") for doc in changed):
                continue
            # 2. 首次 before-image 保留；写入后才记 applied，冲突随 receipt 留存。
            self._write_first(source_ref, "markdown_before_v1", {doc: before.get(doc, "") for doc in changed})
            for doc, text in changed.items():
                path = self._path(doc)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_text(path, text, domain="markdown_memory")
            for conflict in workspace.conflicts:
                logger.warning("Markdown 操作前提失效，已跳过: source_ref=%s %s", source_ref, conflict)
            _ = self._write_once(source_ref, "markdown_ops_applied_v1", {
                "files": {doc: content_digest(text) for doc, text in changed.items()},
                "conflicts": list(workspace.conflicts),
            })
            return
        raise ConcurrentChange(f"Markdown 文件在叠加期间持续变化，保留待处理: {source_ref}")

    def _path(self, doc: str) -> Path:
        parts = Path(doc).parts
        if not parts or any(part in {"", ".", ".."} for part in parts) or Path(doc).is_absolute():
            raise ValueError(f"Markdown 路径越界: {doc}")
        return self.root / doc

    def _init_schema(self) -> None:
        self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.receipts_path))) as conn:
            _ = conn.execute("""CREATE TABLE IF NOT EXISTS consolidation_writes (
                    source_ref TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload TEXT,
                    trailing_blank_line INTEGER NOT NULL DEFAULT 0,
                    done_at TEXT NOT NULL,
                    PRIMARY KEY (source_ref, kind)
                )""")

    def _read_receipt(self, source_ref: str, kind: str) -> dict[str, object] | None:
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            row = conn.execute(
                "SELECT payload FROM consolidation_writes WHERE source_ref=? AND kind=?",
                (source_ref, kind),
            ).fetchone()
        if row is None:
            return None
        raw = row[0]
        if not isinstance(raw, str):
            raise ValueError(f"Markdown profile receipt payload 缺失: {source_ref}/{kind}")
        value = cast(Any, json.loads(raw))
        if not isinstance(value, dict):
            raise ValueError(f"Markdown profile receipt schema 无效: {source_ref}/{kind}")
        return {str(key): item for key, item in cast(dict[object, object], value).items()}

    def _write_once(
        self,
        source_ref: str,
        kind: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        self._write_once_many(source_ref, ((kind, payload),))
        return dict(payload)

    # 同一事务写入多条 receipt；已存在的必须与本次内容一致。
    def _write_once_many(self, source_ref: str, items: tuple[tuple[str, dict[str, object]], ...]) -> None:
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            _ = conn.execute("BEGIN IMMEDIATE")
            for kind, payload in items:
                encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                row = conn.execute(
                    "SELECT payload FROM consolidation_writes WHERE source_ref=? AND kind=?",
                    (source_ref, kind),
                ).fetchone()
                if row is not None:
                    if row[0] != encoded:
                        conn.rollback()
                        raise ValueError(f"Markdown profile receipt 内容冲突: {source_ref}/{kind}")
                    continue
                _ = conn.execute(
                    "INSERT INTO consolidation_writes "
                    "(source_ref, kind, payload, trailing_blank_line, done_at) "
                    "VALUES (?, ?, ?, 0, datetime('now'))",
                    (source_ref, kind, encoded),
                )
            conn.commit()

    def _write_first(self, source_ref: str, kind: str, payload: dict[str, object]) -> None:
        """保留首次尝试的 before-image；重放时后续尝试不覆盖。"""
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with closing(sqlite3.connect(str(self.receipts_path), timeout=30.0)) as conn:
            _ = conn.execute(
                "INSERT OR IGNORE INTO consolidation_writes "
                "(source_ref, kind, payload, trailing_blank_line, done_at) VALUES (?, ?, ?, 0, datetime('now'))",
                (source_ref, kind, encoded),
            )
            conn.commit()
