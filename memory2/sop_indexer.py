"""
SOP 增量重索引：SOP 文件写入后自动更新 memory2 DB
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from memory2.store import MemoryStore2
from memory2.embedder import Embedder

logger = logging.getLogger(__name__)

# 文件名 → memory_type 映射（与 POC 一致，README 跳过）
_SOP_TYPE_MAP: dict[str, str | None] = {
    "user-preferences.md": "preference",
    "core-rules.md": "procedure",
    "bilibili-download-sop.md": "procedure",
    "novel-kb-query.md": "procedure",
    "novel-reporting-sop.md": "procedure",
    "README.md": None,
}
_DEFAULT_TYPE = "procedure"


def _parse_sop_chunks(path: Path, memory_type: str) -> list[dict]:
    """按 ## 二级标题分 chunk，与 POC 逻辑一致。"""
    text = path.read_text(encoding="utf-8")
    chunks = re.split(r"\n(?=## )", text)
    results = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue
        lines = chunk.splitlines()
        title = lines[0].lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()[:400]
        summary = f"[{path.name}] {title}: {body}" if body else f"[{path.name}] {title}"
        summary = summary[:500]
        extra: dict = {"source_file": str(path)}
        if memory_type == "procedure":
            tool_hints = re.findall(r"`(\w+_tool|\w+\.py|\w+_snapshot|\w+-dlp)`", chunk)
            if tool_hints:
                extra["trigger_keywords"] = list(set(tool_hints))
        results.append({"summary": summary, "extra": extra})
    return results


class SopIndexer:
    def __init__(self, store: MemoryStore2, embedder: Embedder, sop_dir: Path) -> None:
        self._store = store
        self._embedder = embedder
        self._sop_dir = sop_dir.resolve()

    def is_sop_file(self, path: Path) -> bool:
        """判断路径是否属于 sop/ 目录下的 .md 文件。"""
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            return False
        return (
            str(resolved).startswith(str(self._sop_dir))
            and resolved.suffix.lower() == ".md"
        )

    async def reindex(self, path: Path) -> str:
        """删除旧 chunks，重新 embed 并写入。返回操作摘要。"""
        resolved = path.expanduser().resolve()
        filename = resolved.name

        # README 不索引
        memory_type = _SOP_TYPE_MAP.get(filename, _DEFAULT_TYPE)
        if memory_type is None:
            return f"跳过（README 不索引）"

        if not resolved.exists():
            # 文件被删除，只清旧数据
            deleted = self._store.delete_by_source_ref(filename)
            return f"文件已删除，清理旧 chunks {deleted} 条"

        # 1. 删旧 chunks
        deleted = self._store.delete_by_source_ref(filename)
        logger.info(f"sop_indexer: 删除旧 chunks {deleted} 条 source_ref={filename}")

        # 2. 解析新内容
        chunks = _parse_sop_chunks(resolved, memory_type)
        if not chunks:
            return f"解析到 0 个 chunk，旧数据已清理"

        # 3. 批量 embed
        texts = [c["summary"] for c in chunks]
        embeddings = await self._embedder.embed_batch(texts)

        # 4. 写入
        new_count = 0
        for chunk, emb in zip(chunks, embeddings):
            result = self._store.upsert_item(
                memory_type=memory_type,
                summary=chunk["summary"],
                embedding=emb,
                source_ref=filename,
                extra=chunk["extra"],
            )
            if result.startswith("new"):
                new_count += 1

        logger.info(
            f"sop_indexer: {filename} 重索引完成，{new_count}/{len(chunks)} 条写入"
        )
        return f"SOP 已重索引：{filename}，{new_count}/{len(chunks)} chunks 更新"
