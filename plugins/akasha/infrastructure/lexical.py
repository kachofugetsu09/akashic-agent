"""在线查询与构建共用的中英分词器。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from functools import cache
import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import re
import sqlite3
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jieba

logger = logging.getLogger(__name__)
TOKEN_CHUNKS = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+")


# jieba 导入与词典初始化只在首次切分中文时发生，不进入插件启动路径。
@cache
def _tokenizer() -> "jieba.Tokenizer":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="pkg_resources is deprecated as an API.*",
            category=UserWarning,
        )
        # jieba 0.42.1 在 Python 3.12 下产生 invalid escape sequence SyntaxWarning，
        # 测试与 Gate 的 -W error 下必须显式忽略，否则导入即失败。
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        import jieba
    return jieba.Tokenizer()


def tokenize(text: str) -> Counter[str]:
    """把中文与 ASCII 文本切成词频。"""

    terms: list[str] = []
    for chunk in TOKEN_CHUNKS.findall(text.lower()):
        pieces = (
            _tokenizer().lcut_for_search(chunk)
            if any("\u3400" <= c <= "\u9fff" for c in chunk)
            else [chunk]
        )
        terms.extend(piece.strip() for piece in pieces if len(piece.strip()) >= 2)
    return Counter(terms)


Terms = tuple[tuple[str, int], ...]
# 切分规则或缓存编码变化时递增，旧缓存整表作废。
_TERMS_VERSION = 1


def terms(text: str) -> Terms:
    return tuple(sorted(tokenize(text).items()))


# 开机恢复按正文哈希复用上次的切分结果；结果只由正文、jieba 版本与切分规则决定，
# 指纹不符即视为没有缓存。缓存损坏或写失败只退回现场切分，不影响学习结果。
class TermCache:
    def __init__(self, path: Path):
        self._path = path
        self._known: dict[str, Terms] = {}
        self._added: dict[str, Terms] = {}

    def terms(self, text: str) -> Terms:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        found = self._known.get(key)
        if found is None:
            found = self._added.get(key)
        if found is None:
            found = self._added[key] = terms(text)
        return found

    def load(self) -> None:
        try:
            with closing(sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)) as connection:
                row = connection.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
                if row is None or row[0] != _fingerprint():
                    return
                self._known = {
                    key: tuple((term, count) for term, count in json.loads(value))
                    for key, value in connection.execute("SELECT hash, terms FROM terms")
                }
        except (sqlite3.Error, ValueError, TypeError):
            logger.warning("Akasha 分词缓存不可读，本次现场切分: %s", self._path, exc_info=True)
            self._known = {}

    def save(self) -> None:
        if not self._added:
            return
        try:
            with closing(sqlite3.connect(self._path, timeout=30)) as connection:
                _write_terms(connection, self._added)
        except sqlite3.Error:
            logger.warning("Akasha 分词缓存写入失败，下次开机重新切分: %s", self._path, exc_info=True)
            return
        self._added = {}

    def release(self) -> None:
        self._known = {}


def _fingerprint() -> str:
    return f"{_TERMS_VERSION}:{version('jieba')}:{TOKEN_CHUNKS.pattern}"


def _write_terms(connection: sqlite3.Connection, added: Mapping[str, Terms]) -> None:
    # 1. 指纹变化时先清空旧切分，再写入本次结果与新指纹。
    with connection:
        _ = connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        _ = connection.execute("CREATE TABLE IF NOT EXISTS terms (hash TEXT PRIMARY KEY, terms TEXT NOT NULL)")
        fingerprint = _fingerprint()
        row = connection.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        if row is None or row[0] != fingerprint:
            _ = connection.execute("DELETE FROM terms")
            _ = connection.execute("INSERT OR REPLACE INTO meta VALUES ('fingerprint', ?)", (fingerprint,))
        # 2. 同一正文的结果确定，并发写入同键时保留先到者即可。
        _ = connection.executemany(
            "INSERT OR IGNORE INTO terms VALUES (?, ?)",
            [(key, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
             for key, value in added.items()],
        )
