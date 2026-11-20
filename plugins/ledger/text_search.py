"""会话内正文查找：每个会话一份只含可见正文的内存索引，按 seq 增量追加。

首建要解析整段消息体（生产主会话 2 万条约 5 秒），所以建好后存一份到库旁的索引目录：
重启后直接加载再追加新消息。它是可随时删除的派生数据，前缀版本不符时整份重建。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from plugins.ledger.contract import TextHit, TextSearch

_logger = logging.getLogger(__name__)

# 最近查找过的会话常驻；最大的生产会话（1.1 万条消息）正文约 155 万字，索引连原文约 6MB。
_CACHED_SESSIONS = 4
# 高频字（如「的」）的命中上限：超过后只报「至少」，不为计数走完整份正文。
_OCCURRENCE_LIMIT = 20_000
_SNIPPET_BEFORE = 32
_SNIPPET_AFTER = 80
_SEPARATOR = "\x00"
_FILE_VERSION = 1
# 增量追加积累到这么多条才重写索引文件；没写进去的部分下次加载后再追加，代价很小。
_SAVE_AFTER_ENTRIES = 50

# 只取输入与回复里的 text 段：消息体里其余部分（上下文材料、历史记录、工具结果）不在聊天正文里显示。
_TEXT_ROWS = (
    "SELECT m.seq, m.id, m.ts, json_extract(m.body,'$.kind') AS kind, json_extract(p.value,'$.value') AS text "
    "FROM messages m, json_each(m.body,'$.parts') p "
    "WHERE m.session_key=? AND m.seq>? AND m.seq<=? AND instr(m.body,'\"kind\":\"text\"')>0 "
    "AND json_extract(m.body,'$.kind') IN ('input','output') AND json_extract(p.value,'$.kind')='text' "
    "ORDER BY m.seq, p.key"
)


@dataclass(slots=True)
class _Entry:
    seq: int
    message_id: str
    kind: str
    timestamp: str
    text: str


@dataclass(slots=True)
class _SessionText:
    revision: int | None
    through: int = -1
    entries: list[_Entry] = field(default_factory=list)
    # 小写正文首尾相接成一个字符串，starts[i] 是第 i 条消息在其中的起点；查找只做 C 层 str.find。
    starts: list[int] = field(default_factory=list)
    haystack: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)
    loaded: bool = False
    saved_entries: int = -1

    # 追加 (through, head] 区间里的正文；消息只追加，前缀改写由 revision 失效整份索引。
    def extend(self, connection: sqlite3.Connection, session_id: str, head: int) -> None:
        if head <= self.through:
            return
        pending: list[_Entry] = []
        for seq, message_id, ts, kind, text in connection.execute(_TEXT_ROWS, (session_id, self.through, head)):
            if not isinstance(text, str) or not text:
                continue
            if pending and pending[-1].seq == seq:
                pending[-1].text += "\n" + text
            else:
                pending.append(_Entry(seq, message_id, kind, ts, text))
        self._append(pending)
        self.through = head

    def _append(self, pending: list[_Entry]) -> None:
        chunks = [self.haystack] if self.haystack else []
        offset = len(self.haystack)
        for entry in pending:
            lowered = entry.text.lower()
            # 少数字符小写后长度会变；用小写原文展示，保证摘要偏移和命中对齐。
            if len(lowered) != len(entry.text):
                entry.text = lowered
            if self.entries or chunks:
                chunks.append(_SEPARATOR)
                offset += 1
            self.starts.append(offset)
            self.entries.append(entry)
            chunks.append(lowered)
            offset += len(lowered)
        self.haystack = "".join(chunks)

    # 读回上次存下的索引；版本、会话或前缀版本不符、文件损坏时当作没有，从头建。
    def load(self, path: Path, session_id: str) -> None:
        self.loaded = True
        try:
            data = json.loads(path.read_bytes())
            if data["version"] != _FILE_VERSION or data["session"] != session_id or data["revision"] != self.revision:
                return
            entries = [_Entry(int(seq), str(message_id), str(kind), str(ts), str(text))
                       for seq, message_id, kind, ts, text in data["entries"]]
            through = int(data["through"])
        except FileNotFoundError:
            return
        except (OSError, ValueError, KeyError, TypeError) as error:
            _logger.warning("会话正文索引无法读取，将重建: %s (%s)", path.name, error)
            return
        self._append(entries)
        self.through = through
        self.saved_entries = len(self.entries)

    # 首建完成或新增够多时原子替换索引文件；写失败只影响下次重启的首查速度。
    def save(self, path: Path, session_id: str) -> None:
        if self.saved_entries >= 0 and len(self.entries) - self.saved_entries < _SAVE_AFTER_ENTRIES:
            return
        payload = json.dumps({
            "version": _FILE_VERSION, "session": session_id, "revision": self.revision, "through": self.through,
            "entries": [[entry.seq, entry.message_id, entry.kind, entry.timestamp, entry.text] for entry in self.entries],
        }, ensure_ascii=False, separators=(",", ":")).encode()
        temporary = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, path)
        except OSError as error:
            _logger.warning("会话正文索引无法保存: %s (%s)", path.name, error)
            return
        self.saved_entries = len(self.entries)

    # 1. 在整份小写正文里逐个找命中，按起点二分归到消息
    # 2. 每条命中消息记命中数与第一处位置，返回最新的 limit 条
    def search(self, needle: str, limit: int) -> tuple[list[TextHit], int, bool]:
        firsts: dict[int, int] = {}
        counts: dict[int, int] = {}
        occurrences = 0
        at = self.haystack.find(needle)
        while at >= 0 and occurrences < _OCCURRENCE_LIMIT:
            index = bisect_right(self.starts, at) - 1
            counts[index] = counts.get(index, 0) + 1
            firsts.setdefault(index, at - self.starts[index])
            occurrences += 1
            at = self.haystack.find(needle, at + len(needle))
        capped = at >= 0
        indexes = sorted(firsts)[-limit:]
        return [self._hit(index, firsts[index], counts[index], len(needle)) for index in indexes], len(firsts), capped

    def _hit(self, index: int, offset: int, count: int, size: int) -> TextHit:
        entry = self.entries[index]
        text = entry.text
        start = max(0, offset - _SNIPPET_BEFORE)
        before = " ".join(text[start:offset].split())
        after = " ".join(text[offset + size:offset + size + _SNIPPET_AFTER].split())
        return TextHit(
            message_id=entry.message_id, seq=entry.seq, kind=entry.kind, timestamp=entry.timestamp, count=count,
            before=("…" if start else "") + before, match=text[offset:offset + size],
            after=after + ("…" if offset + size + _SNIPPET_AFTER < len(text) else ""),
        )


class TextIndexes:
    """MessageLog 持有的会话正文索引；调用者在同一只读快照内传入连接、上界与前缀版本。"""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _SessionText] = OrderedDict()

    def _path(self, session_id: str) -> Path:
        return self._directory / f"{hashlib.sha256(session_id.encode()).hexdigest()[:32]}.json"

    def _session(self, session_id: str, revision: int | None) -> _SessionText:
        with self._lock:
            current = self._sessions.get(session_id)
            if current is None or current.revision != revision:
                current = _SessionText(revision)
                self._sessions[session_id] = current
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > _CACHED_SESSIONS:
                self._sessions.popitem(last=False)
            return current

    def search(
        self, connection: sqlite3.Connection, session_id: str, *,
        query: str, head: int, revision: int | None, limit: int,
    ) -> TextSearch:
        needle = query.strip().lower()
        session = self._session(session_id, revision)
        # 同一会话的首建（或读回）只做一次：并发的第二个查询等它完成后直接查。
        with session.lock:
            path = self._path(session_id)
            if not session.loaded:
                session.load(path, session_id)
            session.extend(connection, session_id, head)
            session.save(path, session_id)
            through = session.through
            if not needle:
                return TextSearch(hits=(), total=0, capped=False, through_seq=through)
            hits, total, capped = session.search(needle, limit)
        return TextSearch(hits=tuple(hits), total=total, capped=capped, through_seq=through)
