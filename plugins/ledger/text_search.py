"""会话内正文查找：每个会话一份只含可见正文的内存索引，按 seq 增量追加。"""

from __future__ import annotations

import sqlite3
import threading
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field

from plugins.ledger.contract import TextHit, TextSearch

# 最近查找过的会话常驻；最大的生产会话（1.1 万条消息）正文约 155 万字，索引连原文约 6MB。
_CACHED_SESSIONS = 4
# 高频字（如「的」）的命中上限：超过后只报「至少」，不为计数走完整份正文。
_OCCURRENCE_LIMIT = 20_000
_SNIPPET_BEFORE = 32
_SNIPPET_AFTER = 80
_SEPARATOR = "\x00"

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
        self.through = head

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

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _SessionText] = OrderedDict()

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
        # 同一会话的首建只做一次：并发的第二个查询等它建完后直接查。
        with session.lock:
            session.extend(connection, session_id, head)
            through = session.through
            if not needle:
                return TextSearch(hits=(), total=0, capped=False, through_seq=through)
            hits, total, capped = session.search(needle, limit)
        return TextSearch(hits=tuple(hits), total=total, capped=capped, through_seq=through)
