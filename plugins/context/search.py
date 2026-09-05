from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from session.message import ContentPart, Control, Input, Message, Output


@dataclass(frozen=True, slots=True)
class SearchResult:
    messages: tuple[Message, ...]
    total: int


class MessageSearch:
    """从日志追赶的临时内容索引；无磁盘状态、持久 cursor 或消息修改权。"""

    def __init__(self, snapshot: Sequence[Message]):
        self._messages: dict[str, Message] = {}
        self._connection = sqlite3.connect(":memory:")
        try:
            _ = self._connection.execute(
                """CREATE VIRTUAL TABLE content_index USING fts5(
                message_id UNINDEXED, session_id UNINDEXED, author UNINDEXED,
                source UNINDEXED, kind UNINDEXED, recorded_at UNINDEXED,
                content, tokenize='trigram'
            )"""
            )
            self.append(snapshot)
        except BaseException:
            self._connection.close()
            raise

    def append(self, messages: Sequence[Message]) -> None:
        """幂等应用已提交消息；重扫失败不推进索引，也不改写源日志。"""
        pending: dict[str, Message] = {}
        for message in messages:
            prior = pending.get(
                message.message_id, self._messages.get(message.message_id)
            )
            if prior is not None:
                if prior != message:
                    raise ValueError("索引收到同 ID 不同消息")
                continue
            pending[message.message_id] = message
        with self._connection:
            _ = self._connection.executemany(
                "INSERT INTO content_index VALUES (?,?,?,?,?,?,?)",
                (
                    (
                        message.message_id,
                        message.session_id,
                        message.author,
                        message.source,
                        (
                            "input"
                            if isinstance(message.body, Input)
                            else (
                                "output"
                                if isinstance(message.body, Output)
                                else "tool_result"
                            )
                        ),
                        message.recorded_at.timestamp(),
                        "".join(
                            cast(str, part.value)
                            for part in message.body.parts
                            if isinstance(part, ContentPart) and part.kind == "text"
                        ).casefold(),
                    )
                    for message in pending.values()
                    if not isinstance(message.body, Control)
                ),
            )
        self._messages.update(pending)

    def close(self) -> None:
        self._connection.close()

    def search(
        self,
        query: str,
        *,
        session_id: str | None = None,
        source: str | None = None,
        author: str | None = None,
        kind: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> SearchResult:
        """按字面词的命中数及时间排序；短中文词和标点不进入 FTS 查询语法。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("搜索词不能为空")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit 必须在 1 到 100 之间")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset 必须是非负整数")
        if kind is not None and kind not in {"input", "output", "tool_result"}:
            raise ValueError("搜索 kind 无效")
        terms = tuple(dict.fromkeys(query.casefold().split()))
        long_terms = [term for term in terms if len(term) >= 3]
        short_terms = [term for term in terms if len(term) < 3]
        # 1. FTS 只缩小长词候选；短词保持字面扫描，不依赖分词器猜语义。
        candidates: list[str] = []
        parameters: list[object] = []
        if long_terms:
            candidates.append(
                "SELECT rowid FROM content_index WHERE content_index MATCH ?"
            )
            parameters.append(
                " OR ".join('"' + term.replace('"', '""') + '"' for term in long_terms)
            )
        if short_terms:
            candidates.append(
                "SELECT rowid FROM content_index WHERE "
                + " OR ".join("instr(content,?)>0" for _ in short_terms)
            )
            parameters.extend(short_terms)
        conditions = ["rowid IN (" + " UNION ".join(candidates) + ")"]
        # 2. 所有过滤在分页和计数前执行；命中正文仍来自原 Message。
        for column, value in (
            ("session_id", session_id),
            ("source", source),
            ("author", author),
            ("kind", kind),
        ):
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"{column} 必须是非空字符串")
                conditions.append(f"{column}=?")
                parameters.append(value)
        score = " + ".join("(instr(content,?)>0)" for _ in terms)
        sql = (
            "WITH matches AS (SELECT message_id, recorded_at, "
            + score
            + " AS score FROM content_index WHERE "
            + " AND ".join(conditions)
            + ") "
        )
        args: list[object] = [*terms, *parameters]
        total = self._connection.execute(
            sql + "SELECT count(*) FROM matches WHERE score>0", args
        ).fetchone()[0]
        rows = self._connection.execute(
            sql
            + "SELECT message_id FROM matches WHERE score>0 ORDER BY score DESC, recorded_at DESC, message_id LIMIT ? OFFSET ?",
            [*args, limit, offset],
        ).fetchall()
        return SearchResult(tuple(self._messages[row[0]] for row in rows), total)
