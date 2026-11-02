"""目标插件测试用的 Ledger owner 夹具。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from plugins.ledger.contract import (
    ContentPart,
    ContentReferences,
    Input,
    MessageReader,
    MessageWriter,
    OwnerStore,
)
from plugins.ledger.log import MessageLog

from plugins.goal.store import GoalStore


class _OwnerState:
    def __init__(self, store: OwnerStore) -> None:
        self._store = store

    def open(self, ctx: object) -> OwnerStore:
        return self._store

    def open_scoped(self, ctx: object, scope: str) -> OwnerStore:
        return self._store


class _Catalog:
    def __init__(self, log: MessageLog) -> None:
        self._log = log

    def reader(self, session_id: str) -> MessageReader:
        return self._log.reader(session_id)


class _Writers:
    def __init__(self, log: MessageLog) -> None:
        self._log = log

    def bind(
        self, ctx: object, *, author: str, source: str, body_types: tuple[type, ...],
        content: Mapping[str, Callable[[ContentPart], ContentReferences]], **_: object,
    ):
        def make(session_id: str) -> MessageWriter:
            return self._log.writer(
                session_id, author=author, source=source,
                body_types=body_types, content=content,
            )
        return make


class _Content:
    def check_text(self, part: object) -> ContentReferences:
        return ContentReferences()


@dataclass
class GoalHarness:
    log: MessageLog
    store: GoalStore


def goal_harness(tmp_path: Path, owner: str = "goal") -> GoalHarness:
    log = MessageLog(tmp_path / "sessions.db")
    store = GoalStore(
        ctx=cast(Any, object()),
        state=_OwnerState(log.owner(owner)),
        catalog=_Catalog(log),
        writers=_Writers(log),
        content=_Content(),
    )
    return GoalHarness(log, store)


def user_input(log: MessageLog, session_id: str, message_id: str, text: str) -> None:
    writer = log.writer(
        session_id, author="user", source="conversation", body_types=(Input,),
        content={"text": lambda part: ContentReferences()},
    )
    writer.append(message_id, Input((_text_part(text),)))
    writer.expire()


def _text_part(text: str):
    from plugins.ledger.contract import ContentPart

    return ContentPart("text", text)
