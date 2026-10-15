"""按需从模型调用账与消息 Turn 计算 KV Cache 用量，不另建投影库。"""
from __future__ import annotations

import heapq
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import cast

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.ledger.contract import MESSAGE_CATALOG, ContentPart, Input, Message, MessageCatalog, Output
from plugins.turn_projection.contract import TURN_PROJECTION, Turn, TurnProjection

from .store import ModelsStore

_PROACTIVE_SOURCES = frozenset({"wake", "drift"})
_RECENT_TURNS = 50
_RECENT_PASSIVE_TURNS = 10
_PREVIEW_CHARS = 90
_SCAN_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class _Fact:
    """扫描时只留下 Turn 用量需要的少量字段，正文随读页释放。"""

    recorded_at: str
    text: str
    call_id: str | None
    output: bool


@dataclass(frozen=True, slots=True)
class TurnUsage:
    session_id: str
    proactive: bool
    recorded_at: str
    preview: str
    prompt_tokens: int
    hit_tokens: int


@dataclass(slots=True)
class _Totals:
    turns: int = 0
    prompt_tokens: int = 0
    hit_tokens: int = 0

    def add(self, turn: TurnUsage) -> None:
        self.turns += 1
        self.prompt_tokens += turn.prompt_tokens
        self.hit_tokens += turn.hit_tokens

    def body(self) -> dict[str, object]:
        return {"turns": self.turns, "prompt_tokens": self.prompt_tokens, "hit_tokens": self.hit_tokens}


@dataclass(slots=True)
class _SourceState:
    """一个 Session 来源已闭合 Turn 的累计量；进程内缓存，重启后按需重算。"""

    after_seq: int = -1
    totals: _Totals = field(default_factory=_Totals)
    recent: deque[TurnUsage] = field(default_factory=lambda: deque(maxlen=_RECENT_TURNS))
    outputs: dict[str, int] = field(default_factory=dict[str, int])


class CacheUsage:
    """Turn 引用的模型调用用量；消息与 Turn 投影在每次调用时借用。"""

    def __init__(self, ctx: Context, store: ModelsStore) -> None:
        self._ctx = ctx
        self._store = store
        self._lock = threading.Lock()
        self._heads: dict[str, int] = {}
        self._sources: dict[tuple[str, str], _SourceState] = {}

    async def overview(self) -> dict[str, object]:
        with self._ctx.borrow(MESSAGE_CATALOG) as catalog, self._ctx.borrow(TURN_PROJECTION) as projection:
            if catalog is None or projection is None:
                raise RuntimeError("消息账本或 Turn 投影不可用")
            return await run_file_io(lambda: self._overview(catalog, projection))

    async def message_output_tokens(self, session_id: str, message_id: str) -> int | None:
        with self._ctx.borrow(MESSAGE_CATALOG) as catalog, self._ctx.borrow(TURN_PROJECTION) as projection:
            if catalog is None or projection is None:
                raise RuntimeError("消息账本或 Turn 投影不可用")
            return await run_file_io(lambda: self._message_output(catalog, projection, session_id, message_id))

    # 汇总所有 Session；被动近期命中只取最近 10 个被动 Turn。
    def _overview(self, catalog: MessageCatalog, projection: TurnProjection) -> dict[str, object]:
        """按需推进扫描，所有当前 Session 完成后才返回完整统计。"""
        with self._lock:
            # 无过滤目录保留软删 Session；按两种可见性读取才是当前有效集合。
            heads = dict(catalog.snapshot_heads(visibility="listed"))
            heads.update(catalog.snapshot_heads(visibility="internal"))
            self._heads = {key: head for key, head in self._heads.items() if key in heads}
            self._sources = {key: state for key, state in self._sources.items() if key[0] in heads}
            # 冷读在闭合 Turn 边界让出请求；下一次接着已有消费位置继续。
            deadline = time.monotonic() + _SCAN_SECONDS
            for session_id, head in heads.items():
                self._refresh(catalog, projection, session_id, head, deadline=deadline)
                if time.monotonic() >= deadline:
                    break
            scanned = sum(self._heads.get(session_id) == head for session_id, head in heads.items())
            if scanned < len(heads):
                return {"complete": False, "scanned_sessions": scanned, "total_sessions": len(heads)}
            states = tuple((source in _PROACTIVE_SOURCES, state) for (_, source), state in self._sources.items())
            passive, proactive = _Totals(), _Totals()
            for is_proactive, state in states:
                target = proactive if is_proactive else passive
                target.turns += state.totals.turns
                target.prompt_tokens += state.totals.prompt_tokens
                target.hit_tokens += state.totals.hit_tokens
            recent = _latest((state.recent for _, state in states), _RECENT_TURNS)
            recent_passive = _Totals()
            for turn in _latest((state.recent for is_proactive, state in states if not is_proactive),
                                _RECENT_PASSIVE_TURNS):
                recent_passive.add(turn)
        return {
            "complete": True,
            "recent_passive": recent_passive.body(), "passive": passive.body(), "proactive": proactive.body(),
            "total": passive.turns + proactive.turns,
            "turns": [{"session_id": turn.session_id, "proactive": turn.proactive, "recorded_at": turn.recorded_at,
                       "preview": turn.preview, "prompt_tokens": turn.prompt_tokens, "hit_tokens": turn.hit_tokens}
                      for turn in recent],
        }

    # 只有被动 Turn 的最终回答带输出用量，与原 Observe 尾注一致。
    def _message_output(
        self, catalog: MessageCatalog, projection: TurnProjection, session_id: str, message_id: str,
    ) -> int | None:
        reader = catalog.reader(session_id)
        if reader.deleted:
            return None
        message = reader.get(message_id)
        if message is None or message.source in _PROACTIVE_SOURCES:
            return None
        with self._lock:
            self._refresh(catalog, projection, session_id, reader.head())
            state = self._sources.get((session_id, message.source))
            return None if state is None else state.outputs.get(message_id)

    # 只扫描上次闭合 Turn 之后的消息；未闭合尾段留到下次。
    def _refresh(
        self, catalog: MessageCatalog, projection: TurnProjection, session_id: str, head: int,
        *, deadline: float | None = None,
    ) -> None:
        """从上次闭合位置读取；时间预算只在完整 Turn 边界生效。"""
        if self._heads.get(session_id) == head:
            return
        reader = catalog.reader(session_id)
        for source in sorted(reader.source_names()):
            state = self._sources.setdefault((session_id, source), _SourceState())
            facts: dict[str, _Fact] = {}
            after = state.after_seq
            stopped = False

            def bounded(rows: Iterable[Message]) -> Iterator[Message]:
                nonlocal stopped
                for message in rows:
                    yield message
                    # 保留完整 Turn；任意消息处截断会反复重读很长的 open 尾段。
                    if (deadline is not None and time.monotonic() >= deadline
                            and isinstance(message.body, Output) and message.body.finish != "continue"):
                        stopped = True
                        return

            def consume(rows: Iterable[Message], source: str = source, after: int = after) -> tuple[Turn, ...]:
                return projection.project(_tap(bounded(rows), facts), source, after_seq=after)

            turns = [turn for turn in reader.scan(consume, after_seq=after, through_seq=head, source=source)
                     if turn.status != "open"]
            call_ids = [facts[identity].call_id for turn in turns for identity in turn.message_ids]
            usages = self._store.read_call_usages([identity for identity in call_ids if identity is not None])
            for turn in turns:
                _record(state, session_id, source, turn, facts, usages)
            if stopped:
                return
        self._heads[session_id] = head


# 1. 闭合 Turn 的全部调用都有 exact 用量才计入；未知不能当零。
# 2. 最终回答的输出 token 供尾注查询；命中与输入只累计被追踪的 Turn。
def _record(
    state: _SourceState, session_id: str, source: str, turn: Turn,
    facts: Mapping[str, _Fact], usages: Mapping[str, Mapping[str, object] | None],
) -> None:
    state.after_seq = turn.through_seq
    members = [(identity, facts[identity]) for identity in turn.message_ids]
    outputs = [(identity, fact) for identity, fact in members if fact.output]
    calls = [fact.call_id for _, fact in outputs if fact.call_id is not None]
    records = [usages.get(identity) for identity in calls]
    exact = [record for record in records if record is not None and record.get("coverage") == "exact"]
    if not calls or len(exact) != len(calls):
        return
    prompt, hit, output = (_sum(exact, name) for name in ("input_tokens", "cached_input_tokens", "output_tokens"))
    if output is not None and outputs and source not in _PROACTIVE_SOURCES:
        state.outputs[outputs[-1][0]] = output
    if prompt is None:
        return
    final = outputs[-1][1] if outputs else members[-1][1]
    text = "\n".join(fact.text for _, fact in members if fact.text).replace("\n", " ").strip()
    preview = text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS].rstrip() + "..."
    usage = TurnUsage(session_id, source in _PROACTIVE_SOURCES, final.recorded_at, preview, prompt, hit or 0)
    state.totals.add(usage)
    state.recent.append(usage)


def _tap(rows: Iterable[Message], facts: dict[str, _Fact]) -> Iterator[Message]:
    for message in rows:
        body = message.body
        call_id: str | None = None
        text = ""
        if isinstance(body, Input):
            text = "\n".join(part.value for part in body.parts
                             if isinstance(part, ContentPart) and part.kind == "text" and isinstance(part.value, str))
        if isinstance(body, Output):
            for part in body.parts:
                if isinstance(part, ContentPart) and part.kind == "model.facts":
                    raw: object = part.value
                    value = cast(Mapping[str, object], raw).get("call_record_id") if isinstance(raw, Mapping) else None
                    call_id = value if isinstance(value, str) and value else None
        facts[message.message_id] = _Fact(
            message.recorded_at.isoformat(), text[:_PREVIEW_CHARS * 2], call_id, isinstance(body, Output),
        )
        yield message


def _sum(records: list[Mapping[str, object]], name: str) -> int | None:
    values = [record.get(name) for record in records]
    return sum(value for value in values if type(value) is int) if all(type(v) is int for v in values) else None


def _latest(groups: Iterable[Iterable[TurnUsage]], limit: int) -> list[TurnUsage]:
    return heapq.nlargest(limit, (turn for group in groups for turn in group), key=lambda turn: turn.recorded_at)
