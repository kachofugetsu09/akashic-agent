"""从模型调用账与消息 Turn 增量计算 KV Cache 用量；已闭合 Turn 的累计量落在可重建的检查点里。"""
from __future__ import annotations

import heapq
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
# 检查点里存的是按当前计数规则算好的结果：改了计数规则（什么算计入、主动来源集合、预览）必须加一，旧结果随之丢弃重算。
_CHECKPOINT_SCHEMA = 1

logger = logging.getLogger(__name__)


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
    """一个 Session 来源已闭合 Turn 的累计量；由检查点跨重启保留。"""

    after_seq: int = -1
    totals: _Totals = field(default_factory=_Totals)
    recent: deque[TurnUsage] = field(default_factory=lambda: deque(maxlen=_RECENT_TURNS))
    outputs: dict[str, int] = field(default_factory=dict[str, int])
    # 以下两项只在内存：尚未写入检查点的输出用量，以及检查点里这个来源的旧输出是否作废。
    unsaved: dict[str, int] = field(default_factory=dict[str, int])
    reset: bool = False


class _Checkpoint:
    """派生检查点：只存已闭合 Turn 的消费位置与累计量，删除文件即从头重扫。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    # 版本不符或文件损坏时丢弃整份检查点，由下一次扫描重建。
    def load(self) -> tuple[dict[str, int], dict[tuple[str, str], _SourceState]]:
        if not self._path.exists():
            return {}, {}
        try:
            with closing(sqlite3.connect(self._path)) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                if version != _CHECKPOINT_SCHEMA:
                    raise sqlite3.DatabaseError(f"checkpoint schema {version}")
                heads = dict(connection.execute("SELECT session_id, head FROM heads"))
                sources: dict[tuple[str, str], _SourceState] = {}
                for session_id, source, after, turns, prompt, hit, recent in connection.execute(
                        "SELECT session_id, source, after_seq, turns, prompt_tokens, hit_tokens, recent FROM sources"):
                    sources[(session_id, source)] = _SourceState(
                        after, _Totals(turns, prompt, hit),
                        deque((TurnUsage(**item) for item in json.loads(recent)), maxlen=_RECENT_TURNS))
                for session_id, source, message_id, tokens in connection.execute(
                        "SELECT session_id, source, message_id, tokens FROM outputs"):
                    state = sources.get((session_id, source))
                    if state is not None:
                        state.outputs[message_id] = tokens
            return heads, sources
        except (sqlite3.DatabaseError, ValueError, TypeError) as error:
            logger.warning("cache usage checkpoint discarded: %s", error)
            self._path.unlink(missing_ok=True)
            return {}, {}

    # 1. 被移除的 Session 整体删除；2. 变化的来源整行替换，重置过的先删旧输出，输出只追加未写入的；
    # 3. head 最后写，保证不越过已写入的累计量。
    def save(self, heads: Mapping[str, int], sources: Mapping[tuple[str, str], _SourceState],
             removed: Iterable[str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self._path)) as connection, connection:
            _ = connection.executescript(f"""
                CREATE TABLE IF NOT EXISTS heads (session_id TEXT PRIMARY KEY, head INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS sources (session_id TEXT, source TEXT, after_seq INTEGER NOT NULL,
                    turns INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL, hit_tokens INTEGER NOT NULL,
                    recent TEXT NOT NULL, PRIMARY KEY (session_id, source));
                CREATE TABLE IF NOT EXISTS outputs (session_id TEXT, source TEXT, message_id TEXT,
                    tokens INTEGER NOT NULL, PRIMARY KEY (session_id, message_id));
                PRAGMA user_version = {_CHECKPOINT_SCHEMA};
            """)
            for session_id in removed:
                for table in ("heads", "sources", "outputs"):
                    _ = connection.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            for (session_id, source), state in sources.items():
                if state.reset:
                    _ = connection.execute("DELETE FROM outputs WHERE session_id = ? AND source = ?", (session_id, source))
                _ = connection.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)", (
                    session_id, source, state.after_seq, state.totals.turns, state.totals.prompt_tokens,
                    state.totals.hit_tokens, json.dumps([asdict(turn) for turn in state.recent], ensure_ascii=False)))
                _ = connection.executemany("INSERT OR REPLACE INTO outputs VALUES (?, ?, ?, ?)",
                                       ((session_id, source, message_id, tokens)
                                        for message_id, tokens in state.unsaved.items()))
            _ = connection.executemany("INSERT OR REPLACE INTO heads VALUES (?, ?)", heads.items())


class CacheUsage:
    """Turn 引用的模型调用用量；消息与 Turn 投影在每次调用时借用。"""

    def __init__(self, ctx: Context, store: ModelsStore, checkpoint: Path) -> None:
        self._ctx = ctx
        self._store = store
        self._lock = threading.Lock()
        self._checkpoint = _Checkpoint(checkpoint)
        self._restored = False
        self._heads: dict[str, int] = {}
        self._sources: dict[tuple[str, str], _SourceState] = {}
        self._dirty_heads: set[str] = set()
        self._dirty_sources: set[tuple[str, str]] = set()
        self._removed: set[str] = set()

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
            self._restore()
            # 无过滤目录保留软删 Session；按两种可见性读取才是当前有效集合。
            heads = dict(catalog.snapshot_heads(visibility="listed"))
            heads.update(catalog.snapshot_heads(visibility="internal"))
            self._removed.update(key for key in self._heads if key not in heads)
            self._removed.update(key[0] for key in self._sources if key[0] not in heads)
            self._heads = {key: head for key, head in self._heads.items() if key in heads}
            self._sources = {key: state for key, state in self._sources.items() if key[0] in heads}
            # 冷读在闭合 Turn 边界让出请求；下一次接着已有消费位置继续。
            deadline = time.monotonic() + _SCAN_SECONDS
            for session_id, head in heads.items():
                self._refresh(catalog, projection, session_id, head, deadline=deadline)
                if time.monotonic() >= deadline:
                    break
            scanned = sum(self._heads.get(session_id) == head for session_id, head in heads.items())
            self._flush()
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
            self._restore()
            self._refresh(catalog, projection, session_id, reader.head())
            self._flush()
            state = self._sources.get((session_id, message.source))
            return None if state is None else state.outputs.get(message_id)

    # 只扫描上次闭合 Turn 之后的消息；未闭合尾段留到下次。
    def _refresh(
        self, catalog: MessageCatalog, projection: TurnProjection, session_id: str, head: int,
        *, deadline: float | None = None,
    ) -> None:
        """从上次闭合位置读取；时间预算只在完整 Turn 边界生效。"""
        stored = self._heads.get(session_id)
        if stored == head:
            return
        # 账本回退到检查点之前（恢复备份或替换）：整个 Session 重算，已消失的来源一并清掉。
        if stored is not None and head < stored:
            self._drop_session(session_id)
        reader = catalog.reader(session_id)
        for source in sorted(reader.source_names()):
            state = self._sources.setdefault((session_id, source), _SourceState())
            # 账本被替换或回退到检查点之前：这个来源从头重算。
            if state.after_seq > head:
                state = self._sources[(session_id, source)] = _SourceState(reset=True)
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
            if turns or state.reset:
                self._dirty_sources.add((session_id, source))
            if stopped:
                return
        self._heads[session_id] = head
        self._dirty_heads.add(session_id)

    # 内存与检查点里都丢掉这个 Session，之后按新账本重建。
    def _drop_session(self, session_id: str) -> None:
        self._removed.add(session_id)
        _ = self._heads.pop(session_id, None)
        self._dirty_heads.discard(session_id)
        for key in [key for key in self._sources if key[0] == session_id]:
            del self._sources[key]
            self._dirty_sources.discard(key)

    # 首次访问时从检查点恢复；之后只在内存推进。
    def _restore(self) -> None:
        if self._restored:
            return
        self._restored = True
        self._heads, self._sources = self._checkpoint.load()

    # 写入本轮变化；失败只记日志，内存结果仍然有效，下次再写。
    def _flush(self) -> None:
        if not (self._dirty_heads or self._dirty_sources or self._removed):
            return
        sources = {key: self._sources[key] for key in self._dirty_sources if key in self._sources}
        try:
            self._checkpoint.save(
                {key: self._heads[key] for key in self._dirty_heads if key in self._heads}, sources, self._removed,
            )
        except (sqlite3.Error, OSError):
            logger.exception("cache usage checkpoint write failed")
            return
        for state in sources.values():
            state.unsaved.clear()
            state.reset = False
        self._dirty_heads.clear()
        self._dirty_sources.clear()
        self._removed.clear()


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
        state.outputs[outputs[-1][0]] = state.unsaved[outputs[-1][0]] = output
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
