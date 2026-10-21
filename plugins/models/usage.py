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
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from agent.plugin_composition import Context
from core.common.file_io import run_file_io
from plugins.ledger.contract import MESSAGE_CATALOG, ContentPart, Input, Message, MessageCatalog, Output
from plugins.turn_projection.contract import TURN_PROJECTION, Turn, TurnProjection

from .store import CallUsage, ModelsStore

_PROACTIVE_SOURCES = frozenset({"wake", "drift"})
_RECENT_TURNS = 50
_RECENT_PASSIVE_TURNS = 10
_PREVIEW_CHARS = 90
_SCAN_SECONDS = 2.0
_STATUS_SECONDS = 0.5
_HOUR_DAYS = 31
_INSIGHT_SESSIONS = 1000
# 检查点里存的是按当前计数规则算好的结果：改了计数规则（什么算计入、主动来源集合、预览、分桶）必须加一，旧结果随之丢弃重算。
_CHECKPOINT_SCHEMA = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Fact:
    """扫描时只留下 Turn 用量需要的少量字段，正文随读页释放。"""

    recorded_at: str
    text: str
    call_id: str | None
    output: bool
    # 这次请求所用压缩摘要的引用；没有摘要时为空。
    summary: str


@dataclass(frozen=True, slots=True)
class TurnUsage:
    session_id: str
    proactive: bool
    recorded_at: str
    preview: str
    prompt_tokens: int
    hit_tokens: int
    output_tokens: int
    calls: int
    # 最后一次调用的实报输入与绑定时的上下文窗口：上下文占用按它算，不是整轮累加。
    context_tokens: int
    context_window: int
    model: str
    compacted: bool


@dataclass(slots=True)
class _Totals:
    turns: int = 0
    prompt_tokens: int = 0
    hit_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, turn: TurnUsage) -> None:
        self.turns += 1
        self.prompt_tokens += turn.prompt_tokens
        self.hit_tokens += turn.hit_tokens
        self.output_tokens += turn.output_tokens
        self.calls += turn.calls

    def merge(self, other: _Totals) -> None:
        self.turns += other.turns
        self.prompt_tokens += other.prompt_tokens
        self.hit_tokens += other.hit_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def body(self) -> dict[str, object]:
        return {"turns": self.turns, "prompt_tokens": self.prompt_tokens, "hit_tokens": self.hit_tokens,
                "output_tokens": self.output_tokens, "calls": self.calls}

    def row(self) -> list[int]:
        return [self.turns, self.prompt_tokens, self.hit_tokens, self.output_tokens, self.calls]


@dataclass(slots=True)
class _SourceState:
    """一个 Session 来源已闭合 Turn 的累计量；由检查点跨重启保留。"""

    after_seq: int = -1
    totals: _Totals = field(default_factory=_Totals)
    recent: deque[TurnUsage] = field(default_factory=lambda: deque(maxlen=_RECENT_TURNS))
    # 按 UTC 小时与模型分桶（键 "YYYY-MM-DDTHH|模型"），只留最近 31 天；界面按本地时区再归天。
    buckets: dict[str, _Totals] = field(default_factory=dict[str, _Totals])
    first_preview: str = ""
    # 最近一次见到的摘要引用与累计压缩次数：沿用同一摘要的后续轮次不算新压缩。
    summary: str = ""
    compactions: int = 0
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
                for session_id, source, after, totals, recent, buckets in connection.execute(
                        "SELECT session_id, source, after_seq, totals, recent, buckets FROM sources"):
                    loaded = cast(dict[str, Any], json.loads(buckets))
                    sources[(session_id, source)] = _SourceState(
                        after, _Totals(*json.loads(totals)),
                        deque((TurnUsage(**item) for item in json.loads(recent)), maxlen=_RECENT_TURNS),
                        buckets={key: _Totals(*row) for key, row in loaded["buckets"].items()},
                        first_preview=loaded["first"], summary=loaded["summary"], compactions=loaded["compactions"])
                for session_id, source, message_id, tokens in connection.execute(
                        "SELECT session_id, source, message_id, tokens FROM outputs"):
                    state = sources.get((session_id, source))
                    if state is not None:
                        state.outputs[message_id] = tokens
            return heads, sources
        except (sqlite3.DatabaseError, ValueError, TypeError, KeyError) as error:
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
                    totals TEXT NOT NULL, recent TEXT NOT NULL, buckets TEXT NOT NULL,
                    PRIMARY KEY (session_id, source));
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
                buckets: dict[str, object] = {"buckets": {key: item.row() for key, item in state.buckets.items()},
                           "first": state.first_preview, "summary": state.summary, "compactions": state.compactions}
                _ = connection.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?, ?)", (
                    session_id, source, state.after_seq, json.dumps(state.totals.row()),
                    json.dumps([asdict(turn) for turn in state.recent], ensure_ascii=False),
                    json.dumps(buckets, ensure_ascii=False)))
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

    async def insights(self, since: str | None) -> dict[str, object]:
        """全部会话的小时桶、来源、模型与会话排行；since 之后的用量另算一列（界面传本地零点）。"""
        with self._ctx.borrow(MESSAGE_CATALOG) as catalog, self._ctx.borrow(TURN_PROJECTION) as projection:
            if catalog is None or projection is None:
                raise RuntimeError("消息账本或 Turn 投影不可用")
            return await run_file_io(lambda: self._insights(catalog, projection, since))

    async def status(self, session_id: str) -> dict[str, object] | None:
        """菜单短值：只在 0.5 秒预算内推进这个会话，冷的长会话先给已扫到的部分，下次接着扫。"""
        with self._ctx.borrow(MESSAGE_CATALOG) as catalog, self._ctx.borrow(TURN_PROJECTION) as projection:
            if catalog is None or projection is None:
                raise RuntimeError("消息账本或 Turn 投影不可用")
            return await run_file_io(lambda: self._status(catalog, projection, session_id))

    async def session(self, session_id: str) -> dict[str, object]:
        """一个会话的累计量与最近逐轮用量；只推进这个会话，不等全量扫描。"""
        with self._ctx.borrow(MESSAGE_CATALOG) as catalog, self._ctx.borrow(TURN_PROJECTION) as projection:
            if catalog is None or projection is None:
                raise RuntimeError("消息账本或 Turn 投影不可用")
            return await run_file_io(lambda: self._session(catalog, projection, session_id))

    # 汇总所有 Session；被动近期命中只取最近 10 个被动 Turn。
    def _overview(self, catalog: MessageCatalog, projection: TurnProjection) -> dict[str, object]:
        """按需推进扫描，所有当前 Session 完成后才返回完整统计。"""
        with self._lock:
            progress = self._advance(catalog, projection)
            if progress is not None:
                return progress
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

    # 1. 小时桶全量给出，界面按范围切；2. 来源与模型只合并 since 之后的桶；3. 会话按 since 之后的输入排行。
    def _insights(self, catalog: MessageCatalog, projection: TurnProjection, since: str | None) -> dict[str, object]:
        cutoff = _ceil_hour_key(since) if since else ""
        with self._lock:
            progress = self._advance(catalog, projection)
            if progress is not None:
                return progress
            hours: dict[str, _Totals] = {}
            sources: dict[str, _Totals] = {}
            models: dict[str, _Totals] = {}
            sessions: dict[str, list[tuple[str, _SourceState]]] = {}
            for (session_id, source), state in self._sources.items():
                if not state.totals.turns:
                    continue
                sessions.setdefault(session_id, []).append((source, state))
                for key, item in state.buckets.items():
                    hour, model = key.split("|", 1)
                    hours.setdefault(hour, _Totals()).merge(item)
                    if hour >= cutoff:
                        sources.setdefault(source, _Totals()).merge(item)
                        models.setdefault(model, _Totals()).merge(item)
            rows = [_session_row(session_id, owned, cutoff) for session_id, owned in sessions.items()]
            rows = [row for row in rows if not cutoff or cast(Mapping[str, int], row["recent"])["turns"]]
            rows.sort(key=lambda row: -cast(Mapping[str, int], row["recent" if cutoff else "totals"])["prompt_tokens"])
        return {
            "complete": True,
            "hours": [[key, *item.row()] for key, item in sorted(hours.items())],
            "sources": [{"source": key, **item.body()} for key, item in sorted(sources.items(), key=lambda e: -e[1].prompt_tokens)],
            "models": [{"model": key, **item.body()} for key, item in sorted(models.items(), key=lambda e: -e[1].prompt_tokens)],
            "sessions": _titled(catalog, rows[:_INSIGHT_SESSIONS]), "total_sessions": len(rows),
        }

    def _status(self, catalog: MessageCatalog, projection: TurnProjection, session_id: str) -> dict[str, object] | None:
        if not catalog.exists(session_id):
            return None
        reader = catalog.reader(session_id)
        if reader.deleted:
            return None
        with self._lock:
            self._restore()
            self._refresh(catalog, projection, session_id, reader.head(), deadline=time.monotonic() + _STATUS_SECONDS)
            self._flush()
            latest = _latest((state.recent for (key, _), state in self._sources.items() if key == session_id), 1)
            totals = _Totals()
            for (key, _), state in self._sources.items():
                if key == session_id:
                    totals.merge(state.totals)
        if not latest:
            return None
        return {"context_tokens": latest[0].context_tokens, "context_window": latest[0].context_window,
                "prompt_tokens": totals.prompt_tokens}

    # 只推进这一个会话；逐轮用量取各来源最近 50 轮按时间合并。
    def _session(self, catalog: MessageCatalog, projection: TurnProjection, session_id: str) -> dict[str, object]:
        if not catalog.exists(session_id):
            return {"session": None, "turns": []}
        reader = catalog.reader(session_id)
        if reader.deleted:
            return {"session": None, "turns": []}
        with self._lock:
            self._restore()
            self._refresh(catalog, projection, session_id, reader.head())
            self._flush()
            owned = [(source, state) for (key, source), state in self._sources.items() if key == session_id and state.totals.turns]
            if not owned:
                return {"session": None, "turns": []}
            turns = sorted(_latest((state.recent for _, state in owned), _RECENT_TURNS), key=lambda turn: turn.recorded_at)
            row = _titled(catalog, [_session_row(session_id, owned, "")])[0]
        return {"session": row, "turns": [asdict(turn) for turn in turns]}

    # 1. 丢掉已删除的 Session；2. 在时间预算内推进扫描；未扫完时返回进度，扫完返回 None。
    def _advance(self, catalog: MessageCatalog, projection: TurnProjection) -> dict[str, object] | None:
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
        return None

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
# 3. 小时桶与模型桶跟累计量同一口径；上下文占用取最后一次调用的输入。
def _record(
    state: _SourceState, session_id: str, source: str, turn: Turn,
    facts: Mapping[str, _Fact], usages: Mapping[str, CallUsage],
) -> None:
    state.after_seq = turn.through_seq
    members = [(identity, facts[identity]) for identity in turn.message_ids]
    outputs = [(identity, fact) for identity, fact in members if fact.output]
    calls = [usages.get(fact.call_id) for _, fact in outputs if fact.call_id is not None]
    known = [call for call in calls if call is not None and call.usage is not None]
    exact = [call.usage for call in known if call.usage is not None and call.usage.get("coverage") == "exact"]
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
    last, context = known[-1], exact[-1].get("input_tokens")
    summaries = [fact.summary for _, fact in members if fact.summary]
    compacted = bool(summaries) and summaries[-1] != state.summary
    usage = TurnUsage(
        session_id, source in _PROACTIVE_SOURCES, final.recorded_at, preview, prompt, hit or 0, output or 0,
        len(calls), context if type(context) is int else 0, last.context_window, last.model, compacted,
    )
    if summaries:
        state.summary = summaries[-1]
    state.compactions += compacted
    state.totals.add(usage)
    state.recent.append(usage)
    state.first_preview = state.first_preview or preview
    state.buckets.setdefault(f"{_hour_key(final.recorded_at)}|{last.model or '未知模型'}", _Totals()).add(usage)
    oldest = _hour_key((datetime.now(UTC) - timedelta(days=_HOUR_DAYS)).isoformat())
    for key in [key for key in state.buckets if key < oldest]:
        del state.buckets[key]


def _tap(rows: Iterable[Message], facts: dict[str, _Fact]) -> Iterator[Message]:
    for message in rows:
        body = message.body
        call_id: str | None = None
        summary = ""
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
                if isinstance(part, ContentPart) and part.kind == "context.summary":
                    held: object = part.value
                    reference = cast(Mapping[str, object], held).get("reference") if isinstance(held, Mapping) else None
                    summary = json.dumps(reference, sort_keys=True, default=str) if reference is not None else "summary"
        facts[message.message_id] = _Fact(
            message.recorded_at.isoformat(), text[:_PREVIEW_CHARS * 2], call_id, isinstance(body, Output), summary,
        )
        yield message


# 会话行：来源合并后的累计量、since 之后的用量，以及最近一轮的上下文占用；标题另由 _titled 补上。
def _session_row(session_id: str, owned: list[tuple[str, _SourceState]], cutoff: str) -> dict[str, object]:
    totals, recent = _Totals(), _Totals()
    for _, state in owned:
        totals.merge(state.totals)
        for key, item in state.buckets.items():
            if cutoff and key >= cutoff:
                recent.merge(item)
    latest = _latest((state.recent for _, state in owned), 1)[0]
    first = min(owned, key=lambda entry: entry[1].recent[0].recorded_at if entry[1].recent else "")[1].first_preview
    return {
        "session_id": session_id, "title": first,
        "sources": sorted(source for source, _ in owned), "last_at": latest.recorded_at,
        "context_tokens": latest.context_tokens, "context_window": latest.context_window, "model": latest.model,
        "compactions": sum(state.compactions for _, state in owned),
        "recent": recent.body(), "totals": totals.body(),
    }


# 只给要返回的行读标题：每个标题是一次账本查询，不能在排序截断前对全部会话做。
def _titled(catalog: MessageCatalog, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    for row in rows:
        row["title"] = catalog.reader(cast(str, row["session_id"])).title or row["title"]
    return rows


# 起点向上取整到整点：半小时时区的零点落在桶中间，那一小时整体归前一天，与界面按桶起点归属一致。
def _ceil_hour_key(value: str) -> str:
    moment = datetime.fromisoformat(value)
    moment = (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).astimezone(UTC)
    floor = moment.replace(minute=0, second=0, microsecond=0)
    return (floor if floor == moment else floor + timedelta(hours=1)).strftime("%Y-%m-%dT%H")


def _hour_key(value: str) -> str:
    moment = datetime.fromisoformat(value)
    return (moment if moment.tzinfo else moment.replace(tzinfo=UTC)).astimezone(UTC).strftime("%Y-%m-%dT%H")


def _sum(records: list[Mapping[str, object]], name: str) -> int | None:
    values = [record.get(name) for record in records]
    return sum(value for value in values if type(value) is int) if all(type(v) is int for v in values) else None


def _latest(groups: Iterable[Iterable[TurnUsage]], limit: int) -> list[TurnUsage]:
    return heapq.nlargest(limit, (turn for group in groups for turn in group), key=lambda turn: turn.recorded_at)
